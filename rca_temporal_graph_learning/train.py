import json
import os
import random
from collections import Counter, defaultdict
from contextlib import nullcontext

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from dataset2 import RCAGraphSequenceDataset
from model import GATGRUMultiTask


SEQ_LEN = 5
WINDOW_SEC = 2
VAL_RATIO = 0.2
MAX_PER_Y1 = 700 #3000
BATCH_SIZE = 64
SERVICE_LOSS_WEIGHT =2.5
EPOCHS = 250
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def save_preprocessing_artifacts(dataset, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)

    node_scaler_path = os.path.join(save_dir, "node_scaler.pkl")
    edge_scaler_path = os.path.join(save_dir, "edge_scaler.pkl")
    service_map_path = os.path.join(save_dir, "service_to_idx.json")
    failure_map_path = os.path.join(save_dir, "failure_to_idx.json")

    if dataset.node_scaler is not None:
        joblib.dump(dataset.node_scaler, node_scaler_path)
        print(f"Node scaler sauvegarde : {node_scaler_path}")
    else:
        print("Node scaler absent, rien a sauvegarder.")

    if dataset.edge_scaler is not None:
        joblib.dump(dataset.edge_scaler, edge_scaler_path)
        print(f"Edge scaler sauvegarde : {edge_scaler_path}")
    else:
        print("Edge scaler absent, rien a sauvegarder.")

    with open(service_map_path, "w", encoding="utf-8") as f:
        json.dump(dataset.service_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Mapping services sauvegarde : {service_map_path}")

    with open(failure_map_path, "w", encoding="utf-8") as f:
        json.dump(dataset.failure_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Mapping failures sauvegarde : {failure_map_path}")


def collate_graph_sequences(batch):
    sequences = []
    labels = []

    for seq_graphs, y in batch:
        sequences.append(seq_graphs)
        labels.append(y)

    labels = torch.stack(labels, dim=0)
    seq_len = len(sequences[0])
    batched_seq = []

    for t in range(seq_len):
        graphs_at_t = [seq[t] for seq in sequences]
        batched_seq.append(Batch.from_data_list(graphs_at_t))

    return batched_seq, labels


def get_dataset_labels(dataset):
    labels = getattr(dataset, "sample_labels", None)
    if isinstance(labels, torch.Tensor) and labels.ndim == 2:
        return labels

    if len(dataset) == 0:
        return torch.empty((0, 2), dtype=torch.long)

    return torch.stack([dataset[idx][1] for idx in range(len(dataset))], dim=0)


def show_label_distribution(dataset, indices, name):
    if not indices:
        print(f"\n{name} distribution: empty")
        return

    labels = get_dataset_labels(dataset)[indices]
    y1_list = labels[:, 0].tolist()
    y2_list = labels[:, 1].tolist()

    print(f"\n{name} service distribution: {Counter(y1_list)}")
    print(f"{name} failure distribution: {Counter(y2_list)}")


def build_balanced_subset_indices_y1(dataset, max_per_y1=600, seed=42, indices=None):
    rng = random.Random(seed)
    indices_by_y1 = defaultdict(list)

    labels = get_dataset_labels(dataset)
    source_indices = list(range(len(dataset))) if indices is None else list(indices)
    for idx in source_indices:
        y1 = int(labels[idx, 0].item())
        indices_by_y1[y1].append(idx)

    balanced_indices = []

    print("\n=== Construction du sous-dataset equilibre sur y1 ===")
    for y1_class in sorted(indices_by_y1.keys()):
        cls_indices = indices_by_y1[y1_class]
        n_available = len(cls_indices)
        n_take = min(max_per_y1, n_available)

        rng.shuffle(cls_indices)
        selected = cls_indices[:n_take]
        balanced_indices.extend(selected)

        print(f"Classe y1={y1_class} | disponible={n_available} | retenu={n_take}")

    rng.shuffle(balanced_indices)
    return balanced_indices


def build_stratified_subset_indices_y1_by_size(dataset, target_size, seed=42):
    rng = random.Random(seed)
    indices_by_y1 = defaultdict(list)
    labels = get_dataset_labels(dataset)

    for idx, y1 in enumerate(labels[:, 0].tolist()):
        indices_by_y1[y1].append(idx)

    total_available = sum(len(class_indices) for class_indices in indices_by_y1.values())
    if total_available == 0:
        return []

    target_size = min(max(int(target_size), 0), total_available)
    if target_size == 0:
        return []

    print("\n=== Construction du sous-dataset validation stratifie sur y1 ===")

    quotas = {}
    remainders = []
    allocated = 0

    for y1_class in sorted(indices_by_y1.keys()):
        class_count = len(indices_by_y1[y1_class])
        raw_quota = target_size * class_count / total_available
        quota = min(class_count, int(np.floor(raw_quota)))
        quotas[y1_class] = quota
        allocated += quota
        remainders.append((raw_quota - quota, y1_class))

    remaining = target_size - allocated
    for _, y1_class in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if quotas[y1_class] < len(indices_by_y1[y1_class]):
            quotas[y1_class] += 1
            remaining -= 1

    selected_indices = []
    for y1_class in sorted(indices_by_y1.keys()):
        cls_indices = indices_by_y1[y1_class][:]
        rng.shuffle(cls_indices)
        selected = cls_indices[:quotas[y1_class]]
        selected_indices.extend(selected)

        print(
            f"Classe y1={y1_class} | disponible={len(cls_indices)} | "
            f"retenu={len(selected)}"
        )

    rng.shuffle(selected_indices)
    return selected_indices


def compute_class_weights_from_indices(
    dataset,
    indices,
    num_classes,
    label_col=0,
    power=0.5,
    max_weight=4.0,
):
    labels = get_dataset_labels(dataset)
    counts = Counter(int(labels[idx, label_col].item()) for idx in indices)

    weights = torch.ones(num_classes, dtype=torch.float32)
    total = sum(counts.values())

    if total == 0:
        return weights

    for class_idx in range(num_classes):
        class_count = counts.get(class_idx, 0)
        if class_count > 0:
            base_weight = total / (num_classes * class_count)
            weights[class_idx] = float(base_weight) ** power
        else:
            weights[class_idx] = 0.0

    positive_mask = weights > 0
    if positive_mask.any():
        if max_weight is not None:
            weights[positive_mask] = torch.clamp(weights[positive_mask], max=max_weight)
        weights[positive_mask] = weights[positive_mask] / weights[positive_mask].mean()

    return weights


def build_loader_kwargs(device):
    num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    kwargs = {
        "collate_fn": collate_graph_sequences,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def run_one_epoch(
    model,
    loader,
    device,
    optimizer=None,
    service_class_weights=None,
    use_amp=False,
    amp_dtype=torch.float16,
    scaler=None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_y1_true = []
    all_y1_pred = []
    all_y2_true = []
    all_y2_pred = []
    non_blocking = device.type == "cuda"

    context = torch.enable_grad if is_train else torch.inference_mode
    with context():
        for batch_sequences, batch_y in loader:
            batch_sequences = [
                batch_t.to(device, non_blocking=non_blocking) for batch_t in batch_sequences
            ]
            batch_y = batch_y.to(device, non_blocking=non_blocking)

            y1 = batch_y[:, 0]
            y2 = batch_y[:, 1]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                service_logits, failure_logits = model(batch_sequences)

                loss1 = F.cross_entropy(service_logits, y1, weight=service_class_weights)
                loss2 = F.cross_entropy(failure_logits, y2)
                loss = SERVICE_LOSS_WEIGHT * loss1 + loss2

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach())

            pred_y1 = service_logits.argmax(dim=1)
            pred_y2 = failure_logits.argmax(dim=1)

            all_y1_true.extend(y1.detach().cpu().tolist())
            all_y1_pred.extend(pred_y1.detach().cpu().tolist())
            all_y2_true.extend(y2.detach().cpu().tolist())
            all_y2_pred.extend(pred_y2.detach().cpu().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    service_acc = accuracy_score(all_y1_true, all_y1_pred)
    failure_acc = accuracy_score(all_y2_true, all_y2_pred)
    service_f1 = f1_score(all_y1_true, all_y1_pred, average="macro", zero_division=0)
    failure_f1 = f1_score(all_y2_true, all_y2_pred, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "service_acc": service_acc,
        "failure_acc": failure_acc,
        "service_f1": service_f1,
        "failure_f1": failure_f1,
    }


def main():
    set_seed(SEED)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Script directory:", script_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = RCAGraphSequenceDataset(
        comm_csv="aggregated_pod_communication.csv",
        res_csv="aggregated_pod_resource_consumption.csv",
        events_csv="aggregated_pod_events.csv",
        seq_len=SEQ_LEN,
        window_sec=WINDOW_SEC,
        drop_services=["unknown"],
        fit_scaler=True,
        cache_graphs=True,
    )

    print(f"Train dataset total : {len(train_dataset)}")

    save_preprocessing_artifacts(train_dataset, save_dir=script_dir)

    balanced_indices = build_balanced_subset_indices_y1(
        train_dataset,
        max_per_y1=MAX_PER_Y1,
        seed=SEED,
    )
    if not balanced_indices:
        raise ValueError("Balanced train subset is empty.")

    validation_dir = os.path.join(script_dir, "tester_model")
    val_dataset_full = RCAGraphSequenceDataset(
        comm_csv=os.path.join(validation_dir, "aggregated_pod_communication.csv"),
        res_csv=os.path.join(validation_dir, "aggregated_pod_resource_consumption.csv"),
        events_csv=os.path.join(validation_dir, "aggregated_pod_events.csv"),
        seq_len=SEQ_LEN,
        window_sec=WINDOW_SEC,
        drop_services=["unknown"],
        fit_scaler=False,
        node_scaler=train_dataset.node_scaler,
        edge_scaler=train_dataset.edge_scaler,
        cache_graphs=True,
        service_to_idx=train_dataset.service_to_idx,
        failure_to_idx=train_dataset.failure_to_idx,
    )
    if len(val_dataset_full) == 0:
        raise ValueError("Validation dataset built from tester_model is empty.")

    target_val_size = max(1, int(round(VAL_RATIO * len(balanced_indices))))
    val_indices = build_stratified_subset_indices_y1_by_size(
        val_dataset_full,
        target_size=target_val_size,
        seed=SEED,
    )
    if not val_indices:
        raise ValueError("Validation subset is empty.")

    print(f"\nTaille du dataset train equilibre : {len(balanced_indices)}")
    print(f"Taille du dataset validation retenu : {len(val_indices)}")
    show_label_distribution(train_dataset, balanced_indices, "TRAIN BALANCED")
    show_label_distribution(val_dataset_full, val_indices, "VALID")

    service_class_weights = compute_class_weights_from_indices(
        train_dataset,
        balanced_indices,
        num_classes=len(train_dataset.all_services),
        label_col=0,
        power=0.5,
        max_weight=4.0,
    ).to(device)
    non_zero_service_weights = service_class_weights[service_class_weights > 0]
    if len(non_zero_service_weights) > 0:
        print(
            "Service class weights | "
            f"min={non_zero_service_weights.min().item():.4f} | "
            f"max={non_zero_service_weights.max().item():.4f}"
        )

    balanced_dataset = Subset(train_dataset, balanced_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    loader_kwargs = build_loader_kwargs(device)

    train_loader = DataLoader(
        balanced_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    model = GATGRUMultiTask(
        num_graph_nodes=len(train_dataset.all_services),
        node_in_dim=len(train_dataset.node_feature_cols),
        edge_dim=len(train_dataset.edge_feature_cols),
        num_service_classes=len(train_dataset.all_services),
        num_failure_classes=len(train_dataset.all_failures),
        gat_hidden_dim=64,
        gru_hidden_dim=128,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    use_amp = device.type == "cuda"
    supports_bf16 = bool(
        use_amp and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp and amp_dtype == torch.float16,
        )
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    print(f"AMP enabled: {use_amp} | dtype={amp_dtype}")

    model_path = os.path.join(script_dir, "model_balanced_y1_batched.pt")
    best_checkpoint_path = os.path.join(script_dir, "best_service_f1_checkpoint.pt")
    best_service_f1 = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = None

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_one_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            service_class_weights=service_class_weights,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            service_class_weights=None,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_service_acc={train_metrics['service_acc']:.4f} | "
            f"train_failure_acc={train_metrics['failure_acc']:.4f} | "
            f"train_service_f1={train_metrics['service_f1']:.4f} | "
            f"train_failure_f1={train_metrics['failure_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_service_acc={val_metrics['service_acc']:.4f} | "
            f"val_failure_acc={val_metrics['failure_acc']:.4f} | "
            f"val_service_f1={val_metrics['service_f1']:.4f} | "
            f"val_failure_f1={val_metrics['failure_f1']:.4f}"
        )

        current_service_f1 = float(val_metrics["service_f1"])
        current_val_loss = float(val_metrics["loss"])
        is_better_checkpoint = (
            current_service_f1 > best_service_f1
            or (
                np.isclose(current_service_f1, best_service_f1)
                and current_val_loss < best_val_loss
            )
        )

        if is_better_checkpoint:
            best_service_f1 = current_service_f1
            best_val_loss = current_val_loss
            best_epoch = epoch
            best_metrics = {
                "train": dict(train_metrics),
                "val": dict(val_metrics),
            }

            torch.save(model.state_dict(), model_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "best_metrics": best_metrics,
                    "config": {
                        "seq_len": SEQ_LEN,
                        "window_sec": WINDOW_SEC,
                        "batch_size": BATCH_SIZE,
                        "service_loss_weight": SERVICE_LOSS_WEIGHT,
                        "epochs": EPOCHS,
                        "seed": SEED,
                    },
                },
                best_checkpoint_path,
            )
            print(
                f"  -> Nouveau meilleur checkpoint | epoch={epoch:03d} | "
                f"val_service_f1={current_service_f1:.4f} | "
                f"val_loss={current_val_loss:.4f}"
            )

    if best_metrics is None:
        raise RuntimeError("Aucun checkpoint n'a pu etre sauvegarde.")

    print(
        f"\nMeilleur checkpoint : epoch {best_epoch:03d} | "
        f"val_service_f1={best_service_f1:.4f} | "
        f"val_loss={best_val_loss:.4f}"
    )
    print(f"Poids du meilleur modele sauvegardes : {model_path}")
    print(f"Checkpoint complet sauvegarde : {best_checkpoint_path}")


if __name__ == "__main__":
    main()
