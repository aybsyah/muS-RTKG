import os
import sys
import json
import joblib
import pickle
import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================================================
# chemins
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PARENT_DIR)

from dataset2 import RCAGraphSequenceDataset
from model import GATGRUMultiTask


# =========================================================
# config
# =========================================================
COMM_CSV = os.path.join(SCRIPT_DIR, "tester_model", "aggregated_pod_communication.csv")
RES_CSV = os.path.join(SCRIPT_DIR, "tester_model", "aggregated_pod_resource_consumption.csv")
EVENTS_CSV = os.path.join(SCRIPT_DIR, "tester_model", "aggregated_pod_events.csv")

#MODEL_PATH = os.path.join(SCRIPT_DIR, "model_balanced_y1_batched.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_service_f1_checkpoint.pt")


NODE_SCALER_PATH = os.path.join(SCRIPT_DIR, "node_scaler.pkl")
EDGE_SCALER_PATH = os.path.join(SCRIPT_DIR, "edge_scaler.pkl")
SERVICE_MAP_PATH = os.path.join(SCRIPT_DIR, "service_to_idx.json")
FAILURE_MAP_PATH = os.path.join(SCRIPT_DIR, "failure_to_idx.json")

OUTPUT_PRED_CSV = os.path.join(SCRIPT_DIR, "predictions_debug_batched.csv")

SEQ_LEN = 5
WINDOW_SEC = 2
BATCH_SIZE = 64


# =========================================================
# collate batché PyG
# =========================================================
def collate_graph_sequences(batch):
    sequences = []
    labels = []

    for seq_graphs, y in batch:
        sequences.append(seq_graphs)
        labels.append(y)

    labels = torch.stack(labels, dim=0)   # [B, 2]

    seq_len = len(sequences[0])
    batched_seq = []

    for t in range(seq_len):
        graphs_at_t = [seq[t] for seq in sequences]
        batch_t = Batch.from_data_list(graphs_at_t)
        batched_seq.append(batch_t)

    return batched_seq, labels


# =========================================================
# utils mappings
# =========================================================
def load_json_mapping(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    mapping = {str(k): int(v) for k, v in mapping.items()}
    idx_to_name = {v: k for k, v in mapping.items()}
    return mapping, idx_to_name


def ordered_names_from_idx(idx_to_name):
    return [idx_to_name[i] for i in sorted(idx_to_name.keys())]


def safe_name(idx_to_name, idx):
    return idx_to_name.get(int(idx), f"<inconnu:{idx}>")


def load_model_state_dict(model_path, device):
    checkpoint = None
    used_unsafe_fallback = False

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )
        used_unsafe_fallback = True

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint, used_unsafe_fallback

    if isinstance(checkpoint, dict):
        return checkpoint, None, used_unsafe_fallback

    raise ValueError(f"Format de checkpoint non supporte pour : {model_path}")


# =========================================================
# top-k
# =========================================================
def top_k_accuracy_from_probs(probs, true_labels, k=3):
    """
    probs: tensor [N, C]
    true_labels: tensor [N]
    """
    if probs.numel() == 0 or len(true_labels) == 0:
        return 0.0

    num_classes = probs.size(1)
    k = min(k, num_classes)

    topk_indices = torch.topk(probs, k=k, dim=1).indices  # [N, k]
    true_labels = true_labels.view(-1, 1)                 # [N, 1]

    correct = (topk_indices == true_labels).any(dim=1).float()
    return correct.mean().item()


# =========================================================
# evaluation
# =========================================================
@torch.no_grad()
def evaluate_model(model, loader, device, idx_to_service, idx_to_failure):
    model.eval()

    all_y1_true = []
    all_y1_pred = []
    all_y2_true = []
    all_y2_pred = []

    all_service_probs = []
    all_failure_probs = []

    debug_rows = []
    sample_id = 0

    for batch_sequences, batch_y in loader:
        # batch_sequences = [Batch_t0, Batch_t1, ..., Batch_t(T-1)]
        batch_sequences = [batch_t.to(device) for batch_t in batch_sequences]
        batch_y = batch_y.to(device)

        y1 = batch_y[:, 0]
        y2 = batch_y[:, 1]

        service_logits, failure_logits = model(batch_sequences)

        service_probs = torch.softmax(service_logits, dim=1)
        failure_probs = torch.softmax(failure_logits, dim=1)

        pred_y1 = service_logits.argmax(dim=1)
        pred_y2 = failure_logits.argmax(dim=1)

        all_y1_true.extend(y1.cpu().tolist())
        all_y1_pred.extend(pred_y1.cpu().tolist())
        all_y2_true.extend(y2.cpu().tolist())
        all_y2_pred.extend(pred_y2.cpu().tolist())

        all_service_probs.append(service_probs.cpu())
        all_failure_probs.append(failure_probs.cpu())

        # top-3 / top-5 indices for debug
        service_top3_idx = torch.topk(service_probs, k=min(3, service_probs.size(1)), dim=1).indices
        service_top5_idx = torch.topk(service_probs, k=min(5, service_probs.size(1)), dim=1).indices
        failure_top3_idx = torch.topk(failure_probs, k=min(3, failure_probs.size(1)), dim=1).indices
        failure_top5_idx = torch.topk(failure_probs, k=min(5, failure_probs.size(1)), dim=1).indices

        for i in range(len(pred_y1)):
            ts = int(y1[i].item())
            tf = int(y2[i].item())
            ps = int(pred_y1[i].item())
            pf = int(pred_y2[i].item())

            s_top3 = service_top3_idx[i].cpu().tolist()
            s_top5 = service_top5_idx[i].cpu().tolist()
            f_top3 = failure_top3_idx[i].cpu().tolist()
            f_top5 = failure_top5_idx[i].cpu().tolist()

            row = {
                "sample_id": sample_id,

                "true_service_idx": ts,
                "true_service_name": safe_name(idx_to_service, ts),
                "pred_service_idx": ps,
                "pred_service_name": safe_name(idx_to_service, ps),
                "pred_service_prob": float(service_probs[i, ps].item()),

                "service_top3_idx": s_top3,
                "service_top3_names": [safe_name(idx_to_service, x) for x in s_top3],
                "service_top5_idx": s_top5,
                "service_top5_names": [safe_name(idx_to_service, x) for x in s_top5],

                "true_failure_idx": tf,
                "true_failure_name": safe_name(idx_to_failure, tf),
                "pred_failure_idx": pf,
                "pred_failure_name": safe_name(idx_to_failure, pf),
                "pred_failure_prob": float(failure_probs[i, pf].item()),

                "failure_top3_idx": f_top3,
                "failure_top3_names": [safe_name(idx_to_failure, x) for x in f_top3],
                "failure_top5_idx": f_top5,
                "failure_top5_names": [safe_name(idx_to_failure, x) for x in f_top5],
            }
            debug_rows.append(row)

            print("-" * 90)
            print(f"Échantillon {sample_id:04d}")
            print(f"Vrai service   : {row['true_service_name']} ({ts})")
            print(f"Prédit service : {row['pred_service_name']} ({ps}) | proba={row['pred_service_prob']:.4f}")
            print(f"Top-3 service  : {row['service_top3_names']}")
            print(f"Top-5 service  : {row['service_top5_names']}")

            print(f"Vraie panne    : {row['true_failure_name']} ({tf})")
            print(f"Panne prédite  : {row['pred_failure_name']} ({pf}) | proba={row['pred_failure_prob']:.4f}")
            print(f"Top-3 panne    : {row['failure_top3_names']}")
            print(f"Top-5 panne    : {row['failure_top5_names']}")

            sample_id += 1

    all_service_probs = torch.cat(all_service_probs, dim=0)
    all_failure_probs = torch.cat(all_failure_probs, dim=0)

    y1_true_tensor = torch.tensor(all_y1_true, dtype=torch.long)
    y2_true_tensor = torch.tensor(all_y2_true, dtype=torch.long)

    service_acc = accuracy_score(all_y1_true, all_y1_pred)
    failure_acc = accuracy_score(all_y2_true, all_y2_pred)

    service_f1 = f1_score(all_y1_true, all_y1_pred, average="macro", zero_division=0)
    failure_f1 = f1_score(all_y2_true, all_y2_pred, average="macro", zero_division=0)

    service_top3 = top_k_accuracy_from_probs(all_service_probs, y1_true_tensor, k=3)
    service_top5 = top_k_accuracy_from_probs(all_service_probs, y1_true_tensor, k=5)

    failure_top3 = top_k_accuracy_from_probs(all_failure_probs, y2_true_tensor, k=3)
    failure_top5 = top_k_accuracy_from_probs(all_failure_probs, y2_true_tensor, k=5)

    return {
        "service_acc": service_acc,
        "failure_acc": failure_acc,
        "service_f1": service_f1,
        "failure_f1": failure_f1,
        "service_top3": service_top3,
        "service_top5": service_top5,
        "failure_top3": failure_top3,
        "failure_top5": failure_top5,
        "y1_true": all_y1_true,
        "y1_pred": all_y1_pred,
        "y2_true": all_y2_true,
        "y2_pred": all_y2_pred,
        "debug_df": pd.DataFrame(debug_rows),
    }


# =========================================================
# main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)
    print("Script directory :", SCRIPT_DIR)

    # -----------------------------------------------------
    # charger scalers
    # -----------------------------------------------------
    node_scaler = joblib.load(NODE_SCALER_PATH)
    edge_scaler = joblib.load(EDGE_SCALER_PATH)
    print("Scalers chargés.")

    # -----------------------------------------------------
    # charger mappings du train
    # -----------------------------------------------------
    service_to_idx, idx_to_service = load_json_mapping(SERVICE_MAP_PATH)
    failure_to_idx, idx_to_failure = load_json_mapping(FAILURE_MAP_PATH)
    print("Mappings chargés.")

    print("\n===== Mapping SERVICE du train =====")
    print(service_to_idx)

    print("\n===== Mapping FAILURE du train =====")
    print(failure_to_idx)

    # -----------------------------------------------------
    # charger dataset
    # -----------------------------------------------------
    dataset = RCAGraphSequenceDataset(
        comm_csv=COMM_CSV,
        res_csv=RES_CSV,
        events_csv=EVENTS_CSV,
        seq_len=SEQ_LEN,
        window_sec=WINDOW_SEC,
        drop_services=["unknown"],
        fit_scaler=False,
        node_scaler=node_scaler,
        edge_scaler=edge_scaler,
        service_to_idx=service_to_idx,
        failure_to_idx=failure_to_idx,
    )

    print(f"\nDataset size : {len(dataset)}")

    print("\n===== Mapping SERVICE du dataset courant =====")
    print(dataset.service_to_idx)

    print("\n===== Mapping FAILURE du dataset courant =====")
    print(dataset.failure_to_idx)

    if len(service_to_idx) != len(dataset.all_services):
        print("\n[ATTENTION] Nombre de classes service différent.")
        print(f"Train : {len(service_to_idx)} | Courant : {len(dataset.all_services)}")

    if len(failure_to_idx) != len(dataset.all_failures):
        print("\n[ATTENTION] Nombre de classes failure différent.")
        print(f"Train : {len(failure_to_idx)} | Courant : {len(dataset.all_failures)}")

    # -----------------------------------------------------
    # dataloader
    # -----------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_graph_sequences,
        num_workers=2
    )

    # -----------------------------------------------------
    # recréer le modèle
    # -----------------------------------------------------
    model = GATGRUMultiTask(
        num_graph_nodes=len(service_to_idx),
        node_in_dim=len(dataset.node_feature_cols),
        edge_dim=len(dataset.edge_feature_cols),
        num_service_classes=len(service_to_idx),
        num_failure_classes=len(failure_to_idx),
        gat_hidden_dim=64,
        gru_hidden_dim=128,
        dropout=0.2,
    ).to(device)

    # -----------------------------------------------------
    # charger poids
    # -----------------------------------------------------
    state_dict, checkpoint_meta, used_unsafe_fallback = load_model_state_dict(
        MODEL_PATH,
        device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\nModèle chargé depuis : {MODEL_PATH}")
    if used_unsafe_fallback:
        print("Chargement avec weights_only=False pour compatibilite PyTorch 2.6+.")
    if checkpoint_meta is not None and "epoch" in checkpoint_meta:
        print(f"Checkpoint epoch : {checkpoint_meta['epoch']}")
    if checkpoint_meta is not None and "best_metrics" in checkpoint_meta:
        best_val_metrics = checkpoint_meta["best_metrics"].get("val", {})
        if "service_f1" in best_val_metrics:
            print(f"Checkpoint val_service_f1 : {best_val_metrics['service_f1']:.4f}")

    # -----------------------------------------------------
    # évaluation
    # -----------------------------------------------------
    results = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        idx_to_service=idx_to_service,
        idx_to_failure=idx_to_failure,
    )

    # -----------------------------------------------------
    # affichage métriques
    # -----------------------------------------------------
    print("\n===== TEST RESULTS =====")
    print(f"Service accuracy  : {results['service_acc']:.4f}")
    print(f"Failure accuracy  : {results['failure_acc']:.4f}")
    print(f"Service F1 macro  : {results['service_f1']:.4f}")
    print(f"Failure F1 macro  : {results['failure_f1']:.4f}")

    print("\n===== TOP-K METRICS =====")
    print(f"Service Top-3 accuracy : {results['service_top3']:.4f}")
    print(f"Service Top-5 accuracy : {results['service_top5']:.4f}")
    print(f"Failure Top-3 accuracy : {results['failure_top3']:.4f}")
    print(f"Failure Top-5 accuracy : {results['failure_top5']:.4f}")

    service_names = ordered_names_from_idx(idx_to_service)
    failure_names = ordered_names_from_idx(idx_to_failure)

    print("\n===== SERVICE REPORT =====")
    print(
        classification_report(
            results["y1_true"],
            results["y1_pred"],
            labels=list(range(len(service_names))),
            target_names=service_names,
            zero_division=0,
        )
    )

    print("\n===== FAILURE REPORT =====")
    print(
        classification_report(
            results["y2_true"],
            results["y2_pred"],
            labels=list(range(len(failure_names))),
            target_names=failure_names,
            zero_division=0,
        )
    )

    print("\n===== CONFUSION MATRIX SERVICE =====")
    print(
        confusion_matrix(
            results["y1_true"],
            results["y1_pred"],
            labels=list(range(len(service_names)))
        )
    )

    print("\n===== CONFUSION MATRIX FAILURE =====")
    print(
        confusion_matrix(
            results["y2_true"],
            results["y2_pred"],
            labels=list(range(len(failure_names)))
        )
    )

    # -----------------------------------------------------
    # sauvegarde csv debug
    # -----------------------------------------------------
    results["debug_df"].to_csv(OUTPUT_PRED_CSV, index=False, encoding="utf-8-sig")
    print(f"\nRésultats sauvegardés dans : {OUTPUT_PRED_CSV}")

    print("\n===== Distribution des services prédits =====")
    print(results["debug_df"]["pred_service_name"].value_counts())

    print("\n===== Distribution des pannes prédites =====")
    print(results["debug_df"]["pred_failure_name"].value_counts())


if __name__ == "__main__":
    main()
