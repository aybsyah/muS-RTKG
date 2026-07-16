#!/usr/bin/env python3
"""
Build Runtime Temporal Knowledge Graph (RTKG) JSON snapshots from:
  1) aggregated_pod_resource_consumption.csv
  2) aggregated_pod_communication.csv
  3) aggregated_pod_events.csv

Each output JSON snapshot contains:
  - microservice resource-consumption nodes
  - worker-node deployment relations
  - communication edges
  - runtime event nodes and event-to-pod relations

Example:
python build_rtkg_snapshots.py \
  --resource_csv aggregated_pod_resource_consumption.csv \
  --communication_csv aggregated_pod_communication.csv \
  --events_csv aggregated_pod_events.csv \
  --output_dir rtkg_snapshots \
  --window 15s
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

EXPERIMENT_COLS = ["namespace", "experiment_id", "run_ts", "failure_type", "target_service"]


def parse_window_seconds(text):
    text = str(text).strip().lower()
    m = re.fullmatch(r"(\d+)\s*(s|sec|secs|second|seconds|min|mins|minute|minutes|m)", text)
    if not m:
        raise ValueError("Invalid --window. Use examples: 1s, 15s, 30s, 1min")
    value = int(m.group(1))
    unit = m.group(2)
    return value * 60 if unit in {"m", "min", "mins", "minute", "minutes"} else value


def parse_timestamp(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"]:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_epoch(dt, window_seconds):
    epoch = int(dt.timestamp())
    return epoch - (epoch % window_seconds)


def iso_from_epoch(epoch):
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def clean_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def clean_int(value, default=0):
    try:
        return int(clean_float(value, default))
    except Exception:
        return default


def clean_str(value, default="unknown"):
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def flag01(value):
    return 1 if clean_int(value, 0) > 0 else 0


def slugify(text):
    text = clean_str(text, "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "unknown"


def add_unique(items, value, limit=30):
    value = clean_str(value, "")
    if value and value not in items and len(items) < limit:
        items.append(value)


def snapshot_key(row, window_epoch):
    return tuple(clean_str(row.get(c), "unknown") for c in EXPERIMENT_COLS) + (int(window_epoch),)


def key_to_meta(key):
    namespace, experiment_id, run_ts, failure_type, target_service, window_epoch = key
    return {
        "namespace": namespace,
        "experiment_id": experiment_id,
        "run_ts": run_ts,
        "failure_type": failure_type,
        "target_service": target_service,
        "window_epoch": int(window_epoch),
    }


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def new_resource_acc():
    return {
        "instances": [],
        "pod_raws": [],
        "count": 0,
        "cpu_usage_sum": 0.0,
        "cpu_usage_max": 0.0,
        "cpu_system_sum": 0.0,
        "cpu_system_max": 0.0,
        "memory_working_set_sum": 0.0,
        "memory_working_set_max": 0.0,
        "memory_rss_sum": 0.0,
        "memory_rss_max": 0.0,
        "network_receive_bytes_sum": 0.0,
        "network_receive_bytes_max": 0.0,
        "network_transmit_packets_sum": 0.0,
        "network_transmit_packets_max": 0.0,
        "in_failure_window": 0,
        "pod_is_target": 0,
        "pod_under_failure": 0,
    }


def new_comm_acc():
    return {
        "source_raws": [],
        "destination_raws": [],
        "count": 0,
        "total_request_max": 0.0,
        "new_request_sum": 0.0,
        "success_count_sum": 0.0,
        "error_count_sum": 0.0,
        "success_rate_sum": 0.0,
        "error_rate_sum": 0.0,
        "average_latency_sum": 0.0,
        "p50_latency_sum": 0.0,
        "p90_latency_sum": 0.0,
        "p99_latency_sum": 0.0,
        "istio_request_bytes_sum": 0.0,
        "throughput_sum": 0.0,
        "request_rate_sum": 0.0,
        "in_failure_window": 0,
        "src_is_target": 0,
        "dst_is_target": 0,
        "src_under_failure": 0,
        "dst_under_failure": 0,
        "edge_under_failure": 0,
    }


def read_resources(path, window_seconds, max_rows=None):
    groups = defaultdict(new_resource_acc)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            dt = parse_timestamp(row.get("timestamp"))
            if dt is None:
                continue
            w = floor_epoch(dt, window_seconds)
            pod = clean_str(row.get("pod"), "unknown")
            key = (snapshot_key(row, w), pod)
            acc = groups[key]
            acc["count"] += 1
            add_unique(acc["instances"], row.get("instance"))
            add_unique(acc["pod_raws"], row.get("pod_raw"))

            cpu_usage = clean_float(row.get("container_cpu_usage_seconds_total"))
            cpu_system = clean_float(row.get("container_cpu_system_seconds_total"))
            memory_ws = clean_float(row.get("container_memory_working_set_bytes"))
            memory_rss = clean_float(row.get("container_memory_rss"))
            net_rx = clean_float(row.get("container_network_receive_bytes_total"))
            net_tx_packets = clean_float(row.get("container_network_transmit_packets_total"))

            acc["cpu_usage_sum"] += cpu_usage
            acc["cpu_usage_max"] = max(acc["cpu_usage_max"], cpu_usage)
            acc["cpu_system_sum"] += cpu_system
            acc["cpu_system_max"] = max(acc["cpu_system_max"], cpu_system)
            acc["memory_working_set_sum"] += memory_ws
            acc["memory_working_set_max"] = max(acc["memory_working_set_max"], memory_ws)
            acc["memory_rss_sum"] += memory_rss
            acc["memory_rss_max"] = max(acc["memory_rss_max"], memory_rss)
            acc["network_receive_bytes_sum"] += net_rx
            acc["network_receive_bytes_max"] = max(acc["network_receive_bytes_max"], net_rx)
            acc["network_transmit_packets_sum"] += net_tx_packets
            acc["network_transmit_packets_max"] = max(acc["network_transmit_packets_max"], net_tx_packets)

            acc["in_failure_window"] = max(acc["in_failure_window"], flag01(row.get("in_failure_window")))
            acc["pod_is_target"] = max(acc["pod_is_target"], flag01(row.get("pod_is_target")))
            acc["pod_under_failure"] = max(acc["pod_under_failure"], flag01(row.get("pod_under_failure")))
    return groups


def read_communications(path, window_seconds, max_rows=None):
    groups = defaultdict(new_comm_acc)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            dt = parse_timestamp(row.get("timestamp"))
            if dt is None:
                continue
            w = floor_epoch(dt, window_seconds)
            source = clean_str(row.get("source"), "unknown")
            destination = clean_str(row.get("destination"), "unknown")
            key = (snapshot_key(row, w), source, destination)
            acc = groups[key]
            acc["count"] += 1
            add_unique(acc["source_raws"], row.get("source_raw"))
            add_unique(acc["destination_raws"], row.get("destination_raw"))

            acc["total_request_max"] = max(acc["total_request_max"], clean_float(row.get("total_request")))
            acc["new_request_sum"] += clean_float(row.get("new_request"))
            acc["success_count_sum"] += clean_float(row.get("success_count"))
            acc["error_count_sum"] += clean_float(row.get("error_count"))
            acc["success_rate_sum"] += clean_float(row.get("success_rate"))
            acc["error_rate_sum"] += clean_float(row.get("error_rate"))
            acc["average_latency_sum"] += clean_float(row.get("average_latency"))
            acc["p50_latency_sum"] += clean_float(row.get("p50_latency"))
            acc["p90_latency_sum"] += clean_float(row.get("p90_latency"))
            acc["p99_latency_sum"] += clean_float(row.get("p99_latency"))
            acc["istio_request_bytes_sum"] += clean_float(row.get("istio_request_bytes"))
            acc["throughput_sum"] += clean_float(row.get("throughput"))
            acc["request_rate_sum"] += clean_float(row.get("request_rate"))

            for col in ["in_failure_window", "src_is_target", "dst_is_target", "src_under_failure", "dst_under_failure", "edge_under_failure"]:
                acc[col] = max(acc[col], flag01(row.get(col)))
    return groups


def read_events(path, window_seconds, max_rows=None, max_message_chars=300):
    events = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            dt = parse_timestamp(row.get("timestamp"))
            if dt is None:
                continue
            w = floor_epoch(dt, window_seconds)
            key = snapshot_key(row, w)
            msg = clean_str(row.get("message"), "")
            if max_message_chars and len(msg) > max_message_chars:
                msg = msg[:max_message_chars] + "..."
            events[key].append({
                "timestamp": dt.isoformat().replace("+00:00", "Z"),
                "event_type": clean_str(row.get("event_type"), "Unknown"),
                "reason": clean_str(row.get("reason"), "Unknown"),
                "message": msg,
                "pod_raw": clean_str(row.get("pod_raw"), ""),
                "pod": clean_str(row.get("pod"), "unknown"),
                "in_failure_window": flag01(row.get("in_failure_window")),
                "pod_is_target": flag01(row.get("pod_is_target")),
                "pod_under_failure": flag01(row.get("pod_under_failure")),
            })
    return events


def add_node(nodes, node_id, node_type, attributes=None):
    node_id = clean_str(node_id)
    if node_id not in nodes:
        nodes[node_id] = {"id": node_id, "type": node_type, "attributes": {}}
    if attributes:
        nodes[node_id]["attributes"].update(attributes)


def add_edge(edges, edge_id, source, target, edge_type, attributes=None):
    edges.append({
        "id": edge_id,
        "source": clean_str(source),
        "target": clean_str(target),
        "type": edge_type,
        "attributes": attributes or {},
    })


def build_snapshot(key, resource_items, comm_items, events, window_seconds, error_rate_threshold):
    meta = key_to_meta(key)
    window_epoch = meta["window_epoch"]
    namespace = meta["namespace"]
    experiment_id = meta["experiment_id"]
    run_ts = meta["run_ts"]
    failure_type = meta["failure_type"]
    target_service = meta["target_service"]

    nodes = {}
    edges = []

    for (_, pod), acc in resource_items:
        count = max(acc["count"], 1)
        is_anom = acc["pod_under_failure"] == 1 and failure_type.lower() not in {"unknown", "normal", "none"}
        anomaly = failure_type if is_anom else "Normal"
        add_node(nodes, pod, "microservice", {
            "namespace": namespace,
            "anomaly": anomaly,
            "resource_anomaly": anomaly,
            "pod_under_failure": acc["pod_under_failure"],
            "pod_is_target": acc["pod_is_target"],
            "in_failure_window": acc["in_failure_window"],
            "cpu_usage_mean": round(acc["cpu_usage_sum"] / count, 6),
            "cpu_usage_max": round(acc["cpu_usage_max"], 6),
            "cpu_system_mean": round(acc["cpu_system_sum"] / count, 6),
            "cpu_system_max": round(acc["cpu_system_max"], 6),
            "memory_working_set_mean_bytes": round(acc["memory_working_set_sum"] / count, 3),
            "memory_working_set_max_bytes": round(acc["memory_working_set_max"], 3),
            "memory_rss_mean_bytes": round(acc["memory_rss_sum"] / count, 3),
            "memory_rss_max_bytes": round(acc["memory_rss_max"], 3),
            "network_receive_bytes_mean": round(acc["network_receive_bytes_sum"] / count, 3),
            "network_receive_bytes_max": round(acc["network_receive_bytes_max"], 3),
            "network_transmit_packets_mean": round(acc["network_transmit_packets_sum"] / count, 3),
            "network_transmit_packets_max": round(acc["network_transmit_packets_max"], 3),
            "instances": acc["instances"],
            "pod_raws": acc["pod_raws"],
        })
        for instance in acc["instances"]:
            worker_id = f"worker:{instance}"
            add_node(nodes, worker_id, "worker_node", {"namespace": namespace, "instance": instance})
            add_edge(edges, f"deploy:{pod}->{worker_id}", pod, worker_id, "deployed_on", {"relation": "deployed_on", "namespace": namespace})

    for (_, source, destination), acc in comm_items:
        count = max(acc["count"], 1)
        add_node(nodes, source, "microservice", {"namespace": namespace})
        add_node(nodes, destination, "microservice", {"namespace": namespace})

        classified_total = acc["success_count_sum"] + acc["error_count_sum"]
        if classified_total > 0:
            success_rate = acc["success_count_sum"] / classified_total
            error_rate = acc["error_count_sum"] / classified_total
        else:
            success_rate = acc["success_rate_sum"] / count
            error_rate = acc["error_rate_sum"] / count

        edge_status = "Unhealthy" if acc["edge_under_failure"] or error_rate > error_rate_threshold or acc["error_count_sum"] > 0 else "Healthy"
        add_edge(edges, f"comm:{source}->{destination}", source, destination, "communication", {
            "relation": "calls",
            "namespace": namespace,
            "communication_anomaly": edge_status,
            "edge_status": edge_status,
            "edge_under_failure": acc["edge_under_failure"],
            "src_under_failure": acc["src_under_failure"],
            "dst_under_failure": acc["dst_under_failure"],
            "src_is_target": acc["src_is_target"],
            "dst_is_target": acc["dst_is_target"],
            "in_failure_window": acc["in_failure_window"],
            "total_request": round(acc["total_request_max"], 3),
            "new_request": round(acc["new_request_sum"], 3),
            "success_count": round(acc["success_count_sum"], 3),
            "error_count": round(acc["error_count_sum"], 3),
            "success_rate": round(success_rate, 6),
            "error_rate": round(error_rate, 6),
            "average_latency_mean": round(acc["average_latency_sum"] / count, 6),
            "p50_latency_mean": round(acc["p50_latency_sum"] / count, 6),
            "p90_latency_mean": round(acc["p90_latency_sum"] / count, 6),
            "p99_latency_mean": round(acc["p99_latency_sum"] / count, 6),
            "istio_request_bytes_sum": round(acc["istio_request_bytes_sum"], 3),
            "throughput_mean": round(acc["throughput_sum"] / count, 6),
            "request_rate_mean": round(acc["request_rate_sum"] / count, 6),
            "source_raws": acc["source_raws"],
            "destination_raws": acc["destination_raws"],
        })

    for idx, event in enumerate(events):
        pod = event["pod"]
        add_node(nodes, pod, "microservice", {"namespace": namespace})
        event_id = f"event:{experiment_id}:{datetime.fromtimestamp(window_epoch, tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}:{idx}"
        add_node(nodes, event_id, "runtime_event", {
            "namespace": namespace,
            "event_type": event["event_type"],
            "reason": event["reason"],
            "message": event["message"],
            "pod": pod,
            "pod_raw": event["pod_raw"],
            "event_timestamp": event["timestamp"],
            "in_failure_window": event["in_failure_window"],
            "pod_is_target": event["pod_is_target"],
            "pod_under_failure": event["pod_under_failure"],
        })
        add_edge(edges, f"affects:{event_id}->{pod}", event_id, pod, "affects", {"relation": "affects", "reason": event["reason"]})

    anomalous_microservices = [
        node_id for node_id, node in nodes.items()
        if node["type"] == "microservice"
        and clean_str(node["attributes"].get("anomaly", "Normal")).lower() not in {"normal", "healthy", "none", "0", "unknown"}
    ]
    unhealthy_communication_edges = [
        e["id"] for e in edges
        if e["type"] == "communication" and e["attributes"].get("edge_status") != "Healthy"
    ]

    snapshot_id = f"{slugify(namespace)}_{slugify(experiment_id)}_{slugify(run_ts)}_{datetime.fromtimestamp(window_epoch, tz=timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    snapshot = {
        "snapshot_id": snapshot_id,
        "timestamp": iso_from_epoch(window_epoch),
        "window": {
            "start": iso_from_epoch(window_epoch),
            "end": iso_from_epoch(window_epoch + window_seconds),
            "size_seconds": window_seconds,
        },
        "experiment": {
            "namespace": namespace,
            "experiment_id": experiment_id,
            "run_ts": run_ts,
            "failure_type": failure_type,
            "target_service": target_service,
        },
        "nodes": list(nodes.values()),
        "edges": edges,
        "graph_summary": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_microservices": len([n for n in nodes.values() if n["type"] == "microservice"]),
            "num_worker_nodes": len([n for n in nodes.values() if n["type"] == "worker_node"]),
            "num_event_nodes": len([n for n in nodes.values() if n["type"] == "runtime_event"]),
            "num_communication_edges": len([e for e in edges if e["type"] == "communication"]),
            "num_deployment_edges": len([e for e in edges if e["type"] == "deployed_on"]),
            "num_event_edges": len([e for e in edges if e["type"] == "affects"]),
            "anomalous_microservices": anomalous_microservices,
            "unhealthy_communication_edges": unhealthy_communication_edges,
        },
    }
    return clean_for_json(snapshot)


def main():
    parser = argparse.ArgumentParser(description="Convert RTKG input CSVs to JSON snapshot files.")
    parser.add_argument("--resource_csv", required=True)
    parser.add_argument("--communication_csv", required=True)
    parser.add_argument("--events_csv", required=True)
    parser.add_argument("--output_dir", default="rtkg_snapshots")
    parser.add_argument("--window", default="15s")
    parser.add_argument("--error_rate_threshold", type=float, default=0.0)
    parser.add_argument("--max_rows", type=int, default=None, help="Debug only: read at most this many rows from each CSV")
    parser.add_argument("--max_event_message_chars", type=int, default=300)
    args = parser.parse_args()

    window_seconds = parse_window_seconds(args.window)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading and aggregating resource-consumption CSV...")
    resource_groups = read_resources(args.resource_csv, window_seconds, args.max_rows)
    print(f"  resource groups: {len(resource_groups):,}")

    print("Reading and aggregating communication CSV...")
    comm_groups = read_communications(args.communication_csv, window_seconds, args.max_rows)
    print(f"  communication groups: {len(comm_groups):,}")

    print("Reading and aligning events CSV...")
    event_groups = read_events(args.events_csv, window_seconds, args.max_rows, args.max_event_message_chars)
    print(f"  snapshot windows with events: {len(event_groups):,}")

    by_snapshot_resources = defaultdict(list)
    for key, acc in resource_groups.items():
        snap_key, _pod = key
        by_snapshot_resources[snap_key].append((key, acc))

    by_snapshot_comms = defaultdict(list)
    for key, acc in comm_groups.items():
        snap_key, _source, _destination = key
        by_snapshot_comms[snap_key].append((key, acc))

    all_snapshot_keys = sorted(
        set(by_snapshot_resources.keys()) | set(by_snapshot_comms.keys()) | set(event_groups.keys()),
        key=lambda k: (k[0], k[1], k[2], k[-1]),
    )
    print(f"Building {len(all_snapshot_keys):,} JSON snapshots...")

    index_path = out_dir / "index.jsonl"
    with open(index_path, "w", encoding="utf-8") as index_file:
        for i, key in enumerate(all_snapshot_keys, start=1):
            snapshot = build_snapshot(
                key,
                by_snapshot_resources.get(key, []),
                by_snapshot_comms.get(key, []),
                event_groups.get(key, []),
                window_seconds,
                args.error_rate_threshold,
            )
            filename = f"{snapshot['snapshot_id']}.json"
            with open(out_dir / filename, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)

            summary = snapshot["graph_summary"]
            index_record = {
                "snapshot_id": snapshot["snapshot_id"],
                "timestamp": snapshot["timestamp"],
                "file": filename,
                **snapshot["experiment"],
                "num_nodes": summary["num_nodes"],
                "num_edges": summary["num_edges"],
                "num_microservices": summary["num_microservices"],
                "num_communication_edges": summary["num_communication_edges"],
                "num_event_nodes": summary["num_event_nodes"],
                "anomalous_microservices": summary["anomalous_microservices"],
                "unhealthy_communication_edges": summary["unhealthy_communication_edges"],
            }
            index_file.write(json.dumps(index_record, ensure_ascii=False) + "\n")

            if i % 500 == 0:
                print(f"  wrote {i:,}/{len(all_snapshot_keys):,}")

    print("Done.")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Index file: {index_path.resolve()}")
    print(f"Total snapshots: {len(all_snapshot_keys):,}")


if __name__ == "__main__":
    main()
