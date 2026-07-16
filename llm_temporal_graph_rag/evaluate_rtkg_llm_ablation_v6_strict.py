#!/usr/bin/env python3
"""Evaluate RTKG-grounded LLMs served by Ollama, with Log-RAG comparison, evidence-only Graph-RAG, and stricter scoring.

This script evaluates local LLM responses over Runtime Temporal Knowledge Graph
(RTKG) snapshots and compares graph-based retrieval against a raw-log retrieval
baseline.

Supported modes:

1) no_rag
   - The LLM receives only the user question.
   - This is kept as an optional sanity-check baseline.

2) log_rag
   - The LLM receives compact evidence retrieved from raw observability logs
     (resource CSV, communication CSV, and event CSV), not from RTKG snapshots.
   - This baseline tests whether text/log retrieval over the same monitoring data
     is sufficient without the structured temporal graph abstraction.

3) latest_snapshot_rag
   - The LLM receives a compact result computed from only the latest short RTKG
     window, e.g., the last 15 seconds.
   - This tests whether a single/current snapshot is enough for temporal queries.

4) temporal_graph_rag_evidence_only
   - The LLM receives structured top-k graph evidence from all RTKG snapshots
     in the requested time window, but NOT the precomputed final answer.
   - This is the main benchmark setting because it tests whether the LLM can
     read and verbalize structured graph evidence.

5) temporal_graph_rag
   - The LLM receives a compact computed result from all RTKG snapshots in the
     requested time window.
   - This represents the deployed operator-facing setting, where deterministic
     graph queries compute the answer and the LLM verbalizes it.

Important evaluation design:
- Ground truth is ALWAYS computed from the full Temporal Graph-RAG evidence.
- The LLM is evaluated with strict scoring: entity correctness, numeric correctness, and unsupported-claim filtering.
- The Log-RAG baseline uses raw CSV/log records and lightweight log retrieval;
  it does not read RTKG snapshots and does not use graph topology as the retrieval
  source.
- Timeouts and Ollama errors are saved in the CSV instead of crashing the run.

Usage example:
  python evaluate_rtkg_llm_ablation_v6_strict.py \
    --snapshots rtkg_snapshots \
    --resource-csv aggregated_pod_resource_consumption.csv \
    --communication-csv aggregated_pod_communication.csv \
    --events-csv aggregated_pod_events.csv \
    --models mistral:latest,llama3.1:8b,qwen3:8b,deepseek-r1:7b \
    --windows "15 seconds,1 minute,5 minutes,10 minutes" \
    --runs 3 \
    --timeout 240 \
    --top-k 3 \
    --num-ctx 2048 \
    --num-predict 128 \
    --exclude-summary \
    --ablation-modes log_rag,latest_snapshot_rag,temporal_graph_rag \
    --out rtkg_llm_lograg_ablation_results_v6.csv \
    --summary-out rtkg_llm_lograg_ablation_summary_v6.csv

This script expects rtkg_graph_rag_ollama_v2.py to be in the same directory or
available in PYTHONPATH.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is only needed for log_rag
    pd = None

from rtkg_graph_rag_ollama_v2 import RTKGGraphRAG, compact_json


QUESTION_TEMPLATES = [
    # ------------------------------------------------------------------
    # Node / microservice anomaly questions
    # ------------------------------------------------------------------
    {
        "task": "highest_node_anomaly_rate",
        "task_group": "node_anomaly",
        "question": "Which microservice has the highest anomaly rate in the last {window}?",
        "ranking": "top_microservices_by_node_anomaly_rate",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "node_anomaly_rate_percent",
        "value_label": "node anomaly rate (%)",
        "valid_if": lambda row: row.get("node_anomalous_snapshots", 0) > 0,
    },
    {
        "task": "most_anomalous_microservice",
        "task_group": "node_anomaly",
        "question": "Which microservice was anomalous most often in the last {window}?",
        "ranking": "top_microservices_by_node_anomaly_rate",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "node_anomaly_rate_percent",
        "value_label": "node anomaly rate (%)",
        "valid_if": lambda row: row.get("node_anomalous_snapshots", 0) > 0,
    },
    {
        "task": "strongest_combined_anomaly_evidence",
        "task_group": "combined_anomaly_evidence",
        "question": "Which microservice has the strongest combined anomaly evidence in the last {window}?",
        "ranking": "top_microservices_by_combined_anomaly_evidence",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "combined_anomaly_evidence_score",
        "value_label": "combined anomaly evidence score",
        "valid_if": lambda row: row.get("combined_anomaly_evidence_score", 0) > 0,
    },

    # ------------------------------------------------------------------
    # Resource metric questions
    # ------------------------------------------------------------------
    {
        "task": "highest_cpu",
        "task_group": "cpu",
        "question": "Which microservice has the highest CPU usage in the last {window}?",
        "ranking": "top_microservices_by_cpu",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "max_cpu",
        "value_label": "max CPU",
        "valid_if": lambda row: row.get("max_cpu", 0) > 0,
    },
    {
        "task": "highest_cpu_paraphrase",
        "task_group": "cpu",
        "question": "Which pod consumed the most CPU in the last {window}?",
        "ranking": "top_microservices_by_cpu",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "max_cpu",
        "value_label": "max CPU",
        "valid_if": lambda row: row.get("max_cpu", 0) > 0,
    },
    {
        "task": "highest_memory",
        "task_group": "memory",
        "question": "Which microservice has the highest memory usage in the last {window}?",
        "ranking": "top_microservices_by_memory",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "max_memory",
        "value_label": "max memory",
        "valid_if": lambda row: row.get("max_memory", 0) > 0,
    },
    {
        "task": "highest_memory_paraphrase",
        "task_group": "memory",
        "question": "Which pod consumed the most memory in the last {window}?",
        "ranking": "top_microservices_by_memory",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "max_memory",
        "value_label": "max memory",
        "valid_if": lambda row: row.get("max_memory", 0) > 0,
    },

    # ------------------------------------------------------------------
    # Communication / edge questions
    # ------------------------------------------------------------------
    {
        "task": "highest_edge_error_rate",
        "task_group": "edge_error",
        "question": "Which communication edge has the highest error rate in the last {window}?",
        "ranking": "top_edges_by_error_rate",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "max_error_rate",
        "value_label": "max error rate",
        "valid_if": lambda row: row.get("max_error_rate", 0) > 0,
    },
    {
        "task": "worst_error_path",
        "task_group": "edge_error",
        "question": "Which service-to-service communication path shows the worst error rate in the last {window}?",
        "ranking": "top_edges_by_error_rate",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "max_error_rate",
        "value_label": "max error rate",
        "valid_if": lambda row: row.get("max_error_rate", 0) > 0,
    },
    {
        "task": "most_unhealthy_communication_edge",
        "task_group": "unhealthy_edge",
        "question": "Which communication edge was unhealthy most often in the last {window}?",
        "ranking": "top_unhealthy_communication_edges",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "unhealthy_rate_percent",
        "value_label": "unhealthy rate (%)",
        "valid_if": lambda row: row.get("unhealthy_snapshots", 0) > 0,
    },
    {
        "task": "highest_edge_latency",
        "task_group": "edge_latency",
        "question": "Which communication edge has the highest latency in the last {window}?",
        "ranking": "top_edges_by_latency",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "max_latency",
        "value_label": "max latency",
        "valid_if": lambda row: row.get("max_latency", 0) > 0,
    },
    {
        "task": "slowest_communication_path",
        "task_group": "edge_latency",
        "question": "Which service-to-service path was the slowest in the last {window}?",
        "ranking": "top_edges_by_latency",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "max_latency",
        "value_label": "max latency",
        "valid_if": lambda row: row.get("max_latency", 0) > 0,
    },

    # ------------------------------------------------------------------
    # Runtime event question
    # ------------------------------------------------------------------
    {
        "task": "most_event_affected_microservice",
        "task_group": "event_affected_service",
        "question": "Which microservice is affected by the most runtime events in the last {window}?",
        "ranking": "top_event_affected_microservices",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "event_count",
        "value_label": "runtime event count",
        "valid_if": lambda row: row.get("event_count", 0) > 0,
    },

    # ------------------------------------------------------------------
    # Additional paraphrased deterministic questions for robustness
    # ------------------------------------------------------------------
    {
        "task": "top_cpu_resource_consumer",
        "task_group": "cpu",
        "question": "Which service is the top CPU resource consumer in the last {window}?",
        "ranking": "top_microservices_by_cpu",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "max_cpu",
        "value_label": "max CPU",
        "valid_if": lambda row: row.get("max_cpu", 0) > 0,
    },
    {
        "task": "peak_memory_service",
        "task_group": "memory",
        "question": "Which service reached the peak memory value in the last {window}?",
        "ranking": "top_microservices_by_memory",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "max_memory",
        "value_label": "max memory",
        "valid_if": lambda row: row.get("max_memory", 0) > 0,
    },
    {
        "task": "most_error_prone_dependency",
        "task_group": "edge_error",
        "question": "Which dependency was the most error-prone in the last {window}?",
        "ranking": "top_edges_by_error_rate",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "max_error_rate",
        "value_label": "max error rate",
        "valid_if": lambda row: row.get("max_error_rate", 0) > 0,
    },
    {
        "task": "highest_error_route",
        "task_group": "edge_error",
        "question": "Which route had the largest communication error rate in the last {window}?",
        "ranking": "top_edges_by_error_rate",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "max_error_rate",
        "value_label": "max error rate",
        "valid_if": lambda row: row.get("max_error_rate", 0) > 0,
    },
    {
        "task": "slowest_dependency",
        "task_group": "edge_latency",
        "question": "Which dependency had the highest communication delay in the last {window}?",
        "ranking": "top_edges_by_latency",
        "entity_type": "edge",
        "entity_field": "edge",
        "value_field": "max_latency",
        "value_label": "max latency",
        "valid_if": lambda row: row.get("max_latency", 0) > 0,
    },
    {
        "task": "k8s_event_impacted_service",
        "task_group": "event_affected_service",
        "question": "Which microservice was most impacted by Kubernetes runtime events in the last {window}?",
        "ranking": "top_event_affected_microservices",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "event_count",
        "value_label": "runtime event count",
        "valid_if": lambda row: row.get("event_count", 0) > 0,
    },
    {
        "task": "highest_graph_evidence_score",
        "task_group": "combined_anomaly_evidence",
        "question": "Which microservice has the highest graph anomaly evidence score in the last {window}?",
        "ranking": "top_microservices_by_combined_anomaly_evidence",
        "entity_type": "service",
        "entity_field": "microservice",
        "value_field": "combined_anomaly_evidence_score",
        "value_label": "combined anomaly evidence score",
        "valid_if": lambda row: row.get("combined_anomaly_evidence_score", 0) > 0,
    },

    # Open-ended qualitative question. Exclude with --exclude-summary for strict accuracy.
    {
        "task": "system_health_summary",
        "task_group": "summary",
        "question": "Summarize the system health in the last {window}.",
        "ranking": None,
        "entity_type": "summary",
        "entity_field": None,
        "value_field": None,
        "value_label": None,
        "valid_if": None,
    },
]

VALID_ABLATION_MODES = [
    "no_rag",
    "log_rag",
    "latest_snapshot_rag",
    "temporal_graph_rag_evidence_only",
    "temporal_graph_rag",
]

# Used when there is no retrieved data at all.
INSUFFICIENT_PATTERNS = [
    "insufficient evidence",
    "no evidence",
    "cannot determine",
    "can't determine",
    "not enough information",
    "no rtkg",
    "no data",
    "not provided",
    "no snapshots",
    "no logs",
    "no log",
]

# Used when evidence exists, but no positive candidate exists, e.g., no anomalies.
NO_POSITIVE_PATTERNS = [
    "no anomaly",
    "no anomalies",
    "no anomalous",
    "not anomalous",
    "no abnormal",
    "no unhealthy",
    "no degraded",
    "no positive",
    "all normal",
    "all healthy",
    "zero anomal",
    "0 anomal",
    "none were anomal",
    "none are anomal",
    "no microservice has",
    "no service has",
    "no communication edge has",
    "no edge has",
    "no runtime events",
    "no events",
    "no event",
]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def norm_text(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def split_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def contains_entity(answer: str, entity: str, entity_type: str) -> bool:
    if not entity:
        return False
    answer_norm = norm_text(answer)
    if entity_type == "edge" and "->" in entity:
        src, dst = [x.strip() for x in entity.split("->", 1)]
        return norm_text(src) in answer_norm and norm_text(dst) in answer_norm
    return norm_text(entity) in answer_norm


def extract_numbers(text: str) -> List[float]:
    values = []
    for m in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text):
        try:
            values.append(float(m.group(0)))
        except Exception:
            pass
    return values


def numeric_correct(answer: str, expected: Optional[float], rel_tol: float, abs_tol: float) -> Optional[bool]:
    if expected is None or not isinstance(expected, (int, float)) or math.isnan(float(expected)):
        return None
    nums = extract_numbers(answer)
    if not nums:
        return False

    expected = float(expected)
    candidates = [expected]

    # If expected is a percent, answers may include either 75 or 0.75.
    if 0 <= expected <= 100:
        candidates.append(expected / 100.0)

    # If expected is a rate, answers may include percent form.
    if 0 <= expected <= 1:
        candidates.append(expected * 100.0)

    for n in nums:
        for c in candidates:
            if math.isclose(n, c, rel_tol=rel_tol, abs_tol=abs_tol):
                return True
    return False


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def expected_from_evidence(evidence: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a deterministic expected answer from full RTKG evidence."""
    if evidence.get("status") != "ok":
        return {"expected_entity": None, "expected_value": None, "expected_status": "no_data", "expected_row": None}

    if spec["ranking"] is None:
        # Summary questions are excluded from strict accuracy.
        return {"expected_entity": None, "expected_value": None, "expected_status": "summary", "expected_row": None}

    rows = evidence.get("rankings", {}).get(spec["ranking"], []) or []
    rows = [r for r in rows if spec["valid_if"](r)]
    if not rows:
        return {"expected_entity": None, "expected_value": None, "expected_status": "no_positive_evidence", "expected_row": None}

    top = rows[0]
    return {
        "expected_entity": str(top.get(spec["entity_field"], "")),
        "expected_value": top.get(spec["value_field"]),
        "expected_status": "ok",
        "expected_row": top,
    }


def safe_compact_json(obj: Any, max_chars: int = 10000) -> str:
    try:
        return compact_json(obj, max_chars=max_chars)
    except TypeError:
        text = compact_json(obj)
        if max_chars and len(text) > max_chars:
            return text[:max_chars] + "\n...TRUNCATED..."
        return text
    except Exception:
        text = json.dumps(obj, ensure_ascii=False, default=str)
        if max_chars and len(text) > max_chars:
            return text[:max_chars] + "\n...TRUNCATED..."
        return text


def parse_window_duration(question_or_window: str) -> Optional[Any]:
    """Return pandas Timedelta for a phrase like 'last 5 minutes' or '5 minutes'."""
    if pd is None:
        return None
    text = question_or_window.strip().lower()
    m = re.search(
        r"(?:last|past|previous)?\s*(\d+)\s*"
        r"(seconds|second|sec|secs|s|minutes|minute|mins|min|m|hours|hour|hrs|hr|h)",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit in ["seconds", "second", "sec", "secs", "s"]:
        return pd.Timedelta(seconds=n)
    if unit in ["minutes", "minute", "mins", "min", "m"]:
        return pd.Timedelta(minutes=n)
    if unit in ["hours", "hour", "hrs", "hr", "h"]:
        return pd.Timedelta(hours=n)
    return None


def replace_time_window(question: str, new_window: str) -> str:
    """Replace a time expression in the question with a new one."""
    pattern = r"(last|past|previous)\s+\d+\s*(seconds|second|sec|secs|s|minutes|minute|mins|min|m|hours|hour|hrs|hr|h)"
    repl = f"last {new_window}"
    if re.search(pattern, question, flags=re.IGNORECASE):
        return re.sub(pattern, repl, question, flags=re.IGNORECASE)
    return question.rstrip(" ?.") + f" in the last {new_window}?"


# -----------------------------------------------------------------------------
# Graph-RAG evidence and prompts
# -----------------------------------------------------------------------------

def graph_result_block(question: str, evidence: Dict[str, Any], spec: Dict[str, Any], direct_answer: str) -> Dict[str, Any]:
    """Build a compact structured result from graph evidence for the prompt."""
    exp = expected_from_evidence(evidence, spec)
    window_info = evidence.get("retrieval_window", {}) if isinstance(evidence, dict) else {}
    graph_summary = evidence.get("graph_summary", {}) if isinstance(evidence, dict) else {}

    result = {
        "question": question,
        "status": exp.get("expected_status"),
        "retrieval_source": "rtkg_snapshot_graph",
        "entity_type": spec.get("entity_type"),
        "top_entity": exp.get("expected_entity"),
        "metric_name": spec.get("value_label") or spec.get("value_field"),
        "metric_value": exp.get("expected_value"),
        "direct_graph_answer": direct_answer,
        "retrieval_window": {
            "start": window_info.get("start"),
            "end": window_info.get("end"),
            "num_snapshots": window_info.get("num_snapshots"),
        },
        "graph_summary": {
            "num_microservices": graph_summary.get("num_microservices_observed"),
            "num_edges": graph_summary.get("num_communication_edges_observed"),
            "num_events": graph_summary.get("num_events"),
        },
    }

    if exp.get("expected_row") is not None:
        # Keep the complete top row for traceability, but not the whole JSON evidence.
        result["top_row"] = exp["expected_row"]

    return result


def graph_evidence_only_block(question: str, evidence: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build structured evidence without exposing the final computed answer.

    This is the stricter benchmark input: the LLM receives the top-k ranking
    rows and must identify the answer from the evidence. It does not receive
    TOP_ENTITY or direct_graph_answer.
    """
    window_info = evidence.get("retrieval_window", {}) if isinstance(evidence, dict) else {}
    graph_summary = evidence.get("graph_summary", {}) if isinstance(evidence, dict) else {}
    ranking_name = spec.get("ranking")
    rows = []
    if ranking_name:
        rows = evidence.get("rankings", {}).get(ranking_name, []) or []

    return {
        "question": question,
        "status": evidence.get("status", "no_data") if isinstance(evidence, dict) else "no_data",
        "retrieval_source": "rtkg_snapshot_graph",
        "benchmark_mode": "evidence_only_no_direct_answer",
        "instruction": "Select the first valid/top candidate from candidate_ranking according to metric_name.",
        "entity_type": spec.get("entity_type"),
        "ranking_name": ranking_name,
        "metric_name": spec.get("value_label") or spec.get("value_field"),
        "value_field": spec.get("value_field"),
        "candidate_ranking": rows,
        "retrieval_window": {
            "start": window_info.get("start"),
            "end": window_info.get("end"),
            "num_snapshots": window_info.get("num_snapshots"),
        },
        "graph_summary": {
            "num_microservices": graph_summary.get("num_microservices_observed"),
            "num_edges": graph_summary.get("num_communication_edges_observed"),
            "num_events": graph_summary.get("num_events"),
        },
    }


def no_rag_prompt(question: str) -> Tuple[str, Dict[str, Any], str]:
    """No-RAG ablation: no evidence is given to the LLM."""
    prompt = f"""
You are an expert SRE assistant for Kubernetes microservice observability.

No RTKG snapshots, logs, metrics, events, or graph evidence are provided.
Do not invent microservices, metrics, edges, timestamps, anomaly types, or values.
If the question requires runtime monitoring data, answer exactly: "There is insufficient evidence to answer from the provided context."
Keep the answer to one sentence.

USER_QUESTION:
{question}

ANSWER:
""".strip()
    return prompt, {}, "No RTKG or log evidence provided."


def latest_snapshot_rag_prompt(
    rag: RTKGGraphRAG,
    question: str,
    spec: Dict[str, Any],
    latest_window: str,
    compact_chars: int,
) -> Tuple[str, Dict[str, Any], str]:
    """Latest-snapshot ablation: only a short/latest window is used."""
    latest_question = replace_time_window(question, latest_window)
    evidence = rag.build_evidence(latest_question)
    direct = rag.direct_answer(question, evidence)
    compact_result = graph_evidence_only_block(question, evidence, spec)

    prompt = f"""
You are an expert SRE assistant for Kubernetes microservice observability.

This is an ablation setting. You are given ONLY the latest RTKG evidence, not the full requested temporal window.
Answer using ONLY the structured graph evidence below.
Do not reason step by step. Do not include <think> tags.
Do not invent services, edges, metrics, timestamps, anomaly types, or values.
Select the first/top valid candidate in CANDIDATE_RANKING.
Your one-sentence answer MUST include both the exact service/edge name and the corresponding metric value.
If STATUS is "no_data", say that there is insufficient evidence.
If the candidate ranking is empty or has no positive/anomalous/error candidate, say that no positive candidate was found in the provided RTKG evidence.

LATEST_SNAPSHOT_GRAPH_EVIDENCE_JSON:
{safe_compact_json(compact_result, max_chars=compact_chars)}

USER_QUESTION:
{question}

ANSWER:
""".strip()
    return prompt, evidence, direct


def temporal_graph_rag_evidence_only_prompt(
    rag: RTKGGraphRAG,
    question: str,
    spec: Dict[str, Any],
    compact_chars: int,
) -> Tuple[str, Dict[str, Any], str]:
    """Full Temporal Graph-RAG evidence-only benchmark.

    The LLM receives structured graph evidence, but not the final direct answer.
    This is the preferred strict benchmark setting.
    """
    evidence = rag.build_evidence(question)
    direct = rag.direct_answer(question, evidence)
    compact_result = graph_evidence_only_block(question, evidence, spec)

    prompt = f"""
You are an expert SRE assistant for Kubernetes microservice observability.

Answer using ONLY the Temporal Graph-RAG evidence below.
Do not reason step by step. Do not include <think> tags.
Do not invent services, edges, metrics, timestamps, anomaly types, or values.
Select the first/top valid candidate in CANDIDATE_RANKING according to the metric.
Your one-sentence answer MUST include both the exact service/edge name and the corresponding metric value.
If STATUS is "no_data", say that there is insufficient evidence.
If the candidate ranking is empty or has no positive/anomalous/error candidate, say that no positive candidate was found in the provided RTKG evidence.

TEMPORAL_GRAPH_RAG_EVIDENCE_ONLY_JSON:
{safe_compact_json(compact_result, max_chars=compact_chars)}

USER_QUESTION:
{question}

ANSWER:
""".strip()
    return prompt, evidence, direct


def temporal_graph_rag_prompt(
    rag: RTKGGraphRAG,
    question: str,
    spec: Dict[str, Any],
    compact_chars: int,
) -> Tuple[str, Dict[str, Any], str]:
    """Full Temporal Graph-RAG: full requested time-window evidence is used."""
    evidence = rag.build_evidence(question)
    direct = rag.direct_answer(question, evidence)
    compact_result = graph_result_block(question, evidence, spec, direct)

    prompt = f"""
You are an expert SRE assistant for Kubernetes microservice observability.

Answer using ONLY the computed Temporal Graph-RAG result below.
Do not reason step by step. Do not include <think> tags.
Do not invent services, edges, metrics, timestamps, anomaly types, or values.
If STATUS is "ok", mention TOP_ENTITY exactly as written and include METRIC_VALUE.
If STATUS is "no_positive_evidence", say that no anomalous or positive candidate was found in the provided RTKG evidence.
If STATUS is "no_data", say that there is insufficient evidence.
Keep the answer to one sentence.

COMPUTED_TEMPORAL_GRAPH_RAG_RESULT_JSON:
{safe_compact_json(compact_result, max_chars=compact_chars)}

USER_QUESTION:
{question}

ANSWER:
""".strip()
    return prompt, evidence, direct


# -----------------------------------------------------------------------------
# Log-RAG corpus, retrieval, and prompt
# -----------------------------------------------------------------------------

class LogRAGCorpus:
    """Raw-log retrieval baseline built from CSV observability logs.

    This class intentionally does not read RTKG JSON snapshots. It loads resource,
    communication, and event CSV files, filters rows by the requested time window,
    and prepares compact textual/log-derived evidence for the LLM.
    """

    RESOURCE_KEEP = {
        "timestamp", "instance", "pod",
        "container_cpu_usage_seconds_total", "container_cpu_system_seconds_total",
        "container_memory_working_set_bytes", "container_memory_rss",
        "container_network_receive_bytes_total", "container_network_transmit_packets_total",
        "namespace", "failure_type", "target_service", "experiment_id", "run_ts",
        "in_failure_window", "pod_is_target", "pod_under_failure",
    }
    COMM_KEEP = {
        "timestamp", "source", "destination", "source_raw", "destination_raw",
        "total_request", "new_request", "success_count", "error_count", "success_rate", "error_rate",
        "average_latency", "p50_latency", "p90_latency", "p99_latency",
        "throughput", "request_rate", "namespace", "failure_type", "target_service",
        "experiment_id", "run_ts", "in_failure_window", "edge_under_failure",
        "src_under_failure", "dst_under_failure",
    }
    EVENTS_KEEP = {
        "timestamp", "event_type", "reason", "message", "namespace", "failure_type", "target_service",
        "experiment_id", "run_ts", "pod", "pod_raw", "in_failure_window", "pod_under_failure",
    }

    def __init__(self, resource_csv: str, communication_csv: str, events_csv: str) -> None:
        self.resource_csv = Path(resource_csv) if resource_csv else None
        self.communication_csv = Path(communication_csv) if communication_csv else None
        self.events_csv = Path(events_csv) if events_csv else None
        self.resource_df = self._load_csv(self.resource_csv, self.RESOURCE_KEEP, "resource")
        self.communication_df = self._load_csv(self.communication_csv, self.COMM_KEEP, "communication")
        self.events_df = self._load_csv(self.events_csv, self.EVENTS_KEEP, "events")
        self.latest_ts = self._latest_timestamp()

    @staticmethod
    def _load_csv(path: Optional[Path], keep_cols: set, label: str) -> Any:
        if pd is None:
            return None
        if path is None or not path.exists():
            return None
        try:
            df = pd.read_csv(path, usecols=lambda c: c in keep_cols, low_memory=False)
            if "timestamp" not in df.columns:
                return None
            df["_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
            df = df.dropna(subset=["_ts"])
            return df
        except Exception as e:
            print(f"[WARN] Could not load {label} log CSV {path}: {e}")
            return None

    def _latest_timestamp(self) -> Any:
        if pd is None:
            return None
        values = []
        for df in [self.resource_df, self.communication_df, self.events_df]:
            if df is not None and not df.empty and "_ts" in df.columns:
                values.append(df["_ts"].max())
        if not values:
            return None
        return max(values)

    def _window_bounds(self, question: str) -> Tuple[Any, Any, Optional[Any]]:
        if pd is None or self.latest_ts is None:
            return None, None, None
        delta = parse_window_duration(question)
        if delta is None:
            delta = pd.Timedelta(minutes=5)
        end = self.latest_ts
        start = end - delta
        return start, end, delta

    @staticmethod
    def _filter_window(df: Any, start: Any, end: Any) -> Any:
        if pd is None or df is None or start is None or end is None or df.empty:
            return None
        return df[(df["_ts"] >= start) & (df["_ts"] <= end)].copy()

    @staticmethod
    def _records(df: Any, n: int, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if pd is None or df is None or df.empty:
            return []
        if columns is not None:
            cols = [c for c in columns if c in df.columns]
            df = df[cols]
        else:
            df = df.drop(columns=["_ts"], errors="ignore")
        out = []
        for rec in df.head(n).to_dict(orient="records"):
            clean = {}
            for k, v in rec.items():
                if pd.isna(v):
                    continue
                if hasattr(v, "isoformat"):
                    clean[k] = v.isoformat()
                else:
                    clean[k] = v
            out.append(clean)
        return out

    def _empty_evidence(self, question: str, reason: str) -> Dict[str, Any]:
        return {
            "status": "no_data",
            "retrieval_source": "raw_observability_logs",
            "question": question,
            "reason": reason,
            "retrieval_window": {"start": None, "end": None, "num_resource_rows": 0, "num_communication_rows": 0, "num_event_rows": 0},
        }

    def build_evidence(self, question: str, spec: Dict[str, Any], max_records: int = 40) -> Dict[str, Any]:
        if pd is None:
            return self._empty_evidence(question, "pandas is not installed; log_rag cannot read CSV logs")
        if self.latest_ts is None:
            return self._empty_evidence(question, "no log CSV data found or timestamp parsing failed")

        start, end, _ = self._window_bounds(question)
        rwin = self._filter_window(self.resource_df, start, end)
        cwin = self._filter_window(self.communication_df, start, end)
        ewin = self._filter_window(self.events_df, start, end)

        num_r = 0 if rwin is None else int(len(rwin))
        num_c = 0 if cwin is None else int(len(cwin))
        num_e = 0 if ewin is None else int(len(ewin))

        evidence: Dict[str, Any] = {
            "status": "ok" if (num_r + num_c + num_e) > 0 else "no_data",
            "retrieval_source": "raw_observability_logs",
            "question": question,
            "task": spec.get("task"),
            "note": "Log-RAG baseline: retrieved from raw CSV observability logs, not from RTKG snapshots or graph topology.",
            "retrieval_window": {
                "start": str(start) if start is not None else None,
                "end": str(end) if end is not None else None,
                "num_resource_rows": num_r,
                "num_communication_rows": num_c,
                "num_event_rows": num_e,
            },
        }

        task = spec.get("task_group", spec.get("task"))
        top_n = max(1, min(max_records, 50))

        # Include recent event snippets for context in every mode, but keep them short.
        if ewin is not None and not ewin.empty:
            ev = ewin.sort_values("_ts", ascending=False)
            evidence["recent_event_log_records"] = self._records(
                ev,
                min(10, top_n),
                ["timestamp", "pod", "event_type", "reason", "message", "failure_type", "target_service"],
            )

        if task == "node_anomaly":
            if rwin is not None and not rwin.empty and "pod" in rwin.columns:
                tmp = rwin.copy()
                if "pod_under_failure" in tmp.columns:
                    tmp["_anom"] = pd.to_numeric(tmp["pod_under_failure"], errors="coerce").fillna(0) > 0
                elif "in_failure_window" in tmp.columns and "pod_is_target" in tmp.columns:
                    tmp["_anom"] = (pd.to_numeric(tmp["in_failure_window"], errors="coerce").fillna(0) > 0) & (pd.to_numeric(tmp["pod_is_target"], errors="coerce").fillna(0) > 0)
                else:
                    tmp["_anom"] = False
                grp = tmp.groupby("pod", dropna=True).agg(
                    log_records=("pod", "size"),
                    anomalous_log_records=("_anom", "sum"),
                ).reset_index()
                grp["anomaly_rate_percent_from_logs"] = (grp["anomalous_log_records"] / grp["log_records"] * 100.0).round(4)
                grp = grp.sort_values(["anomaly_rate_percent_from_logs", "anomalous_log_records", "log_records"], ascending=False)
                evidence["retrieved_log_summary"] = "Candidate services ranked by anomaly rate computed from raw resource log records."
                evidence["candidate_services_from_logs"] = self._records(grp, top_n)
                anom_rows = tmp[tmp["_anom"]].sort_values("_ts", ascending=False)
                evidence["sample_anomalous_resource_log_records"] = self._records(
                    anom_rows,
                    min(10, top_n),
                    ["timestamp", "pod", "pod_under_failure", "in_failure_window", "failure_type", "target_service"],
                )

        elif task == "cpu":
            cpu_col = "container_cpu_usage_seconds_total"
            if rwin is not None and not rwin.empty and {"pod", cpu_col}.issubset(rwin.columns):
                tmp = rwin.copy()
                tmp[cpu_col] = pd.to_numeric(tmp[cpu_col], errors="coerce")
                grp = tmp.groupby("pod", dropna=True).agg(
                    max_cpu_from_logs=(cpu_col, "max"),
                    mean_cpu_from_logs=(cpu_col, "mean"),
                    log_records=("pod", "size"),
                ).reset_index()
                grp = grp.sort_values("max_cpu_from_logs", ascending=False)
                evidence["retrieved_log_summary"] = "Candidate services ranked by CPU metric from raw resource log records."
                evidence["candidate_services_from_logs"] = self._records(grp, top_n)
                rows = tmp.sort_values(cpu_col, ascending=False)
                evidence["top_resource_log_records"] = self._records(
                    rows,
                    top_n,
                    ["timestamp", "pod", cpu_col, "container_cpu_system_seconds_total", "failure_type", "target_service", "pod_under_failure"],
                )

        elif task == "memory":
            mem_col = "container_memory_working_set_bytes"
            if rwin is not None and not rwin.empty and {"pod", mem_col}.issubset(rwin.columns):
                tmp = rwin.copy()
                tmp[mem_col] = pd.to_numeric(tmp[mem_col], errors="coerce")
                grp = tmp.groupby("pod", dropna=True).agg(
                    max_memory_from_logs=(mem_col, "max"),
                    mean_memory_from_logs=(mem_col, "mean"),
                    log_records=("pod", "size"),
                ).reset_index()
                grp = grp.sort_values("max_memory_from_logs", ascending=False)
                evidence["retrieved_log_summary"] = "Candidate services ranked by memory metric from raw resource log records."
                evidence["candidate_services_from_logs"] = self._records(grp, top_n)
                rows = tmp.sort_values(mem_col, ascending=False)
                evidence["top_resource_log_records"] = self._records(
                    rows,
                    top_n,
                    ["timestamp", "pod", mem_col, "container_memory_rss", "failure_type", "target_service", "pod_under_failure"],
                )

        elif task == "edge_error":
            metric = "error_rate"
            if cwin is not None and not cwin.empty and {"source", "destination", metric}.issubset(cwin.columns):
                tmp = cwin.copy()
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                if "error_count" in tmp.columns:
                    tmp["error_count"] = pd.to_numeric(tmp["error_count"], errors="coerce").fillna(0)
                else:
                    tmp["error_count"] = 0
                grp = tmp.groupby(["source", "destination"], dropna=True).agg(
                    max_error_rate_from_logs=(metric, "max"),
                    mean_error_rate_from_logs=(metric, "mean"),
                    total_error_count_from_logs=("error_count", "sum"),
                    log_records=(metric, "size"),
                ).reset_index()
                grp["edge"] = grp["source"].astype(str) + " -> " + grp["destination"].astype(str)
                grp = grp.sort_values(["max_error_rate_from_logs", "total_error_count_from_logs"], ascending=False)
                evidence["retrieved_log_summary"] = "Candidate communication edges ranked by error rate from raw communication log records."
                evidence["candidate_edges_from_logs"] = self._records(grp, top_n)
                rows = tmp.sort_values(metric, ascending=False)
                evidence["top_communication_log_records"] = self._records(
                    rows,
                    top_n,
                    ["timestamp", "source", "destination", "error_rate", "error_count", "total_request", "failure_type", "target_service", "edge_under_failure"],
                )

        elif task == "edge_latency":
            metric = "average_latency"
            if cwin is not None and not cwin.empty and {"source", "destination", metric}.issubset(cwin.columns):
                tmp = cwin.copy()
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                if "p99_latency" in tmp.columns:
                    tmp["p99_latency"] = pd.to_numeric(tmp["p99_latency"], errors="coerce")
                grp = tmp.groupby(["source", "destination"], dropna=True).agg(
                    max_average_latency_from_logs=(metric, "max"),
                    mean_average_latency_from_logs=(metric, "mean"),
                    log_records=(metric, "size"),
                ).reset_index()
                if "p99_latency" in tmp.columns:
                    p99 = tmp.groupby(["source", "destination"], dropna=True)["p99_latency"].max().reset_index(name="max_p99_latency_from_logs")
                    grp = grp.merge(p99, on=["source", "destination"], how="left")
                grp["edge"] = grp["source"].astype(str) + " -> " + grp["destination"].astype(str)
                grp = grp.sort_values("max_average_latency_from_logs", ascending=False)
                evidence["retrieved_log_summary"] = "Candidate communication edges ranked by latency from raw communication log records."
                evidence["candidate_edges_from_logs"] = self._records(grp, top_n)
                rows = tmp.sort_values(metric, ascending=False)
                evidence["top_communication_log_records"] = self._records(
                    rows,
                    top_n,
                    ["timestamp", "source", "destination", "average_latency", "p50_latency", "p90_latency", "p99_latency", "failure_type", "target_service", "edge_under_failure"],
                )

        elif task == "combined_anomaly_evidence":
            scores: Dict[str, float] = {}
            details: Dict[str, Dict[str, Any]] = {}

            def add_score(service: Any, score: float, reason: str) -> None:
                if service is None or str(service).lower() in {"", "nan", "unknown"}:
                    return
                svc = str(service)
                scores[svc] = scores.get(svc, 0.0) + float(score)
                d = details.setdefault(svc, {"microservice": svc, "log_score": 0.0, "reasons": []})
                d["log_score"] = round(scores[svc], 4)
                if reason not in d["reasons"]:
                    d["reasons"].append(reason)

            if rwin is not None and not rwin.empty and "pod" in rwin.columns:
                tmp = rwin.copy()
                if "pod_under_failure" in tmp.columns:
                    tmp["_anom"] = pd.to_numeric(tmp["pod_under_failure"], errors="coerce").fillna(0) > 0
                elif "in_failure_window" in tmp.columns and "pod_is_target" in tmp.columns:
                    tmp["_anom"] = (pd.to_numeric(tmp["in_failure_window"], errors="coerce").fillna(0) > 0) & (pd.to_numeric(tmp["pod_is_target"], errors="coerce").fillna(0) > 0)
                else:
                    tmp["_anom"] = False
                for svc, cnt in tmp.groupby("pod")["_anom"].sum().items():
                    if cnt > 0:
                        add_score(svc, float(cnt), "resource anomaly/failure log records")

            if cwin is not None and not cwin.empty:
                tmp = cwin.copy()
                if "error_rate" in tmp.columns:
                    tmp["error_rate"] = pd.to_numeric(tmp["error_rate"], errors="coerce").fillna(0)
                else:
                    tmp["error_rate"] = 0
                if "edge_under_failure" in tmp.columns:
                    tmp["_bad_edge"] = pd.to_numeric(tmp["edge_under_failure"], errors="coerce").fillna(0) > 0
                else:
                    tmp["_bad_edge"] = tmp["error_rate"] > 0
                bad = tmp[tmp["_bad_edge"]]
                for col in ["source", "destination"]:
                    if col in bad.columns:
                        for svc, cnt in bad.groupby(col).size().items():
                            if cnt > 0:
                                add_score(svc, float(cnt), "communication error/failure log records")

            if ewin is not None and not ewin.empty:
                pod_col = "pod" if "pod" in ewin.columns else ("target_service" if "target_service" in ewin.columns else None)
                if pod_col:
                    for svc, cnt in ewin.groupby(pod_col).size().items():
                        if cnt > 0:
                            add_score(svc, float(cnt), "runtime event log records")

            ranked = sorted(details.values(), key=lambda r: r["log_score"], reverse=True)
            evidence["retrieved_log_summary"] = "Candidate services ranked by combined anomaly evidence from raw resource, communication, and event logs."
            evidence["candidate_services_from_logs"] = ranked[:top_n]

        elif task == "unhealthy_edge":
            if cwin is not None and not cwin.empty and {"source", "destination"}.issubset(cwin.columns):
                tmp = cwin.copy()
                if "edge_under_failure" in tmp.columns:
                    tmp["_unhealthy"] = pd.to_numeric(tmp["edge_under_failure"], errors="coerce").fillna(0) > 0
                elif "error_rate" in tmp.columns:
                    tmp["_unhealthy"] = pd.to_numeric(tmp["error_rate"], errors="coerce").fillna(0) > 0
                else:
                    tmp["_unhealthy"] = False
                grp = tmp.groupby(["source", "destination"], dropna=True).agg(
                    log_records=("source", "size"),
                    unhealthy_log_records=("_unhealthy", "sum"),
                ).reset_index()
                grp["unhealthy_rate_percent_from_logs"] = (grp["unhealthy_log_records"] / grp["log_records"] * 100.0).round(4)
                grp["edge"] = grp["source"].astype(str) + " -> " + grp["destination"].astype(str)
                grp = grp.sort_values(["unhealthy_rate_percent_from_logs", "unhealthy_log_records", "log_records"], ascending=False)
                evidence["retrieved_log_summary"] = "Candidate communication edges ranked by unhealthy/failure frequency from raw communication log records."
                evidence["candidate_edges_from_logs"] = self._records(grp, top_n)
                bad = tmp[tmp["_unhealthy"]].sort_values("_ts", ascending=False)
                evidence["sample_unhealthy_communication_log_records"] = self._records(
                    bad,
                    min(10, top_n),
                    ["timestamp", "source", "destination", "edge_under_failure", "error_rate", "failure_type", "target_service"],
                )

        elif task == "event_affected_service":
            if ewin is not None and not ewin.empty:
                pod_col = "pod" if "pod" in ewin.columns else ("target_service" if "target_service" in ewin.columns else None)
                if pod_col:
                    grp = ewin.groupby(pod_col, dropna=True).size().reset_index(name="event_count_from_logs")
                    grp = grp.rename(columns={pod_col: "microservice"})
                    grp = grp.sort_values("event_count_from_logs", ascending=False)
                    evidence["retrieved_log_summary"] = "Candidate services ranked by runtime event count from raw Kubernetes/event log records."
                    evidence["candidate_services_from_logs"] = self._records(grp, top_n)
                    evidence["top_event_log_records"] = self._records(
                        ewin.sort_values("_ts", ascending=False),
                        top_n,
                        ["timestamp", pod_col, "event_type", "reason", "message", "failure_type", "target_service"],
                    )

        else:
            # Summary baseline: provide a compact mix of log contexts.
            evidence["retrieved_log_summary"] = "Compact raw-log context for system-health summary."
            if rwin is not None and not rwin.empty:
                evidence["sample_resource_log_records"] = self._records(
                    rwin.sort_values("_ts", ascending=False),
                    min(top_n, 20),
                )
            if cwin is not None and not cwin.empty:
                evidence["sample_communication_log_records"] = self._records(
                    cwin.sort_values("_ts", ascending=False),
                    min(top_n, 20),
                )

        # If the task-specific retrieval produced no candidate fields, make that explicit.
        has_candidates = any(k in evidence for k in [
            "candidate_services_from_logs",
            "candidate_edges_from_logs",
            "top_resource_log_records",
            "top_communication_log_records",
            "sample_resource_log_records",
            "sample_communication_log_records",
        ])
        if evidence["status"] == "ok" and not has_candidates:
            evidence["status"] = "no_data"
            evidence["reason"] = "log rows exist in the time window, but no task-relevant log columns were found"

        return evidence


def log_rag_prompt(
    log_corpus: Optional[LogRAGCorpus],
    question: str,
    spec: Dict[str, Any],
    compact_chars: int,
    max_records: int,
) -> Tuple[str, Dict[str, Any], str]:
    """Log-RAG baseline: retrieve compact evidence from raw CSV logs, not graph snapshots."""
    if log_corpus is None:
        evidence = {
            "status": "no_data",
            "retrieval_source": "raw_observability_logs",
            "reason": "log_rag selected but no LogRAGCorpus was initialized",
            "question": question,
        }
    else:
        evidence = log_corpus.build_evidence(question, spec, max_records=max_records)

    if evidence.get("status") == "ok":
        direct = "Log-RAG retrieved raw log evidence for the requested time window. The answer must be inferred from the retrieved log records."
    else:
        direct = "No sufficient raw-log evidence was retrieved for this question."

    prompt = f"""
You are an expert SRE assistant for Kubernetes microservice observability.

This is a Log-RAG baseline. You are given retrieved evidence from raw observability logs, not from the RTKG graph.
Use ONLY the raw-log evidence below.
Do not use outside knowledge.
Do not reason step by step. Do not include <think> tags.
Do not invent services, edges, metrics, timestamps, anomaly types, or values.
For service questions, answer with the service/pod name found in the candidate log records.
For edge questions, answer with the source -> destination edge found in the candidate log records.
If the log evidence has STATUS "no_data", say that there is insufficient log evidence.
If the candidate log records show no positive/anomalous/error candidate, say that no positive candidate was found in the retrieved logs.
Keep the answer to one sentence.

RETRIEVED_LOG_RAG_EVIDENCE_JSON:
{safe_compact_json(evidence, max_chars=compact_chars)}

USER_QUESTION:
{question}

ANSWER:
""".strip()
    return prompt, evidence, direct


# -----------------------------------------------------------------------------
# Prompt dispatcher
# -----------------------------------------------------------------------------

def build_prompt_for_mode(
    rag: RTKGGraphRAG,
    log_corpus: Optional[LogRAGCorpus],
    question: str,
    spec: Dict[str, Any],
    mode: str,
    latest_window: str,
    compact_chars: int,
    log_rag_max_records: int,
) -> Tuple[str, Dict[str, Any], str]:
    if mode == "no_rag":
        return no_rag_prompt(question)
    if mode == "log_rag":
        return log_rag_prompt(log_corpus, question, spec, compact_chars, log_rag_max_records)
    if mode == "latest_snapshot_rag":
        return latest_snapshot_rag_prompt(rag, question, spec, latest_window, compact_chars)
    if mode == "temporal_graph_rag_evidence_only":
        return temporal_graph_rag_evidence_only_prompt(rag, question, spec, compact_chars)
    if mode == "temporal_graph_rag":
        return temporal_graph_rag_prompt(rag, question, spec, compact_chars)
    raise ValueError(f"Unknown ablation mode: {mode}")


# -----------------------------------------------------------------------------
# Ollama + evaluation
# -----------------------------------------------------------------------------

def ollama_generate(
    ollama_url: str,
    model: str,
    prompt: str,
    temperature: float,
    num_ctx: int,
    timeout: int,
    num_predict: int,
) -> Tuple[str, Dict[str, Any], float]:
    """Call Ollama safely. Timeouts/errors are returned instead of raised."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    t0 = time.perf_counter()
    try:
        res = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
        wall_ms = (time.perf_counter() - t0) * 1000
        if res.status_code != 200:
            return (
                f"OLLAMA_ERROR_{res.status_code}: {res.text[:500]}",
                {"status_code": res.status_code, "error": True},
                wall_ms,
            )
        data = res.json()
        return data.get("response", ""), data, wall_ms

    except requests.exceptions.Timeout:
        wall_ms = (time.perf_counter() - t0) * 1000
        return "OLLAMA_TIMEOUT", {"error": True, "timeout": True}, wall_ms

    except requests.exceptions.RequestException as e:
        wall_ms = (time.perf_counter() - t0) * 1000
        return f"OLLAMA_REQUEST_ERROR: {str(e)}", {"error": True}, wall_ms


def ns_to_ms(x: Any) -> Optional[float]:
    try:
        return round(float(x) / 1_000_000.0, 3)
    except Exception:
        return None


def tokens_per_second(eval_count: Any, eval_duration_ns: Any) -> Optional[float]:
    try:
        eval_count = float(eval_count)
        sec = float(eval_duration_ns) / 1_000_000_000.0
        if sec <= 0:
            return None
        return round(eval_count / sec, 3)
    except Exception:
        return None


def answer_mentions_insufficient_evidence(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in INSUFFICIENT_PATTERNS)


def answer_mentions_no_positive_evidence(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in NO_POSITIVE_PATTERNS)


def answer_has_unsupported_claims(answer: str, expected_status: str) -> bool:
    """Lightweight unsupported-claim filter for strict scoring.

    This is intentionally conservative: it penalizes answers that expose hidden
    reasoning tags or answer with insufficient/no-evidence wording when the
    deterministic ground truth contains a positive candidate.
    """
    low = answer.lower()
    if "<think>" in low or "</think>" in low:
        return True
    if expected_status == "ok" and (answer_mentions_insufficient_evidence(answer) or answer_mentions_no_positive_evidence(answer)):
        return True
    return False


def evaluate_correctness(
    answer: str,
    meta: Dict[str, Any],
    expected: Dict[str, Any],
    spec: Dict[str, Any],
    rel_tol: float,
    abs_tol: float,
    strict: bool = True,
) -> Tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool]]:
    """Return (answer_correct, entity_correct, numeric_correct, unsupported_claim).

    In strict mode, answer_correct requires:
      entity_correct AND numeric_correct AND no unsupported claim.
    This makes the benchmark harder than a simple entity mention check.
    """
    if meta.get("timeout") or meta.get("error") or answer.startswith("OLLAMA_"):
        return False, False, False, True

    expected_entity = expected.get("expected_entity")
    expected_value = expected.get("expected_value")
    expected_status = expected.get("expected_status")
    unsupported = answer_has_unsupported_claims(answer, str(expected_status))

    if expected_status == "ok":
        entity_ok = contains_entity(answer, expected_entity, spec["entity_type"])
        value_ok = numeric_correct(answer, expected_value, rel_tol, abs_tol)
        if strict:
            answer_ok = bool(entity_ok) and bool(value_ok) and not unsupported
        else:
            answer_ok = bool(entity_ok)
        return answer_ok, entity_ok, value_ok, unsupported

    if expected_status == "summary":
        return None, None, None, None

    if expected_status == "no_data":
        answer_ok = answer_mentions_insufficient_evidence(answer) and not unsupported
        return answer_ok, answer_ok, None, unsupported

    if expected_status == "no_positive_evidence":
        # Important: "insufficient evidence" is NOT counted as correct here.
        # The graph was retrieved successfully; the correct answer is that there is no positive/anomalous candidate.
        answer_ok = answer_mentions_no_positive_evidence(answer) and not unsupported
        return answer_ok, answer_ok, None, unsupported

    return False, False, None, unsupported


def evidence_counts(used_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comparable evidence-size fields for any retrieval mode."""
    if not used_evidence:
        return {
            "num_snapshots_used": 0,
            "num_microservices_used": 0,
            "num_edges_used": 0,
            "num_events_used": 0,
            "num_resource_log_rows_used": 0,
            "num_communication_log_rows_used": 0,
            "num_event_log_rows_used": 0,
        }

    rw = used_evidence.get("retrieval_window", {}) if isinstance(used_evidence, dict) else {}
    gs = used_evidence.get("graph_summary", {}) if isinstance(used_evidence, dict) else {}

    return {
        "num_snapshots_used": rw.get("num_snapshots", 0),
        "num_microservices_used": gs.get("num_microservices_observed", 0),
        "num_edges_used": gs.get("num_communication_edges_observed", 0),
        "num_events_used": gs.get("num_events", 0),
        "num_resource_log_rows_used": rw.get("num_resource_rows", 0),
        "num_communication_log_rows_used": rw.get("num_communication_rows", 0),
        "num_event_log_rows_used": rw.get("num_event_rows", 0),
    }


def evaluate(args: argparse.Namespace) -> List[Dict[str, Any]]:
    models = split_csv_arg(args.models)
    windows = split_csv_arg(args.windows)
    modes = split_csv_arg(args.ablation_modes)
    rows: List[Dict[str, Any]] = []

    for mode in modes:
        if mode not in VALID_ABLATION_MODES:
            raise ValueError(f"Invalid ablation mode: {mode}. Valid modes: {VALID_ABLATION_MODES}")

    question_specs = [q for q in QUESTION_TEMPLATES if not (args.exclude_summary and q["task"] == "system_health_summary")]

    # Use one Graph-RAG instance and switch model names in Ollama calls.
    rag = RTKGGraphRAG(
        snapshots_dir=args.snapshots,
        model=models[0] if models else "mistral:latest",
        ollama_url=args.ollama_url,
        default_minutes=args.default_minutes,
        top_k=args.top_k,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
    )

    log_corpus: Optional[LogRAGCorpus] = None
    if "log_rag" in modes:
        log_corpus = LogRAGCorpus(
            resource_csv=args.resource_csv,
            communication_csv=args.communication_csv,
            events_csv=args.events_csv,
        )
        if log_corpus.latest_ts is None:
            print("[WARN] log_rag selected, but no usable CSV log data was loaded. Check --resource-csv, --communication-csv, and --events-csv.")

    for window in windows:
        for spec in question_specs:
            question = spec["question"].format(window=window)

            # Ground truth is always full Temporal Graph-RAG evidence.
            gt_evidence = rag.build_evidence(question)
            gt_direct_answer = rag.direct_answer(question, gt_evidence)
            expected = expected_from_evidence(gt_evidence, spec)

            for mode in modes:
                prompt, used_evidence, used_direct_answer = build_prompt_for_mode(
                    rag=rag,
                    log_corpus=log_corpus,
                    question=question,
                    spec=spec,
                    mode=mode,
                    latest_window=args.latest_window,
                    compact_chars=args.compact_chars,
                    log_rag_max_records=args.log_rag_max_records,
                )

                evidence_json_chars = len(safe_compact_json(used_evidence, max_chars=10_000_000)) if used_evidence else 0
                prompt_chars = len(prompt)
                counts = evidence_counts(used_evidence)

                for model in models:
                    for run_idx in range(1, args.runs + 1):
                        answer, meta, wall_ms = ollama_generate(
                            args.ollama_url,
                            model,
                            prompt,
                            args.temperature,
                            args.num_ctx,
                            args.timeout,
                            args.num_predict,
                        )

                        answer_correct, entity_ok, value_ok, unsupported_claim = evaluate_correctness(
                            answer=answer,
                            meta=meta,
                            expected=expected,
                            spec=spec,
                            rel_tol=args.rel_tol,
                            abs_tol=args.abs_tol,
                            strict=not args.entity_only_accuracy,
                        )

                        row = {
                            "ablation_mode": mode,
                            "model": model,
                            "run": run_idx,
                            "task": spec["task"],
                            "question": question,
                            "window": window,
                            "expected_status": expected.get("expected_status"),
                            "expected_entity": expected.get("expected_entity"),
                            "expected_value": expected.get("expected_value"),
                            "entity_correct": entity_ok,
                            "numeric_correct": value_ok,
                            "unsupported_claim": unsupported_claim,
                            "strict_correct": answer_correct,
                            "answer_correct": answer_correct,
                            "timeout": bool(meta.get("timeout", False)),
                            "error": bool(meta.get("error", False)),
                            "wall_latency_ms": round(wall_ms, 3),
                            "ollama_total_duration_ms": ns_to_ms(meta.get("total_duration")),
                            "ollama_load_duration_ms": ns_to_ms(meta.get("load_duration")),
                            "ollama_prompt_eval_duration_ms": ns_to_ms(meta.get("prompt_eval_duration")),
                            "ollama_eval_duration_ms": ns_to_ms(meta.get("eval_duration")),
                            "prompt_eval_count": meta.get("prompt_eval_count"),
                            "eval_count": meta.get("eval_count"),
                            "tokens_per_second": tokens_per_second(meta.get("eval_count"), meta.get("eval_duration")),
                            "num_snapshots_ground_truth": gt_evidence.get("retrieval_window", {}).get("num_snapshots") if isinstance(gt_evidence, dict) else None,
                            **counts,
                            "prompt_chars": prompt_chars,
                            "evidence_json_chars": evidence_json_chars,
                            "ground_truth_direct_graph_answer": gt_direct_answer,
                            "used_direct_answer_or_retrieval_note": used_direct_answer,
                            "llm_answer": answer.replace("\n", " ").strip(),
                        }
                        rows.append(row)

                        print(
                            f"[{model}] {mode} | {spec['task']} | {window} | run {run_idx}: "
                            f"correct={answer_correct}, latency={wall_ms:.1f} ms"
                        )
    return rows


# -----------------------------------------------------------------------------
# CSV writers
# -----------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    # Preserve first-row order, while supporting any later extra keys.
    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    idx = max(0, int(math.ceil(p * len(values))) - 1)
    return values[idx]


def write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r.get("ablation_mode", "temporal_graph_rag"), r["model"]), []).append(r)

    summary_rows = []
    for (mode, model), rs in sorted(by_key.items()):
        acc_rows = [r for r in rs if r["answer_correct"] in [True, False]]
        entity_rows = [r for r in rs if r["entity_correct"] in [True, False]]
        num_rows = [r for r in rs if r["numeric_correct"] in [True, False]]
        lat = [float(r["wall_latency_ms"]) for r in rs]
        tps = [float(r["tokens_per_second"]) for r in rs if r.get("tokens_per_second") not in [None, ""]]
        timeouts = [r for r in rs if r.get("timeout") is True]
        errors = [r for r in rs if r.get("error") is True]
        unsupported_rows = [r for r in rs if r.get("unsupported_claim") is True]

        summary_rows.append({
            "ablation_mode": mode,
            "model": model,
            "n_answers_total": len(rs),
            "n_accuracy_answers": len(acc_rows),
            "strict_accuracy": round(sum(bool(r["answer_correct"]) for r in acc_rows) / len(acc_rows), 4) if acc_rows else None,
            "accuracy": round(sum(bool(r["answer_correct"]) for r in acc_rows) / len(acc_rows), 4) if acc_rows else None,
            "entity_accuracy": round(sum(bool(r["entity_correct"]) for r in entity_rows) / len(entity_rows), 4) if entity_rows else None,
            "numeric_accuracy_when_number_expected": round(sum(bool(r["numeric_correct"]) for r in num_rows) / len(num_rows), 4) if num_rows else None,
            "avg_latency_ms": round(statistics.mean(lat), 3) if lat else None,
            "median_latency_ms": round(statistics.median(lat), 3) if lat else None,
            "p95_latency_ms": round(percentile(lat, 0.95), 3) if lat else None,
            "avg_tokens_per_second": round(statistics.mean(tps), 3) if tps else None,
            "unsupported_claim_rate": round(len(unsupported_rows) / len(rs), 4) if rs else None,
            "timeout_rate": round(len(timeouts) / len(rs), 4) if rs else None,
            "error_rate": round(len(errors) / len(rs), 4) if rs else None,
        })

    write_csv(summary_rows, path)

    print("\nSUMMARY")
    print("=" * 80)
    for r in summary_rows:
        print(json.dumps(r, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True, help="Folder containing RTKG JSON snapshots")
    ap.add_argument("--models", default="mistral:latest", help="Comma-separated Ollama model names")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--windows", default="15 seconds,1 minute,5 minutes,10 minutes")
    ap.add_argument("--runs", type=int, default=3, help="Repeat each model/question this many times")
    ap.add_argument("--out", default="rtkg_llm_lograg_ablation_results_v6.csv")
    ap.add_argument("--summary-out", default="rtkg_llm_lograg_ablation_summary_v6.csv")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--num-predict", type=int, default=128, help="Maximum generated tokens per response")
    ap.add_argument("--default-minutes", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--rel-tol", type=float, default=0.05, help="Relative tolerance for numeric correctness")
    ap.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute tolerance for numeric correctness")
    ap.add_argument(
        "--ablation-modes",
        default="log_rag,latest_snapshot_rag,temporal_graph_rag_evidence_only,temporal_graph_rag",
        help="Comma-separated ablation modes: no_rag,log_rag,latest_snapshot_rag,temporal_graph_rag_evidence_only,temporal_graph_rag",
    )
    ap.add_argument(
        "--latest-window",
        default="15 seconds",
        help="Window used for latest-snapshot RAG ablation",
    )
    ap.add_argument(
        "--compact-chars",
        type=int,
        default=6000,
        help="Maximum chars for compact computed evidence in the prompt",
    )
    ap.add_argument(
        "--exclude-summary",
        action="store_true",
        help="Exclude open-ended system-health summary questions from the run",
    )
    ap.add_argument(
        "--resource-csv",
        default="aggregated_pod_resource_consumption.csv",
        help="Raw resource log CSV used by the Log-RAG baseline",
    )
    ap.add_argument(
        "--communication-csv",
        default="aggregated_pod_communication.csv",
        help="Raw communication log CSV used by the Log-RAG baseline",
    )
    ap.add_argument(
        "--events-csv",
        default="aggregated_pod_events.csv",
        help="Raw Kubernetes/event log CSV used by the Log-RAG baseline",
    )
    ap.add_argument(
        "--log-rag-max-records",
        type=int,
        default=25,
        help="Maximum candidate log records included in the Log-RAG prompt",
    )
    ap.add_argument(
        "--entity-only-accuracy",
        action="store_true",
        help="Use lenient entity-only accuracy instead of strict entity+numeric accuracy",
    )
    args = ap.parse_args()

    rows = evaluate(args)
    write_csv(rows, Path(args.out))
    write_summary(rows, Path(args.summary_out))
    print(f"\nDetailed results: {args.out}")
    print(f"Summary results:  {args.summary_out}")


if __name__ == "__main__":
    main()
