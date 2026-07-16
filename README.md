# muS-RTKG: Runtime Temporal Knowledge Graphs for Microservice Observability

**muS-RTKG** is a Kubernetes-native observability framework for microservice systems. It builds **Runtime Temporal Knowledge Graphs (RTKGs)** from resource metrics, service-to-service communication metrics, and runtime events.

The objective of muS-RTKG is to provide a unified temporal graph representation of the runtime system state. Instead of analyzing logs, metrics, traces, events, and anomaly outputs separately, muS-RTKG integrates them into time-indexed graph snapshots that support root-cause analysis, graph visualization, and natural-language querying.

---

## Framework Overview

muS-RTKG represents a running microservice system as a sequence of temporal graph snapshots. Each RTKG snapshot captures the system state within a fixed time window.

In each RTKG snapshot:

- nodes represent microservices, pods, worker nodes, or runtime events;
- edges represent service-to-service communication, deployment relationships, or event-to-service links;
- node attributes store resource metrics, anomaly labels, and runtime metadata;
- edge attributes store request rate, latency, error rate, throughput, and communication anomaly labels;
- runtime events are linked to the affected microservices.

This temporal graph representation enables operators and analytics modules to reason about how anomalies appear, evolve, and propagate across the microservice system.

The framework is organized around three developed components:

1. **RCA with Temporal Graph Learning**
2. **RTKG Visualization Tool**
3. **Local LLM Interface with Temporal Graph-RAG**

---

## Main Developed Components

### 1. RCA with Temporal Graph Learning

The RCA module performs root-cause analysis over sequences of RTKG-derived graphs. It identifies:

- the faulty microservice,
- the failure type,
- and top-k root-cause candidates.

The model combines graph attention and temporal sequence modeling to capture both service dependencies and anomaly evolution over time.

Folder:

```text
rca_temporal_graph_learning/
```

Main files:

```text
dataset2.py
model.py
train.py
top-k.py
service_to_idx.json
failure_to_idx.json
```

Model checkpoints and scalers are stored under:

```text
rca_temporal_graph_learning/artifacts/
```

---

### 2. RTKG Visualization Tool

The visualization module provides a graph-level view of RTKG snapshots. It helps operators inspect:

- microservice nodes,
- service-to-service communication edges,
- resource metrics,
- communication metrics,
- anomaly labels,
- runtime events,
- temporal evolution across snapshots.

Example RTKG visualization:

![RTKG visualization example](docs/figures/rtkg_example.png)

Folder:

```text
visualization/
```

The visualization module complements the RCA and LLM modules. While the RCA module provides automated root-cause localization and the LLM module provides natural-language querying, the visualization module gives operators a direct graph-level view of the runtime system state.

---

### 3. Local LLM Interface with Temporal Graph-RAG

The LLM module provides a local natural-language interface over RTKG snapshots. It uses local LLMs served through Ollama and retrieves structured graph evidence from RTKG snapshots.

The LLM is not used as an unconstrained reasoning engine. Instead, the RTKG retrieval layer provides graph-grounded evidence, and the LLM verbalizes this evidence in an operator-friendly form.

The evaluation compares:

- **Log-RAG**: retrieval from raw textual observability records;
- **Latest-Snapshot Graph-RAG**: retrieval from only the latest RTKG snapshot window;
- **Temporal Graph-RAG**: retrieval from all RTKG snapshots in the requested temporal window.

Folder:

```text
llm_temporal_graph_rag/
```

Main files:

```text
rtkg_graph_rag_ollama_v2.py
evaluate_rtkg_llm_ablation_v6_strict.py
```

Evaluation outputs are stored under:

```text
llm_temporal_graph_rag/results/
```

---

## Repository Structure

```text
.
├── docs/
│   ├── framework_overview.md
│   └── figures/
│       ├── model_architecture_with_layers.png
│       └── rtkg_example.png
│
├── rtkg_generation/
│   ├── README.md
│   └── build_rtkg_snapshots.py
│
├── rca_temporal_graph_learning/
│   ├── README.md
│   ├── dataset2.py
│   ├── model.py
│   ├── train.py
│   ├── top-k.py
│   ├── service_to_idx.json
│   ├── failure_to_idx.json
│   └── artifacts/
│
├── visualization/
│   ├── README.md
│   └── examples/
│       └── rtkg_example.png
│
├── llm_temporal_graph_rag/
│   ├── README.md
│   ├── rtkg_graph_rag_ollama_v2.py
│   ├── evaluate_rtkg_llm_ablation_v6_strict.py
│   └── results/
│
├── examples/
│   └── data/
│       ├── aggregated_pod_resource_consumption.csv
│       ├── aggregated_pod_communication.csv
│       └── aggregated_pod_events.csv
│
├── artifacts/
├── requirements.txt
└── README.md
```

---

## Input Data

The RTKG generation pipeline expects three CSV files:

```text
aggregated_pod_resource_consumption.csv
aggregated_pod_communication.csv
aggregated_pod_events.csv
```

Example input files are stored under:

```text
examples/data/
```

The three input files correspond to:

- **Resource metrics**: CPU, memory, network, pod-level metrics, and failure metadata.
- **Communication metrics**: source service, destination service, request count, error rate, latency, throughput, and communication status.
- **Runtime events**: Kubernetes events, pod events, event reason, event message, and affected services.

---

## Quick Start

### 1. Build RTKG snapshots

```bash
python rtkg_generation/build_rtkg_snapshots.py \
  --resource_csv examples/data/aggregated_pod_resource_consumption.csv \
  --communication_csv examples/data/aggregated_pod_communication.csv \
  --events_csv examples/data/aggregated_pod_events.csv \
  --output_dir rtkg_snapshots \
  --window 15s
```

This command generates timestamped RTKG snapshots in:

```text
rtkg_snapshots/
```

---

### 2. Query RTKG snapshots with a local LLM

```bash
python llm_temporal_graph_rag/rtkg_graph_rag_ollama_v2.py \
  --snapshots rtkg_snapshots \
  --model mistral:latest \
  --question "Which microservice has the highest CPU usage in the last 5 minutes?" \
  --show-evidence
```

Example questions:

```text
Which microservice has the highest CPU usage in the last 5 minutes?
Which communication edge has the highest error rate in the last 1 minute?
Which microservice is affected by the most runtime events in the last 10 minutes?
Which service-to-service communication link was the slowest in the last 5 minutes?
```

---

### 3. Run the LLM retrieval-source evaluation

```bash
python llm_temporal_graph_rag/evaluate_rtkg_llm_ablation_v6_strict.py \
  --snapshots rtkg_snapshots \
  --resource-csv examples/data/aggregated_pod_resource_consumption.csv \
  --communication-csv examples/data/aggregated_pod_communication.csv \
  --events-csv examples/data/aggregated_pod_events.csv \
  --models mistral:latest,llama3.1:8b \
  --windows "15 seconds,1 minute,5 minutes,10 minutes" \
  --runs 3 \
  --timeout 240 \
  --top-k 3 \
  --num-ctx 2048 \
  --num-predict 128 \
  --exclude-summary \
  --ablation-modes log_rag,latest_snapshot_rag,temporal_graph_rag
```

The evaluation reports:

- entity accuracy,
- numeric accuracy,
- strict accuracy,
- response latency,
- timeout rate,
- error rate.

Strict accuracy requires the correct service or communication edge, the correct numerical value when applicable, and no unsupported claims.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
requests
pandas
numpy
networkx
```

The LLM interface also requires a local Ollama server.

Example Ollama commands:

```bash
ollama list
ollama pull mistral
ollama pull llama3.1:8b
```

---

## Main Commands by Component

### RTKG generation

```bash
python rtkg_generation/build_rtkg_snapshots.py \
  --resource_csv examples/data/aggregated_pod_resource_consumption.csv \
  --communication_csv examples/data/aggregated_pod_communication.csv \
  --events_csv examples/data/aggregated_pod_events.csv \
  --output_dir rtkg_snapshots \
  --window 15s
```

### RCA training

```bash
cd rca_temporal_graph_learning
python train.py
```

### RCA top-k evaluation

```bash
cd rca_temporal_graph_learning
python top-k.py
```

### LLM querying

```bash
python llm_temporal_graph_rag/rtkg_graph_rag_ollama_v2.py \
  --snapshots rtkg_snapshots \
  --model mistral:latest \
  --question "Which microservice has the highest CPU usage in the last 5 minutes?" \
  --show-evidence
```

---

## Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@article{yahyaoui2026musrtkg,
  title   = {muS-RTKG: Runtime Temporal Knowledge Graphs for Microservice Observability},
  author  = {Yahyaoui, Aymen and others},
  journal = {Under review},
  year    = {2026}
}
```

---

## License

This repository is released for research purposes. See the LICENSE file for details.
