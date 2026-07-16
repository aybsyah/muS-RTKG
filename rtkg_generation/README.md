# RTKG Generation

This folder contains the script used to generate Runtime Temporal Knowledge Graphs (RTKGs) from Kubernetes microservice observability data.

## Input files

The generator expects three CSV files:

```text
aggregated_pod_resource_consumption.csv
aggregated_pod_communication.csv
aggregated_pod_events.csv
python build_rtkg_snapshots.py \
  --resource_csv aggregated_pod_resource_consumption.csv \
  --communication_csv aggregated_pod_communication.csv \
  --events_csv aggregated_pod_events.csv \
  --output_dir rtkg_snapshots \
  --window 15s
python build_rtkg_snapshots.py \
  --resource_csv aggregated_pod_resource_consumption.csv \
  --communication_csv aggregated_pod_communication.csv \
  --events_csv aggregated_pod_events.csv \
  --output_dir rtkg_snapshots_test \
  --window 15s \
  --max_rows 500

### 2. Create the LLM README

```bash
cat > llm_temporal_graph_rag/README.md <<'EOF'
# LLM Interface with Temporal Graph-RAG

This folder contains the local LLM interface used to query RTKG snapshots using natural language.

The interface uses local models served through Ollama. The LLM is used as a natural-language explanation layer over retrieved RTKG evidence, not as an unconstrained reasoning engine.

## Main scripts

```text
rtkg_graph_rag_ollama_v2.py
evaluate_rtkg_llm_ablation_v6_strict.py
python rtkg_graph_rag_ollama_v2.py \
  --snapshots rtkg_snapshots \
  --model mistral:latest \
  --question "Which microservice has the highest CPU usage in the last 5 minutes?" \
  --show-evidence
python evaluate_rtkg_llm_ablation_v6_strict.py \
  --snapshots rtkg_snapshots \
  --resource-csv aggregated_pod_resource_consumption.csv \
  --communication-csv aggregated_pod_communication.csv \
  --events-csv aggregated_pod_events.csv \
  --models mistral:latest,llama3.1:8b \
  --windows "15 seconds,1 minute,5 minutes,10 minutes" \
  --runs 3 \
  --timeout 240 \
  --top-k 3 \
  --num-ctx 2048 \
  --num-predict 128 \
  --exclude-summary \
  --ablation-modes log_rag,latest_snapshot_rag,temporal_graph_rag

### 3. Update the main README

This will replace the current README. Run:

```bash
cat > README.md <<'EOF'
# muS-RTKG: Runtime Temporal Knowledge Graphs for Microservice Observability

**muS-RTKG** is a runtime observability framework for Kubernetes-based microservice systems. It builds **Runtime Temporal Knowledge Graphs (RTKGs)** from resource metrics, service-to-service communication metrics, and runtime events.

The generated temporal graph snapshots provide a unified representation of system state over time and support anomaly detection, root-cause analysis, visualization, and LLM-assisted querying.

## Main Components

### 1. RTKG Generation

The RTKG generation pipeline converts raw observability data into timestamped graph snapshots. The input data include:

- pod and microservice resource metrics,
- service-to-service communication metrics,
- Kubernetes/runtime events.

Each snapshot captures the system state within a time window.

### 2. RCA with Temporal Graph Learning

The repository includes temporal graph learning code for root-cause analysis. The RCA model identifies:

- the faulty microservice,
- the failure type,
- and root-cause candidates from graph sequences.

### 3. LLM Interface with Temporal Graph-RAG

The repository also includes a local LLM interface for querying RTKG snapshots using natural language.

The LLM is not used as an unconstrained reasoning engine. Instead, RTKG evidence is retrieved from graph snapshots, and the LLM verbalizes this evidence in an operator-friendly form.

The evaluation compares:

- **Log-RAG**: retrieval from raw textual observability records,
- **Latest-Snapshot Graph-RAG**: retrieval from only the latest RTKG snapshot window,
- **Temporal Graph-RAG**: retrieval from all RTKG snapshots in the requested temporal window.

## Repository Structure

```text
.
├── rtkg_generation/             # RTKG snapshot generation
├── llm_temporal_graph_rag/      # Local LLM and Temporal Graph-RAG interface
├── train.py                     # RCA model training
├── top-k.py                     # Top-k RCA evaluation
├── model.py                     # RCA model architecture
├── dataset2.py                  # Dataset loading utilities
└── README.md
aggregated_pod_resource_consumption.csv
aggregated_pod_communication.csv
aggregated_pod_events.csv
python rtkg_generation/build_rtkg_snapshots.py \
  --resource_csv aggregated_pod_resource_consumption.csv \
  --communication_csv aggregated_pod_communication.csv \
  --events_csv aggregated_pod_events.csv \
  --output_dir rtkg_snapshots \
  --window 15s
python llm_temporal_graph_rag/rtkg_graph_rag_ollama_v2.py \
  --snapshots rtkg_snapshots \
  --model mistral:latest \
  --question "Which microservice has the highest CPU usage in the last 5 minutes?" \
  --show-evidence
python llm_temporal_graph_rag/evaluate_rtkg_llm_ablation_v6_strict.py \
  --snapshots rtkg_snapshots \
  --resource-csv aggregated_pod_resource_consumption.csv \
  --communication-csv aggregated_pod_communication.csv \
  --events-csv aggregated_pod_events.csv \
  --models mistral:latest,llama3.1:8b \
  --windows "15 seconds,1 minute,5 minutes,10 minutes" \
  --runs 3 \
  --timeout 240 \
  --top-k 3 \
  --num-ctx 2048 \
  --num-predict 128 \
  --exclude-summary \
  --ablation-modes log_rag,latest_snapshot_rag,temporal_graph_rag
llm_temporal_graph_rag/results/
llm_temporal_graph_rag/results/

### 4. Add requirements

```bash
cat > requirements.txt <<'EOF'
requests
pandas
numpy
networkx
