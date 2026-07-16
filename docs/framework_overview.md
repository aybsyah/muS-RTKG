# muS-RTKG Framework Overview

muS-RTKG is a Kubernetes-native observability framework based on Runtime Temporal Knowledge Graphs (RTKGs).

The framework continuously transforms heterogeneous telemetry into time-indexed graph snapshots. These snapshots unify:

- microservice topology,
- deployment relationships,
- service-to-service communication metrics,
- pod-level resource metrics,
- Kubernetes runtime events,
- anomaly labels,
- diagnostic evidence.

Each RTKG snapshot represents the runtime state of the monitored microservice system within a fixed time window. A sequence of snapshots captures how the system evolves over time.

## Main Developed Components

### 1. RCA Module

The RCA module applies temporal graph learning over RTKG sequences to localize faulty services and classify failure types.

It uses the RTKG as a structured input representation and learns how anomalies evolve and propagate across microservice dependencies.

### 2. RTKG Visualization Tool

The visualization tool exposes RTKG snapshots as graph views. It helps operators inspect service dependencies, anomaly labels, runtime events, and metric values directly from the temporal graph.

### 3. LLM Interface with Temporal Graph-RAG

The LLM module provides a local natural-language interface over RTKG snapshots.

Instead of querying raw logs only, the LLM retrieves structured RTKG evidence from the relevant time window. This allows operators to ask diagnostic questions such as:

- Which microservice has the highest CPU usage in the last five minutes?
- Which communication edge has the highest error rate?
- Which service is affected by the most runtime events?

The LLM is used as an explanation layer over graph evidence, not as an unconstrained reasoning engine.
