# muS-RTKG Visualization Tool

This module contains the visualization component of the muS-RTKG framework.

The visualization tool is designed to help operators inspect Runtime Temporal Knowledge Graph (RTKG) snapshots generated from Kubernetes microservice telemetry. It exposes the graph structure and runtime context, including:

- microservice nodes,
- Kubernetes worker or deployment nodes,
- service-to-service communication edges,
- resource metrics,
- communication metrics,
- anomaly labels,
- runtime events,
- temporal evolution across snapshots.

## Example

The figure below shows an example RTKG snapshot visualization.

![RTKG visualization example](examples/rtkg_example.png)

## Role in the framework

The visualization module supports interactive exploration of the RTKG and helps operators understand how anomalies evolve and propagate across microservices. It complements the RCA module and the LLM interface by providing a direct graph-level view of the runtime system state.
