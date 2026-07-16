# RCA with Temporal Graph Learning

This module contains the root-cause analysis (RCA) component of the muS-RTKG framework.

The RCA model operates on sequences of RTKG-derived graphs and predicts:

- the faulty microservice,
- the failure type,
- and top-k root-cause candidates.

## Main idea

The model combines graph attention and temporal sequence modeling. Each RTKG snapshot captures the state of the microservice system at a time window. The model encodes each graph snapshot, processes the temporal sequence, and predicts the most likely root-cause service and failure type.

## Main files

```text
dataset2.py
model.py
train.py
top-k.py
service_to_idx.json
failure_to_idx.json
