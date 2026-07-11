import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data


class RCAGraphSequenceDataset(Dataset):
    def __init__(
        self,
        comm_csv,
        res_csv,
        events_csv,
        seq_len=5,
        window_sec=1,
        drop_services=None,
        fit_scaler=True,
        node_scaler=None,
        edge_scaler=None,
        cache_graphs=True,
        service_to_idx=None,
        failure_to_idx=None,
    ):
        self.comm_csv = comm_csv
        self.res_csv = res_csv
        self.events_csv = events_csv
        self.seq_len = seq_len
        self.window_sec = window_sec
        self.drop_services = set(drop_services or [])

        self.node_scaler = node_scaler
        self.edge_scaler = edge_scaler
        self.fit_scaler = fit_scaler
        self.cache_graphs = cache_graphs
        self.fixed_service_to_idx = service_to_idx
        self.fixed_failure_to_idx = failure_to_idx

        self.df_comm = pd.read_csv(comm_csv)
        self.df_res = pd.read_csv(res_csv)
        self.df_evt = pd.read_csv(events_csv)
        self.df_comm.columns = self.df_comm.columns.str.strip()
        self.df_res.columns = self.df_res.columns.str.strip()
        self.df_evt.columns = self.df_evt.columns.str.strip()

        self._prepare_frames()
        self._build_feature_tables()
        self._build_window_indices()
        self._graph_cache = {} if cache_graphs else None
        self._build_sequences()

    @staticmethod
    def _normalize_name(x):
        if not isinstance(x, str):
            return "unknown"
        x = x.strip().lower()
        x = re.sub(r"[^a-z0-9_-]", "", x)
        return x if x else "unknown"

    @staticmethod
    def _safe_mode(values):
        values = [v for v in values if pd.notna(v)]
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]

    @staticmethod
    def _add_temporal_deltas(frame, group_cols, feature_cols):
        if len(frame) == 0 or not feature_cols:
            return frame, []

        frame = frame.sort_values([*group_cols, "window_ts"]).copy()
        delta_cols = []

        grouped = frame.groupby(group_cols, sort=False)
        for feature_col in feature_cols:
            delta_col = f"delta_{feature_col}"
            frame[delta_col] = grouped[feature_col].diff().fillna(0.0)
            delta_cols.append(delta_col)

        if "average_latency" in feature_cols and "delta_average_latency" in frame.columns:
            frame["delta_latency"] = frame["delta_average_latency"]
            delta_cols.append("delta_latency")

        if "error_rate" in feature_cols and "delta_error_rate" in frame.columns:
            frame["delta_error"] = frame["delta_error_rate"]
            delta_cols.append("delta_error")

        return frame, delta_cols

    def _add_causal_score(self, edges):
        if len(edges) == 0:
            edges = edges.copy()
            causal_feature_cols = [
                "causal_score",
                "causal_score_pos",
                "causal_score_norm",
                "causal_rank_pct",
                "causal_window_share",
                "source_causal_pressure",
                "destination_causal_pressure",
                "causal_source_share",
                "causal_destination_share",
                "causal_prev_score",
                "causal_persistence",
                "causal_acceleration",
                "causal_path_score",
                "causal_is_top_edge",
            ]
            for col in causal_feature_cols:
                edges[col] = pd.Series(dtype=np.float32)
            return edges, causal_feature_cols

        edges = edges.copy()
        delta_error = (
            edges["delta_error"].astype(np.float32)
            if "delta_error" in edges.columns
            else pd.Series(np.zeros(len(edges), dtype=np.float32), index=edges.index)
        )
        delta_latency = (
            edges["delta_latency"].astype(np.float32)
            if "delta_latency" in edges.columns
            else pd.Series(np.zeros(len(edges), dtype=np.float32), index=edges.index)
        )
        delta_throughput = (
            edges["delta_throughput"].astype(np.float32)
            if "delta_throughput" in edges.columns
            else pd.Series(np.zeros(len(edges), dtype=np.float32), index=edges.index)
        )

        edges["causal_score"] = (
            0.4 * delta_error
            + 0.3 * delta_latency
            + 0.3 * (-delta_throughput)
        ).fillna(0.0).astype(np.float32)
        edges["causal_score_pos"] = edges["causal_score"].clip(lower=0.0).astype(np.float32)

        window_groups = edges.groupby("window_ts", sort=False)
        source_groups = edges.groupby(["window_ts", "source"], sort=False)
        destination_groups = edges.groupby(["window_ts", "destination"], sort=False)

        window_max = window_groups["causal_score_pos"].transform("max").astype(np.float32)
        window_sum = window_groups["causal_score_pos"].transform("sum").astype(np.float32)
        source_pressure = source_groups["causal_score_pos"].transform("sum").astype(np.float32)
        destination_pressure = destination_groups["causal_score_pos"].transform("sum").astype(np.float32)

        edges["source_causal_pressure"] = source_pressure
        edges["destination_causal_pressure"] = destination_pressure

        edges["causal_score_norm"] = (
            edges["causal_score_pos"] / window_max.clip(lower=1e-6)
        ).astype(np.float32)
        edges["causal_window_share"] = (
            edges["causal_score_pos"] / window_sum.clip(lower=1e-6)
        ).astype(np.float32)

        rank_pct = window_groups["causal_score_pos"].rank(method="average", pct=True)
        rank_desc = window_groups["causal_score_pos"].rank(method="dense", ascending=False)
        no_signal_mask = window_sum <= 1e-6

        edges["causal_rank_pct"] = rank_pct.astype(np.float32)
        edges.loc[no_signal_mask, "causal_rank_pct"] = 0.0
        edges["causal_is_top_edge"] = (
            ((rank_desc <= 3) & (edges["causal_score_pos"] > 0)).astype(np.float32)
        )

        source_share = (
            edges["causal_score_pos"] / source_pressure.clip(lower=1e-6)
        ).astype(np.float32)
        destination_share = (
            edges["causal_score_pos"] / destination_pressure.clip(lower=1e-6)
        ).astype(np.float32)
        edges["causal_source_share"] = source_share
        edges["causal_destination_share"] = destination_share

        edge_groups = edges.groupby(["source", "destination"], sort=False)
        edges["causal_prev_score"] = (
            edge_groups["causal_score_pos"].shift(1).fillna(0.0).astype(np.float32)
        )
        edges["causal_persistence"] = (
            0.65 * edges["causal_score_pos"] + 0.35 * edges["causal_prev_score"]
        ).astype(np.float32)
        edges["causal_acceleration"] = (
            (edges["causal_score_pos"] - edges["causal_prev_score"]).clip(lower=0.0)
        ).astype(np.float32)

        persistence_max = window_groups["causal_persistence"].transform("max").astype(np.float32)
        persistence_norm = (
            edges["causal_persistence"] / persistence_max.clip(lower=1e-6)
        ).astype(np.float32)

        # Capture whether an edge is locally suspicious, persistent over time,
        # and part of a stronger propagation path in the same window.
        edges["causal_path_score"] = (
            0.30 * edges["causal_score_norm"]
            + 0.20 * source_share
            + 0.20 * destination_share
            + 0.15 * edges["causal_window_share"]
            + 0.15 * persistence_norm
        ).astype(np.float32)
        edges.loc[no_signal_mask, "causal_path_score"] = 0.0


        causal_feature_cols = [
            "causal_score",
            "causal_score_pos",
            "causal_score_norm",
            "causal_rank_pct",
            "causal_window_share",
            "source_causal_pressure",
            "destination_causal_pressure",
            "causal_source_share",
            "causal_destination_share",
            "causal_prev_score",
            "causal_persistence",
            "causal_acceleration",
            "causal_path_score",
            "causal_is_top_edge",
        ]
        return edges, causal_feature_cols

    @staticmethod
    def _build_node_causal_features(edges):
        causal_node_feature_cols = [
            "causal_out_persistence",
            "causal_in_persistence",
            "causal_out_path_sum",
            "causal_in_path_sum",
            "causal_total_pressure",
            "causal_net_flow",
            "causal_root_hint",
        ]

        if len(edges) == 0:
            return pd.DataFrame(columns=["window_ts", "service", *causal_node_feature_cols])

        source_features = (
            edges.groupby(["window_ts", "source"], as_index=False, sort=False)
            .agg(
                {
                    "causal_persistence": "sum",
                    "causal_path_score": "sum",
                }
            )
            .rename(
                columns={
                    "source": "service",
                    "causal_persistence": "causal_out_persistence",
                    "causal_path_score": "causal_out_path_sum",
                }
            )
        )

        destination_features = (
            edges.groupby(["window_ts", "destination"], as_index=False, sort=False)
            .agg(
                {
                    "causal_persistence": "sum",
                    "causal_path_score": "sum",
                }
            )
            .rename(
                columns={
                    "destination": "service",
                    "causal_persistence": "causal_in_persistence",
                    "causal_path_score": "causal_in_path_sum",
                }
            )
        )

        node_causal_features = pd.merge(
            source_features,
            destination_features,
            on=["window_ts", "service"],
            how="outer",
        ).fillna(0.0)
        node_causal_features["causal_total_pressure"] = (
            node_causal_features["causal_out_persistence"]
            + node_causal_features["causal_in_persistence"]
        ).astype(np.float32)
        node_causal_features["causal_net_flow"] = (
            node_causal_features["causal_out_persistence"]
            - node_causal_features["causal_in_persistence"]
        ).astype(np.float32)
        node_causal_features["causal_root_hint"] = (
            node_causal_features["causal_out_path_sum"]
            - node_causal_features["causal_in_path_sum"]
        ).astype(np.float32)

        for col in causal_node_feature_cols:
            node_causal_features[col] = node_causal_features[col].astype(np.float32)

        return node_causal_features[["window_ts", "service", *causal_node_feature_cols]]

    @staticmethod
    def _align_frame_to_scaler(frame, feature_cols, scaler, frame_name):
        if scaler is None or not feature_cols:
            return frame, feature_cols

        expected_cols = list(getattr(scaler, "feature_names_in_", []))
        if expected_cols:
            frame = frame.copy()
            for col in expected_cols:
                if col not in frame.columns:
                    frame[col] = 0.0
            return frame, expected_cols

        expected_n_features = getattr(scaler, "n_features_in_", None)
        if expected_n_features is not None and expected_n_features != len(feature_cols):
            raise ValueError(
                f"{frame_name} features ({len(feature_cols)}) do not match the scaler "
                f"expectation ({expected_n_features}). Refit the scaler or keep the same features."
            )

        return frame, feature_cols

    def _prepare_frames(self):
        self.df_comm["timestamp"] = pd.to_datetime(
            self.df_comm["timestamp"], utc=True, errors="coerce"
        )
        self.df_res["timestamp"] = pd.to_datetime(
            self.df_res["timestamp"], utc=True, errors="coerce"
        )
        self.df_evt["timestamp"] = pd.to_datetime(
            self.df_evt["timestamp"], utc=True, errors="coerce"
        )

        self.df_comm = self.df_comm.dropna(subset=["timestamp"]).copy()
        self.df_res = self.df_res.dropna(subset=["timestamp"]).copy()
        self.df_evt = self.df_evt.dropna(subset=["timestamp"]).copy()

        if "source" in self.df_comm.columns:
            self.df_comm["source"] = self.df_comm["source"].apply(self._normalize_name)
        elif "source_workload" in self.df_comm.columns:
            self.df_comm["source"] = self.df_comm["source_workload"].apply(self._normalize_name)
        else:
            raise ValueError("Colonne source/source_workload absente dans le CSV communication.")

        if "destination" in self.df_comm.columns:
            self.df_comm["destination"] = self.df_comm["destination"].apply(self._normalize_name)
        elif "destination_workload" in self.df_comm.columns:
            self.df_comm["destination"] = self.df_comm["destination_workload"].apply(self._normalize_name)
        else:
            raise ValueError("Colonne destination/destination_workload absente dans le CSV communication.")

        if "target_service" in self.df_comm.columns:
            self.df_comm["target_service"] = self.df_comm["target_service"].apply(self._normalize_name)
        elif "target_ms_detected" in self.df_comm.columns:
            self.df_comm["target_service"] = self.df_comm["target_ms_detected"].apply(self._normalize_name)
        else:
            raise ValueError("Colonne target_service/target_ms_detected absente dans le CSV communication.")

        if "pod" in self.df_res.columns:
            self.df_res["service"] = self.df_res["pod"].apply(self._normalize_name)
        else:
            raise ValueError("Colonne pod absente dans le CSV ressource.")

        if "pod" in self.df_evt.columns:
            self.df_evt["service"] = self.df_evt["pod"].apply(self._normalize_name)
        elif "pod_name" in self.df_evt.columns:
            self.df_evt["service"] = self.df_evt["pod_name"].apply(self._normalize_name)
        else:
            self.df_evt["service"] = "unknown"

        if "failure_type" not in self.df_comm.columns:
            self.df_comm["failure_type"] = "normal"
        if "in_failure_window" not in self.df_comm.columns:
            self.df_comm["in_failure_window"] = 0

        if self.drop_services:
            self.df_comm = self.df_comm[
                ~self.df_comm["source"].isin(self.drop_services)
                & ~self.df_comm["destination"].isin(self.drop_services)
                & ~self.df_comm["target_service"].isin(self.drop_services)
            ].copy()

            self.df_res = self.df_res[
                ~self.df_res["service"].isin(self.drop_services)
            ].copy()

            self.df_evt = self.df_evt[
                ~self.df_evt["service"].isin(self.drop_services)
            ].copy()

        window_freq = f"{self.window_sec}s"
        self.df_comm["window_ts"] = self.df_comm["timestamp"].dt.floor(window_freq)
        self.df_res["window_ts"] = self.df_res["timestamp"].dt.floor(window_freq)
        self.df_evt["window_ts"] = self.df_evt["timestamp"].dt.floor(window_freq)

        all_services = set(self.df_comm["source"].unique())
        all_services.update(self.df_comm["destination"].unique())
        all_services.update(self.df_comm["target_service"].unique())
        all_services.update(self.df_res["service"].unique())
        all_services.update(self.df_evt["service"].unique())
        self.all_services = sorted(s for s in all_services if pd.notna(s) and s != "")

        all_failures = self.df_comm["failure_type"].dropna().unique().tolist()
        self.all_failures = sorted(f for f in all_failures if f != "")

        if self.fixed_service_to_idx is not None:
            self.service_to_idx = {
                str(name): int(idx) for name, idx in self.fixed_service_to_idx.items()
            }
            self.all_services = [
                name for name, _ in sorted(self.service_to_idx.items(), key=lambda item: item[1])
            ]
        else:
            if "normal" not in self.all_services:
                self.all_services = ["normal"] + self.all_services
            self.service_to_idx = {s: i for i, s in enumerate(self.all_services)}

        self.idx_to_service = {i: s for s, i in self.service_to_idx.items()}

        if self.fixed_failure_to_idx is not None:
            self.failure_to_idx = {
                str(name): int(idx) for name, idx in self.fixed_failure_to_idx.items()
            }
            self.all_failures = [
                name for name, _ in sorted(self.failure_to_idx.items(), key=lambda item: item[1])
            ]
        else:
            if "normal" not in self.all_failures:
                self.all_failures = ["normal"] + self.all_failures
            self.failure_to_idx = {s: i for i, s in enumerate(self.all_failures)}

        self.idx_to_failure = {i: s for s, i in self.failure_to_idx.items()}

    def _build_feature_tables(self):
        self.labels_by_window = (
            self.df_comm.groupby("window_ts", as_index=False, sort=False)
            .agg(
                {
                    "target_service": self._safe_mode,
                    "failure_type": self._safe_mode,
                    "in_failure_window": "max",
                }
            )
            .rename(columns={"target_service": "target"})
        )

        self.labels_by_window = self.labels_by_window[
            self.labels_by_window["target"].isin(self.all_services)
        ].copy()
        self.labels_by_window.loc[
            self.labels_by_window["in_failure_window"] == 0,
            ["target", "failure_type"],
        ] = "normal"
        self.labels_by_window["y1"] = self.labels_by_window["target"].map(self.service_to_idx)
        self.labels_by_window["y2"] = self.labels_by_window["failure_type"].map(self.failure_to_idx)
        self.labels_by_window.dropna(subset=["y1", "y2"], inplace=True)

        edge_candidate_cols = [
            "new_request",
            "success_rate",
            "error_rate",
            "average_latency",
            "p50_latency",
            "p90_latency",
            "p99_latency",
            "throughput",
            "request_rate",
        ]
        self.edge_feature_cols = [c for c in edge_candidate_cols if c in self.df_comm.columns]
        if not self.edge_feature_cols:
            self.df_comm["dummy_edge_feat"] = 1.0
            self.edge_feature_cols = ["dummy_edge_feat"]

        self.edges = (
            self.df_comm.groupby(
                ["window_ts", "source", "destination"],
                as_index=False,
                sort=False,
            )[self.edge_feature_cols]
            .mean()
            .fillna(0)
        )
        self.edges, edge_delta_cols = self._add_temporal_deltas(
            self.edges,
            group_cols=["source", "destination"],
            feature_cols=self.edge_feature_cols,
        )
        self.edge_feature_cols = [*self.edge_feature_cols, *edge_delta_cols]
        self.edges, causal_feature_cols = self._add_causal_score(self.edges)
        self.edge_feature_cols.extend(causal_feature_cols)

        node_candidate_cols = [
            "container_cpu_usage_seconds_total",
            "container_cpu_system_seconds_total",
            "container_memory_working_set_bytes",
            "container_memory_rss",
            "container_network_receive_bytes_total",
            "container_network_transmit_packets_total",
        ]
        node_cols = [c for c in node_candidate_cols if c in self.df_res.columns]
        if not node_cols:
            self.df_res["dummy_node_feat"] = 1.0
            node_cols = ["dummy_node_feat"]

        self.nodes = (
            self.df_res.groupby(["window_ts", "service"], as_index=False, sort=False)[node_cols]
            .mean()
        )

        evt = self.df_evt.copy()
        if len(evt) > 0:
            evt["event_type"] = evt.get("event_type", "").fillna("").astype(str).str.lower()
            evt["reason"] = evt.get("reason", "").fillna("").astype(str).str.lower()
            evt["evt_count"] = 1
            evt["evt_warning"] = (evt["event_type"] == "warning").astype(int)
            evt["evt_normal"] = (evt["event_type"] == "normal").astype(int)

            critical_reasons = {
                "failed",
                "backoff",
                "unhealthy",
                "killing",
                "crashloopbackoff",
                "oomkilled",
                "pulling",
                "pulled",
            }
            evt["evt_critical_reason"] = evt["reason"].isin(critical_reasons).astype(int)
            evt_features = (
                evt.groupby(["window_ts", "service"], as_index=False, sort=False)[
                    ["evt_count", "evt_warning", "evt_normal", "evt_critical_reason"]
                ]
                .sum()
            )
        else:
            evt_features = pd.DataFrame(
                columns=[
                    "window_ts",
                    "service",
                    "evt_count",
                    "evt_warning",
                    "evt_normal",
                    "evt_critical_reason",
                ]
            )

        self.node_features = pd.merge(
            self.nodes,
            evt_features,
            on=["window_ts", "service"],
            how="left",
        ).fillna(0)
        node_causal_features = self._build_node_causal_features(self.edges)
        self.node_features = pd.merge(
            self.node_features,
            node_causal_features,
            on=["window_ts", "service"],
            how="left",
        ).fillna(0)

        self.node_feature_cols = [
            c for c in self.node_features.columns if c not in ["window_ts", "service"]
        ]

        if self.fit_scaler:
            self.node_scaler = StandardScaler() if self.node_feature_cols else None
            self.edge_scaler = StandardScaler() if self.edge_feature_cols else None

            if self.node_scaler is not None and len(self.node_features) > 0:
                self.node_features[self.node_feature_cols] = self.node_scaler.fit_transform(
                    self.node_features[self.node_feature_cols]
                )

            if self.edge_scaler is not None and len(self.edges) > 0:
                self.edges[self.edge_feature_cols] = self.edge_scaler.fit_transform(
                    self.edges[self.edge_feature_cols]
                )
        else:
            if self.node_scaler is not None and self.node_feature_cols:
                self.node_features, self.node_feature_cols = self._align_frame_to_scaler(
                    self.node_features,
                    self.node_feature_cols,
                    self.node_scaler,
                    "Node",
                )
                self.node_features[self.node_feature_cols] = self.node_scaler.transform(
                    self.node_features[self.node_feature_cols]
                )
            if self.edge_scaler is not None and self.edge_feature_cols:
                self.edges, self.edge_feature_cols = self._align_frame_to_scaler(
                    self.edges,
                    self.edge_feature_cols,
                    self.edge_scaler,
                    "Edge",
                )
                self.edges[self.edge_feature_cols] = self.edge_scaler.transform(
                    self.edges[self.edge_feature_cols]
                )

    def _build_window_indices(self):
        self.node_rows_by_window = {}
        if len(self.node_features) > 0 and self.node_feature_cols:
            node_service_idx = self.node_features["service"].map(self.service_to_idx)
            valid_nodes = node_service_idx.notna()
            indexed_nodes = self.node_features.loc[
                valid_nodes,
                ["window_ts", *self.node_feature_cols],
            ].copy()
            indexed_nodes["_service_idx"] = (
                node_service_idx.loc[valid_nodes].astype(np.int64).to_numpy()
            )

            for window_ts, frame in indexed_nodes.groupby("window_ts", sort=False):
                self.node_rows_by_window[window_ts] = (
                    frame["_service_idx"].to_numpy(dtype=np.int64, copy=True),
                    frame[self.node_feature_cols].to_numpy(dtype=np.float32, copy=True),
                )

        self.edge_rows_by_window = {}
        if len(self.edges) > 0:
            src_idx = self.edges["source"].map(self.service_to_idx)
            dst_idx = self.edges["destination"].map(self.service_to_idx)
            valid_edges = src_idx.notna() & dst_idx.notna()
            indexed_edges = self.edges.loc[
                valid_edges,
                ["window_ts", *self.edge_feature_cols],
            ].copy()
            indexed_edges["_src_idx"] = src_idx.loc[valid_edges].astype(np.int64).to_numpy()
            indexed_edges["_dst_idx"] = dst_idx.loc[valid_edges].astype(np.int64).to_numpy()

            for window_ts, frame in indexed_edges.groupby("window_ts", sort=False):
                self.edge_rows_by_window[window_ts] = (
                    frame[["_src_idx", "_dst_idx"]].to_numpy(dtype=np.int64, copy=True),
                    frame[self.edge_feature_cols].to_numpy(dtype=np.float32, copy=True),
                )

        self.label_lookup = {
            row.window_ts: (int(row.y1), int(row.y2))
            for row in self.labels_by_window[["window_ts", "y1", "y2"]].itertuples(index=False)
        }

    def _build_single_graph(self, window_ts):
        num_nodes = len(self.all_services)

        x = np.zeros((num_nodes, len(self.node_feature_cols)), dtype=np.float32)
        node_rows = self.node_rows_by_window.get(window_ts)
        if node_rows is not None:
            service_indices, node_values = node_rows
            x[service_indices] = node_values

        edge_rows = self.edge_rows_by_window.get(window_ts)
        if edge_rows is None:
            loop_idx = np.arange(num_nodes, dtype=np.int64)
            edge_index_arr = np.stack([loop_idx, loop_idx], axis=1)
            edge_attr_arr = np.zeros((num_nodes, len(self.edge_feature_cols)), dtype=np.float32)
        else:
            edge_index_arr, edge_attr_arr = edge_rows

        labels = self.label_lookup.get(window_ts)
        if labels is None:
            y1 = -1
            y2 = -1
        else:
            y1, y2 = labels

        graph = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index_arr.T.copy()).long().contiguous(),
            edge_attr=torch.from_numpy(edge_attr_arr.copy()).float(),
            y=torch.tensor([y1, y2], dtype=torch.long),
        )
        graph.window_ts = str(window_ts)
        return graph

    def _build_graphs(self):
        ordered_windows = sorted(self.labels_by_window["window_ts"].unique())
        if self.cache_graphs:
            for window_ts in ordered_windows:
                if window_ts not in self._graph_cache:
                    self._graph_cache[window_ts] = self._build_single_graph(window_ts)
            self.graphs = [self._graph_cache[window_ts] for window_ts in ordered_windows]
        else:
            self.graphs = [self._build_single_graph(window_ts) for window_ts in ordered_windows]

    def _build_sequences(self):
        self.samples = []
        sample_labels = []
        ordered_windows = sorted(self.labels_by_window["window_ts"].unique())

        for i in range(len(ordered_windows) - self.seq_len + 1):
            seq_windows = ordered_windows[i:i + self.seq_len]
            seq = []

            for window_ts in seq_windows:
                if self.cache_graphs:
                    if window_ts not in self._graph_cache:
                        self._graph_cache[window_ts] = self._build_single_graph(window_ts)
                    graph = self._graph_cache[window_ts]
                else:
                    graph = self._build_single_graph(window_ts)
                seq.append(graph)

            y1 = int(seq[-1].y[0].item())  # type: ignore
            y2 = int(seq[-1].y[1].item())  # type: ignore

            if y1 >= 0 and y2 >= 0:
                label = torch.tensor([y1, y2], dtype=torch.long)
                self.samples.append((seq, label))
                sample_labels.append((y1, y2))

        if sample_labels:
            self.sample_labels = torch.tensor(sample_labels, dtype=torch.long)
        else:
            self.sample_labels = torch.empty((0, 2), dtype=torch.long)

        if self.cache_graphs:
            self.graphs = [
                self._graph_cache[window_ts]
                for window_ts in ordered_windows
                if window_ts in self._graph_cache
            ]
        else:
            self.graphs = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, y = self.samples[idx]
        return seq, y
