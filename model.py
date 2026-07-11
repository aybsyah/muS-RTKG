import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
)


class GATEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_dim,
        hidden_dim=32,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout_p = dropout

        self.input_proj = nn.Linear(node_in_dim, hidden_dim * heads)

        self.gat1 = GATv2Conv(
            in_channels=node_in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * heads)

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim * heads)

        self.out_proj = nn.Linear(hidden_dim * heads, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x_res = self.input_proj(x)

        x1 = self.gat1(x, edge_index, edge_attr)
        x1 = self.norm1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        x1 = x1 + x_res

        x2 = self.gat2(x1, edge_index, edge_attr)
        x2 = self.norm2(x2)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)
        x2 = x2 + x1

        x_out = self.out_proj(x2)
        x_out = self.out_norm(x_out)
        x_out = F.elu(x_out)

        return x_out


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, D]
        attn_logits = self.score(x).squeeze(-1)      # [B, T]
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        return context, attn_weights


class MLPHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        hidden = max(input_dim // 2, 32)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class GATGRUMultiTask(nn.Module):
    def __init__(
        self,
        num_graph_nodes,
        node_in_dim,
        edge_dim,
        num_service_classes,
        num_failure_classes,
        gat_hidden_dim=32,
        gru_hidden_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.num_graph_nodes = num_graph_nodes
        self.gat_hidden_dim = gat_hidden_dim
        self.dropout_p = dropout

        self.gat_encoder = GATEncoder(
            node_in_dim=node_in_dim,
            edge_dim=edge_dim,
            hidden_dim=gat_hidden_dim,
            heads=4,
            dropout=dropout,
        )

        # mean + max + std
        self.graph_emb_dim = gat_hidden_dim * 3

        self.pre_gru = nn.Sequential(
            nn.Linear(self.graph_emb_dim, self.graph_emb_dim),
            nn.LayerNorm(self.graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=self.graph_emb_dim,
            hidden_size=gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.temporal_dim = gru_hidden_dim * 2

        self.temporal_attn = TemporalAttentionPooling(self.temporal_dim)

        self.fusion_dim = self.temporal_dim * 2

        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, self.temporal_dim),
            nn.LayerNorm(self.temporal_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.service_head = MLPHead(
            input_dim=self.temporal_dim,
            num_classes=num_service_classes,
            dropout=dropout,
        )

        self.failure_head = MLPHead(
            input_dim=self.temporal_dim,
            num_classes=num_failure_classes,
            dropout=dropout,
        )

    def _batched_std_pool(self, z, batch_idx, num_graphs):
        """
        z: [N, H]
        batch_idx: [N]
        return: [B, H]
        """
        device = z.device
        dtype = z.dtype
        H = z.size(1)

        counts = torch.bincount(batch_idx, minlength=num_graphs).to(device=device, dtype=dtype)

        sum_x = torch.zeros(num_graphs, H, device=device, dtype=dtype)
        sum_x2 = torch.zeros(num_graphs, H, device=device, dtype=dtype)

        sum_x.index_add_(0, batch_idx, z)
        sum_x2.index_add_(0, batch_idx, z * z)

        counts_clamped = counts.clamp_min(1.0).unsqueeze(1)

        mean = sum_x / counts_clamped
        mean_sq = sum_x2 / counts_clamped

        var = mean_sq - mean * mean
        var = torch.clamp(var, min=1e-12)

        std = torch.sqrt(var)
        return std

    def encode_batched_graph(self, batch_graph):
        """
        batch_graph.x         : [N_total, F]
        batch_graph.edge_index: [2, E_total]
        batch_graph.edge_attr : [E_total, edge_dim]
        batch_graph.batch     : [N_total]
        return: [B, 3H]
        """
        z = self.gat_encoder(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.edge_attr
        )  # [N_total, H]

        batch_idx = batch_graph.batch
        num_graphs = int(batch_graph.num_graphs)

        z_mean = global_mean_pool(z, batch_idx)   # [B, H]
        z_max = global_max_pool(z, batch_idx)     # [B, H]
        z_std = self._batched_std_pool(z, batch_idx, num_graphs)  # [B, H]

        g_emb = torch.cat([z_mean, z_max, z_std], dim=1)  # [B, 3H]
        g_emb = self.pre_gru(g_emb)

        return g_emb

    def forward(self, batched_seq):
        """
        batched_seq = [Batch_t0, Batch_t1, ..., Batch_t(T-1)]
        """
        graph_embs_over_time = []

        for batch_t in batched_seq:
            g_emb_t = self.encode_batched_graph(batch_t)   # [B, 3H]
            graph_embs_over_time.append(g_emb_t)

        x = torch.stack(graph_embs_over_time, dim=1)  # [B, T, 3H]

        gru_out, _ = self.gru(x)                      # [B, T, 2G]

        h_last = gru_out[:, -1, :]                    # [B, 2G]
        h_attn, _ = self.temporal_attn(gru_out)       # [B, 2G]

        h = torch.cat([h_last, h_attn], dim=-1)       # [B, 4G]
        h = self.fusion(h)                            # [B, 2G]

        service_logits = self.service_head(h)
        failure_logits = self.failure_head(h)

        return service_logits, failure_logits