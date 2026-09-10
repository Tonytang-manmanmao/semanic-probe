"""Source-guided hypergraph modules (Route B "SSM -> HyperGraph" soft-hyperedge probe).

Pipeline (source -> HyperGraph forward guidance):

    h_ssm = NodeSSM(x, t)                     # node-level selective scan (S6)
    h_src = <source>(h_ssm, x, pos, t)        # ssm / semantic / motion / joint
    H     = AdaptiveHyperedge(h_src)          # source states -> soft hyperedge incidence
    H_hyp = HyperConv(x, H)                   # V -> E -> V hypergraph convolution
    out   = h_ssm + H_hyp                     # residual fusion

The four probe sources change exactly ONE thing -- what feeds
``AdaptiveHyperedge`` (the hyperedge incidence matrix ``H``):

    ssm       : h_ssm                     SSM hidden state (Route B's original claim)
    semantic  : x                         graph node features
    motion    : [x_pos, y_pos, t]         position / time physical prior
    joint     : concat(h_ssm, x, motion)  all three concatenated

``H`` is soft (no hard top-k selection) and row-normalised over hyperedges
(``F.softmax(..., dim=-1)``), i.e. every node distributes unit mass over the
``num_hyperedges`` hyperedges. Three structural regularisers (entropy /
sharpness / consistency) are added to the total loss -- see
``hyperedge_regularization``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeSSM(nn.Module):
    """Node-level selective state space model (Mamba-style S6)."""

    def __init__(self, d_model, d_state=16, dt_rank=None, d_conv=3):
        super().__init__()
        d_inner = 2 * d_model
        dt_rank = dt_rank or math.ceil(d_model / 16)

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=0, groups=d_inner
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1)
        A = A.repeat(d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _selective_scan_batched(self, x, dt, B, C):
        # x: [B, L, d_inner], dt: [B, L, d_inner], B/C: [B, L, d_state]
        n_graphs, seq_len, _ = x.shape
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        h = x.new_zeros(n_graphs, self.d_inner, self.d_state)
        y_list = []
        for t in range(seq_len):
            dt_t = dt[:, t]                             # [B, d_inner]
            a_t = torch.exp(dt_t.unsqueeze(-1) * A)     # [B, d_inner, d_state]
            b_t = dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h = a_t * h + b_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1) + self.D * x[:, t]
            y_list.append(y_t)
        return torch.stack(y_list, dim=1)               # [B, L, d_inner]

    def _forward_batched(self, x_pad):
        # x_pad: [B, L, d_model], rows already time-sorted per graph.
        xz = self.in_proj(x_pad)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)               # [B, d_inner, L]
        x_conv = F.pad(x_conv, (self.conv1d.kernel_size[0] - 1, 0))
        x_conv = self.conv1d(x_conv)                    # [B, d_inner, L]
        x_conv = x_conv.transpose(1, 2)                 # [B, L, d_inner]
        x_conv = F.silu(x_conv)

        params = self.x_proj(x_conv)
        dt, B, C = params.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        y = self._selective_scan_batched(x_conv, dt, B, C)
        y = y * F.silu(z)
        return self.out_proj(y)

    def forward(self, x, t, batch=None):
        if x.numel() == 0:
            return x

        if batch is None or batch.numel() != x.shape[0]:
            batch_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            batch_ids = batch.reshape(-1).to(device=x.device, dtype=torch.long)

        # Group nodes by graph, sort each group by time, pad to a common length,
        # then run the SSM over the whole batch at once: one loop over the
        # longest sequence instead of one loop per node per graph.
        groups = []
        max_len = 0
        for graph_id in torch.unique(batch_ids):
            node_indices = (batch_ids == graph_id).nonzero(as_tuple=True)[0]
            if node_indices.numel() == 0:
                continue
            t_graph = t[node_indices]
            local_order = torch.argsort(t_graph, stable=True)
            sorted_nodes = node_indices[local_order]
            groups.append(sorted_nodes)
            max_len = max(max_len, int(sorted_nodes.numel()))

        if not groups:
            return torch.zeros_like(x)

        n_graphs = len(groups)
        x_pad = x.new_zeros(n_graphs, max_len, x.shape[-1])
        for b, idx in enumerate(groups):
            x_pad[b, : idx.numel()] = x[idx]

        out_pad = self._forward_batched(x_pad)

        out_full = torch.zeros_like(x)
        for b, idx in enumerate(groups):
            out_full[idx] = out_pad[b, : idx.numel()]
        return out_full


class AdaptiveHyperedge(nn.Module):
    """Project a source feature to a soft hyperedge incidence matrix.

    ``in_dim`` is the width of the feature that drives the incidence matrix.
    It is the ONLY thing the four probe sources change:

        ssm      -> in_dim = node_dim          (SSM hidden state)
        semantic -> in_dim = node_dim          (graph node features)
        motion   -> in_dim = 3                 ([x, y, t])
        joint    -> in_dim = 2 * node_dim + 3  (concatenation of the three)
    """

    def __init__(self, in_dim, num_hyperedges, temperature=0.5):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.temperature = temperature
        self.proj = nn.Linear(in_dim, num_hyperedges)

    def forward(self, h_src):
        logits = self.proj(h_src)
        # row-normalised: each node spreads unit mass over the hyperedges
        return F.softmax(logits / self.temperature, dim=-1)


class HyperConv(nn.Module):
    """Hypergraph convolution (V -> E -> V, HGNN-style)."""

    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x, H):
        eps = 1e-6
        d_v = H.sum(dim=1).clamp(min=eps)
        d_e = H.sum(dim=0).clamp(min=eps)

        e = torch.einsum("nk,nd->kd", H, x)
        e = e / d_e.unsqueeze(-1)

        out = torch.einsum("nk,kd->nd", H, e)
        out = out / d_v.sqrt().unsqueeze(-1)
        out = self.lin(out)

        return self.act(out) + x


def hyperedge_regularization(
    h_src,
    H,
    entropy_weight=0.0,
    sharpness_weight=0.0,
    consistency_weight=0.0,
    eps=1e-8,
):
    """Return hypergraph structural losses.

    ``entropy_weight`` encourages a balanced hyperedge-load distribution.
    ``sharpness_weight`` discourages a node from spreading mass uniformly over
    all hyperedges. ``consistency_weight`` pulls nodes assigned to the same
    hyperedge toward similar source states (``h_src`` = whatever drove ``H``).

    Note: the entropy term is the same anti-collapse pressure that keeps the
    soft hyperedges from degenerating into a few dominant ones.
    """
    if H is None or H.numel() == 0:
        return None, {}

    loss = torch.zeros((), device=H.device, dtype=H.dtype)
    logs = {}

    if abs(entropy_weight) > 0:
        edge_load = H.sum(dim=0).clamp(min=eps)
        edge_load = edge_load / edge_load.sum()
        term = (edge_load * edge_load.log()).sum()
        loss = loss + entropy_weight * term
        logs["ssm_hyper_entropy_loss"] = term.detach()

    if abs(sharpness_weight) > 0:
        row_entropy = -(H * (H + eps).log()).sum(dim=-1).mean()
        loss = loss + sharpness_weight * row_entropy
        logs["ssm_hyper_sharpness_loss"] = row_entropy.detach()

    if abs(consistency_weight) > 0 and h_src is not None and h_src.numel() > 0:
        h = F.normalize(h_src, dim=-1, eps=eps)
        edge_center = torch.einsum("nk,nd->kd", H, h)
        center_norm = torch.linalg.vector_norm(edge_center, dim=-1)
        total_mass = H.sum().clamp(min=eps)
        weighted_cosine = center_norm.sum()
        term = 1.0 - weighted_cosine / total_mass
        loss = loss + consistency_weight * term
        logs["ssm_hyper_consistency_loss"] = term.detach()

    return loss, logs
