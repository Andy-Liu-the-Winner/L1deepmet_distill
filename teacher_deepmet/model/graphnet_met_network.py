"""L1-GraphNet teacher: message passing on a fixed dR graph, with a global node.

See teacher_GraphNet_DESIGN.md. Controlled counterpart to PartMETNetwork: same
inputs, same output heads, same zero-init — only the trunk differs (radius-graph
message passing instead of full attention).

Works on the same padded batches (B, N, 8) + mask as ParT. The design doc calls
for precomputing edge_index in the shards; here the graph is built on the fly as
a dense (B, N, N) adjacency from dR < r with WRAPPED delta-phi, which is exactly
the same graph (correct across the +-pi seam, unlike v1's raw-phi radius_graph)
and costs a few tensor ops per batch instead of a repack pass.

Outputs (identical contract to PartMETNetwork):
    w (B, N): w = relu(puppiWeight + delta), zero on padding
    s (B,)  : s = 1 + 0.5*tanh(head(g)), g = global node
Zero-init heads: untrained model == PUPPI baseline exactly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PDG_VOCAB = [11, 13, 22, 130, 211]


class MPBlock(nn.Module):
    """One message-passing block with a global exchange node.

    message   m_ij = MLP_e([x_i, x_j, e_ij])   (first linear factorized so the
                                                (B,N,N,2d+3) concat is never built)
    aggregate a_i  = mean over neighbors j
    update    x_i <- LayerNorm(x_i + MLP_n([x_i, a_i, g]))
    global    g   <- g + MLP_g([mean_i x_i, max_i x_i])
    """

    def __init__(self, d):
        super().__init__()
        self.msg_dst = nn.Linear(d, d)          # receiver x_i
        self.msg_src = nn.Linear(d, d, bias=False)  # sender x_j
        self.msg_edge = nn.Linear(3, d, bias=False)
        self.msg_out = nn.Linear(d, d)
        self.upd = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, h, g, adj, deg, edge_attr, mask):
        # h: (B,N,d)  g: (B,d)  adj: (B,N,N) float  deg: (B,N,1)  edge_attr: (B,N,N,3)
        m = (self.msg_dst(h).unsqueeze(2)          # i (receiver) over rows
             + self.msg_src(h).unsqueeze(1)        # j (sender) over cols
             + self.msg_edge(edge_attr))           # (B,N,N,d)
        m = self.msg_out(F.gelu(m))
        a = (m * adj.unsqueeze(-1)).sum(2) / deg   # masked mean over neighbors
        h = self.ln(h + self.upd(torch.cat([h, a, g.unsqueeze(1).expand_as(h)], dim=-1)))
        h = h * mask.unsqueeze(-1)
        n_real = mask.sum(1, keepdim=True).clamp(min=1)
        h_mean = h.sum(1) / n_real
        h_max = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        g = g + self.glob(torch.cat([h_mean, h_max], dim=-1))
        return h, g


class GraphMETTeacher(nn.Module):
    def __init__(self, hidden_dim=64, depth=3, emb_dim=8, radius=0.4):
        super().__init__()
        self.radius = radius
        self.embed_pdg = nn.Embedding(len(PDG_VOCAB), emb_dim)
        self.embed_charge = nn.Embedding(3, emb_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(3 + 2 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList([MPBlock(hidden_dim) for _ in range(depth)])
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.scale_head = nn.Linear(hidden_dim, 1)
        # zero-init both heads: untrained model == PUPPI baseline (w=puppi, s=1)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, x_pad, mask):
        # x_pad: (B, N, 8) [pt, px, py, eta, phi, puppiWeight, pdgid, charge]; mask (B, N)
        B, N, _ = x_pad.shape
        pt = x_pad[..., 0].clamp(min=1e-3)
        eta = x_pad[..., 3]
        phi = x_pad[..., 4]
        puppi = x_pad[..., 5]
        logpt = torch.log(pt)

        pdg = x_pad[..., 6].long().abs()
        pdg_remap = torch.zeros_like(pdg)
        for i, p in enumerate(PDG_VOCAB):
            pdg_remap = torch.where(pdg == p, torch.full_like(pdg, i), pdg_remap)
        charge = (x_pad[..., 7].long() + 1).clamp(0, 2)

        cont = torch.stack([logpt, eta, puppi], dim=-1)
        h = self.node_encoder(
            torch.cat([cont, self.embed_pdg(pdg_remap), self.embed_charge(charge)], dim=-1))

        # fixed dR graph with wrapped dphi -> correct across the +-pi seam
        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi
        dR = torch.sqrt(deta ** 2 + dphi ** 2 + 1e-12)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        adj = ((dR < self.radius) & pair_mask).float()
        adj = adj * (1.0 - torch.eye(N, device=adj.device))  # no self-loops
        deg = adj.sum(2, keepdim=True).clamp(min=1.0)
        edge_attr = torch.stack([deta, dphi, dR], dim=-1) * adj.unsqueeze(-1)

        h = h * mask.unsqueeze(-1)
        g = h.new_zeros(B, h.shape[-1])
        for blk in self.blocks:
            h, g = blk(h, g, adj, deg, edge_attr, mask)

        delta = self.delta_head(h).squeeze(-1)
        w = F.relu(puppi + delta) * mask
        s = 1.0 + 0.5 * torch.tanh(self.scale_head(g)).squeeze(-1)
        return w, s

    @torch.no_grad()
    def effective_weights(self, x_pad, mask):
        """s * w — the per-particle distillation target with the calibration folded in."""
        w, s = self.forward(x_pad, mask)
        return s.unsqueeze(1) * w
