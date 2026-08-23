"""L1-ParT teacher: compact Particle Transformer for L1 MET.

See teacher_ParT_DESIGN.md. Works on padded batches (B, N, 8) with a boolean mask —
no torch_geometric / torch_cluster needed.

Input columns (verified true format): [pt, px, py, eta, phi, puppiWeight, pdgid, charge]

Rotation invariance by construction: absolute phi / px / py never enter the network;
phi only appears through pairwise wrapped delta-phi in the attention bias.

Outputs:
    w (B, N): per-particle weights, w = relu(puppiWeight + delta), zero on padding
    s (B,)  : per-event response-calibration scalar, s = 1 + 0.5*tanh(head(pooled))
Both heads are zero-initialized, so an untrained model reproduces the PUPPI baseline
(w == puppiWeight, s == 1) exactly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PDG_VOCAB = [11, 13, 22, 130, 211]
NEG_INF = -1e9


class AttnBlock(nn.Module):
    """Pre-LN transformer block with additive per-head attention bias (manual MHA,
    torch 1.12 has no scaled_dot_product_attention)."""

    def __init__(self, d, heads, ffn):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.dh = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)

    def forward(self, h, attn_bias):
        # h: (B, N, d); attn_bias: (B, heads, N, N) with NEG_INF on padded keys
        B, N, d = h.shape
        x = self.ln1(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.heads, self.dh).transpose(1, 2)  # (B, H, N, dh)
        k = k.view(B, N, self.heads, self.dh).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dh).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh) + attn_bias
        attn = torch.softmax(logits, dim=-1)
        a = torch.matmul(attn, v)  # (B, H, N, dh)
        a = a.transpose(1, 2).reshape(B, N, d)
        h = h + self.proj(a)
        h = h + self.fc2(F.gelu(self.fc1(self.ln2(h))))
        return h


class PartMETNetwork(nn.Module):
    def __init__(self, hidden_dim=64, heads=4, depth=4, ffn_dim=128, emb_dim=8,
                 pair_hidden=32):
        super().__init__()
        self.heads = heads
        self.embed_pdg = nn.Embedding(len(PDG_VOCAB), emb_dim)
        self.embed_charge = nn.Embedding(3, emb_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(3 + 2 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        # pairwise bias MLP, shared across layers: [deta, dphi, log dR, log ptmin, log ptprod]
        self.pair_mlp = nn.Sequential(
            nn.Linear(5, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, heads),
        )
        self.blocks = nn.ModuleList(
            [AttnBlock(hidden_dim, heads, ffn_dim) for _ in range(depth)])
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pool_query = nn.Parameter(torch.randn(hidden_dim) / math.sqrt(hidden_dim))
        self.scale_head = nn.Linear(hidden_dim, 1)
        # zero-init both heads: untrained model == PUPPI baseline (w=puppi, s=1)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, x_pad, mask):
        # x_pad: (B, N, 8) raw features, zero-padded; mask: (B, N) bool, True = real
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

        # pairwise geometric features -> per-head attention bias
        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dphi = torch.remainder(dphi + math.pi, 2 * math.pi) - math.pi  # wrap to (-pi, pi]
        dR = torch.sqrt(deta ** 2 + dphi ** 2 + 1e-12)
        log_ptmin = torch.min(logpt.unsqueeze(2), logpt.unsqueeze(1))
        log_ptprod = logpt.unsqueeze(2) + logpt.unsqueeze(1)
        U = torch.stack([deta, dphi, torch.log(dR + 1e-4), log_ptmin, log_ptprod], dim=-1)
        bias = self.pair_mlp(U).permute(0, 3, 1, 2)  # (B, heads, N, N)

        # mask padded KEYS only; padded queries still get finite rows (discarded at output)
        key_mask = torch.where(mask, torch.zeros_like(mask, dtype=h.dtype),
                               torch.full_like(mask, NEG_INF, dtype=h.dtype))
        attn_bias = bias + key_mask.unsqueeze(1).unsqueeze(2)  # broadcast over heads, queries

        for blk in self.blocks:
            h = blk(h, attn_bias)

        delta = self.delta_head(h).squeeze(-1)
        w = F.relu(puppi + delta) * mask

        # attention pooling with a learned query, masked
        pool_logits = (h * self.pool_query).sum(-1) / math.sqrt(h.shape[-1])
        pool_logits = pool_logits.masked_fill(~mask, NEG_INF)
        alpha = torch.softmax(pool_logits, dim=-1)
        pooled = torch.einsum('bn,bnd->bd', alpha, h)
        s = 1.0 + 0.5 * torch.tanh(self.scale_head(pooled)).squeeze(-1)

        return w, s

    @torch.no_grad()
    def effective_weights(self, x_pad, mask):
        """s * w — the per-particle distillation target with the calibration folded in."""
        w, s = self.forward(x_pad, mask)
        return s.unsqueeze(1) * w
