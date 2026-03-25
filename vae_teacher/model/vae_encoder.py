"""
VAEEncoder: maps Student S-format input (6 cont, 2 cat) to a latent vector z of
dimension hidden_dim=32.  z is intended to be injected directly into Teacher T's
graph-convolution layers, bypassing T's input-embedding block (which requires a
3rd categorical feature: PV association).

Architecture mirrors StudentGraphMETNetwork's embedding block, then adds a VAE
mu/logvar head on top.

Input
-----
  x_cont : [N_particles, 6]   continuous features (pt, px, py, eta, d0, dz)
                               NOTE: caller must NOT pre-normalise; this module
                               applies its own datanorm (same as StudentNet).
  x_cat  : [N_particles, 2]   integer categoricals  col-0 = pdgid, col-1 = charge

Output
------
  z       : [N_particles, hidden_dim]   sampled latent  (reparameterised)
  mu      : [N_particles, hidden_dim]   latent mean
  logvar  : [N_particles, hidden_dim]   log variance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    pdgs = [1, 2, 11, 13, 22, 130, 211]   # same remapping as both teacher & student

    def __init__(self, continuous_dim: int = 6, norm=None, hidden_dim: int = 32):
        super().__init__()

        self.hidden_dim = hidden_dim
        # Input normalisation (mirrors StudentGraphMETNetwork)
        if norm is None:
            norm = torch.ones(continuous_dim)
        self.register_buffer('datanorm', norm)

        # Categorical embeddings (no PV — only charge and pdgid)
        self.embed_charge = nn.Embedding(3, hidden_dim // 4)       # [N, 8]
        self.embed_pdgid  = nn.Embedding(7, hidden_dim // 4)       # [N, 8]

        # Continuous embedding:  6 → hidden_dim//2
        self.embed_continuous = nn.Sequential(
            nn.Linear(continuous_dim, hidden_dim // 2),
            nn.ELU(),
        )

        # Categorical fusion: 2*(hidden_dim//4) → hidden_dim//2
        self.embed_categorical = nn.Sequential(
            nn.Linear(2 * (hidden_dim // 4), hidden_dim // 2),
            nn.ELU(),
        )

        # Joint encoder: hidden_dim → hidden_dim
        self.encode_all = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # VAE heads
        self.fc_mu     = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)

    # ------------------------------------------------------------------
    def _reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu          # deterministic at eval time

    # ------------------------------------------------------------------
    def _embed(self, x_cont, x_cat):
        """Shared embedding forward — returns pre-VAE representation h [N, hidden_dim]."""
        # Normalise continuous features (in-place clone to avoid mutating input)
        x_cont = x_cont * self.datanorm

        emb_cont = self.embed_continuous(x_cont)               # [N, 16]

        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)          # [N, 8]

        # Remap raw pdgid values to indices 0-6
        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(
                pdg_remap == pdgval,
                torch.full_like(pdg_remap, i),
                pdg_remap,
            )
        emb_pdg = self.embed_pdgid(pdg_remap)                   # [N, 8]

        emb_cat = self.embed_categorical(
            torch.cat([emb_chrg, emb_pdg], dim=1)               # [N, 16]
        )                                                        # [N, 16]

        h = self.encode_all(torch.cat([emb_cat, emb_cont], dim=1))  # [N, 32]
        return h

    # ------------------------------------------------------------------
    def forward(self, x_cont, x_cat):
        h      = self._embed(x_cont, x_cat)        # [N, hidden_dim]
        mu     = self.fc_mu(h)                     # [N, hidden_dim]
        logvar = self.fc_logvar(h)                 # [N, hidden_dim]
        z      = self._reparameterize(mu, logvar)  # [N, hidden_dim]
        return z, mu, logvar


def kl_loss(mu, logvar):
    """
    Per-element KL divergence: KL( q(z|x) || N(0,I) ).
    Returns the mean over all particles and latent dimensions.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
