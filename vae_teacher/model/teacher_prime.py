"""
TeacherPrime (T')
=================
T' bridges the input-dimension gap between the full Teacher T (which requires
PV association as a 3rd categorical feature) and the Student S (which has no PV).

Architecture
------------
  S_input (x_cont [N,6], x_cat_2 [N,2])
        │
        ▼
   VAEEncoder            (trained in Phase 2, may be frozen in Phase 4)
        │
        ▼
   z  [N, hidden_dim=32]   (latent — approximates T's encode_all output)
        │
        ▼
   T.bn_all               (frozen BatchNorm from Teacher T checkpoint)
        │
        ▼
   T.conv_continuous[0..depth-1]  (frozen EdgeConv layers from T)
        │
        ▼
   T.output               (frozen Linear head from T)
        │
        ▼
   per-particle weights [N]  (ReLU applied by caller or included here)

Loading
-------
Call TeacherPrime.from_checkpoints(teacher_ckpt_path, vae_ckpt_path, device)
after Phase 1 and Phase 2 have completed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vae_encoder import VAEEncoder


class TeacherPrime(nn.Module):
    def __init__(self, vae_encoder: VAEEncoder, teacher_graphnet: nn.Module):
        """
        Parameters
        ----------
        vae_encoder     : trained VAEEncoder instance
        teacher_graphnet: the GraphMETNetwork from the Teacher T checkpoint
                          (student_deepmet GraphMETNetwork with embed_pv active)
        """
        super().__init__()

        self.vae_encoder = vae_encoder

        # Extract and register T's post-embedding layers.
        # These are kept frozen during both Phase 2 (VAE training) and Phase 4
        # (student distillation). The VAEEncoder is the only trainable part.
        self.bn_all          = teacher_graphnet.bn_all
        self.conv_continuous = teacher_graphnet.conv_continuous
        self.output_head     = teacher_graphnet.output

        self._freeze_teacher_layers()

    # ------------------------------------------------------------------
    def _freeze_teacher_layers(self):
        for module in [self.bn_all, self.conv_continuous, self.output_head]:
            for p in module.parameters():
                p.requires_grad = False
        # Keep bn_all in eval mode so running stats are used (not batch stats)
        self.bn_all.eval()

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """Override train() so that bn_all stays in eval mode always."""
        super().train(mode)
        self.bn_all.eval()        # always eval — frozen running stats
        return self

    # ------------------------------------------------------------------
    def forward(self, x_cont, x_cat_2, edge_index, batch):
        """
        Parameters
        ----------
        x_cont    : [N, 6]   continuous features (unnormalised — VAEEncoder normalises internally)
        x_cat_2   : [N, 2]   integer categoricals (pdgid col-0, charge col-1)
        edge_index: [2, E]   graph edges
        batch     : [N]      batch assignment

        Returns
        -------
        weights : [N]      per-particle weights (after ReLU)
        mu      : [N, 32]  VAE latent mean   (needed for KL loss during training)
        logvar  : [N, 32]  VAE log-variance  (needed for KL loss during training)
        """
        # Step 1: encode S input → latent z
        z, mu, logvar = self.vae_encoder(x_cont, x_cat_2)   # [N, 32]

        # Step 2: normalise z with T's frozen BatchNorm
        emb = self.bn_all(z)                                  # [N, 32]

        # Step 3: T's graph convolutions (residual)
        for co_conv in self.conv_continuous:
            emb = emb + co_conv[1](co_conv[0](emb, edge_index))

        # Step 4: T's output head
        out = self.output_head(emb).squeeze(-1)               # [N]

        return F.relu(out), mu, logvar

    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoints(cls, teacher_ckpt_path: str, vae_ckpt_path: str,
                         device, norm=None):
        """
        Convenience constructor: load T from Phase-1 checkpoint and VAEEncoder
        from Phase-2 checkpoint, then assemble T'.

        Parameters
        ----------
        teacher_ckpt_path : path to teacher_ckpts_L1_withPV/best.pth.tar
        vae_ckpt_path     : path to teacher_prime_ckpts/best_vae.pth.tar
        device            : torch device
        norm              : datanorm tensor for VAEEncoder (default: ones)

        Returns
        -------
        TeacherPrime instance with all Teacher layers frozen, VAEEncoder loaded.
        """
        import teacher_model.net as _tn
        TeacherNet = _tn.Net                # cat_dim=2, no PV

        scale_momentum = 128.
        if norm is None:
            norm = torch.tensor(
                [1./scale_momentum, 1./scale_momentum, 1./scale_momentum,
                 1., 1., 1.]
            ).to(device)

        # Load Teacher T (cat_dim=2, no PV — matches existing checkpoints)
        teacher_net = TeacherNet(6, 2, norm).to(device)
        ckpt_t = torch.load(teacher_ckpt_path, map_location=device)
        teacher_net.load_state_dict(ckpt_t['state_dict'])
        teacher_net.eval()

        teacher_graphnet = teacher_net.graphnet

        # Build VAEEncoder
        vae_enc = VAEEncoder(continuous_dim=6, norm=norm, hidden_dim=32).to(device)
        ckpt_v = torch.load(vae_ckpt_path, map_location=device)
        vae_enc.load_state_dict(ckpt_v['vae_state_dict'])

        return cls(vae_enc, teacher_graphnet).to(device)
