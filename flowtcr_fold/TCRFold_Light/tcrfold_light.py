"""
Evoformer-lite wrapper acting as a geometry critic.

This reuses the legacy MSA-free Evoformer block from conditioned/src/Evoformer.py.
Full structure heads are left as TODOs; current scaffold returns transformed
sequence/pair representations.
"""

from typing import Dict

import torch
from torch import nn

try:
    from conditioned.src import Evoformer as legacy_evoformer  # type: ignore
except Exception:
    legacy_evoformer = None  # type: ignore


class TCRFoldLight(nn.Module):
    def __init__(self, s_dim: int = 512, z_dim: int = 128, n_layers: int = 12):
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        if legacy_evoformer is None:
            raise ImportError("conditioned.src.Evoformer not found; ensure legacy code is available.")
        self.backbone = legacy_evoformer.Evoformer(s_dim, z_dim, N_elayers=n_layers)
        self.dist_head = nn.Linear(z_dim, 1)
        self.contact_head = nn.Linear(z_dim, 1)
        self.energy_head = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, 1))

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        s: [B, L, s_dim] sequence embeddings
        z: [B, L, L, z_dim] pair embeddings
        """
        if s.dim() == 3:
            s_in = s[0]
        else:
            s_in = s
        if z.dim() == 4:
            z_in = z[0]
        else:
            z_in = z

        m_out, z_out = self.backbone(s_in, z_in)

        # reshape back to batch-less outputs; callers can reshape if needed
        dist = self.dist_head(z_out)
        contact = torch.sigmoid(self.contact_head(z_out))
        energy = self.energy_head(z_out).mean()
        return {"m": m_out, "z": z_out, "distance": dist, "contact": contact, "energy": energy}
