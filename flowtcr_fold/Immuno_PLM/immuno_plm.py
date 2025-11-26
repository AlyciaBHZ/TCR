"""
Hybrid encoder: ESM (or basic embedding) + PSI/Topology bias (hierarchical pairs + collapse).

- Encodes tokens to sequence embeddings
- Builds hierarchical pair bias (inspired by psi_model/create_hierarchical_pairs) with collapse token (use CLS as collapse)
- Provides pooled representations for InfoNCE; optional MLM
"""

from typing import Dict, Optional, List, Tuple

import torch
from torch import nn

try:
    import esm  # type: ignore
except Exception:
    esm = None  # type: ignore


class TopologyBias(nn.Module):
    """
    Hierarchical pair bias with collapse token (index 0), region-aware levels.
    """

    def __init__(self, z_dim: int = 128, max_pairs: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.max_pairs = max_pairs
        self.pair_linear = nn.Linear(max_pairs, z_dim)

    def create_hierarchical_pairs(self, L: int, idx_map: List[Tuple[int, int]]):
        pair_id = torch.zeros((L, L), dtype=torch.long)
        # collapse token self
        pair_id[0, 0] = 0
        # collapse with others
        pair_id[0, 1:] = 1
        pair_id[1:, 0] = 1

        if not idx_map:
            return pair_id.clamp(max=self.max_pairs - 1)

        hd_start, hd_end = idx_map[0]
        # HD neighbors
        for i in range(hd_start, max(hd_end - 1, hd_start)):
            if i + 1 < hd_end:
                pair_id[i, i + 1] = 2
                pair_id[i + 1, i] = 2
        # HD internal
        for i in range(hd_start, hd_end):
            for j in range(hd_start, hd_end):
                if i != j and pair_id[i, j] == 0:
                    pair_id[i, j] = 3
        # HD ↔ conditioning
        for i in range(hd_start, hd_end):
            for region_start, region_end in idx_map[1:]:
                pair_id[i, region_start:region_end] = 4
                pair_id[region_start:region_end, i] = 4

        counter = 5
        # conditioning internal
        for region_start, region_end in idx_map[1:]:
            pair_id[region_start:region_end, region_start:region_end] = counter
            counter += 1
        # conditioning ↔ conditioning
        conditioning_regions = idx_map[1:]
        for i, (r1_start, r1_end) in enumerate(conditioning_regions):
            for j, (r2_start, r2_end) in enumerate(conditioning_regions[i + 1 :], i + 1):
                pair_id[r1_start:r1_end, r2_start:r2_end] = counter
                pair_id[r2_start:r2_end, r1_start:r1_end] = counter
                counter += 1

        return pair_id.clamp(max=self.max_pairs - 1)

    def forward(self, idx_pairs: List[Tuple[int, int]], total_L: int, device):
        if not idx_pairs:
            return None
        pair_id = self.create_hierarchical_pairs(total_L, idx_pairs).to(device)
        z = torch.nn.functional.one_hot(pair_id, num_classes=self.max_pairs).float()
        z = self.pair_linear(z)
        return z


class ImmunoPLM(nn.Module):
    def __init__(self, hidden_dim: int = 256, z_dim: int = 128, use_esm: bool = False, vocab_size: int = 256):
        super().__init__()
        self.use_esm = use_esm and esm is not None
        self.z_dim = z_dim
        if self.use_esm:
            self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.embed_dim = self.esm_model.embed_dim
        else:
            self.alphabet = None
            self.embed_dim = hidden_dim
            self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.topology_bias = TopologyBias(z_dim=z_dim)
        self.proj = nn.Linear(self.embed_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        region_slices: Optional[List[Dict[str, slice]]] = None,
    ):
        """
        tokens: [B, L] token ids (CLS at index 0 treated as collapse token)
        mask: [B, L] attention mask (1=real,0=pad)
        region_slices: list[dict] per sample, mapping region->slice (pep/mhc/cdr3b...). cdr3b treated as HD region.
        """
        if self.use_esm:
            outputs = self.esm_model(tokens, repr_layers=[6], return_contacts=False)
            s = outputs["representations"][6]  # [B, L, D]
        else:
            s = self.token_embed(tokens)
        s = self.proj(s)

        if mask is None:
            pooled = s.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1)
            pooled = (s * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(pooled)

        z_topo = None
        if region_slices:
            zs = []
            for b_idx in range(tokens.size(0)):
                region_dict = region_slices[b_idx] if b_idx < len(region_slices) else None
                if region_dict is None:
                    continue
                # reorder: HD (cdr3b) first, then others
                idx_pairs = []
                if "cdr3b" in region_dict:
                    idx_pairs.append((region_dict["cdr3b"].start, region_dict["cdr3b"].stop))
                for key, sl in region_dict.items():
                    if key == "cdr3b":
                        continue
                    idx_pairs.append((sl.start, sl.stop))
                if not idx_pairs:
                    continue
                z_bias = self.topology_bias(idx_pairs, tokens.size(1), tokens.device)
                if z_bias is not None:
                    zs.append(z_bias)
            if zs:
                z_topo = torch.stack(zs, dim=0)

        return {"s": s, "pooled": pooled, "z_bias": z_topo}
