"""
Immuno-PLM: Hybrid encoder with topology bias, V/J conditioning, and optional ESM + LoRA.
- Topology bias (psi_model style): collapse token + hierarchical pairs.
- V/J conditioning: h_v, h_j, l_v, l_j embeddings fused into CLS.
- Backbones: BasicTokenizer embedding, or ESM feature extractor; LoRA adapters optional (built-in implementation).
- Training targets: InfoNCE (batch negatives) + optional MLM (handled in training script).
"""
from typing import Dict, Optional, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import math

# Optional ESM backbone
try:
    import esm  # type: ignore
    ESM_AVAILABLE = True
except Exception:
    esm = None  # type: ignore
    ESM_AVAILABLE = False

# Built-in LoRA implementation (no external PEFT dependency)
class LoRALinear(nn.Module):
    """
    Lightweight LoRA for Linear layers: frozen base + trainable low-rank update.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for p in self.base.parameters():
            p.requires_grad = False
        in_dim = base.in_features
        out_dim = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out


def inject_lora_linear(module: nn.Module, rank: int = 8, alpha: int = 32, dropout: float = 0.1,
                       target_keys: Optional[List[str]] = None):
    """
    Recursively wrap Linear layers whose names contain any of target_keys with LoRALinear.
    """
    target_keys = target_keys or ["q_proj", "k_proj", "v_proj", "out_proj"]
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(key in name for key in target_keys):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            inject_lora_linear(child, rank=rank, alpha=alpha, dropout=dropout, target_keys=target_keys)


def inject_lora(model: nn.Module, rank: int = 8, target_layers: Optional[List[str]] = None, alpha: int = 32,
                dropout: float = 0.1) -> nn.Module:
    """
    Convenience wrapper: apply LoRALinear injection in-place and return the model.
    """
    inject_lora_linear(model, rank=rank, alpha=alpha, dropout=dropout, target_keys=target_layers)
    return model


# =============================================================================
# Topology Bias (psi_model-style hierarchical pairs)
# =============================================================================
class TopologyBias(nn.Module):
    """Hierarchical pair bias with collapse token at index 0."""

    def __init__(self, z_dim: int = 128, max_pairs: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.max_pairs = max_pairs
        self.pair_embed_lvl1 = nn.Linear(8, z_dim // 2)
        self.pair_embed_lvl2 = nn.Linear(4, z_dim // 2)

    def create_hierarchical_pairs(self, L: int, idx_map: List[Tuple[int, int]]) -> torch.Tensor:
        pair_id = torch.zeros((L, L), dtype=torch.long)
        pair_id[0, 0] = 0  # collapse self
        pair_id[0, 1:] = 1
        pair_id[1:, 0] = 1
        if not idx_map:
            return pair_id.clamp(max=self.max_pairs - 1)

        hd_start, hd_end = idx_map[0]
        # CDR3 neighbors
        for i in range(hd_start, max(hd_end - 1, hd_start)):
            if i + 1 < hd_end:
                pair_id[i, i + 1] = pair_id[i + 1, i] = 2
        # CDR3 internal
        for i in range(hd_start, hd_end):
            for j in range(hd_start, hd_end):
                if i != j and pair_id[i, j] == 0:
                    pair_id[i, j] = 3
        # CDR3 <-> others
        for i in range(hd_start, hd_end):
            for region_start, region_end in idx_map[1:]:
                pair_id[i, region_start:region_end] = 4
                pair_id[region_start:region_end, i] = 4
        counter = 5
        # others internal
        for region_start, region_end in idx_map[1:]:
            pair_id[region_start:region_end, region_start:region_end] = counter
            counter += 1
        # others <-> others
        conditioning_regions = idx_map[1:]
        for i, (r1_start, r1_end) in enumerate(conditioning_regions):
            for j, (r2_start, r2_end) in enumerate(conditioning_regions[i + 1 :], i + 1):
                pair_id[r1_start:r1_end, r2_start:r2_end] = counter
                pair_id[r2_start:r2_end, r1_start:r1_end] = counter
                counter += 1
        return pair_id.clamp(max=self.max_pairs - 1)

    def forward(self, idx_pairs: List[Tuple[int, int]], total_L: int, device) -> Optional[torch.Tensor]:
        if not idx_pairs:
            return None
        pair_id = self.create_hierarchical_pairs(total_L, idx_pairs).to(device)
        z_lvl1 = self.pair_embed_lvl1(F.one_hot(pair_id // 4, num_classes=8).float())
        z_lvl2 = self.pair_embed_lvl2(F.one_hot(pair_id % 4, num_classes=4).float())
        return torch.cat([z_lvl1, z_lvl2], dim=-1)  # [L, L, z_dim]


# =============================================================================
# Immuno-PLM
# =============================================================================
class ImmunoPLM(nn.Module):
    """Hybrid encoder with optional ESM/LoRA, topology bias, and V/J conditioning."""

    def __init__(
        self,
        hidden_dim: int = 256,
        z_dim: int = 128,
        use_esm: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        vocab_size: int = 256,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        gene_vocab_size: int = 256,
        gene_embed_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_esm = use_esm and ESM_AVAILABLE
        self.use_lora = use_lora and self.use_esm

        # Backbone
        if self.use_esm:
            if esm_model_name == "esm2_t33_650M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.repr_layer = 33
            elif esm_model_name == "esm2_t12_35M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.repr_layer = 12
            else:
                self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.repr_layer = 6
            self.embed_dim = self.esm_model.embed_dim
            # LoRA
            if self.use_lora:
                inject_lora_linear(self.esm_model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
                # report trainable params
                t_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
                a_params = sum(p.numel() for p in self.esm_model.parameters())
                print(f"LoRA injected. Trainable params: {t_params} / {a_params} ({t_params / a_params:.2%})")
            # Freeze base weights when not using LoRA
            if not self.use_lora:
                for p in self.esm_model.parameters():
                    p.requires_grad = False
        else:
            self.alphabet = None
            self.embed_dim = hidden_dim
            self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            self.decoder = nn.Linear(hidden_dim, vocab_size)

        # Topology and fusion
        self.topology_bias = TopologyBias(z_dim=z_dim)
        self.seq_proj = nn.Linear(self.embed_dim, hidden_dim)
        self.pair_fusion = nn.Linear(z_dim, hidden_dim)

        # V/J conditioning
        self.gene_embed_dim = gene_embed_dim
        self.gene_vocab_size = gene_vocab_size
        self.gene_embeds = nn.ModuleDict(
            {k: nn.Embedding(gene_vocab_size, gene_embed_dim) for k in ["h_v", "h_j", "l_v", "l_j"]}
        )
        self.gene_proj = nn.Linear(gene_embed_dim * 4, hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.contrastive_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        region_slices: Optional[List[Dict[str, slice]]] = None,
        gene_ids: Optional[Dict[str, torch.Tensor]] = None,
        return_pair_repr: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, L = tokens.shape
        device = tokens.device

        # Backbone
        if self.use_esm:
            esm_output = self.esm_model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            s = esm_output["representations"][self.repr_layer]
        else:
            s = self.token_embed(tokens)
        s = self.seq_proj(s)

        # Topology bias
        z_topo = None
        z_context = torch.zeros(B, L, self.z_dim, device=device)
        if region_slices:
            for b_idx in range(B):
                region_dict = region_slices[b_idx] if b_idx < len(region_slices) else None
                if not region_dict:
                    continue
                idx_pairs = []
                if "cdr3b" in region_dict:
                    sl = region_dict["cdr3b"]
                    idx_pairs.append((sl.start, sl.stop))
                for key, sl in region_dict.items():
                    if key == "cdr3b":
                        continue
                    idx_pairs.append((sl.start, sl.stop))
                if not idx_pairs:
                    continue
                z_pair = self.topology_bias(idx_pairs, L, device)
                if z_pair is not None:
                    z_ctx = z_pair.max(dim=1)[0]
                    z_context[b_idx] = z_ctx
                    if return_pair_repr and b_idx == 0:
                        z_topo = z_pair
            s = s + self.pair_fusion(z_context)

        # Gene conditioning
        if gene_ids:
            gene_vecs = []
            for key in ["h_v", "h_j", "l_v", "l_j"]:
                if key in gene_ids:
                    gene_vecs.append(self.gene_embeds[key](gene_ids[key].to(device)))
                else:
                    gene_vecs.append(torch.zeros(B, self.gene_embed_dim, device=device))
            gene_cond = torch.cat(gene_vecs, dim=-1)
            gene_cond = self.gene_proj(gene_cond)
            s_cls = s[:, 0, :] + gene_cond
            s = torch.cat([s_cls.unsqueeze(1), s[:, 1:, :]], dim=1)

        # Pooling
        s = self.dropout(s)
        if mask is None:
            pooled = s.mean(dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (s * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        pooled = self.norm(pooled)
        contrastive = self.contrastive_head(pooled)

        return {
            "s": s,
            "pooled": pooled,
            "z_bias": z_context,
            "contrastive": contrastive,
            "z_pair": z_topo,
        }

    @classmethod
    def load(cls, path: str, **kwargs) -> "ImmunoPLM":
        state_dict = torch.load(path, map_location="cpu")
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model

    def save(self, path: str):
        torch.save(self.state_dict(), path)


# =============================================================================
# Contrastive losses
# =============================================================================
def compute_batch_infonce(tcr_emb: torch.Tensor, pmhc_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    tcr_emb = F.normalize(tcr_emb, p=2, dim=-1)
    pmhc_emb = F.normalize(pmhc_emb, p=2, dim=-1)
    logits = torch.matmul(tcr_emb, pmhc_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def compute_infonce_with_negatives(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    pos_sim = (anchor * positive).sum(dim=-1) / temperature
    denom = torch.exp(pos_sim)
    if negatives is not None and negatives.size(0) > 0:
        negatives = F.normalize(negatives, p=2, dim=-1)
        neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.unsqueeze(0).transpose(-2, -1))
        neg_sim = neg_sim.squeeze(1) / temperature
        denom = denom + torch.exp(neg_sim).sum(dim=-1)
    loss = -torch.log(torch.exp(pos_sim) / denom).mean()
    return loss


if __name__ == "__main__":
    print("Quick test (BasicTokenizer mode)...")
    model = ImmunoPLM(hidden_dim=256, z_dim=128, use_esm=False)
    tokens = torch.randint(0, 20, (2, 50))
    mask = torch.ones(2, 50)
    region_slices = [
        {"pep": slice(1, 10), "mhc": slice(11, 40), "cdr3b": slice(41, 50)},
        {"pep": slice(1, 10), "mhc": slice(11, 40), "cdr3b": slice(41, 50)},
    ]
    gene_ids = {k: torch.randint(0, 10, (2,)) for k in ["h_v", "h_j", "l_v", "l_j"]}
    out = model(tokens, mask, region_slices, gene_ids=gene_ids)
    print("s:", out["s"].shape, "pooled:", out["pooled"].shape, "contrastive:", out["contrastive"].shape)
