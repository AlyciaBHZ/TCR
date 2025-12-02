from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Built-in LoRA (no external PEFT dependency)
# =============================================================================
class LoRALinear(nn.Module):
    """Lightweight LoRA for Linear layers: frozen base + trainable low-rank update."""

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


def inject_lora_linear(
    module: nn.Module,
    rank: int = 8,
    alpha: int = 32,
    dropout: float = 0.1,
    target_keys: Optional[List[str]] = None,
):
    """Recursively wrap Linear layers whose names contain any of target_keys with LoRALinear."""
    target_keys = target_keys or ["q_proj", "k_proj", "v_proj", "out_proj"]
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(key in name for key in target_keys):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        else:
            inject_lora_linear(child, rank=rank, alpha=alpha, dropout=dropout, target_keys=target_keys)


# =============================================================================
# ScaffoldRetriever Model
# =============================================================================
class ScaffoldRetriever(nn.Module):
    """
    Shared encoder + allele embedding for pMHC; multi-head logits for HV/HJ/LV/LJ.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_hv: int,
        num_hj: int,
        num_lv: int,
        num_lj: int,
        num_alleles: int,
        use_esm: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        vocab_size: int = 256,
        esm_model_name: str = "esm2_t6_8M_UR50D",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Backbone
        try:
            import esm

            ESM_AVAILABLE = True
        except Exception:
            esm = None
            ESM_AVAILABLE = False

        self.use_esm = use_esm and ESM_AVAILABLE
        self.use_lora = use_lora and self.use_esm

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

            if self.use_lora:
                # Freeze base ESM weights first
                for p in self.esm_model.parameters():
                    p.requires_grad = False
                # Inject LoRA adapters (only LoRA params will be trainable)
                inject_lora_linear(self.esm_model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
                # Report effective trainable ratio
                t_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
                a_params = sum(p.numel() for p in self.esm_model.parameters())
                print(f"[LoRA] Trainable params: {t_params:,}/{a_params:,} ({t_params/a_params:.2%})")
            else:
                for p in self.esm_model.parameters():
                    p.requires_grad = False
        else:
            self.alphabet = None
            self.embed_dim = hidden_dim
            self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.proj = nn.Sequential(nn.Linear(self.embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)

        # Allele embedding (added to pooled pMHC)
        self.allele_embed = nn.Embedding(num_alleles, hidden_dim)

        # Multi-label heads
        self.hv_classifier = nn.Linear(hidden_dim, num_hv)
        self.hj_classifier = nn.Linear(hidden_dim, num_hj)
        self.lv_classifier = nn.Linear(hidden_dim, num_lv)
        self.lj_classifier = nn.Linear(hidden_dim, num_lj)

    # ------------------------------------------------------------------ #
    def _encode_tokens(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode tokens to pooled embedding."""
        if self.use_esm:
            out = self.esm_model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            s = out["representations"][self.repr_layer]
            pooled = s[:, 0, :]
        else:
            s = self.token_embed(tokens)
            pooled = s[:, 0, :]
        pooled = self.dropout(pooled)
        return self.norm(self.proj(pooled))

    def encode_pmhc(self, tokens: torch.Tensor, mask: torch.Tensor, allele_ids: torch.Tensor) -> torch.Tensor:
        z = self._encode_tokens(tokens, mask)
        z = z + self.allele_embed(allele_ids)
        return z

    def forward(
        self,
        pmhc_tokens: torch.Tensor,
        pmhc_mask: torch.Tensor,
        allele_ids: torch.Tensor,
        hv_tokens: torch.Tensor,
        hv_mask: torch.Tensor,
        hj_tokens: torch.Tensor,
        hj_mask: torch.Tensor,
        lv_tokens: torch.Tensor,
        lv_mask: torch.Tensor,
        lj_tokens: torch.Tensor,
        lj_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_pmhc = self.encode_pmhc(pmhc_tokens, pmhc_mask, allele_ids)
        z_hv = self._encode_tokens(hv_tokens, hv_mask)
        z_hj = self._encode_tokens(hj_tokens, hj_mask)
        z_lv = self._encode_tokens(lv_tokens, lv_mask)
        z_lj = self._encode_tokens(lj_tokens, lj_mask)

        hv_logits = self.hv_classifier(z_pmhc)
        hj_logits = self.hj_classifier(z_pmhc)
        lv_logits = self.lv_classifier(z_pmhc)
        lj_logits = self.lj_classifier(z_pmhc)

        return {
            "z_pmhc": z_pmhc,
            "z_hv": z_hv,
            "z_hj": z_hj,
            "z_lv": z_lv,
            "z_lj": z_lj,
            "hv_logits": hv_logits,
            "hj_logits": hj_logits,
            "lv_logits": lv_logits,
            "lj_logits": lj_logits,
        }
