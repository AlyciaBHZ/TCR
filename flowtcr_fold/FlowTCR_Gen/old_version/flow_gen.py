"""
Discrete flow matching generator (categorical simplex approximation).

- Uses convex interpolant x_t = (1 - t) * x0 + t * onehot(y)
- Predicts vector field v_theta(x_t, t, cond) ~ y_onehot - x0
- Conditioning uses Immuno-PLM scaffold encoder outputs (pMHC + HV/HJ/LV/LJ).
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from flowtcr_fold.Immuno_PLM.train_scaffold_retrieval import ScaffoldRetriever
from flowtcr_fold.data.tokenizer import BasicTokenizer, get_tokenizer


def one_hot(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(tokens, num_classes=vocab_size).float()


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -(torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t * freqs  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(t)], dim=-1)
        return emb


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 4,
        cond_dim: int = 0,
        pad_id: int = 0,
        max_len: int = 64,
        n_heads: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def _embed_tokens(self, x_t: torch.Tensor) -> torch.Tensor:
        # Expectation of embeddings under soft distribution x_t
        emb_weights = self.token_embed.weight  # [V, D]
        return torch.matmul(x_t, emb_weights)  # [B, L, D]

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, L, vocab] soft distribution on simplex
            t: [B, 1] time in [0,1]
            cond_emb: [B, cond_dim] conditioning embedding (pMHC + scaffold)
            pad_mask: [B, L] binary mask (1 for valid tokens)
        """
        B, L, _ = x_t.shape
        h = self._embed_tokens(x_t)

        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, L)
        h = h + self.pos_embed(pos_ids)

        t_proj = self.time_embed(t)  # [B, D]
        h = h + t_proj.unsqueeze(1)

        if cond_emb is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond_emb).unsqueeze(1)

        key_padding = None
        if pad_mask is not None:
            key_padding = pad_mask == 0
        h = self.transformer(h, src_key_padding_mask=key_padding)
        return self.head(h)  # [B, L, V]

    def sample_x0(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Uniform base distribution on simplex."""
        B, L = shape
        base = torch.full((B, L, self.vocab_size), 1.0 / self.vocab_size, device=self.token_embed.weight.device)
        return base

    def flow_matching_loss(
        self,
        tokens: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Categorical flow matching with convex interpolant.
        tokens: [B, L] target discrete ids
        pad_mask: [B, L] where 1=valid, 0=pad
        """
        B, L = tokens.shape
        if pad_mask is None:
            pad_mask = torch.ones_like(tokens, dtype=torch.float32)
        else:
            pad_mask = pad_mask.float()
        y_onehot = one_hot(tokens.clamp(min=0), self.vocab_size)
        x0 = self.sample_x0((B, L))
        t = torch.rand(B, 1, device=tokens.device)
        x_t = (1 - t) * x0 + t * y_onehot
        v_target = y_onehot - x0
        v_pred = self.forward(x_t, t, cond_emb=cond_emb, pad_mask=pad_mask)
        diff = (v_pred - v_target).pow(2)
        diff = diff * pad_mask.unsqueeze(-1)
        denom = pad_mask.sum() * self.vocab_size
        denom = denom.clamp(min=1.0)
        return diff.sum() / denom

    @torch.no_grad()
    def sample(
        self,
        cond_emb: Optional[torch.Tensor],
        seq_len: int,
        steps: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Euler sampler from uniform x0 to x1 using learned vector field.
        Returns token ids [B, L].
        """
        device = self.token_embed.weight.device
        B = cond_emb.size(0) if cond_emb is not None else 1
        x = self.sample_x0((B, seq_len)).to(device)
        for i in range(steps):
            t = torch.full((B, 1), (i + 1) / steps, device=device)
            v = self.forward(x, t, cond_emb=cond_emb)
            x = x + (1.0 / steps) * v
            x = torch.clamp(x, min=1e-6)
            x = x / x.sum(dim=-1, keepdim=True)
        logits = x / temperature
        return logits.argmax(dim=-1)


class ConditionEmbedder:
    """
    Wraps the trained ScaffoldRetriever to produce conditioning embeddings for FlowTCR-Gen.
    """

    def __init__(
        self,
        ckpt_path: str,
        tokenizer=None,
        device: Optional[torch.device] = None,
        gene_vocab_path: Optional[str] = None,
        hidden_dim: int = 256,
        use_esm: bool = False,
        esm_model: str = "esm2_t12_35M_UR50D",
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        max_len: int = 512,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer or get_tokenizer()
        ckpt_path = Path(ckpt_path)
        gene_vocab_path = Path(gene_vocab_path) if gene_vocab_path else ckpt_path.with_name("gene_vocab.json")
        self.gene_vocab = self._load_gene_vocab(gene_vocab_path)
        self.esm_model = esm_model
        self.model = ScaffoldRetriever(
            hidden_dim=hidden_dim,
            num_hv=len(self.gene_vocab.get("h_v", {})),
            num_hj=len(self.gene_vocab.get("h_j", {})),
            num_lv=len(self.gene_vocab.get("l_v", {})),
            num_lj=len(self.gene_vocab.get("l_j", {})),
            use_esm=use_esm,
            esm_model=esm_model,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            vocab_size=self._vocab_size(),
        )
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_len = max_len
        self.pad_id = self._pad_id()
        self.cond_dim = hidden_dim * 5

    def _pad_id(self) -> int:
        tok = self.tokenizer
        if hasattr(tok, "pad_token_id"):
            return int(tok.pad_token_id)
        if isinstance(tok, BasicTokenizer):
            return tok.stoi["[PAD]"]
        return 0

    def _vocab_size(self) -> int:
        tok = self.tokenizer
        if isinstance(tok, BasicTokenizer):
            return len(tok.itos)
        if hasattr(tok, "all_toks"):
            return len(tok.all_toks)
        if hasattr(tok, "vocab"):
            return len(tok.vocab)
        return 256

    def _load_gene_vocab(self, path: Path) -> Dict:
        if path.exists():
            with path.open() as f:
                return json.load(f)
        return {"h_v": {}, "h_j": {}, "l_v": {}, "l_j": {}}

    def _tokenize_seq(self, seq: str) -> torch.Tensor:
        tok = self.tokenizer
        if hasattr(tok, "cls_token_id"):
            cls_idx = tok.cls_token_id
            eos_idx = tok.eos_token_id
        else:
            cls_idx = tok.stoi["[CLS]"]
            eos_idx = tok.stoi["[SEP]"]
        tokens = [cls_idx]
        tokens.extend(tok.encode(seq))
        tokens.append(eos_idx)
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len - 1] + [eos_idx]
        return torch.tensor(tokens, dtype=torch.long)

    def _tokenize_pmhc(self, peptide: str, mhc: str) -> torch.Tensor:
        tok = self.tokenizer
        if hasattr(tok, "cls_token_id"):
            cls_idx = tok.cls_token_id
            sep_idx = tok.eos_token_id
            eos_idx = tok.eos_token_id
        else:
            cls_idx = tok.stoi["[CLS]"]
            sep_idx = tok.stoi["[SEP]"]
            eos_idx = tok.stoi.get("[SEP]", 0)
        tokens = [cls_idx]
        tokens.extend(tok.encode(peptide))
        tokens.append(sep_idx)
        tokens.extend(tok.encode(mhc))
        tokens.append(eos_idx)
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len - 1] + [eos_idx]
        return torch.tensor(tokens, dtype=torch.long)

    def _pad_batch(self, seqs: List[torch.Tensor]) -> torch.Tensor:
        if not seqs:
            return torch.zeros(0, 0, dtype=torch.long, device=self.device)
        max_len = max(t.size(0) for t in seqs)
        padded = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        for i, t in enumerate(seqs):
            padded[i, : t.size(0)] = t
        return padded.to(self.device)

    @torch.no_grad()
    def __call__(self, metas: Iterable[Dict], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        pmhc_tokens = []
        hv_tokens = []
        hj_tokens = []
        lv_tokens = []
        lj_tokens = []
        for meta in metas:
            pmhc_tokens.append(self._tokenize_pmhc(meta.get("peptide", ""), meta.get("mhc", "")))
            hv_tokens.append(self._tokenize_seq(meta.get("h_v_seq", "") or meta.get("h_v", "")))
            hj_tokens.append(self._tokenize_seq(meta.get("h_j_seq", "") or meta.get("h_j", "")))
            lv_tokens.append(self._tokenize_seq(meta.get("l_v_seq", "") or meta.get("l_v", "")))
            lj_tokens.append(self._tokenize_seq(meta.get("l_j_seq", "") or meta.get("l_j", "")))

        pmhc = self._pad_batch(pmhc_tokens)
        hv = self._pad_batch(hv_tokens)
        hj = self._pad_batch(hj_tokens)
        lv = self._pad_batch(lv_tokens)
        lj = self._pad_batch(lj_tokens)

        z_pmhc = self.model.encode(pmhc, None)
        z_hv = self.model.encode(hv, None)
        z_hj = self.model.encode(hj, None)
        z_lv = self.model.encode(lv, None)
        z_lj = self.model.encode(lj, None)
        return torch.cat([z_pmhc, z_hv, z_hj, z_lv, z_lj], dim=-1)
