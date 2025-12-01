"""
Scaffold Retrieval Training (InfoNCE + Classification)

Confirmed Pipeline:
==================

A. Training:
   - Input: Batch of (pMHC, HV, HJ, LV, LJ sequences, Gene IDs)
   - Forward: 5 encodings (shared encoder)
   - Loss: 
     * L_NCE = InfoNCE(z_pmhc, z_hv) + InfoNCE(z_pmhc, z_hj) + ...  (4 parallel)
     * L_CLS = CrossEntropy(logits_hv, hv_id) + ...  (4 parallel, auxiliary)
     * Total = L_NCE + weight * L_CLS

B. Indexing (after training, done once):
   - Collect unique HV/HJ/LV/LJ sequences from training data
   - Encode each → store in Bank tensors

C. Inference:
   - Input: new pMHC
   - Encode: query = Encoder(pMHC)
   - Retrieve: score = query @ Bank.T → argmax → best sequences
   - Output: HV_seq, HJ_seq, LV_seq, LJ_seq

Usage:
    # Training
    python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \\
        --data flowtcr_fold/data/trn.jsonl \\
        --epochs 100 --batch_size 32

    # With ESM-2 + LoRA
    python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \\
        --data flowtcr_fold/data/trn.jsonl \\
        --use_esm --use_lora --lora_rank 8 \\
        --epochs 100 --batch_size 16
"""

# Force unbuffered output for real-time logging (especially with SLURM)
import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Dataset
# =============================================================================
class ScaffoldRetrievalDataset(Dataset):
    """
    Dataset for scaffold retrieval training.
    
    Returns:
    - pMHC tokens
    - 4 V/J sequence tokens (HV, HJ, LV, LJ)
    - 4 Gene IDs (for classification loss)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        max_len: int = 512,
        gene_vocab: Optional[Dict] = None,
        build_vocab: bool = False,
    ):
        self.path = Path(data_path)
        self.max_len = max_len
        
        if tokenizer is None:
            from flowtcr_fold.data.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()
        self.tokenizer = tokenizer
        
        # Gene vocabulary: Gene Name → ID
        self.gene_vocab = gene_vocab or {
            "h_v": {"<UNK>": 0},
            "h_j": {"<UNK>": 0},
            "l_v": {"<UNK>": 0},
            "l_j": {"<UNK>": 0},
        }
        
        self.samples = self._load()
        
        if build_vocab:
            self._build_vocab()
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Gene vocab sizes: HV={len(self.gene_vocab['h_v'])}, "
              f"HJ={len(self.gene_vocab['h_j'])}, "
              f"LV={len(self.gene_vocab['l_v'])}, "
              f"LJ={len(self.gene_vocab['l_j'])}")
    
    def _load(self) -> List[Dict]:
        samples = []
        
        if self.path.suffix == ".csv":
            with self.path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(self._parse_row(row))
        else:
            with self.path.open() as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        samples.append(self._parse_row(obj))
        
        # Filter: must have pMHC and at least HV+HJ
        valid = [s for s in samples 
                 if s["peptide"] and s["mhc"] and s["h_v_seq"] and s["h_j_seq"]]
        print(f"Filtered: {len(samples)} → {len(valid)} (require pMHC + h_v_seq + h_j_seq)")
        return valid
    
    def _parse_row(self, row: dict) -> Dict:
        return {
            "peptide": row.get("peptide", "") or "",
            "mhc": row.get("mhc", "") or row.get("mhc_seq", "") or "",
            "h_v": row.get("h_v", "") or "",  # Gene Name
            "h_j": row.get("h_j", "") or "",
            "l_v": row.get("l_v", "") or "",
            "l_j": row.get("l_j", "") or "",
            "h_v_seq": row.get("h_v_seq", "") or "",  # Gene Sequence
            "h_j_seq": row.get("h_j_seq", "") or "",
            "l_v_seq": row.get("l_v_seq", "") or "",
            "l_j_seq": row.get("l_j_seq", "") or "",
        }
    
    def _build_vocab(self):
        """Build gene name → ID vocabulary."""
        for sample in self.samples:
            for gene_type in ["h_v", "h_j", "l_v", "l_j"]:
                name = sample[gene_type]
                if name and name not in self.gene_vocab[gene_type]:
                    self.gene_vocab[gene_type][name] = len(self.gene_vocab[gene_type])
    
    def __len__(self):
        return len(self.samples)
    
    def _tokenize(self, seq: str, add_special: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a sequence."""
        if not seq:
            tok = self.tokenizer
            if hasattr(tok, "cls_token_id"):
                tokens = [tok.cls_token_id, tok.eos_token_id]
            else:
                tokens = [tok.stoi["[CLS]"], tok.stoi["[SEP]"]]
            return torch.tensor(tokens, dtype=torch.long), torch.tensor([1, 1], dtype=torch.long)
        
        tok = self.tokenizer
        if hasattr(tok, "cls_token_id"):
            cls_idx = tok.cls_token_id
            eos_idx = tok.eos_token_id
        else:
            cls_idx = tok.stoi["[CLS]"]
            eos_idx = tok.stoi["[SEP]"]
        
        tokens = [cls_idx] if add_special else []
        tokens.extend(tok.encode(seq))
        if add_special:
            tokens.append(eos_idx)
        
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len - 1] + [eos_idx]
        
        mask = [1] * len(tokens)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
    
    def _tokenize_pmhc(self, sample: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize pMHC: [CLS] peptide [SEP] mhc [EOS]"""
        tok = self.tokenizer
        if hasattr(tok, "cls_token_id"):
            cls_idx = tok.cls_token_id
            sep_idx = tok.eos_token_id
            eos_idx = tok.eos_token_id
        else:
            cls_idx = tok.stoi["[CLS]"]
            sep_idx = tok.stoi["[SEP]"]
            eos_idx = tok.stoi.get("[EOS]", sep_idx)
        
        tokens = [cls_idx]
        tokens.extend(tok.encode(sample["peptide"]))
        tokens.append(sep_idx)
        tokens.extend(tok.encode(sample["mhc"]))
        tokens.append(eos_idx)
        
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len - 1] + [eos_idx]
        
        mask = [1] * len(tokens)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Tokenize pMHC
        pmhc_tokens, pmhc_mask = self._tokenize_pmhc(sample)
        
        # Tokenize V/J sequences
        hv_tokens, hv_mask = self._tokenize(sample["h_v_seq"])
        hj_tokens, hj_mask = self._tokenize(sample["h_j_seq"])
        lv_tokens, lv_mask = self._tokenize(sample["l_v_seq"])
        lj_tokens, lj_mask = self._tokenize(sample["l_j_seq"])
        
        # Gene IDs (for classification loss)
        hv_id = self.gene_vocab["h_v"].get(sample["h_v"], 0)
        hj_id = self.gene_vocab["h_j"].get(sample["h_j"], 0)
        lv_id = self.gene_vocab["l_v"].get(sample["l_v"], 0)
        lj_id = self.gene_vocab["l_j"].get(sample["l_j"], 0)
        
        return {
            "pmhc_tokens": pmhc_tokens,
            "pmhc_mask": pmhc_mask,
            "hv_tokens": hv_tokens,
            "hv_mask": hv_mask,
            "hj_tokens": hj_tokens,
            "hj_mask": hj_mask,
            "lv_tokens": lv_tokens,
            "lv_mask": lv_mask,
            "lj_tokens": lj_tokens,
            "lj_mask": lj_mask,
            "hv_id": hv_id,
            "hj_id": hj_id,
            "lv_id": lv_id,
            "lj_id": lj_id,
            # Keep sequences for bank building
            "h_v_seq": sample["h_v_seq"],
            "h_j_seq": sample["h_j_seq"],
            "l_v_seq": sample["l_v_seq"],
            "l_j_seq": sample["l_j_seq"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function with padding."""
    
    def pad_sequences(tokens_list, masks_list):
        B = len(tokens_list)
        max_len = max(t.size(0) for t in tokens_list)
        tokens = torch.zeros(B, max_len, dtype=torch.long)
        masks = torch.zeros(B, max_len, dtype=torch.long)
        for i, (t, m) in enumerate(zip(tokens_list, masks_list)):
            tokens[i, :t.size(0)] = t
            masks[i, :m.size(0)] = m
        return tokens, masks
    
    pmhc_tokens, pmhc_mask = pad_sequences(
        [item["pmhc_tokens"] for item in batch],
        [item["pmhc_mask"] for item in batch]
    )
    hv_tokens, hv_mask = pad_sequences(
        [item["hv_tokens"] for item in batch],
        [item["hv_mask"] for item in batch]
    )
    hj_tokens, hj_mask = pad_sequences(
        [item["hj_tokens"] for item in batch],
        [item["hj_mask"] for item in batch]
    )
    lv_tokens, lv_mask = pad_sequences(
        [item["lv_tokens"] for item in batch],
        [item["lv_mask"] for item in batch]
    )
    lj_tokens, lj_mask = pad_sequences(
        [item["lj_tokens"] for item in batch],
        [item["lj_mask"] for item in batch]
    )
    
    return {
        "pmhc_tokens": pmhc_tokens,
        "pmhc_mask": pmhc_mask,
        "hv_tokens": hv_tokens,
        "hv_mask": hv_mask,
        "hj_tokens": hj_tokens,
        "hj_mask": hj_mask,
        "lv_tokens": lv_tokens,
        "lv_mask": lv_mask,
        "lj_tokens": lj_tokens,
        "lj_mask": lj_mask,
        "hv_id": torch.tensor([item["hv_id"] for item in batch], dtype=torch.long),
        "hj_id": torch.tensor([item["hj_id"] for item in batch], dtype=torch.long),
        "lv_id": torch.tensor([item["lv_id"] for item in batch], dtype=torch.long),
        "lj_id": torch.tensor([item["lj_id"] for item in batch], dtype=torch.long),
        # valid_seq: sequence exists → use for InfoNCE (don't waste signal!)
        "valid_hv_seq": torch.tensor([bool(item["h_v_seq"]) for item in batch], dtype=torch.bool),
        "valid_hj_seq": torch.tensor([bool(item["h_j_seq"]) for item in batch], dtype=torch.bool),
        "valid_lv_seq": torch.tensor([bool(item["l_v_seq"]) for item in batch], dtype=torch.bool),
        "valid_lj_seq": torch.tensor([bool(item["l_j_seq"]) for item in batch], dtype=torch.bool),
        # valid_id: gene_id exists → use for classification loss
        "valid_hv_id": torch.tensor([item["hv_id"] > 0 for item in batch], dtype=torch.bool),
        "valid_hj_id": torch.tensor([item["hj_id"] > 0 for item in batch], dtype=torch.bool),
        "valid_lv_id": torch.tensor([item["lv_id"] > 0 for item in batch], dtype=torch.bool),
        "valid_lj_id": torch.tensor([item["lj_id"] > 0 for item in batch], dtype=torch.bool),
    }


# =============================================================================
# Model: Shared Encoder + Classification Heads
# =============================================================================
class ScaffoldRetriever(nn.Module):
    """
    Scaffold retrieval model.
    
    - Shared ESM-2/Transformer encoder for all inputs
    - 4 classification heads (auxiliary loss)
    - Single encode() function for both pMHC and V/J sequences
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_hv: int = 100,
        num_hj: int = 20,
        num_lv: int = 100,
        num_lj: int = 20,
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
        
        # ESM backbone
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
                from flowtcr_fold.Immuno_PLM.immuno_plm import inject_lora_linear
                inject_lora_linear(self.esm_model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
                t_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
                a_params = sum(p.numel() for p in self.esm_model.parameters())
                print(f"LoRA injected. Trainable: {t_params:,} / {a_params:,} ({t_params/a_params:.2%})")
            else:
                for p in self.esm_model.parameters():
                    p.requires_grad = False
        else:
            self.alphabet = None
            self.embed_dim = hidden_dim
            self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        # Projection head (shared)
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Classification heads (auxiliary)
        self.hv_classifier = nn.Linear(hidden_dim, num_hv)
        self.hj_classifier = nn.Linear(hidden_dim, num_hj)
        self.lv_classifier = nn.Linear(hidden_dim, num_lv)
        self.lj_classifier = nn.Linear(hidden_dim, num_lj)
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encode any sequence → [B, hidden_dim] embedding.
        
        Used for both pMHC and V/J sequences (shared encoder).
        """
        if self.use_esm:
            out = self.esm_model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            s = out["representations"][self.repr_layer]
        else:
            s = self.token_embed(tokens)
        
        # Pool using CLS token
        pooled = s[:, 0, :]
        pooled = self.dropout(pooled)
        projected = self.proj(pooled)
        return self.norm(projected)
    
    def forward(
        self,
        pmhc_tokens: torch.Tensor,
        pmhc_mask: torch.Tensor,
        hv_tokens: torch.Tensor,
        hv_mask: torch.Tensor,
        hj_tokens: torch.Tensor,
        hj_mask: torch.Tensor,
        lv_tokens: torch.Tensor,
        lv_mask: torch.Tensor,
        lj_tokens: torch.Tensor,
        lj_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        5 encodings (shared encoder):
        - z_pmhc: pMHC embedding
        - z_hv, z_hj, z_lv, z_lj: V/J embeddings
        
        4 classification logits (from z_pmhc):
        - hv_logits, hj_logits, lv_logits, lj_logits
        """
        # 5 encodings (shared encoder)
        z_pmhc = self.encode(pmhc_tokens, pmhc_mask)
        z_hv = self.encode(hv_tokens, hv_mask)
        z_hj = self.encode(hj_tokens, hj_mask)
        z_lv = self.encode(lv_tokens, lv_mask)
        z_lj = self.encode(lj_tokens, lj_mask)
        
        # Classification logits (from pMHC embedding)
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


# =============================================================================
# Loss Functions
# =============================================================================
def compute_infonce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    InfoNCE loss with in-batch negatives.
    
    anchor[i] should be similar to positive[i], dissimilar to positive[j] (j != i)
    """
    if valid_mask is not None:
        anchor = anchor[valid_mask]
        positive = positive[valid_mask]
        if anchor.size(0) == 0:
            return torch.tensor(0.0, device=anchor.device)
    
    # Normalize
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    
    # Similarity matrix [B, B]
    logits = torch.matmul(anchor, positive.T) / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Symmetric loss
    loss_a2p = F.cross_entropy(logits, labels)
    loss_p2a = F.cross_entropy(logits.T, labels)
    
    return (loss_a2p + loss_p2a) / 2


def compute_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Classification loss with accuracy."""
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device), 0.0
    
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    
    return loss, acc


# =============================================================================
# Bank Building & Retrieval
# =============================================================================
class ScaffoldBank:
    """
    Bank of V/J sequence embeddings for retrieval.
    
    Structure:
    - hv_bank: {seq: (embedding, gene_name), ...}
    - Similar for hj, lv, lj
    """
    
    def __init__(self):
        self.banks = {
            "h_v": OrderedDict(),  # seq → (embedding, gene_name)
            "h_j": OrderedDict(),
            "l_v": OrderedDict(),
            "l_j": OrderedDict(),
        }
        self.bank_tensors = {}  # gene_type → [N, D] tensor
        self.bank_seqs = {}     # gene_type → [seq1, seq2, ...]
    
    def add(self, gene_type: str, seq: str, embedding: torch.Tensor, gene_name: str = ""):
        """Add a sequence to the bank."""
        if seq and seq not in self.banks[gene_type]:
            self.banks[gene_type][seq] = (embedding.detach().cpu(), gene_name)
    
    def build_tensors(self, device: torch.device):
        """Build tensor banks for fast retrieval."""
        for gene_type in ["h_v", "h_j", "l_v", "l_j"]:
            if not self.banks[gene_type]:
                self.bank_tensors[gene_type] = torch.zeros(1, 256, device=device)
                self.bank_seqs[gene_type] = [""]
                continue
            
            seqs = list(self.banks[gene_type].keys())
            embeddings = torch.stack([self.banks[gene_type][s][0] for s in seqs])
            
            self.bank_seqs[gene_type] = seqs
            self.bank_tensors[gene_type] = F.normalize(embeddings, p=2, dim=-1).to(device)
    
    def retrieve(
        self,
        query: torch.Tensor,  # [1, D] or [B, D]
        gene_type: str,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k sequences for a query.
        
        Returns: [(seq, score), ...]
        """
        query = F.normalize(query, p=2, dim=-1)
        bank = self.bank_tensors[gene_type]
        
        # [B, N]
        scores = torch.matmul(query, bank.T)
        
        results = []
        for i in range(query.size(0)):
            top_scores, top_indices = scores[i].topk(min(top_k, len(self.bank_seqs[gene_type])))
            sample_results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                seq = self.bank_seqs[gene_type][idx]
                sample_results.append((seq, score))
            results.append(sample_results)
        
        return results if query.size(0) > 1 else results[0]
    
    def save(self, path: str):
        """Save bank to file."""
        data = {
            gene_type: {seq: (emb.tolist(), name) for seq, (emb, name) in bank.items()}
            for gene_type, bank in self.banks.items()
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str, device: torch.device) -> "ScaffoldBank":
        """Load bank from file."""
        with open(path) as f:
            data = json.load(f)
        
        bank = cls()
        for gene_type, entries in data.items():
            for seq, (emb_list, name) in entries.items():
                emb = torch.tensor(emb_list)
                bank.banks[gene_type][seq] = (emb, name)
        
        bank.build_tensors(device)
        return bank
    
    def __repr__(self):
        sizes = {k: len(v) for k, v in self.banks.items()}
        return f"ScaffoldBank(HV={sizes['h_v']}, HJ={sizes['h_j']}, LV={sizes['l_v']}, LJ={sizes['l_j']})"


def collect_unique_sequences_from_file(path: str) -> Dict[str, Dict[str, str]]:
    """
    Collect unique V/J sequences from a data file (CSV or JSONL).
    
    Returns:
        Dict mapping gene_type → {seq → gene_name}
    """
    unique_seqs = {
        "h_v": {},  # seq → gene_name
        "h_j": {},
        "l_v": {},
        "l_j": {},
    }
    
    path = Path(path)
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return unique_seqs
    
    if path.suffix == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("h_v_seq"):
                    unique_seqs["h_v"][row["h_v_seq"]] = row.get("h_v", "")
                if row.get("h_j_seq"):
                    unique_seqs["h_j"][row["h_j_seq"]] = row.get("h_j", "")
                if row.get("l_v_seq"):
                    unique_seqs["l_v"][row["l_v_seq"]] = row.get("l_v", "")
                if row.get("l_j_seq"):
                    unique_seqs["l_j"][row["l_j_seq"]] = row.get("l_j", "")
    else:  # JSONL
        with path.open() as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if obj.get("h_v_seq"):
                        unique_seqs["h_v"][obj["h_v_seq"]] = obj.get("h_v", "")
                    if obj.get("h_j_seq"):
                        unique_seqs["h_j"][obj["h_j_seq"]] = obj.get("h_j", "")
                    if obj.get("l_v_seq"):
                        unique_seqs["l_v"][obj["l_v_seq"]] = obj.get("l_v", "")
                    if obj.get("l_j_seq"):
                        unique_seqs["l_j"][obj["l_j_seq"]] = obj.get("l_j", "")
    
    return unique_seqs


def build_bank(
    model: ScaffoldRetriever,
    dataset: ScaffoldRetrievalDataset,
    device: torch.device,
    mode: str = "benchmark",
    extra_data_paths: Optional[List[str]] = None,
) -> ScaffoldBank:
    """
    Build scaffold bank from dataset(s).
    
    Args:
        model: Trained ScaffoldRetriever model
        dataset: Training dataset (used for tokenization)
        device: torch device
        mode: "benchmark" (only training data) or "production" (all data)
        extra_data_paths: Additional data files to include (for production mode)
    
    Returns:
        ScaffoldBank with encoded V/J sequences
    
    Notes:
        - benchmark mode: Only uses training data. For fair evaluation,
          the bank should NOT contain test/val data to avoid data leakage.
        - production mode: Uses all available data for maximum coverage.
          Use this when deploying for real TCR design tasks.
    """
    model.eval()
    bank = ScaffoldBank()
    
    # Collect unique sequences from training data
    unique_seqs = {
        "h_v": {},  # seq → gene_name
        "h_j": {},
        "l_v": {},
        "l_j": {},
    }
    
    # Always include training data
    for sample in dataset.samples:
        if sample["h_v_seq"]:
            unique_seqs["h_v"][sample["h_v_seq"]] = sample["h_v"]
        if sample["h_j_seq"]:
            unique_seqs["h_j"][sample["h_j_seq"]] = sample["h_j"]
        if sample["l_v_seq"]:
            unique_seqs["l_v"][sample["l_v_seq"]] = sample["l_v"]
        if sample["l_j_seq"]:
            unique_seqs["l_j"][sample["l_j_seq"]] = sample["l_j"]
    
    print(f"[{mode} mode] Training data: HV={len(unique_seqs['h_v'])}, "
          f"HJ={len(unique_seqs['h_j'])}, LV={len(unique_seqs['l_v'])}, LJ={len(unique_seqs['l_j'])}")
    
    # For production mode, add extra data sources
    if mode == "production" and extra_data_paths:
        for extra_path in extra_data_paths:
            print(f"  Adding extra data: {extra_path}")
            extra_seqs = collect_unique_sequences_from_file(extra_path)
            for gene_type in ["h_v", "h_j", "l_v", "l_j"]:
                for seq, name in extra_seqs[gene_type].items():
                    if seq not in unique_seqs[gene_type]:
                        unique_seqs[gene_type][seq] = name
        
        print(f"[{mode} mode] After adding extra data: HV={len(unique_seqs['h_v'])}, "
              f"HJ={len(unique_seqs['h_j'])}, LV={len(unique_seqs['l_v'])}, LJ={len(unique_seqs['l_j'])}")
    
    # Encode each unique sequence
    with torch.no_grad():
        for gene_type, seqs in unique_seqs.items():
            for seq, gene_name in seqs.items():
                tokens, mask = dataset._tokenize(seq)
                tokens = tokens.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)
                
                emb = model.encode(tokens, mask)  # [1, D]
                bank.add(gene_type, seq, emb[0], gene_name)
    
    bank.build_tensors(device)
    print(f"Bank built: {bank}")
    
    return bank


# =============================================================================
# Training
# =============================================================================
def train_epoch(
    model: ScaffoldRetriever,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_nce = 0.0
    total_cls = 0.0
    acc_hv, acc_hj, acc_lv, acc_lj = 0.0, 0.0, 0.0, 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        pmhc_tokens = batch["pmhc_tokens"].to(device)
        pmhc_mask = batch["pmhc_mask"].to(device)
        hv_tokens = batch["hv_tokens"].to(device)
        hv_mask = batch["hv_mask"].to(device)
        hj_tokens = batch["hj_tokens"].to(device)
        hj_mask = batch["hj_mask"].to(device)
        lv_tokens = batch["lv_tokens"].to(device)
        lv_mask = batch["lv_mask"].to(device)
        lj_tokens = batch["lj_tokens"].to(device)
        lj_mask = batch["lj_mask"].to(device)
        
        hv_id = batch["hv_id"].to(device)
        hj_id = batch["hj_id"].to(device)
        lv_id = batch["lv_id"].to(device)
        lj_id = batch["lj_id"].to(device)
        
        # valid_seq: sequence exists → use for InfoNCE
        valid_hv_seq = batch["valid_hv_seq"].to(device)
        valid_hj_seq = batch["valid_hj_seq"].to(device)
        valid_lv_seq = batch["valid_lv_seq"].to(device)
        valid_lj_seq = batch["valid_lj_seq"].to(device)
        # valid_id: gene_id exists → use for classification
        valid_hv_id = batch["valid_hv_id"].to(device)
        valid_hj_id = batch["valid_hj_id"].to(device)
        valid_lv_id = batch["valid_lv_id"].to(device)
        valid_lj_id = batch["valid_lj_id"].to(device)
        
        # Forward
        out = model(
            pmhc_tokens, pmhc_mask,
            hv_tokens, hv_mask,
            hj_tokens, hj_mask,
            lv_tokens, lv_mask,
            lj_tokens, lj_mask,
        )
        
        # InfoNCE losses (main) - use valid_seq (sequence exists)
        loss_nce_hv = compute_infonce(out["z_pmhc"], out["z_hv"], args.tau, valid_hv_seq)
        loss_nce_hj = compute_infonce(out["z_pmhc"], out["z_hj"], args.tau, valid_hj_seq)
        loss_nce_lv = compute_infonce(out["z_pmhc"], out["z_lv"], args.tau, valid_lv_seq)
        loss_nce_lj = compute_infonce(out["z_pmhc"], out["z_lj"], args.tau, valid_lj_seq)
        loss_nce = loss_nce_hv + loss_nce_hj + loss_nce_lv + loss_nce_lj
        
        # Classification losses (auxiliary) - use valid_id (gene_id exists)
        loss_cls_hv, a_hv = compute_classification_loss(out["hv_logits"], hv_id, valid_hv_id)
        loss_cls_hj, a_hj = compute_classification_loss(out["hj_logits"], hj_id, valid_hj_id)
        loss_cls_lv, a_lv = compute_classification_loss(out["lv_logits"], lv_id, valid_lv_id)
        loss_cls_lj, a_lj = compute_classification_loss(out["lj_logits"], lj_id, valid_lj_id)
        loss_cls = loss_cls_hv + loss_cls_hj + loss_cls_lv + loss_cls_lj
        
        # Total loss
        loss = loss_nce + args.cls_weight * loss_cls
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        total_nce += loss_nce.item()
        total_cls += loss_cls.item()
        acc_hv += a_hv
        acc_hj += a_hj
        acc_lv += a_lv
        acc_lj += a_lj
        num_batches += 1
        
        if (batch_idx + 1) % args.log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}: "
                  f"loss={loss.item():.4f} (NCE={loss_nce.item():.3f}, CLS={loss_cls.item():.3f})")
    
    return {
        "loss": total_loss / num_batches,
        "nce": total_nce / num_batches,
        "cls": total_cls / num_batches,
        "acc_hv": acc_hv / num_batches,
        "acc_hj": acc_hj / num_batches,
        "acc_lv": acc_lv / num_batches,
        "acc_lj": acc_lj / num_batches,
    }


def evaluate(
    model: ScaffoldRetriever,
    loader: DataLoader,
    device: torch.device,
    args,
) -> Dict[str, float]:
    """Evaluate model on validation set with retrieval metrics."""
    model.eval()
    
    all_pmhc_emb = []
    all_hv_emb = []
    all_hj_emb = []
    all_lv_emb = []
    all_lj_emb = []
    all_valid_hv = []
    all_valid_hj = []
    all_valid_lv = []
    all_valid_lj = []
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            pmhc_tokens = batch["pmhc_tokens"].to(device)
            pmhc_mask = batch["pmhc_mask"].to(device)
            hv_tokens = batch["hv_tokens"].to(device)
            hv_mask = batch["hv_mask"].to(device)
            hj_tokens = batch["hj_tokens"].to(device)
            hj_mask = batch["hj_mask"].to(device)
            lv_tokens = batch["lv_tokens"].to(device)
            lv_mask = batch["lv_mask"].to(device)
            lj_tokens = batch["lj_tokens"].to(device)
            lj_mask = batch["lj_mask"].to(device)
            
            # Use valid_seq for InfoNCE (sequence exists)
            valid_hv_seq = batch["valid_hv_seq"].to(device)
            valid_hj_seq = batch["valid_hj_seq"].to(device)
            valid_lv_seq = batch["valid_lv_seq"].to(device)
            valid_lj_seq = batch["valid_lj_seq"].to(device)
            
            out = model(
                pmhc_tokens, pmhc_mask,
                hv_tokens, hv_mask,
                hj_tokens, hj_mask,
                lv_tokens, lv_mask,
                lj_tokens, lj_mask,
            )
            
            # InfoNCE losses - use valid_seq
            loss_nce_hv = compute_infonce(out["z_pmhc"], out["z_hv"], args.tau, valid_hv_seq)
            loss_nce_hj = compute_infonce(out["z_pmhc"], out["z_hj"], args.tau, valid_hj_seq)
            loss_nce_lv = compute_infonce(out["z_pmhc"], out["z_lv"], args.tau, valid_lv_seq)
            loss_nce_lj = compute_infonce(out["z_pmhc"], out["z_lj"], args.tau, valid_lj_seq)
            loss = loss_nce_hv + loss_nce_hj + loss_nce_lv + loss_nce_lj
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect embeddings for retrieval metrics
            all_pmhc_emb.append(out["z_pmhc"])
            all_hv_emb.append(out["z_hv"])
            all_hj_emb.append(out["z_hj"])
            all_lv_emb.append(out["z_lv"])
            all_lj_emb.append(out["z_lj"])
            all_valid_hv.append(valid_hv_seq)
            all_valid_hj.append(valid_hj_seq)
            all_valid_lv.append(valid_lv_seq)
            all_valid_lj.append(valid_lj_seq)
    
    # Compute retrieval metrics (Recall@K)
    all_pmhc_emb = torch.cat(all_pmhc_emb, dim=0)
    all_hv_emb = torch.cat(all_hv_emb, dim=0)
    all_valid_hv = torch.cat(all_valid_hv, dim=0)
    
    def compute_recall(pmhc_emb, gene_emb, valid_mask, k=10):
        if not valid_mask.any():
            return 0.0
        pmhc_emb = pmhc_emb[valid_mask]
        gene_emb = gene_emb[valid_mask]
        
        pmhc_emb = F.normalize(pmhc_emb, p=2, dim=-1)
        gene_emb = F.normalize(gene_emb, p=2, dim=-1)
        
        sim = torch.matmul(pmhc_emb, gene_emb.T)
        N = sim.size(0)
        labels = torch.arange(N, device=sim.device)
        
        _, top_k_idx = sim.topk(min(k, N), dim=1)
        recall = (top_k_idx == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        return recall
    
    recall_hv = compute_recall(all_pmhc_emb, all_hv_emb, all_valid_hv, k=10)
    
    return {
        "loss": total_loss / max(num_batches, 1),
        "recall@10_hv": recall_hv,
    }


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train Scaffold Retrieval Model")
    
    # Data
    p.add_argument("--data", type=str, default="flowtcr_fold/data/trn.jsonl",
                   help="Training data (JSONL or CSV)")
    p.add_argument("--val_data", type=str, default=None,
                   help="Validation data (optional)")
    
    # Bank building mode
    p.add_argument("--bank_mode", type=str, default="benchmark",
                   choices=["benchmark", "production"],
                   help="Bank building mode: 'benchmark' uses only training data (strict evaluation), "
                        "'production' uses all available data (maximum coverage)")
    p.add_argument("--bank_extra_data", type=str, nargs="*", default=[],
                   help="Extra data files to include in bank (only for production mode)")
    
    # Model architecture
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--use_esm", action="store_true")
    p.add_argument("--esm_model", type=str, default="esm2_t6_8M_UR50D")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--cls_weight", type=float, default=0.2,
                   help="Weight for classification loss (auxiliary)")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (epochs without improvement)")
    
    # Output & Checkpointing
    p.add_argument("--out_dir", type=str, default="flowtcr_fold/Immuno_PLM/checkpoints")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--ckpt_interval", type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--auto_resume", action="store_true", default=True,
                   help="Automatically resume from latest checkpoint in out_dir")
    p.add_argument("--no_resume", action="store_true",
                   help="Force fresh start, ignore existing checkpoints")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Scaffold Retrieval Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"ESM-2: {args.use_esm} (model: {args.esm_model})")
    print(f"LoRA: {args.use_lora} (rank: {args.lora_rank})")
    print(f"Classification weight: {args.cls_weight}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Bank mode: {args.bank_mode}")
    if args.bank_extra_data:
        print(f"Bank extra data: {args.bank_extra_data}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Dataset
    print("\nLoading dataset...")
    from flowtcr_fold.data.tokenizer import get_tokenizer, vocab_size
    tokenizer = get_tokenizer()
    
    train_ds = ScaffoldRetrievalDataset(
        args.data, tokenizer=tokenizer, build_vocab=True
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Validation dataset (if provided)
    val_loader = None
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        val_ds = ScaffoldRetrievalDataset(
            args.val_data, tokenizer=tokenizer, gene_vocab=train_ds.gene_vocab
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        print(f"Validation samples: {len(val_ds)}")
    
    # Model
    print("\nInitializing model...")
    model = ScaffoldRetriever(
        hidden_dim=args.hidden_dim,
        num_hv=len(train_ds.gene_vocab["h_v"]),
        num_hj=len(train_ds.gene_vocab["h_j"]),
        num_lv=len(train_ds.gene_vocab["l_v"]),
        num_lj=len(train_ds.gene_vocab["l_j"]),
        use_esm=args.use_esm,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        vocab_size=vocab_size(tokenizer),
        esm_model_name=args.esm_model,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    
    from flowtcr_fold.common.utils import EarlyStopper, save_checkpoint
    stopper = EarlyStopper(patience=args.patience)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-resume: find latest checkpoint in out_dir
    start_epoch = 0
    best_loss = float("inf")
    
    def find_latest_checkpoint(out_dir: Path):
        """Find the latest checkpoint (highest epoch or newest model_best.pt)."""
        import re
        
        # Find all epoch checkpoints
        epoch_ckpts = list(out_dir.glob("scaffold_epoch_*.pt"))
        max_epoch = 0
        max_epoch_ckpt = None
        
        for ckpt in epoch_ckpts:
            match = re.search(r"epoch_(\d+)", ckpt.name)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    max_epoch_ckpt = ckpt
        
        # Check model_best.pt
        best_ckpt = out_dir / "model_best.pt"
        
        if max_epoch_ckpt is None and not best_ckpt.exists():
            return None, 0  # No checkpoints found
        
        # Compare modification times: use whichever is newer
        if max_epoch_ckpt and best_ckpt.exists():
            if max_epoch_ckpt.stat().st_mtime > best_ckpt.stat().st_mtime:
                return max_epoch_ckpt, max_epoch
            else:
                # model_best is newer, but we don't know its epoch
                # Use max_epoch_ckpt for resuming (has optimizer state)
                return max_epoch_ckpt, max_epoch
        elif max_epoch_ckpt:
            return max_epoch_ckpt, max_epoch
        else:
            # Only model_best exists (no epoch checkpoints)
            return best_ckpt, 0
    
    if args.auto_resume and not args.no_resume:
        resume_path, resume_epoch = find_latest_checkpoint(out_dir)
        
        if resume_path is not None:
            print(f"\n{'='*60}")
            print("AUTO-RESUME: Found existing checkpoint!")
            print(f"{'='*60}")
            print(f"  Loading: {resume_path}")
            
            model.load_state_dict(torch.load(resume_path, map_location=device))
            
            # Try to load optimizer state
            opt_path = resume_path.with_suffix(".opt")
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                print(f"  Loaded optimizer state: {opt_path.name}")
            
            start_epoch = resume_epoch
            print(f"  Resuming from epoch {start_epoch}")
            print(f"{'='*60}")
        else:
            print("\nNo existing checkpoints found, starting fresh training.")
    elif args.no_resume:
        print("\n--no_resume specified, starting fresh training.")
    
    # Training
    print("\nStarting training...")
    print(f"Checkpoint interval: every {args.ckpt_interval} epochs")
    print("-" * 60)
    
    # Save gene vocab
    with open(out_dir / "gene_vocab.json", "w") as f:
        json.dump(train_ds.gene_vocab, f, indent=2)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        stats = train_epoch(model, train_loader, optimizer, device, args)
        print(f"  Train: loss={stats['loss']:.4f} (NCE={stats['nce']:.3f}, CLS={stats['cls']:.3f})")
        print(f"  Acc: HV={stats['acc_hv']:.3f}, HJ={stats['acc_hj']:.3f}, "
              f"LV={stats['acc_lv']:.3f}, LJ={stats['acc_lj']:.3f}")
        
        # Validate (if val_loader exists)
        current_loss = stats["loss"]
        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device, args)
            print(f"  Val: loss={val_stats['loss']:.4f}, R@10_HV={val_stats['recall@10_hv']:.3f}")
            current_loss = val_stats["loss"]  # Use validation loss for model selection
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), out_dir / "model_best.pt")
            print(f"  New best model! (loss={best_loss:.4f})")
        
        # Checkpoint every N epochs
        if (epoch + 1) % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, str(out_dir), epoch + 1, tag="scaffold")
            print(f"  Checkpoint saved: scaffold_epoch_{epoch + 1}.pt")
        
        # Early stopping
        if stopper.update(current_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    
    # Build and save bank
    print("\n" + "=" * 60)
    print(f"Building scaffold bank (mode: {args.bank_mode})...")
    print("=" * 60)
    
    # Collect extra data paths for production mode
    extra_paths = []
    if args.bank_mode == "production":
        # Add val_data if specified
        if args.val_data:
            extra_paths.append(args.val_data)
        # Add any additional data files
        extra_paths.extend(args.bank_extra_data)
        if extra_paths:
            print(f"Production mode: including extra data from {len(extra_paths)} file(s)")
    else:
        print("Benchmark mode: using only training data (no data leakage)")
    
    bank = build_bank(model, train_ds, device, mode=args.bank_mode, extra_data_paths=extra_paths)
    
    bank_filename = f"scaffold_bank_{args.bank_mode}.json"
    bank.save(str(out_dir / bank_filename))
    print(f"Bank saved to {out_dir / bank_filename}")
    
    # Demo retrieval
    print("\n" + "=" * 60)
    print("Demo: Retrieving scaffolds for first sample")
    print("=" * 60)
    
    sample = train_ds[0]
    pmhc_tokens = sample["pmhc_tokens"].unsqueeze(0).to(device)
    pmhc_mask = sample["pmhc_mask"].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        query = model.encode(pmhc_tokens, pmhc_mask)
    
    print(f"Ground truth:")
    print(f"  HV: {sample['h_v_seq'][:40]}...")
    print(f"  HJ: {sample['h_j_seq']}")
    
    for gene_type in ["h_v", "h_j", "l_v", "l_j"]:
        results = bank.retrieve(query, gene_type, top_k=3)
        print(f"\n{gene_type.upper()} top-3:")
        for i, (seq, score) in enumerate(results):
            print(f"  {i+1}. score={score:.3f}, seq={seq[:40]}...")
    
    print(f"\nTraining complete! Models saved to {out_dir}")


if __name__ == "__main__":
    main()

