"""
Triplet-aware dataset with hard-negative hooks for Immuno-PLM.

Reads CSV (paired pMHC + TCR) or JSONL records and yields Anchor/Positive/Negative
examples. Negatives are sampled on the fly using simple decoy strategies; replace
with similarity-based decoys/mutants when available.
"""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import get_tokenizer, BasicTokenizer

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

def _looks_like_seq(val: str, min_len: int = 8) -> bool:
    return len(val) >= min_len and all(c in AA_SET for c in val)

@dataclass
class Sample:
    peptide: str
    mhc: str
    cdr3b: str
    h_v: Optional[str] = None  # beta V (or heavy) raw field (may be name or seq)
    h_j: Optional[str] = None  # beta J (or heavy)
    l_v: Optional[str] = None  # alpha V (or light)
    l_j: Optional[str] = None  # alpha J (or light)
    h_v_name: Optional[str] = None
    h_j_name: Optional[str] = None
    l_v_name: Optional[str] = None
    l_j_name: Optional[str] = None
    h_v_seq: Optional[str] = None
    h_j_seq: Optional[str] = None
    l_v_seq: Optional[str] = None
    l_j_seq: Optional[str] = None


class FlowDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        negative_fraction: float = 0.5,
        tokenizer=None,
    ):
        self.path = Path(data_path)
        self.split = split
        self.negative_fraction = negative_fraction
        self.tokenizer = tokenizer or get_tokenizer()
        # identity threshold (fraction) for decoy selection
        self.identity_threshold = 0.75
        # probability of mutating TCR for type-B decoys
        self.mutate_neg_prob = 0.6
        # allowed amino acids for controlled mutation (avoid drastic changes)
        self.allowed_aas = list("ACDEFGHIKLMNPQRSTVWY")
        self.samples: List[Sample] = self._load(self.path)
        # Index by MHC for decoys and by peptide for similarity
        self.by_mhc: Dict[str, List[int]] = {}
        for i, s in enumerate(self.samples):
            self.by_mhc.setdefault(s.mhc, []).append(i)
        if not self.samples:
            raise ValueError(f"No samples loaded from {data_path}")

    def _load(self, path: Path) -> List[Sample]:
        samples: List[Sample] = []
        if path.suffix == ".csv":
            with path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(
                        Sample(
                            peptide=row.get("peptide", ""),
                            mhc=row.get("mhc", ""),
                            cdr3b=row.get("cdr3_b", "") or row.get("cdr3", ""),
                            h_v=row.get("h_v"),
                            h_j=row.get("h_j"),
                            l_v=row.get("l_v"),
                            l_j=row.get("l_j"),
                            h_v_name=row.get("h_v_name"),
                            h_j_name=row.get("h_j_name"),
                            l_v_name=row.get("l_v_name"),
                            l_j_name=row.get("l_j_name"),
                            h_v_seq=row.get("h_v_seq"),
                            h_j_seq=row.get("h_j_seq"),
                            l_v_seq=row.get("l_v_seq"),
                            l_j_seq=row.get("l_j_seq"),
                        )
                    )
        else:
            with path.open() as f:
                for line in f:
                    obj = json.loads(line)
                    samples.append(
                        Sample(
                            peptide=obj.get("peptide", ""),
                            mhc=obj.get("mhc", ""),
                            cdr3b=obj.get("cdr3b", obj.get("cdr3", "")),
                            h_v=obj.get("h_v"),
                            h_j=obj.get("h_j"),
                            l_v=obj.get("l_v"),
                            l_j=obj.get("l_j"),
                            h_v_name=obj.get("h_v_name"),
                            h_j_name=obj.get("h_j_name"),
                            l_v_name=obj.get("l_v_name"),
                            l_j_name=obj.get("l_j_name"),
                            h_v_seq=obj.get("h_v_seq"),
                            h_j_seq=obj.get("h_j_seq"),
                            l_v_seq=obj.get("l_v_seq"),
                            l_j_seq=obj.get("l_j_seq"),
                        )
                    )
        return [s for s in samples if s.peptide and s.mhc and s.cdr3b]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resolve_name(raw: Optional[str], seq: Optional[str], mapped_name: Optional[str]) -> str:
        """
        Prefer mapped_name if provided; else if raw looks like a name (short or contains '*'), use raw; else empty.
        """
        if mapped_name and str(mapped_name).strip():
            return str(mapped_name).strip()
        if raw and (len(raw) <= 20 or "*" in raw):
            return str(raw).strip()
        return ""

    def _build_tokens(self, sample: Sample) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, slice]]:
        """
        Build token ids and region slices. Uses self.tokenizer directly.
        """
        tok = self.tokenizer
        # Handle different tokenizer interfaces (ESM vs Basic)
        if hasattr(tok, "cls_token_id"):
             # ESM tokenizer
            cls_idx = tok.cls_token_id
            sep_idx = tok.eos_token_id # ESM uses eos as sep/end
            pad_idx = tok.pad_token_id
        else:
            # BasicTokenizer
            cls_idx = tok.stoi["[CLS]"]
            sep_idx = tok.stoi["[SEP]"]
            pad_idx = tok.stoi["[PAD]"]

        tokens: List[int] = [cls_idx]

        start_pep = len(tokens)
        tokens.extend(tok.encode(sample.peptide))
        end_pep = len(tokens)
        tokens.append(sep_idx)

        start_mhc = len(tokens)
        tokens.extend(tok.encode(sample.mhc))
        end_mhc = len(tokens)
        tokens.append(sep_idx)

        start_cdr3 = len(tokens)
        tokens.extend(tok.encode(sample.cdr3b))
        end_cdr3 = len(tokens)
        tokens.append(sep_idx)

        mask = [1] * len(tokens)
        slices = {"pep": slice(start_pep, end_pep), "mhc": slice(start_mhc, end_mhc), "cdr3b": slice(start_cdr3, end_cdr3)}
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long), slices

    def _sample_negative(self, anchor_idx: int) -> Tuple[Sample, str]:
        """
        Decoy strategies (rule-based):
        - Type A: same TCR, similar but different peptide on same MHC (identity in [0.6, 0.9])
        - Type B: same pMHC, different CDR3 (cdr3b), optionally controlled mutation (no first/last position)
        """
        anchor = self.samples[anchor_idx]
        choice = random.random()
        # Type A: swap peptide within same MHC
        if choice < 0.5:
            candidates = [
                self.samples[i]
                for i in self.by_mhc.get(anchor.mhc, [])
                if self.samples[i].peptide != anchor.peptide
                and 0.6 <= self._seq_identity(self.samples[i].peptide, anchor.peptide) <= 0.9
            ]
            if candidates:
                pick = random.choice(candidates)
                return Sample(peptide=pick.peptide, mhc=anchor.mhc, cdr3b=anchor.cdr3b, v_gene=anchor.v_gene, j_gene=anchor.j_gene), "decoy_peptide"
        # Type B: swap CDR3 within same MHC
        candidates = [self.samples[i] for i in self.by_mhc.get(anchor.mhc, []) if self.samples[i].cdr3b != anchor.cdr3b]
        if candidates:
            pick = random.choice(candidates)
            cdr3 = pick.cdr3b
            if random.random() < self.mutate_neg_prob:
                cdr3 = self._mutate_seq(cdr3, num_mutations=2, avoid_terminals=True)
                neg_type = "mutant_cdr3"
            else:
                neg_type = "swap_cdr3"
            return Sample(peptide=anchor.peptide, mhc=anchor.mhc, cdr3b=cdr3, v_gene=anchor.v_gene, j_gene=anchor.j_gene), neg_type
        # fallback: random shuffle
        neg_idx = anchor_idx
        while neg_idx == anchor_idx:
            neg_idx = random.randrange(len(self.samples))
        pick = self.samples[neg_idx]
        return Sample(peptide=pick.peptide, mhc=pick.mhc, cdr3b=anchor.cdr3b, v_gene=anchor.v_gene, j_gene=anchor.j_gene), "random"

    @staticmethod
    def _seq_identity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        L = min(len(a), len(b))
        if L == 0:
            return 0.0
        matches = sum(1 for x, y in zip(a[:L], b[:L]) if x == y)
        return matches / L

    @staticmethod
    def _mutate_seq(seq: str, num_mutations: int = 2, avoid_terminals: bool = False) -> str:
        if not seq:
            return seq
        aas = list("ACDEFGHIKLMNPQRSTVWY")
        seq_list = list(seq)
        for _ in range(num_mutations):
            if avoid_terminals and len(seq_list) > 2:
                idx = random.randrange(1, len(seq_list) - 1)
            else:
                idx = random.randrange(len(seq_list))
            choices = [a for a in aas if a != seq_list[idx]]
            seq_list[idx] = random.choice(choices)
        return "".join(seq_list)

    def __getitem__(self, idx: int):
        pos = self.samples[idx]
        tokens_pos, mask_pos, slices_pos = self._build_tokens(pos)

        tokens_neg = None
        mask_neg = None
        slices_neg = None
        neg_type = None
        if random.random() < self.negative_fraction:
            neg_sample, neg_type = self._sample_negative(idx)
            tokens_neg, mask_neg, slices_neg = self._build_tokens(neg_sample)

        return {
            "tokens_pos": tokens_pos,
            "mask_pos": mask_pos,
            "slices_pos": slices_pos,
            "tokens_neg": tokens_neg,
            "mask_neg": mask_neg,
            "slices_neg": slices_neg,
            "meta": {
                "peptide": pos.peptide,
                "mhc": pos.mhc,
                "cdr3b": pos.cdr3b,
                "h_v": self._resolve_name(pos.h_v, pos.h_v_seq, pos.h_v_name),
                "h_j": self._resolve_name(pos.h_j, pos.h_j_seq, pos.h_j_name),
                "l_v": self._resolve_name(pos.l_v, pos.l_v_seq, pos.l_v_name),
                "l_j": self._resolve_name(pos.l_j, pos.l_j_seq, pos.l_j_name),
                "h_v_seq": pos.h_v_seq if pos.h_v_seq else (pos.h_v if pos.h_v and _looks_like_seq(pos.h_v) else ""),
                "h_j_seq": pos.h_j_seq if pos.h_j_seq else (pos.h_j if pos.h_j and _looks_like_seq(pos.h_j) else ""),
                "l_v_seq": pos.l_v_seq if pos.l_v_seq else (pos.l_v if pos.l_v and _looks_like_seq(pos.l_v) else ""),
                "l_j_seq": pos.l_j_seq if pos.l_j_seq else (pos.l_j if pos.l_j and _looks_like_seq(pos.l_j) else ""),
                "neg_type": neg_type,
            },
        }
