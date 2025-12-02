from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def tokenize_sequence(seq: str, tokenizer, max_len: int, add_special: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a single sequence with optional special tokens and return (tokens, mask)."""
    tok = tokenizer
    if hasattr(tok, "cls_token_id"):
        cls_idx = tok.cls_token_id
        sep_idx = tok.eos_token_id
    else:
        cls_idx = tok.stoi["[CLS]"]
        sep_idx = tok.stoi["[SEP]"]

    if not seq:
        tokens = [cls_idx, sep_idx] if add_special else []
        mask = [1] * len(tokens)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    tokens: List[int] = []
    if add_special:
        tokens.append(cls_idx)
    tokens.extend(tok.encode(seq))
    if add_special:
        tokens.append(sep_idx)

    if len(tokens) > max_len:
        tokens = tokens[: max_len - 1] + [sep_idx]

    mask = [1] * len(tokens)
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def tokenize_pmhc(peptide: str, mhc: str, tokenizer, max_len: int, mask_peptide: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize pMHC as [CLS] peptide [SEP] mhc [SEP], optionally masking peptide."""
    tok = tokenizer
    if hasattr(tok, "cls_token_id"):
        cls_idx = tok.cls_token_id
        sep_idx = tok.eos_token_id
    else:
        cls_idx = tok.stoi["[CLS]"]
        sep_idx = tok.stoi["[SEP]"]

    tokens: List[int] = [cls_idx]
    if not mask_peptide and peptide:
        tokens.extend(tok.encode(peptide))
    tokens.append(sep_idx)
    tokens.extend(tok.encode(mhc or ""))
    tokens.append(sep_idx)

    if len(tokens) > max_len:
        tokens = tokens[: max_len - 1] + [sep_idx]

    mask = [1] * len(tokens)
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long)


def build_pos_mask(groups: List[Any]) -> torch.Tensor:
    """Build a [B, B] boolean mask where mask[i, j] = True if groups match."""
    B = len(groups)
    mask = torch.zeros(B, B, dtype=torch.bool)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if gi == gj:
                mask[i, j] = True
    return mask


class ScaffoldRetrievalDataset(Dataset):
    """Dataset for scaffold retrieval (Stage 1)."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_len: int = 512,
        gene_vocab: Optional[Dict[str, Dict[str, int]]] = None,
        allele_vocab: Optional[Dict[str, int]] = None,
        build_vocab: bool = True,
    ):
        self.path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = self._load()

        # Gene vocab
        self.gene_vocab = gene_vocab or {"h_v": {"<UNK>": 0}, "h_j": {"<UNK>": 0}, "l_v": {"<UNK>": 0}, "l_j": {"<UNK>": 0}}
        if build_vocab:
            self._build_gene_vocab()

        # Allele vocab
        self.allele_vocab = allele_vocab or {"<UNK>": 0}
        if build_vocab:
            self._build_allele_vocab()

        # Group-level multi-hot targets
        self.group_targets_mhc = self._build_group_targets(use_peptide=False)
        self.group_targets_pmhc = self._build_group_targets(use_peptide=True)

        # Pos weights for BCE (inverse frequency)
        self.pos_weight = self._compute_pos_weight()

        print(f"[Dataset] Samples: {len(self.samples)} | Gene vocab HV/HJ/LV/LJ = "
              f"{len(self.gene_vocab['h_v'])}/{len(self.gene_vocab['h_j'])}/"
              f"{len(self.gene_vocab['l_v'])}/{len(self.gene_vocab['l_j'])} | Alleles={len(self.allele_vocab)}")

    # --------------------------------------------------------------------- #
    # Loading and vocab
    # --------------------------------------------------------------------- #
    def _load(self) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
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
        # Only require peptide; MHC can be empty (handled separately in training)
        valid = [s for s in samples if s["peptide"]]
        has_mhc = sum(1 for s in valid if s["mhc"])
        print(f"[Dataset] Filtered {len(samples)} -> {len(valid)} (require peptide)")
        print(f"[Dataset] Has MHC: {has_mhc} ({has_mhc/len(valid)*100:.1f}%), Missing MHC: {len(valid)-has_mhc}")
        return valid

    @staticmethod
    def _parse_row(row: Dict[str, str]) -> Dict[str, str]:
        """Parse row with field name normalization (handle both *_seq and *_sequence)."""
        return {
            "peptide": row.get("peptide", "") or "",
            "mhc": row.get("mhc", "") or "",
            "mhc_seq": row.get("mhc_sequence", "") or row.get("mhc_seq", "") or "",
            "h_v": row.get("h_v", "") or "",
            "h_j": row.get("h_j", "") or "",
            "l_v": row.get("l_v", "") or "",
            "l_j": row.get("l_j", "") or "",
            "h_v_seq": row.get("h_v_sequence", "") or row.get("h_v_seq", "") or "",
            "h_j_seq": row.get("h_j_sequence", "") or row.get("h_j_seq", "") or "",
            "l_v_seq": row.get("l_v_sequence", "") or row.get("l_v_seq", "") or "",
            "l_j_seq": row.get("l_j_sequence", "") or row.get("l_j_seq", "") or "",
            "cdr3_b": row.get("cdr3_b", "") or "",
        }

    def _build_gene_vocab(self):
        for sample in self.samples:
            for gtype in ["h_v", "h_j", "l_v", "l_j"]:
                name = sample[gtype]
                if name and name not in self.gene_vocab[gtype]:
                    self.gene_vocab[gtype][name] = len(self.gene_vocab[gtype])

    def _build_allele_vocab(self):
        for sample in self.samples:
            name = sample["mhc"]
            if name and name not in self.allele_vocab:
                self.allele_vocab[name] = len(self.allele_vocab)

    def _build_group_targets(self, use_peptide: bool) -> Dict[Any, Dict[str, List[int]]]:
        """
        Build group → multi-hot list per gene type.
        group key: mhc or (peptide, mhc) if use_peptide=True.
        Only includes samples with MHC (since BCE targets require known allele).
        """
        group_targets: Dict[Any, Dict[str, set]] = defaultdict(lambda: {"h_v": set(), "h_j": set(), "l_v": set(), "l_j": set()})
        for sample in self.samples:
            # Only build targets for samples with MHC
            if not sample["mhc"]:
                continue
            key = (sample["peptide"], sample["mhc"]) if use_peptide else sample["mhc"]
            for gtype in ["h_v", "h_j", "l_v", "l_j"]:
                gid = self.gene_vocab[gtype].get(sample[gtype], 0)
                if gid > 0:
                    group_targets[key][gtype].add(gid)

        # Convert to multi-hot vectors
        multi_hot: Dict[Any, Dict[str, List[int]]] = {}
        for key, entries in group_targets.items():
            multi_hot[key] = {}
            for gtype, ids in entries.items():
                vec = [0] * len(self.gene_vocab[gtype])
                for gid in ids:
                    vec[gid] = 1
                multi_hot[key][gtype] = vec
        return multi_hot

    def _compute_pos_weight(self) -> Dict[str, torch.Tensor]:
        """Compute inverse-frequency pos_weight per gene type with dampening and capping."""
        counts: Dict[str, Counter] = {g: Counter() for g in ["h_v", "h_j", "l_v", "l_j"]}
        for s in self.samples:
            for g in counts:
                gid = self.gene_vocab[g].get(s[g], 0)
                if gid > 0:
                    counts[g][gid] += 1

        pos_weight: Dict[str, torch.Tensor] = {}
        for g, cnt in counts.items():
            total = sum(cnt.values()) + 1e-6
            weights = []
            for i in range(len(self.gene_vocab[g])):
                c = cnt.get(i, 0)
                if c == 0:
                    # Gene never seen → moderate weight
                    w = 10.0
                else:
                    # sqrt dampening + cap to prevent extreme weights
                    w = (total / c) ** 0.5
                    w = min(w, 50.0)  # cap at 50
                weights.append(w)
            pos_weight[g] = torch.tensor(weights, dtype=torch.float)
        return pos_weight

    # --------------------------------------------------------------------- #
    # Dataset interface
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "peptide": s["peptide"],
            "mhc": s["mhc"],
            "allele_id": self.allele_vocab.get(s["mhc"], 0),
            "h_v_seq": s["h_v_seq"],
            "h_j_seq": s["h_j_seq"],
            "l_v_seq": s["l_v_seq"],
            "l_j_seq": s["l_j_seq"],
            "h_v_id": self.gene_vocab["h_v"].get(s["h_v"], 0),
            "h_j_id": self.gene_vocab["h_j"].get(s["h_j"], 0),
            "l_v_id": self.gene_vocab["l_v"].get(s["l_v"], 0),
            "l_j_id": self.gene_vocab["l_j"].get(s["l_j"], 0),
        }


def collate_fn_factory(
    tokenizer,
    max_len: int,
    dataset: ScaffoldRetrievalDataset,
    mask_peptide: bool = False,
):
    """Create a collate_fn capturing tokenizer/dataset for both peptide-on/off."""

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        def pad(batch_tokens: List[torch.Tensor], batch_masks: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            B = len(batch_tokens)
            max_l = max(t.size(0) for t in batch_tokens)
            tokens = torch.zeros(B, max_l, dtype=torch.long)
            masks = torch.zeros(B, max_l, dtype=torch.long)
            for i, (t, m) in enumerate(zip(batch_tokens, batch_masks)):
                tokens[i, : t.size(0)] = t
                masks[i, : m.size(0)] = m
            return tokens, masks

        # Tokenize pMHC
        pmhc_tokens_list, pmhc_masks_list = [], []
        hv_tokens_list, hv_masks_list = [], []
        hj_tokens_list, hj_masks_list = [], []
        lv_tokens_list, lv_masks_list = [], []
        lj_tokens_list, lj_masks_list = [], []
        allele_ids, mhc_groups, pmhc_groups, peptide_groups = [], [], [], []
        has_mhc_list = []

        multihot_hv, multihot_hj, multihot_lv, multihot_lj = [], [], [], []
        valid_hv_seq, valid_hj_seq, valid_lv_seq, valid_lj_seq = [], [], [], []

        for idx, sample in enumerate(batch):
            pmhc_t, pmhc_m = tokenize_pmhc(sample["peptide"], sample["mhc"], tokenizer, max_len, mask_peptide=mask_peptide)
            pmhc_tokens_list.append(pmhc_t)
            pmhc_masks_list.append(pmhc_m)

            hv_t, hv_m = tokenize_sequence(sample["h_v_seq"], tokenizer, max_len)
            hj_t, hj_m = tokenize_sequence(sample["h_j_seq"], tokenizer, max_len)
            lv_t, lv_m = tokenize_sequence(sample["l_v_seq"], tokenizer, max_len)
            lj_t, lj_m = tokenize_sequence(sample["l_j_seq"], tokenizer, max_len)
            hv_tokens_list.append(hv_t)
            hv_masks_list.append(hv_m)
            hj_tokens_list.append(hj_t)
            hj_masks_list.append(hj_m)
            lv_tokens_list.append(lv_t)
            lv_masks_list.append(lv_m)
            lj_tokens_list.append(lj_t)
            lj_masks_list.append(lj_m)

            allele_ids.append(sample["allele_id"])
            
            # Track whether sample has MHC
            sample_has_mhc = bool(sample["mhc"])
            has_mhc_list.append(sample_has_mhc)
            
            # MHC groups: only meaningful for samples with MHC
            # For samples without MHC, assign unique key so they don't group together
            if sample_has_mhc:
                mhc_groups.append(sample["mhc"])
                pmhc_groups.append((sample["peptide"], sample["mhc"]))
            else:
                # Each sample without MHC gets unique group (self-contrast only)
                mhc_groups.append(f"__NO_MHC_{idx}__")
                pmhc_groups.append((sample["peptide"], f"__NO_MHC_{idx}__"))
            
            # Peptide groups: for peptide-only InfoNCE (all samples)
            peptide_groups.append(sample["peptide"])

            # Multi-hot targets (MHC group) - empty for samples without MHC
            if sample_has_mhc:
                tgt_mhc = dataset.group_targets_mhc.get(sample["mhc"], {"h_v": [], "h_j": [], "l_v": [], "l_j": []})
            else:
                tgt_mhc = {"h_v": [], "h_j": [], "l_v": [], "l_j": []}
            multihot_hv.append(torch.tensor(tgt_mhc["h_v"], dtype=torch.float) if tgt_mhc["h_v"] else torch.zeros(len(dataset.gene_vocab["h_v"])))
            multihot_hj.append(torch.tensor(tgt_mhc["h_j"], dtype=torch.float) if tgt_mhc["h_j"] else torch.zeros(len(dataset.gene_vocab["h_j"])))
            multihot_lv.append(torch.tensor(tgt_mhc["l_v"], dtype=torch.float) if tgt_mhc["l_v"] else torch.zeros(len(dataset.gene_vocab["l_v"])))
            multihot_lj.append(torch.tensor(tgt_mhc["l_j"], dtype=torch.float) if tgt_mhc["l_j"] else torch.zeros(len(dataset.gene_vocab["l_j"])))

            valid_hv_seq.append(bool(sample["h_v_seq"]))
            valid_hj_seq.append(bool(sample["h_j_seq"]))
            valid_lv_seq.append(bool(sample["l_v_seq"]))
            valid_lj_seq.append(bool(sample["l_j_seq"]))

        batch_dict: Dict[str, Any] = {}
        batch_dict["pmhc_tokens"], batch_dict["pmhc_mask"] = pad(pmhc_tokens_list, pmhc_masks_list)
        batch_dict["hv_tokens"], batch_dict["hv_mask"] = pad(hv_tokens_list, hv_masks_list)
        batch_dict["hj_tokens"], batch_dict["hj_mask"] = pad(hj_tokens_list, hj_masks_list)
        batch_dict["lv_tokens"], batch_dict["lv_mask"] = pad(lv_tokens_list, lv_masks_list)
        batch_dict["lj_tokens"], batch_dict["lj_mask"] = pad(lj_tokens_list, lj_masks_list)

        batch_dict["allele_id"] = torch.tensor(allele_ids, dtype=torch.long)
        
        # has_mhc mask: True for samples with MHC info
        batch_dict["has_mhc"] = torch.tensor(has_mhc_list, dtype=torch.bool)

        batch_dict["pos_mask_mhc"] = build_pos_mask(mhc_groups)
        batch_dict["pos_mask_pmhc"] = build_pos_mask(pmhc_groups)
        # Peptide-only grouping: for peptide-only InfoNCE branch
        batch_dict["pos_mask_pep"] = build_pos_mask(peptide_groups)

        # Multi-hot targets (MHC-level)
        def pad_multihot(tensors: List[torch.Tensor], dim: int) -> torch.Tensor:
            out = torch.zeros(len(tensors), dim, dtype=torch.float)
            for i, t in enumerate(tensors):
                if t.numel() > 0:
                    out[i, : t.numel()] = t
            return out

        batch_dict["target_hv_mhc"] = pad_multihot(multihot_hv, len(dataset.gene_vocab["h_v"]))
        batch_dict["target_hj_mhc"] = pad_multihot(multihot_hj, len(dataset.gene_vocab["h_j"]))
        batch_dict["target_lv_mhc"] = pad_multihot(multihot_lv, len(dataset.gene_vocab["l_v"]))
        batch_dict["target_lj_mhc"] = pad_multihot(multihot_lj, len(dataset.gene_vocab["l_j"]))

        batch_dict["valid_hv_seq"] = torch.tensor(valid_hv_seq, dtype=torch.bool)
        batch_dict["valid_hj_seq"] = torch.tensor(valid_hj_seq, dtype=torch.bool)
        batch_dict["valid_lv_seq"] = torch.tensor(valid_lv_seq, dtype=torch.bool)
        batch_dict["valid_lj_seq"] = torch.tensor(valid_lj_seq, dtype=torch.bool)

        # MHC list for KL divergence evaluation
        batch_dict["mhc_list"] = mhc_groups
        batch_dict["peptide_list"] = peptide_groups

        return batch_dict

    return collate
