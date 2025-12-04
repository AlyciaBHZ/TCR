"""
Data processing for FlowTCR-Gen (Stage 2).

Handles:
- Loading paired TCR-pMHC data
- Tokenization for CDR3β, peptide, MHC, scaffolds
- Creating training batches with proper conditioning
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Standard amino acid vocabulary
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']


def build_vocab() -> Dict[str, int]:
    """Build token vocabulary."""
    vocab = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = i
    for i, aa in enumerate(AMINO_ACIDS):
        vocab[aa] = len(SPECIAL_TOKENS) + i
    return vocab


def get_vocab() -> Tuple[Dict[str, int], List[str]]:
    """Get vocabulary mappings."""
    stoi = build_vocab()
    itos = {v: k for k, v in stoi.items()}
    itos_list = [itos[i] for i in range(len(itos))]
    return stoi, itos_list


class FlowTCRGenTokenizer:
    """Simple tokenizer for FlowTCR-Gen."""

    def __init__(self):
        self.stoi, self.itos = get_vocab()
        self.pad_idx = self.stoi['<PAD>']
        self.unk_idx = self.stoi['<UNK>']
        self.cls_idx = self.stoi['<CLS>']
        self.sep_idx = self.stoi['<SEP>']
        self.mask_idx = self.stoi['<MASK>']
        self.vocab_size = len(self.stoi)

    def encode(self, seq: str) -> List[int]:
        """Encode sequence to token indices."""
        return [self.stoi.get(aa, self.unk_idx) for aa in seq.upper()]

    def decode(self, tokens: List[int]) -> str:
        """Decode token indices to sequence."""
        return ''.join(self.itos[t] for t in tokens if t < len(self.itos) and t not in [self.pad_idx, self.cls_idx, self.sep_idx, self.mask_idx])

    def to_one_hot(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert token indices to one-hot."""
        return F.one_hot(tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()


class FlowTCRGenDataset(Dataset):
    """
    Dataset for FlowTCR-Gen training.
    
    Each sample contains:
    - CDR3β: Target for generation
    - Peptide: Antigenic peptide
    - MHC: MHC allele sequence
    - Scaffold: HV/HJ/LV/LJ gene sequences
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[FlowTCRGenTokenizer] = None,
        max_cdr3_len: int = 30,
        max_pep_len: int = 15,
        max_mhc_len: int = 300,
        max_scaffold_len: int = 150,
    ):
        self.path = Path(data_path)
        self.tokenizer = tokenizer or FlowTCRGenTokenizer()
        self.max_cdr3_len = max_cdr3_len
        self.max_pep_len = max_pep_len
        self.max_mhc_len = max_mhc_len
        self.max_scaffold_len = max_scaffold_len
        
        self.samples = self._load()
        print(f"[FlowTCRGenDataset] Loaded {len(self.samples)} samples from {data_path}")

    def _load(self) -> List[Dict[str, str]]:
        """Load data from CSV or JSONL."""
        samples = []
        
        if self.path.suffix == '.csv':
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
        
        # Filter: require CDR3β
        valid = [s for s in samples if s['cdr3_b']]
        print(f"[FlowTCRGenDataset] Filtered {len(samples)} -> {len(valid)} (require cdr3_b)")
        
        return valid

    @staticmethod
    def _parse_row(row: Dict[str, Any]) -> Dict[str, str]:
        """Parse row with field normalization."""
        return {
            'peptide': row.get('peptide', '') or '',
            'mhc': row.get('mhc', '') or '',
            'mhc_seq': row.get('mhc_sequence', '') or row.get('mhc_seq', '') or '',
            'cdr3_b': row.get('cdr3_b', '') or '',
            'h_v': row.get('h_v', '') or '',
            'h_j': row.get('h_j', '') or '',
            'l_v': row.get('l_v', '') or '',
            'l_j': row.get('l_j', '') or '',
            'h_v_seq': row.get('h_v_sequence', '') or row.get('h_v_seq', '') or '',
            'h_j_seq': row.get('h_j_sequence', '') or row.get('h_j_seq', '') or '',
            'l_v_seq': row.get('l_v_sequence', '') or row.get('l_v_seq', '') or '',
            'l_j_seq': row.get('l_j_sequence', '') or row.get('l_j_seq', '') or '',
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        s = self.samples[idx]
        
        # Tokenize CDR3β
        cdr3_tokens = self.tokenizer.encode(s['cdr3_b'])[:self.max_cdr3_len]
        
        # Tokenize peptide
        pep_tokens = self.tokenizer.encode(s['peptide'])[:self.max_pep_len] if s['peptide'] else []
        
        # Tokenize MHC sequence (use sequence if available, else allele name)
        mhc_seq = s['mhc_seq'] or s['mhc']
        mhc_tokens = self.tokenizer.encode(mhc_seq)[:self.max_mhc_len] if mhc_seq else []
        
        # Tokenize scaffold sequences
        hv_tokens = self.tokenizer.encode(s['h_v_seq'])[:self.max_scaffold_len] if s['h_v_seq'] else []
        hj_tokens = self.tokenizer.encode(s['h_j_seq'])[:self.max_scaffold_len] if s['h_j_seq'] else []
        lv_tokens = self.tokenizer.encode(s['l_v_seq'])[:self.max_scaffold_len] if s['l_v_seq'] else []
        lj_tokens = self.tokenizer.encode(s['l_j_seq'])[:self.max_scaffold_len] if s['l_j_seq'] else []
        
        return {
            'cdr3_tokens': torch.tensor(cdr3_tokens, dtype=torch.long),
            'pep_tokens': torch.tensor(pep_tokens, dtype=torch.long) if pep_tokens else torch.zeros(0, dtype=torch.long),
            'mhc_tokens': torch.tensor(mhc_tokens, dtype=torch.long) if mhc_tokens else torch.zeros(0, dtype=torch.long),
            'hv_tokens': torch.tensor(hv_tokens, dtype=torch.long) if hv_tokens else torch.zeros(0, dtype=torch.long),
            'hj_tokens': torch.tensor(hj_tokens, dtype=torch.long) if hj_tokens else torch.zeros(0, dtype=torch.long),
            'lv_tokens': torch.tensor(lv_tokens, dtype=torch.long) if lv_tokens else torch.zeros(0, dtype=torch.long),
            'lj_tokens': torch.tensor(lj_tokens, dtype=torch.long) if lj_tokens else torch.zeros(0, dtype=torch.long),
            'cdr3_seq': s['cdr3_b'],
            'peptide': s['peptide'],
            'mhc': s['mhc'],
        }


def collate_fn_flow(
    batch: List[Dict[str, Any]],
    tokenizer: FlowTCRGenTokenizer,
) -> Dict[str, Any]:
    """
    Collate function for FlowTCR-Gen with per-sample conditioning.
    
    Returns batch dict with:
    - cdr3_tokens, cdr3_mask: [B, L_cdr3_max]
    - pep_tokens, pep_mask: [B, L_pep_max]
    - mhc_tokens, mhc_mask: [B, L_mhc_max]
    - scaffold_tokens, scaffold_mask: Dict[str, [B, L_scaffold_max]]
    
    All conditioning is per-sample, properly padded and masked.
    """
    B = len(batch)
    vocab_size = tokenizer.vocab_size
    
    def pad_sequences(tokens_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad list of variable-length tensors to [B, L_max] with mask."""
        lens = [t.shape[0] if t.shape[0] > 0 else 1 for t in tokens_list]  # min len 1 for empty
        max_len = max(lens)
        
        padded = torch.zeros(B, max_len, dtype=torch.long)
        mask = torch.zeros(B, max_len, dtype=torch.float)
        
        for i, t in enumerate(tokens_list):
            L = t.shape[0]
            if L > 0:
                padded[i, :L] = t
                mask[i, :L] = 1.0
        
        return padded, mask
    
    # CDR3: [B, L_cdr3_max]
    cdr3_tokens, cdr3_mask = pad_sequences([b['cdr3_tokens'] for b in batch])
    
    # Peptide: [B, L_pep_max]
    pep_tokens, pep_mask = pad_sequences([b['pep_tokens'] for b in batch])
    
    # MHC: [B, L_mhc_max]
    mhc_tokens, mhc_mask = pad_sequences([b['mhc_tokens'] for b in batch])
    
    # Scaffolds: each is [B, L_scaffold_max]
    scaffold_tokens = {}
    scaffold_mask = {}
    for key in ['hv', 'hj', 'lv', 'lj']:
        tokens_key = f'{key}_tokens'
        scaffold_tokens[key], scaffold_mask[key] = pad_sequences([b[tokens_key] for b in batch])
    
    # Meta info
    cdr3_seqs = [b['cdr3_seq'] for b in batch]
    peptides = [b['peptide'] for b in batch]
    mhcs = [b['mhc'] for b in batch]
    
    return {
        # CDR3 target
        'cdr3_tokens': cdr3_tokens,      # [B, L_cdr3]
        'cdr3_mask': cdr3_mask,          # [B, L_cdr3]
        
        # Per-sample conditioning (tokens, will be converted to one-hot in model)
        'pep_tokens': pep_tokens,        # [B, L_pep]
        'pep_mask': pep_mask,            # [B, L_pep]
        'mhc_tokens': mhc_tokens,        # [B, L_mhc]
        'mhc_mask': mhc_mask,            # [B, L_mhc]
        'scaffold_tokens': scaffold_tokens,  # Dict[str, [B, L]]
        'scaffold_mask': scaffold_mask,      # Dict[str, [B, L]]
        
        # Meta
        'cdr3_seqs': cdr3_seqs,
        'peptides': peptides,
        'mhcs': mhcs,
    }


def create_collate_fn(tokenizer: FlowTCRGenTokenizer):
    """Create collate function with tokenizer."""
    def collate(batch):
        return collate_fn_flow(batch, tokenizer)
    return collate


def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader], FlowTCRGenTokenizer]:
    """
    Create train and validation dataloaders.
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    tokenizer = FlowTCRGenTokenizer()
    
    train_ds = FlowTCRGenDataset(train_path, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True,
    )
    
    val_loader = None
    if val_path:
        val_ds = FlowTCRGenDataset(val_path, tokenizer=tokenizer)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=create_collate_fn(tokenizer),
            pin_memory=True,
        )
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Quick test
    tokenizer = FlowTCRGenTokenizer()
    print(f"✅ Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Test encoding/decoding
    seq = "CASSLGQFF"
    tokens = tokenizer.encode(seq)
    decoded = tokenizer.decode(tokens)
    print(f"✅ Encode/decode: {seq} -> {tokens} -> {decoded}")
    
    # Test one-hot
    tokens_t = torch.tensor(tokens)
    one_hot = tokenizer.to_one_hot(tokens_t)
    print(f"✅ One-hot shape: {one_hot.shape}")
    
    # Test dataset (if data exists)
    data_path = Path("flowtcr_fold/data/trn.jsonl")
    if data_path.exists():
        dataset = FlowTCRGenDataset(str(data_path))
        print(f"✅ Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"✅ Sample keys: {sample.keys()}")
        print(f"   CDR3: {sample['cdr3_seq']}, tokens shape: {sample['cdr3_tokens'].shape}")

