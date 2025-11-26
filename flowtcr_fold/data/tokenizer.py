"""
Lightweight tokenizer utilities for FlowTCR.

Priority is compatibility with legacy data and the planned ESM-2 backbone. If
facebook/esm is available, we wrap its tokenizer. Otherwise we fall back to a
simple amino-acid vocabulary with special tokens.
"""

from typing import Dict, List, Optional

import torch


AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")  # standard 20 aa
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[TRA]", "[TRB]", "[PEP]", "[MHC]"]


class BasicTokenizer:
    def __init__(self, add_special_tokens: Optional[List[str]] = None):
        specials = SPECIAL_TOKENS + (add_special_tokens or [])
        self.itos = specials + AA_VOCAB
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, seq: str, prefix: Optional[str] = None, suffix: Optional[str] = None) -> List[int]:
        tokens = []
        if prefix:
            tokens.append(self.stoi.get(prefix, self.stoi["[UNK]"]))
        for aa in seq:
            tokens.append(self.stoi.get(aa, self.stoi["[UNK]"]))
        if suffix:
            tokens.append(self.stoi.get(suffix, self.stoi["[UNK]"]))
        return tokens

    def pad_batch(self, batch: List[List[int]], pad_token: str = "[PAD]"):
        pad_id = self.stoi[pad_token]
        max_len = max(len(x) for x in batch)
        padded = []
        attn = []
        for x in batch:
            pad_len = max_len - len(x)
            padded.append(x + [pad_id] * pad_len)
            attn.append([1] * len(x) + [0] * pad_len)
        if torch is None:
            return padded, attn
        return torch.tensor(padded, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


def get_tokenizer():
    """
    Returns an ESM tokenizer if available; otherwise a basic tokenizer.
    """
    try:
        import esm  # type: ignore

        return esm.Alphabet.from_architecture("esm2_t33_650M_UR50D")
    except Exception:
        return BasicTokenizer()


def vocab_size(tokenizer) -> int:
    if isinstance(tokenizer, BasicTokenizer):
        return len(tokenizer.itos)
    # esm alphabet has property cls_idx etc; use all tokens
    if hasattr(tokenizer, "all_toks"):
        return len(tokenizer.all_toks)
    if hasattr(tokenizer, "padding_idx") and hasattr(tokenizer, "size"):
        return tokenizer.size
    raise ValueError("Unsupported tokenizer type for vocab size")
