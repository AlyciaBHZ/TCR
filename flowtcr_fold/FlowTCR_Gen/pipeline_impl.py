"""
End-to-end pipeline scaffold: Gen -> Critique -> Refine.

Stages:
- Stage 1: FlowTCR-Gen samples sequences
- Stage 2: TCRFold-Light filters/scores
- Stage 3: Physics refinement (EvoEF2)
"""

import os
from typing import List, Optional

import torch

from flowtcr_fold.FlowTCR_Gen.flow_gen import FlowMatchingModel
from flowtcr_fold.TCRFold_Light.tcrfold_light import TCRFoldLight
from flowtcr_fold.physics import TCRStructureOptimizer
from flowtcr_fold.data.tokenizer import (
    BasicTokenizer,
    SPECIAL_TOKENS,
    get_tokenizer,
    vocab_size,
)


def _tokens_to_sequences(tokenizer: BasicTokenizer, tokens: torch.Tensor) -> List[str]:
    """
    Convert token ids to amino-acid strings, dropping special tokens.
    """
    specials = set(SPECIAL_TOKENS)
    vocab = tokenizer.itos if hasattr(tokenizer, "itos") else []
    seqs: List[str] = []
    for row in tokens.cpu().tolist():
        chars: List[str] = []
        for idx in row:
            if 0 <= idx < len(vocab):
                tok = vocab[int(idx)]
                if tok in specials or len(tok) != 1:
                    continue
                chars.append(tok)
        seqs.append("".join(chars))
    return seqs


def generate(flow_model: FlowMatchingModel, num: int = 8, seq_len: int = 20, steps: int = 10, device=None):
    """
    Simple Euler sampler on the flow field from uniform x0 to x1.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_model.to(device).eval()
    with torch.no_grad():
        x = flow_model.sample_x0((num, seq_len))
        for i in range(steps):
            t = torch.full((num, 1), (i + 1) / steps, device=device)
            v = flow_model.forward(x, t, conditioning=None)
            x = x + (1.0 / steps) * v
        tokens = x.argmax(dim=-1)
    return tokens


def critique(critic: TCRFoldLight, tokens: torch.Tensor):
    """
    Lightweight critic using TCRFold-Light contact/energy surrogate.
    """
    critic.to(tokens.device).eval()
    emb_table = torch.nn.Embedding(256, critic.s_dim, device=tokens.device)
    s = emb_table(tokens)
    z = torch.zeros(tokens.size(0), tokens.size(1), tokens.size(1), critic.z_dim, device=tokens.device)
    with torch.no_grad():
        out = critic(s, z)
    scores = out["contact"].mean(dim=(1, 2)) - out["energy"]
    return scores


def refine(tokens: torch.Tensor, scaffold_pdb: Optional[str], tokenizer: BasicTokenizer, output_dir: str = "refined_structures"):
    """
    Call EvoEF2 refinement when scaffold is available.
    Returns ranked [(pdb_path, binding_energy)] or empty list if skipped.
    """
    if not scaffold_pdb or not os.path.exists(scaffold_pdb):
        return []
    optimizer = TCRStructureOptimizer()
    sequences = _tokens_to_sequences(tokenizer, tokens)
    return optimizer.refine_generated_sequences(scaffold_pdb=scaffold_pdb, sequences=sequences, output_dir=output_dir)


def run_pipeline(num: int = 16, top_k: int = 8, scaffold_pdb: Optional[str] = None, output_dir: str = "refined_structures"):
    """
    Generate -> critique -> (optional) EvoEF2 refine.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    flow = FlowMatchingModel(vocab_size=vocab_size(tokenizer)).to(device)
    critic = TCRFoldLight().to(device)
    candidates = generate(flow, num=num, device=device)
    scores = critique(critic, candidates)
    topk = torch.topk(scores, k=min(top_k, scores.numel()))
    selected = candidates[topk.indices]
    refined = refine(selected, scaffold_pdb=scaffold_pdb, tokenizer=tokenizer if isinstance(tokenizer, BasicTokenizer) else BasicTokenizer(), output_dir=output_dir)
    if refined:
        return refined, scores[topk.indices]
    return selected, scores[topk.indices]


if __name__ == "__main__":
    cands, scores = run_pipeline()
    print("Top candidates:", cands if isinstance(cands, list) else cands.shape[0])
