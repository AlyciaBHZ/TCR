"""
End-to-end pipeline scaffold: Gen -> Critique -> Refine.

Stages:
- Stage 1: FlowTCR-Gen samples sequences
- Stage 2: TCRFold-Light filters/scores
- Stage 3: Physics refinement (EvoEF2) [TODO]
"""

import torch

from flowtcr_fold.models.flow_gen import FlowMatchingModel, one_hot
from flowtcr_fold.models.tcrfold_light import TCRFoldLight
# from flowtcr_fold.physics.evoef_runner import compute_binding_energy  # when available


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
    critic.to(tokens.device).eval()
    emb_table = torch.nn.Embedding(256, critic.s_dim, device=tokens.device)
    s = emb_table(tokens)
    z = torch.zeros(tokens.size(0), tokens.size(1), tokens.size(1), critic.z_dim, device=tokens.device)
    with torch.no_grad():
        out = critic(s, z)
    scores = out["contact"].mean(dim=(1, 2)) - out["energy"]
    return scores


def refine(tokens: torch.Tensor):
    # TODO: call EvoEF2 for MC repacking; return refined scores
    return tokens


def run_pipeline(num: int = 16, top_k: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow = FlowMatchingModel().to(device)
    critic = TCRFoldLight().to(device)
    candidates = generate(flow, num=num, device=device)
    scores = critique(critic, candidates)
    topk = torch.topk(scores, k=min(top_k, scores.numel()))
    selected = candidates[topk.indices]
    refined = refine(selected)
    return refined, scores[topk.indices]


if __name__ == "__main__":
    cands, scores = run_pipeline()
    print("Top candidates:", cands.shape[0])
