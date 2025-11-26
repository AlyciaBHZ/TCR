"""
Discrete flow matching generator (categorical simplex approximation).

- Uses convex interpolant x_t = (1 - t) * x0 + t * onehot(y)
- Predicts vector field v_theta(x_t, t, cond) ~ y_onehot - x0
- Conditioning placeholder accepts optional embeddings (e.g., pMHC, geometry)
"""

from typing import Dict, Optional, Tuple

import torch
from torch import nn


def one_hot(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(tokens, num_classes=vocab_size).float()


class FlowMatchingModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, conditioning: Optional[Dict[str, torch.Tensor]] = None):
        """
        x_t: [B, L, vocab] soft distribution on simplex
        t: [B, 1] time in [0,1]
        conditioning: optional dict; if contains "cond_emb" [B, D], add to token features
        """
        # project soft distributions via expected embedding
        emb_weights = self.embed.weight  # [V, D]
        h = torch.matmul(x_t, emb_weights)  # [B, L, D]
        t_proj = t.view(-1, 1, 1)
        h = h * (1 + t_proj)
        if conditioning and "cond_emb" in conditioning:
            cond = conditioning["cond_emb"].unsqueeze(1)  # [B,1,D]
            h = h + self.cond_proj(cond)
        h = self.net(h)
        v_pred = self.head(h)  # [B, L, V]
        return v_pred

    def sample_x0(self, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Uniform base distribution on simplex: start from uniform probabilities.
        """
        B, L = shape
        return torch.full((B, L, self.vocab_size), 1.0 / self.vocab_size, device=self.embed.weight.device)

    def flow_matching_loss(self, tokens: torch.Tensor, conditioning: Optional[Dict[str, torch.Tensor]] = None):
        """
        Implements categorical flow matching with convex interpolant.
        tokens: [B, L] target discrete ids
        """
        B, L = tokens.shape
        y_onehot = one_hot(tokens, self.vocab_size)
        x0 = self.sample_x0((B, L))
        t = torch.rand(B, 1, device=tokens.device)
        x_t = (1 - t) * x0 + t * y_onehot
        v_target = y_onehot - x0
        v_pred = self.forward(x_t, t, conditioning)
        loss = torch.mean((v_pred - v_target).pow(2))
        return loss
