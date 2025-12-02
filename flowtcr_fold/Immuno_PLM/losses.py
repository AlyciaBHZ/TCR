from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_infonce(anchor: torch.Tensor, positive: torch.Tensor, pos_mask: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Multi-positive InfoNCE with precomputed pos_mask [B, B] (bool).
    Loss for sample i: -log( sum_{j in P(i)} exp(sim_ij / tau) / sum_k exp(sim_ik / tau) )
    """
    if anchor.numel() == 0 or positive.numel() == 0:
        return torch.tensor(0.0, device=anchor.device)

    # Normalize
    anchor_n = F.normalize(anchor, p=2, dim=-1)
    positive_n = F.normalize(positive, p=2, dim=-1)

    logits = torch.matmul(anchor_n, positive_n.T) / temperature  # [B, B]
    # Numerical stability
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)

    # Masked sums
    pos_mask = pos_mask.to(anchor.device)
    pos_sum = (exp_logits * pos_mask).sum(dim=1)  # [B]
    all_sum = exp_logits.sum(dim=1) + 1e-8

    # Avoid log(0) if a sample has no positives
    pos_sum = pos_sum + (pos_sum == 0).float() * 1e-8

    loss = -torch.log(pos_sum / all_sum + 1e-8)
    # Exclude rows with no positives from contributing
    valid_rows = (pos_mask.sum(dim=1) > 0).float()
    if valid_rows.sum() == 0:
        return torch.tensor(0.0, device=anchor.device)
    return (loss * valid_rows).sum() / valid_rows.sum()


def multilabel_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-label BCE with optional valid_mask (B,) to skip empty targets.
    """
    if logits.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    pos_weight = pos_weight.to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.device), pos_weight=pos_weight, reduction="none")
    loss = loss.mean(dim=1)
    if valid_mask is None:
        return loss.mean()
    valid_mask = valid_mask.to(logits.device).float()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return (loss * valid_mask).sum() / valid_mask.sum()
