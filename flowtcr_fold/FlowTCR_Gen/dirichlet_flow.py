"""
Dirichlet Flow Matching for CDR3β sequence generation.

Core components:
- Dirichlet interpolation: x_t = (1-t) * x_0 + t * x_1
- Flow matching loss: MSE(v_pred, v_target) where v_target = x_1 - x_0
- CFG (Classifier-Free Guidance) support

Key innovation: Uses Dirichlet distribution on amino acid simplex
for smooth interpolation and probability-preserving flow.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_x0_dirichlet(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample prior distribution from Dirichlet(α, α, ..., α).
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size (number of amino acids)
        alpha: Dirichlet concentration parameter
            - α = 1: Uniform distribution on simplex
            - α < 1: Sparse, concentrated on corners
            - α > 1: Dense, concentrated near center
        device: Target device
    
    Returns:
        x0: [B, L, vocab] samples on probability simplex
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create Dirichlet distribution
    concentration = torch.ones(vocab_size, device=device) * alpha
    dist = torch.distributions.Dirichlet(concentration)
    
    # Sample
    x0 = dist.sample((batch_size, seq_len))  # [B, L, vocab]
    return x0


def sample_x0_uniform(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Uniform base distribution (special case of Dirichlet with α → ∞).
    
    Returns:
        x0: [B, L, vocab] uniform distribution
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x0 = torch.ones(batch_size, seq_len, vocab_size, device=device) / vocab_size
    return x0


def dirichlet_interpolate(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Linear interpolation on simplex (convex combination).
    
    x_t = (1 - t) * x_0 + t * x_1
    
    Args:
        x0: [B, L, vocab] prior distribution
        x1: [B, L, vocab] target distribution (one-hot)
        t: [B, 1, 1] or [B, 1] or scalar, time in [0, 1]
    
    Returns:
        x_t: [B, L, vocab] interpolated distribution
    """
    # Ensure t has correct shape for broadcasting
    while t.dim() < x0.dim():
        t = t.unsqueeze(-1)
    
    x_t = (1 - t) * x0 + t * x1
    return x_t


def compute_velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Compute target velocity field: v* = x_1 - x_0
    
    This is the optimal transport velocity for linear interpolation.
    """
    return x1 - x0


class FlowHead(nn.Module):
    """
    Velocity prediction head for flow matching.
    
    Takes the CDR3 region representations from encoder
    and predicts the velocity field v_θ(x_t, t, cond).
    """

    def __init__(
        self,
        s_dim: int = 256,
        vocab_size: int = 21,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.vocab_size = vocab_size
        
        # Multi-layer prediction head
        self.head = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, cdr3_repr: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity for CDR3 region.
        
        Args:
            cdr3_repr: [L_cdr3, s_dim] CDR3 representation from encoder
        
        Returns:
            v_pred: [L_cdr3, vocab] predicted velocity field
        """
        return self.head(cdr3_repr)


class DirichletFlowMatcher(nn.Module):
    """
    Complete Dirichlet Flow Matching module.
    
    Combines:
    - Dirichlet prior sampling
    - Linear interpolation
    - Velocity prediction
    - Flow matching loss with optional regularization
    """

    def __init__(
        self,
        s_dim: int = 256,
        vocab_size: int = 21,
        alpha: float = 1.0,
        use_uniform_prior: bool = True,
        lambda_entropy: float = 0.01,
        lambda_profile: float = 0.01,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.use_uniform_prior = use_uniform_prior
        self.lambda_entropy = lambda_entropy
        self.lambda_profile = lambda_profile
        
        # Velocity prediction head
        self.flow_head = FlowHead(s_dim=s_dim, vocab_size=vocab_size)

    def sample_x0(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sample prior distribution."""
        if self.use_uniform_prior:
            return sample_x0_uniform(batch_size, seq_len, self.vocab_size, device)
        else:
            return sample_x0_dirichlet(batch_size, seq_len, self.vocab_size, self.alpha, device)

    def flow_matching_loss(
        self,
        cdr3_repr: torch.Tensor,
        target_tokens: torch.Tensor,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.
        
        Args:
            cdr3_repr: [L_cdr3, s_dim] CDR3 representation from encoder
            target_tokens: [L_cdr3] target token indices
            x_t: [L_cdr3, vocab] current interpolated state
            x_0: [L_cdr3, vocab] prior distribution
            pad_mask: [L_cdr3] 1 for valid, 0 for pad
        
        Returns:
            Dict with 'loss', 'mse_loss', 'entropy_loss', 'profile_loss'
        """
        L, V = x_t.shape
        device = x_t.device
        
        if pad_mask is None:
            pad_mask = torch.ones(L, device=device)
        
        # Target: one-hot of target tokens
        x_1 = F.one_hot(target_tokens.clamp(min=0), self.vocab_size).float()  # [L, vocab]
        
        # Target velocity
        v_target = compute_velocity_target(x_0, x_1)  # [L, vocab]
        
        # Predicted velocity
        v_pred = self.flow_head(cdr3_repr)  # [L, vocab]
        
        # MSE loss (masked)
        diff = (v_pred - v_target).pow(2)  # [L, vocab]
        diff = diff * pad_mask.unsqueeze(-1)  # [L, vocab]
        
        mse_loss = diff.sum() / (pad_mask.sum() * self.vocab_size + 1e-8)
        
        # Entropy regularization (masked - only on valid positions)
        pred_probs = F.softmax(v_pred, dim=-1)
        entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1)  # [L]
        entropy = entropy * pad_mask  # Apply mask [L]
        valid_count = pad_mask.sum() + 1e-8
        entropy_loss = -entropy.sum() / valid_count  # Negative because we want to maximize entropy
        
        # Profile regularization (masked - only on valid positions)
        profile_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = mse_loss + self.lambda_entropy * entropy_loss + self.lambda_profile * profile_loss
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'entropy_loss': entropy_loss,
            'profile_loss': profile_loss,
        }

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time uniformly from [0, 1]."""
        return torch.rand(batch_size, 1, device=device)

    @torch.no_grad()
    def sample(
        self,
        encoder_fn,
        cdr3_len: int,
        peptide: torch.Tensor,
        peptide_idx: torch.Tensor,
        mhc: torch.Tensor,
        mhc_idx: torch.Tensor,
        scaffold_seqs: Dict[str, torch.Tensor],
        scaffold_idx: Dict[str, torch.Tensor],
        conditioning_info: Optional[list] = None,
        n_steps: int = 100,
        cfg_weight: float = 1.0,
        uncond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample CDR3 sequence using Euler ODE solver.
        
        Args:
            encoder_fn: Function to encode inputs (returns s, z, idx_map)
            cdr3_len: Target CDR3 length
            peptide, mhc, scaffold_seqs: Conditioning inputs
            n_steps: Number of ODE integration steps
            cfg_weight: Classifier-free guidance weight
            uncond_emb: Unconditional embedding for CFG
        
        Returns:
            tokens: [L_cdr3] predicted token indices
        """
        device = peptide.device
        
        # Initialize x_0 (uniform prior)
        x = sample_x0_uniform(1, cdr3_len, self.vocab_size, device).squeeze(0)  # [L, vocab]
        
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.tensor([step / n_steps], device=device)
            
            # Encode with current x_t
            s, z, idx_map = encoder_fn(
                cdr3_xt=x,
                t=t,
                peptide=peptide,
                peptide_idx=peptide_idx,
                mhc=mhc,
                mhc_idx=mhc_idx,
                scaffold_seqs=scaffold_seqs,
                scaffold_idx=scaffold_idx,
                conditioning_info=conditioning_info,
            )
            
            # Extract CDR3 representation
            cdr3_start, cdr3_end = idx_map['cdr3']
            cdr3_repr = s[cdr3_start:cdr3_end]  # [L_cdr3, s_dim]
            
            # Predict velocity
            v = self.flow_head(cdr3_repr)  # [L_cdr3, vocab]
            
            # CFG (if weight > 1 and unconditional embedding provided)
            if cfg_weight > 1.0 and uncond_emb is not None:
                # Would need unconditioned forward pass here
                # For now, just use conditional velocity
                pass
            
            # Euler step
            x = x + v * dt
            
            # Project back to simplex (normalize)
            x = F.softmax(x, dim=-1)
        
        # Final prediction: argmax
        tokens = x.argmax(dim=-1)
        return tokens

    def compute_log_prob(
        self,
        encoder_fn,
        target_tokens: torch.Tensor,
        peptide: torch.Tensor,
        peptide_idx: torch.Tensor,
        mhc: torch.Tensor,
        mhc_idx: torch.Tensor,
        scaffold_seqs: Dict[str, torch.Tensor],
        scaffold_idx: Dict[str, torch.Tensor],
        conditioning_info: Optional[list] = None,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """
        Compute approximate log probability via integration.
        
        This is used for model score hook (Stage 3 integration).
        """
        device = target_tokens.device
        L = target_tokens.shape[0]
        
        # Target one-hot
        x_1 = F.one_hot(target_tokens.clamp(min=0), self.vocab_size).float()
        
        # Prior
        x_0 = sample_x0_uniform(1, L, self.vocab_size, device).squeeze(0)
        
        # Integrate velocity field to compute flow cost
        total_cost = 0.0
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t_val = step / n_steps
            t = torch.tensor([t_val], device=device)
            
            # Interpolate
            x_t = dirichlet_interpolate(x_0.unsqueeze(0), x_1.unsqueeze(0), t).squeeze(0)
            
            # Encode
            s, z, idx_map = encoder_fn(
                cdr3_xt=x_t,
                t=t,
                peptide=peptide,
                peptide_idx=peptide_idx,
                mhc=mhc,
                mhc_idx=mhc_idx,
                scaffold_seqs=scaffold_seqs,
                scaffold_idx=scaffold_idx,
                conditioning_info=conditioning_info,
            )
            
            # CDR3 representation
            cdr3_start, cdr3_end = idx_map['cdr3']
            cdr3_repr = s[cdr3_start:cdr3_end]
            
            # Predicted velocity
            v_pred = self.flow_head(cdr3_repr)
            
            # True velocity
            v_true = x_1 - x_0
            
            # Accumulate squared error
            total_cost += ((v_pred - v_true) ** 2).sum()
        
        # Negative log prob approximation
        return total_cost * dt


class CFGWrapper(nn.Module):
    """
    Classifier-Free Guidance wrapper for training and inference.
    
    During training: Randomly drop conditioning with probability p
    During inference: Combine conditional and unconditional predictions
    """

    def __init__(
        self,
        encoder: nn.Module,
        flow_matcher: DirichletFlowMatcher,
        drop_prob: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.flow_matcher = flow_matcher
        self.drop_prob = drop_prob
        
        # Learnable unconditional embedding
        self.uncond_emb = nn.Parameter(torch.zeros(1, encoder.s_dim))
        nn.init.normal_(self.uncond_emb, std=0.02)

    def forward(
        self,
        cdr3_xt: torch.Tensor,
        t: torch.Tensor,
        peptide: torch.Tensor,
        peptide_idx: torch.Tensor,
        mhc: torch.Tensor,
        mhc_idx: torch.Tensor,
        scaffold_seqs: Dict[str, torch.Tensor],
        scaffold_idx: Dict[str, torch.Tensor],
        conditioning_info: Optional[list] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward with optional condition dropping.
        
        Returns:
            s: Sequence representation
            z: Pair representation
            idx_map: Region indices
        """
        # During training, randomly drop conditioning
        if training and torch.rand(1).item() < self.drop_prob:
            # Use empty conditioning
            conditioning_info = []
        
        return self.encoder(
            cdr3_xt=cdr3_xt,
            t=t,
            peptide=peptide,
            peptide_idx=peptide_idx,
            mhc=mhc,
            mhc_idx=mhc_idx,
            scaffold_seqs=scaffold_seqs,
            scaffold_idx=scaffold_idx,
            conditioning_info=conditioning_info,
        )

    @torch.no_grad()
    def sample_with_cfg(
        self,
        cdr3_len: int,
        peptide: torch.Tensor,
        peptide_idx: torch.Tensor,
        mhc: torch.Tensor,
        mhc_idx: torch.Tensor,
        scaffold_seqs: Dict[str, torch.Tensor],
        scaffold_idx: Dict[str, torch.Tensor],
        conditioning_info: Optional[list] = None,
        n_steps: int = 100,
        cfg_weight: float = 1.5,
    ) -> torch.Tensor:
        """
        Sample with Classifier-Free Guidance.
        
        v_final = v_uncond + w * (v_cond - v_uncond)
        
        Args:
            cfg_weight: Guidance strength (1.0 = no guidance)
        """
        device = peptide.device
        
        # Initialize x_0
        x = sample_x0_uniform(1, cdr3_len, self.flow_matcher.vocab_size, device).squeeze(0)
        
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.tensor([step / n_steps], device=device)
            
            # Conditional forward
            s_cond, z_cond, idx_map_cond = self.encoder(
                cdr3_xt=x,
                t=t,
                peptide=peptide,
                peptide_idx=peptide_idx,
                mhc=mhc,
                mhc_idx=mhc_idx,
                scaffold_seqs=scaffold_seqs,
                scaffold_idx=scaffold_idx,
                conditioning_info=conditioning_info,
            )
            
            cdr3_start, cdr3_end = idx_map_cond['cdr3']
            cdr3_repr_cond = s_cond[cdr3_start:cdr3_end]
            v_cond = self.flow_matcher.flow_head(cdr3_repr_cond)
            
            # Unconditional forward (empty conditioning)
            s_uncond, z_uncond, idx_map_uncond = self.encoder(
                cdr3_xt=x,
                t=t,
                peptide=peptide,
                peptide_idx=peptide_idx,
                mhc=mhc,
                mhc_idx=mhc_idx,
                scaffold_seqs=scaffold_seqs,
                scaffold_idx=scaffold_idx,
                conditioning_info=[],
            )
            
            cdr3_repr_uncond = s_uncond[cdr3_start:cdr3_end]
            v_uncond = self.flow_matcher.flow_head(cdr3_repr_uncond)
            
            # CFG combination
            v = v_uncond + cfg_weight * (v_cond - v_uncond)
            
            # Euler step
            x = x + v * dt
            x = F.softmax(x, dim=-1)
        
        return x.argmax(dim=-1)


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Dirichlet sampling
    x0 = sample_x0_dirichlet(2, 15, 21, alpha=1.0, device=device)
    print(f"✅ Dirichlet x0 shape: {x0.shape}, sum per position: {x0.sum(dim=-1).mean():.4f}")
    
    # Test interpolation
    x1 = F.one_hot(torch.randint(0, 21, (2, 15), device=device), 21).float()
    t = torch.tensor([[0.5]], device=device)
    x_t = dirichlet_interpolate(x0, x1, t)
    print(f"✅ Interpolated x_t shape: {x_t.shape}, sum: {x_t.sum(dim=-1).mean():.4f}")
    
    # Test flow matcher
    flow_matcher = DirichletFlowMatcher(s_dim=128, vocab_size=21).to(device)
    
    cdr3_repr = torch.randn(15, 128, device=device)
    target_tokens = torch.randint(0, 21, (15,), device=device)
    x_t_single = x_t[0]
    x_0_single = x0[0]
    
    losses = flow_matcher.flow_matching_loss(
        cdr3_repr=cdr3_repr,
        target_tokens=target_tokens,
        x_t=x_t_single,
        x_0=x_0_single,
    )
    
    print(f"✅ Flow matching losses:")
    for k, v in losses.items():
        print(f"   {k}: {v.item():.4f}")

