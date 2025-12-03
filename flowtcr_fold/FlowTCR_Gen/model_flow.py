"""
FlowTCR-Gen: Complete model for topology-aware CDR3β generation.

Combines:
- FlowTCRGenEncoder: Collapse Token + Hierarchical Pairs + Evoformer
- DirichletFlowMatcher: Dirichlet flow matching with CFG support
- Model Score Hook: For Stage 3 integration

This is the main model class that serves as the entry point for training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowtcr_fold.FlowTCR_Gen.encoder import FlowTCRGenEncoder
from flowtcr_fold.FlowTCR_Gen.dirichlet_flow import (
    DirichletFlowMatcher,
    dirichlet_interpolate,
    sample_x0_uniform,
    CFGWrapper,
)


class FlowTCRGen(nn.Module):
    """
    FlowTCR-Gen: Topology-aware Dirichlet Flow Matching for CDR3β generation.
    
    Key innovations (论文主打):
    1. Collapse Token (ψ): Learnable global observer
    2. Hierarchical Pair Embeddings: 7-level topology encoding
    3. Dirichlet Flow Matching: Continuous generation on simplex
    4. CFG Support: Classifier-free guidance for controllable generation
    
    Usage:
        model = FlowTCRGen(...)
        
        # Training
        loss = model.training_step(batch)
        
        # Inference
        tokens = model.generate(peptide, mhc, scaffold_seqs, ...)
        
        # Model score (for Stage 3)
        score = model.get_model_score(cdr3_tokens, cond)
    """

    def __init__(
        self,
        s_dim: int = 256,
        z_dim: int = 64,
        n_layers: int = 6,
        vocab_size: int = 25,  # 20 AA + 5 special tokens
        max_len: int = 512,
        dirichlet_alpha: float = 1.0,
        use_uniform_prior: bool = True,
        lambda_entropy: float = 0.01,
        lambda_profile: float = 0.01,
        cfg_drop_prob: float = 0.1,
        use_collapse: bool = True,
        use_hier_pairs: bool = True,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.vocab_size = vocab_size
        self.cfg_drop_prob = cfg_drop_prob
        self.use_collapse = use_collapse
        self.use_hier_pairs = use_hier_pairs
        
        # Encoder with collapse token and hierarchical pairs
        self.encoder = FlowTCRGenEncoder(
            s_dim=s_dim,
            z_dim=z_dim,
            n_layers=n_layers,
            vocab_size=vocab_size,
            max_len=max_len,
            use_collapse=use_collapse,
            use_hier_pairs=use_hier_pairs,
        )
        
        # Dirichlet flow matcher
        self.flow_matcher = DirichletFlowMatcher(
            s_dim=s_dim,
            vocab_size=vocab_size,
            alpha=dirichlet_alpha,
            use_uniform_prior=use_uniform_prior,
            lambda_entropy=lambda_entropy,
            lambda_profile=lambda_profile,
        )
        
        # Learnable unconditional embedding for CFG
        self.uncond_emb = nn.Parameter(torch.zeros(1, s_dim))
        nn.init.normal_(self.uncond_emb, std=0.02)

    def forward(
        self,
        cdr3_tokens: torch.Tensor,
        cdr3_mask: torch.Tensor,
        pep_tokens: torch.Tensor,
        pep_mask: torch.Tensor,
        mhc_tokens: torch.Tensor,
        mhc_mask: torch.Tensor,
        scaffold_tokens: Dict[str, torch.Tensor],
        scaffold_mask: Dict[str, torch.Tensor],
        conditioning_info: Optional[List[str]] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with per-sample conditioning.
        
        Args:
            cdr3_tokens: [B, L_cdr3] target CDR3 token indices
            cdr3_mask: [B, L_cdr3] 1 for valid, 0 for pad
            pep_tokens: [B, L_pep] peptide tokens
            pep_mask: [B, L_pep] peptide mask
            mhc_tokens: [B, L_mhc] MHC tokens
            mhc_mask: [B, L_mhc] MHC mask
            scaffold_tokens: Dict of scaffold tokens [B, L]
            scaffold_mask: Dict of scaffold masks [B, L]
            conditioning_info: List of conditioning regions to use
            training: Whether in training mode
        
        Returns:
            Dict with losses and metrics
        """
        if conditioning_info is None:
            conditioning_info = ['pep', 'mhc', 'hv', 'hj', 'lv', 'lj']
        
        device = cdr3_tokens.device
        B, L = cdr3_tokens.shape
        
        # CFG: Randomly drop conditioning during training
        if training and torch.rand(1).item() < self.cfg_drop_prob:
            conditioning_info = []
        
        # Sample time t ~ Uniform(0, 1)
        t = torch.rand(B, 1, device=device)
        
        # Get target one-hot
        x_1 = F.one_hot(cdr3_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()  # [B, L, vocab]
        
        # Sample prior (route through flow_matcher to respect config)
        x_0 = self.flow_matcher.sample_x0(B, L, device)  # [B, L, vocab]
        
        # Interpolate
        x_t = dirichlet_interpolate(x_0, x_1, t.unsqueeze(-1))  # [B, L, vocab]
        
        # Convert tokens to one-hot for conditioning
        pep_one_hot = F.one_hot(pep_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        mhc_one_hot = F.one_hot(mhc_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        scaffold_one_hot = {}
        for key, tokens in scaffold_tokens.items():
            scaffold_one_hot[key] = F.one_hot(tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        # Encode with per-sample conditioning
        s_list, z_list, idx_maps = self.encoder(
            cdr3_xt=x_t,
            cdr3_mask=cdr3_mask,
            t=t,
            pep_one_hot=pep_one_hot,
            pep_mask=pep_mask,
            mhc_one_hot=mhc_one_hot,
            mhc_mask=mhc_mask,
            scaffold_one_hot=scaffold_one_hot,
            scaffold_mask=scaffold_mask,
            conditioning_info=conditioning_info,
        )
        
        # Compute loss for each sample
        total_loss = 0.0
        total_mse = 0.0
        total_entropy = 0.0
        
        for i in range(B):
            s = s_list[i]
            idx_map = idx_maps[i]
            
            # Get CDR3 representation
            cdr3_start, cdr3_end = idx_map['cdr3']
            cdr3_repr = s[cdr3_start:cdr3_end]  # [L_cdr3_valid, s_dim]
            
            # Get valid length for this sample
            valid_len = int(cdr3_mask[i].sum().item())
            
            # Compute loss (only on valid positions)
            losses = self.flow_matcher.flow_matching_loss(
                cdr3_repr=cdr3_repr,
                target_tokens=cdr3_tokens[i, :valid_len],
                x_t=x_t[i, :valid_len],
                x_0=x_0[i, :valid_len],
                pad_mask=cdr3_mask[i, :valid_len],
            )
            
            total_loss += losses['loss']
            total_mse += losses['mse_loss']
            total_entropy += losses['entropy_loss']
        
        return {
            'loss': total_loss / B,
            'mse_loss': total_mse / B,
            'entropy_loss': total_entropy / B,
        }

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convenience wrapper for training with per-sample conditioning."""
        return self.forward(
            cdr3_tokens=batch['cdr3_tokens'],
            cdr3_mask=batch['cdr3_mask'],
            pep_tokens=batch['pep_tokens'],
            pep_mask=batch['pep_mask'],
            mhc_tokens=batch['mhc_tokens'],
            mhc_mask=batch['mhc_mask'],
            scaffold_tokens=batch['scaffold_tokens'],
            scaffold_mask=batch['scaffold_mask'],
            training=True,
        )

    @torch.no_grad()
    def generate(
        self,
        cdr3_len: int,
        pep_tokens: torch.Tensor,
        pep_mask: torch.Tensor,
        mhc_tokens: torch.Tensor,
        mhc_mask: torch.Tensor,
        scaffold_tokens: Dict[str, torch.Tensor],
        scaffold_mask: Dict[str, torch.Tensor],
        conditioning_info: Optional[List[str]] = None,
        n_steps: int = 100,
        cfg_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate CDR3β sequence with per-sample conditioning.
        
        Args:
            cdr3_len: Target sequence length
            pep_tokens: [1, L_pep] peptide tokens
            pep_mask: [1, L_pep] peptide mask
            mhc_tokens: [1, L_mhc] MHC tokens
            mhc_mask: [1, L_mhc] MHC mask
            scaffold_tokens: Dict of [1, L] scaffold tokens
            scaffold_mask: Dict of [1, L] scaffold masks
            conditioning_info: List of conditioning regions to use
            n_steps: Number of ODE integration steps
            cfg_weight: CFG guidance strength (1.0 = no guidance)
        
        Returns:
            tokens: [L] generated token indices
        """
        if conditioning_info is None:
            conditioning_info = ['pep', 'mhc', 'hv', 'hj', 'lv', 'lj']
        
        device = pep_tokens.device if pep_tokens.numel() > 0 else (
            mhc_tokens.device if mhc_tokens.numel() > 0 else 
            next(self.parameters()).device
        )
        
        # Initialize x_0 (uniform prior) with batch dim
        x = sample_x0_uniform(1, cdr3_len, self.vocab_size, device)  # [1, L, vocab]
        cdr3_mask = torch.ones(1, cdr3_len, device=device)
        
        # Convert tokens to one-hot
        pep_one_hot = F.one_hot(pep_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        mhc_one_hot = F.one_hot(mhc_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        scaffold_one_hot = {}
        for key, tokens in scaffold_tokens.items():
            scaffold_one_hot[key] = F.one_hot(tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.tensor([[step / n_steps]], device=device)
            
            # Conditional encoding
            s_list, z_list, idx_maps = self.encoder(
                cdr3_xt=x,
                cdr3_mask=cdr3_mask,
                t=t,
                pep_one_hot=pep_one_hot,
                pep_mask=pep_mask,
                mhc_one_hot=mhc_one_hot,
                mhc_mask=mhc_mask,
                scaffold_one_hot=scaffold_one_hot,
                scaffold_mask=scaffold_mask,
                conditioning_info=conditioning_info,
            )
            
            s_cond = s_list[0]
            idx_map = idx_maps[0]
            cdr3_start, cdr3_end = idx_map['cdr3']
            cdr3_repr_cond = s_cond[cdr3_start:cdr3_end]
            v_cond = self.flow_matcher.flow_head(cdr3_repr_cond)
            
            if cfg_weight > 1.0:
                # Unconditional encoding
                s_list_uncond, _, idx_maps_uncond = self.encoder(
                    cdr3_xt=x,
                    cdr3_mask=cdr3_mask,
                    t=t,
                    pep_one_hot=pep_one_hot,
                    pep_mask=pep_mask,
                    mhc_one_hot=mhc_one_hot,
                    mhc_mask=mhc_mask,
                    scaffold_one_hot=scaffold_one_hot,
                    scaffold_mask=scaffold_mask,
                    conditioning_info=[],  # Empty conditioning
                )
                
                s_uncond = s_list_uncond[0]
                idx_map_uncond = idx_maps_uncond[0]
                cdr3_start_uncond, cdr3_end_uncond = idx_map_uncond['cdr3']
                cdr3_repr_uncond = s_uncond[cdr3_start_uncond:cdr3_end_uncond]
                v_uncond = self.flow_matcher.flow_head(cdr3_repr_uncond)
                
                # CFG combination
                v = v_uncond + cfg_weight * (v_cond - v_uncond)
            else:
                v = v_cond
            
            # Euler step (squeeze and unsqueeze for batch dim)
            x = x.squeeze(0) + v * dt
            
            # Project back to simplex
            x = F.softmax(x, dim=-1)
            x = x.unsqueeze(0)  # Add batch dim back
        
        # Final prediction (squeeze batch dim)
        tokens = x.squeeze(0).argmax(dim=-1)
        return tokens

    def get_model_score(
        self,
        cdr3_tokens: torch.Tensor,
        cdr3_mask: torch.Tensor,
        pep_tokens: torch.Tensor,
        pep_mask: torch.Tensor,
        mhc_tokens: torch.Tensor,
        mhc_mask: torch.Tensor,
        scaffold_tokens: Dict[str, torch.Tensor],
        scaffold_mask: Dict[str, torch.Tensor],
        conditioning_info: Optional[List[str]] = None,
        n_t_samples: int = 4,
    ) -> torch.Tensor:
        """
        Compute model score for Stage 3 integration using masked flow loss.
        
        This is the recommended "flow loss per sample" proxy that:
        - Uses multiple random t samples for better estimation
        - Masks padding positions properly
        - Normalizes by actual sequence length
        
        Lower score = higher probability = better sequence.
        
        Args:
            cdr3_tokens: [1, L] CDR3 token indices (with padding)
            cdr3_mask: [1, L] 1 for valid, 0 for pad
            pep_tokens: [1, L_pep] peptide tokens
            pep_mask: [1, L_pep] peptide mask
            mhc_tokens: [1, L_mhc] MHC tokens
            mhc_mask: [1, L_mhc] MHC mask
            scaffold_tokens: Dict of [1, L] scaffold tokens
            scaffold_mask: Dict of [1, L] scaffold masks
            conditioning_info: List of conditioning regions
            n_t_samples: Number of random t samples for expectation
        
        Returns:
            score: Scalar model score (masked MSE, normalized by length)
        """
        device = cdr3_tokens.device
        L = cdr3_tokens.shape[1]
        valid_len = int(cdr3_mask.sum().item())
        
        if conditioning_info is None:
            conditioning_info = ['pep', 'mhc', 'hv', 'hj', 'lv', 'lj']
        
        # Target one-hot [1, L, vocab]
        x_1 = F.one_hot(cdr3_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        # Convert conditioning to one-hot
        pep_one_hot = F.one_hot(pep_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        mhc_one_hot = F.one_hot(mhc_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        scaffold_one_hot = {}
        for key, tokens in scaffold_tokens.items():
            scaffold_one_hot[key] = F.one_hot(tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        total_mse = 0.0
        eps = 1e-8
        
        with torch.no_grad():
            for _ in range(n_t_samples):
                # Sample x_0 and t
                x_0 = self.flow_matcher.sample_x0(1, L, device)  # [1, L, vocab]
                t = torch.rand(1, 1, device=device)
                
                # Interpolate
                x_t = dirichlet_interpolate(x_0, x_1, t.unsqueeze(-1))  # [1, L, vocab]
                
                # True velocity (only for valid positions)
                v_true = x_1 - x_0  # [1, L, vocab]
                
                # Encode with per-sample conditioning
                s_list, z_list, idx_maps = self.encoder(
                    cdr3_xt=x_t,
                    cdr3_mask=cdr3_mask,
                    t=t,
                    pep_one_hot=pep_one_hot,
                    pep_mask=pep_mask,
                    mhc_one_hot=mhc_one_hot,
                    mhc_mask=mhc_mask,
                    scaffold_one_hot=scaffold_one_hot,
                    scaffold_mask=scaffold_mask,
                    conditioning_info=conditioning_info,
                )
                
                s = s_list[0]
                idx_map = idx_maps[0]
                
                # CDR3 representation (only valid positions)
                cdr3_start, cdr3_end = idx_map['cdr3']
                cdr3_repr = s[cdr3_start:cdr3_end]  # [valid_len, s_dim]
                
                # Predicted velocity
                v_pred = self.flow_matcher.flow_head(cdr3_repr)  # [valid_len, vocab]
                
                # Masked MSE (only on valid positions, normalized by length and vocab)
                v_true_valid = v_true[0, :valid_len]  # [valid_len, vocab]
                diff = (v_pred - v_true_valid) ** 2  # [valid_len, vocab]
                
                # Per-position MSE, then average over valid positions
                pos_mse = diff.sum(dim=-1)  # [valid_len]
                mse = pos_mse.mean()  # scalar
                
                total_mse += mse
        
        # Return average MSE across t samples
        return total_mse / n_t_samples

    def get_collapse_scalar(
        self,
        cdr3_tokens: torch.Tensor,
        cdr3_mask: torch.Tensor,
        pep_tokens: torch.Tensor,
        pep_mask: torch.Tensor,
        mhc_tokens: torch.Tensor,
        mhc_mask: torch.Tensor,
        scaffold_tokens: Dict[str, torch.Tensor],
        scaffold_mask: Dict[str, torch.Tensor],
        conditioning_info: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Extract collapse token scalar for alternative model score.
        
        The collapse token aggregates global context and can be used
        as a quick quality indicator.
        """
        if not self.use_collapse:
            return torch.tensor(0.0, device=cdr3_tokens.device)
        
        device = cdr3_tokens.device
        
        # Use target as x_t at t=1
        x_1 = F.one_hot(cdr3_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        t = torch.tensor([[1.0]], device=device)
        
        # Convert conditioning to one-hot
        pep_one_hot = F.one_hot(pep_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        mhc_one_hot = F.one_hot(mhc_tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        scaffold_one_hot = {}
        for key, tokens in scaffold_tokens.items():
            scaffold_one_hot[key] = F.one_hot(tokens.clamp(min=0, max=self.vocab_size - 1), self.vocab_size).float()
        
        with torch.no_grad():
            s_list, z_list, idx_maps = self.encoder(
                cdr3_xt=x_1,
                cdr3_mask=cdr3_mask,
                t=t,
                pep_one_hot=pep_one_hot,
                pep_mask=pep_mask,
                mhc_one_hot=mhc_one_hot,
                mhc_mask=mhc_mask,
                scaffold_one_hot=scaffold_one_hot,
                scaffold_mask=scaffold_mask,
                conditioning_info=conditioning_info,
            )
        
        return self.encoder.get_collapse_scalar(s_list[0])

    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                's_dim': self.s_dim,
                'z_dim': self.z_dim,
                'vocab_size': self.vocab_size,
                'use_collapse': self.use_collapse,
                'use_hier_pairs': self.use_hier_pairs,
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'FlowTCRGen':
        """Load model from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ckpt = torch.load(path, map_location=device)
        config = ckpt['config']
        
        model = cls(
            s_dim=config['s_dim'],
            z_dim=config.get('z_dim', 64),
            vocab_size=config['vocab_size'],
            use_collapse=config.get('use_collapse', True),
            use_hier_pairs=config.get('use_hier_pairs', True),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        
        return model


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FlowTCRGen(
        s_dim=128,
        z_dim=32,
        n_layers=2,
        vocab_size=25,
    ).to(device)
    
    print(f"✅ FlowTCRGen created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Mock batch
    B, L = 2, 15
    vocab_size = 25
    
    batch = {
        'cdr3_tokens': torch.randint(0, vocab_size, (B, L), device=device),
        'cdr3_mask': torch.ones(B, L, device=device),
        'pep_one_hot': F.one_hot(torch.randint(0, vocab_size, (9,), device=device), vocab_size).float(),
        'pep_idx': torch.arange(9, device=device),
        'mhc_one_hot': F.one_hot(torch.randint(0, vocab_size, (50,), device=device), vocab_size).float(),
        'mhc_idx': torch.arange(50, device=device),
        'scaffold_seqs': {
            'hv': F.one_hot(torch.randint(0, vocab_size, (30,), device=device), vocab_size).float(),
        },
        'scaffold_idx': {
            'hv': torch.arange(30, device=device),
        },
    }
    
    # Test forward
    losses = model.training_step(batch)
    print(f"✅ Forward pass: loss={losses['loss'].item():.4f}, mse={losses['mse_loss'].item():.4f}")
    
    # Test generation
    tokens = model.generate(
        cdr3_len=L,
        pep_one_hot=batch['pep_one_hot'],
        pep_idx=batch['pep_idx'],
        mhc_one_hot=batch['mhc_one_hot'],
        mhc_idx=batch['mhc_idx'],
        scaffold_seqs=batch['scaffold_seqs'],
        scaffold_idx=batch['scaffold_idx'],
        n_steps=10,
    )
    print(f"✅ Generation: tokens shape = {tokens.shape}")
    
    # Test model score
    score = model.get_model_score(
        cdr3_tokens=batch['cdr3_tokens'][0],
        pep_one_hot=batch['pep_one_hot'],
        pep_idx=batch['pep_idx'],
        mhc_one_hot=batch['mhc_one_hot'],
        mhc_idx=batch['mhc_idx'],
        scaffold_seqs=batch['scaffold_seqs'],
        scaffold_idx=batch['scaffold_idx'],
        n_steps=5,
    )
    print(f"✅ Model score: {score.item():.4f}")
    
    # Test save/load
    model.save('/tmp/test_flowtcrgen.pt')
    model2 = FlowTCRGen.load('/tmp/test_flowtcrgen.pt', device=device)
    print(f"✅ Save/load test passed")

