"""
TCRFoldProphet: Structure Predictor + Energy Surrogate for Stage 3.

Architecture:
- Sequence Embedding: AA index → learned embedding + position encoding
- Pair Initialization: Outer product + relative position + pair type embedding
- Evoformer Trunk: 12 layers of MSA row/col attention + pair updates
- Distance Head: Pair → distance bins (64 bins, 2-22Å)
- Contact Head: Pair → contact probability (binary)
- Energy Head: Global pooling → E_φ surrogate

Training Phases:
- Phase 3A: PPI structure pretraining (distance + contact loss)
- Phase 3B: Energy surrogate fitting (MSE to EvoEF2)
- Phase 3C: TCR-specific finetuning (structure + energy)

Usage:
    from flowtcr_fold.TCRFold_Light.tcrfold_prophet import TCRFoldProphet
    
    model = TCRFoldProphet(s_dim=384, z_dim=128, n_layers=8)
    outputs = model(seq_tokens, pair_type)
    # outputs: {'s', 'z', 'dist_logits', 'contact_logits', 'energy'}
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Optional, Tuple

try:
    from conditioned.src.Evoformer import Evoformer
except ImportError:
    Evoformer = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, D] or [L, D]"""
        if x.dim() == 3:
            x = x + self.pe[:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class RelativePositionBias(nn.Module):
    """Relative position encoding for pair representation."""
    
    def __init__(self, z_dim: int, max_rel_pos: int = 32):
        super().__init__()
        self.max_rel_pos = max_rel_pos
        # 2 * max_rel_pos + 1 bins: [-max, ..., 0, ..., +max]
        self.embed = nn.Embedding(2 * max_rel_pos + 1, z_dim)
    
    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        """Returns: [L, L, z_dim]"""
        pos = torch.arange(L, device=device)
        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(0)  # [L, L]
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        return self.embed(rel_pos)  # [L, L, z_dim]


class PairTypeEmbedding(nn.Module):
    """
    Embedding for pair types in PPI:
    - 0: intra-chain A (same chain)
    - 1: intra-chain B (same chain)
    - 2: inter-chain AB (different chains)
    - 3: padding
    """
    
    def __init__(self, z_dim: int, n_types: int = 4):
        super().__init__()
        self.embed = nn.Embedding(n_types, z_dim)
    
    def forward(self, pair_type: torch.Tensor) -> torch.Tensor:
        """pair_type: [B, L, L] or [L, L]"""
        return self.embed(pair_type)


class EnergyHead(nn.Module):
    """
    E_φ: Energy surrogate head.
    
    Takes pair representation z and predicts total binding energy.
    Uses global pooling with interface masking.
    """
    
    def __init__(self, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        z: torch.Tensor,
        interface_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z: [B, L, L, z_dim] pair representation
            interface_mask: [B, L, L] optional mask for interface pairs
        
        Returns:
            energy: [B] predicted binding energy
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)
        
        B, L1, L2, D = z.shape
        
        if interface_mask is not None:
            # Weighted average over interface pairs
            mask = interface_mask.unsqueeze(-1).float()  # [B, L, L, 1]
            z_pooled = (z * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
        else:
            # Simple mean pooling
            z_pooled = z.mean(dim=(1, 2))  # [B, D]
        
        return self.mlp(z_pooled).squeeze(-1)  # [B]


class DistanceHead(nn.Module):
    """
    Distance bin prediction head.
    
    Predicts distance between residue pairs binned into 64 bins (2-22Å).
    """
    
    def __init__(self, z_dim: int, n_bins: int = 64, min_dist: float = 2.0, max_dist: float = 22.0):
        super().__init__()
        self.n_bins = n_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        self.proj = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, n_bins),
        )
        
        # Pre-compute bin edges
        self.register_buffer(
            'bin_edges',
            torch.linspace(min_dist, max_dist, n_bins + 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, L, L, z_dim] or [L, L, z_dim]
        
        Returns:
            logits: [B, L, L, n_bins] distance bin logits
        """
        return self.proj(z)
    
    def distance_to_bins(self, dist: torch.Tensor) -> torch.Tensor:
        """Convert continuous distances to bin indices."""
        dist = dist.clamp(self.min_dist, self.max_dist - 1e-6)
        bins = torch.bucketize(dist, self.bin_edges[1:-1])
        return bins
    
    def bins_to_distance(self, bins: torch.Tensor) -> torch.Tensor:
        """Convert bin indices to bin centers."""
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        return bin_centers[bins]


class ContactHead(nn.Module):
    """Binary contact prediction head (CA-CA distance < 8Å)."""
    
    def __init__(self, z_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2),
            nn.ReLU(),
            nn.Linear(z_dim // 2, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, L, L, z_dim] or [L, L, z_dim]
        
        Returns:
            logits: [B, L, L] or [L, L] contact logits
        """
        return self.proj(z).squeeze(-1)


class TCRFoldProphet(nn.Module):
    """
    TCRFold-Prophet: Structure Predictor + Energy Surrogate.
    
    Combines Evoformer backbone with task-specific heads for:
    - Distance prediction (binned)
    - Contact prediction (binary)
    - Energy prediction (E_φ surrogate)
    
    Args:
        s_dim: Sequence representation dimension (default: 384)
        z_dim: Pair representation dimension (default: 128)
        n_layers: Number of Evoformer layers (default: 8)
        n_heads: Number of attention heads (default: 8)
        vocab_size: Number of amino acid types + padding (default: 22)
        max_len: Maximum sequence length (default: 1024)
        dist_bins: Number of distance bins (default: 64)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        s_dim: int = 384,
        z_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 8,
        vocab_size: int = 22,  # 20 AA + X + PAD
        max_len: int = 1024,
        dist_bins: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        # Sequence embedding
        self.aa_embed = nn.Embedding(vocab_size, s_dim, padding_idx=vocab_size - 1)
        self.pos_enc = PositionalEncoding(s_dim, max_len, dropout)
        
        # Pair initialization
        self.pair_proj = nn.Linear(s_dim * 2, z_dim)
        self.relpos = RelativePositionBias(z_dim, max_rel_pos=32)
        self.pair_type_embed = PairTypeEmbedding(z_dim, n_types=4)
        
        # Evoformer backbone
        if Evoformer is None:
            raise ImportError(
                "conditioned.src.Evoformer not found. "
                "Ensure the legacy Evoformer code is available."
            )
        self.evoformer = Evoformer(s_dim, z_dim, N_elayers=n_layers)
        
        # Task heads
        self.dist_head = DistanceHead(z_dim, n_bins=dist_bins)
        self.contact_head = ContactHead(z_dim)
        self.energy_head = EnergyHead(z_dim, hidden_dim=256)
        
        # Layer norms
        self.s_norm = nn.LayerNorm(s_dim)
        self.z_norm = nn.LayerNorm(z_dim)
    
    def init_pair_rep(
        self,
        s: torch.Tensor,
        pair_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Initialize pair representation from sequence embedding.
        
        Args:
            s: [B, L, s_dim] sequence embeddings
            pair_type: [B, L, L] pair type IDs (0=intra-A, 1=intra-B, 2=inter-AB)
        
        Returns:
            z: [B, L, L, z_dim] initial pair representation
        """
        B, L, D = s.shape
        
        # Outer product
        s_i = s.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, D]
        s_j = s.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, D]
        z = self.pair_proj(torch.cat([s_i, s_j], dim=-1))  # [B, L, L, z_dim]
        
        # Add relative position bias
        relpos = self.relpos(L, s.device)  # [L, L, z_dim]
        z = z + relpos.unsqueeze(0)
        
        # Add pair type embedding
        if pair_type is not None:
            z = z + self.pair_type_embed(pair_type)
        
        return z
    
    def forward(
        self,
        seq_tokens: torch.Tensor,
        pair_type: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        interface_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            seq_tokens: [B, L] or [L] sequence token indices
            pair_type: [B, L, L] pair type IDs
            mask: [B, L] sequence mask (1=valid, 0=padding)
            interface_mask: [B, L, L] interface pair mask for energy head
        
        Returns:
            dict with keys:
                - s: [B, L, s_dim] final sequence representation
                - z: [B, L, L, z_dim] final pair representation
                - dist_logits: [B, L, L, n_bins] distance bin logits
                - contact_logits: [B, L, L] contact logits
                - energy: [B] predicted binding energy
        """
        # Handle unbatched input
        if seq_tokens.dim() == 1:
            seq_tokens = seq_tokens.unsqueeze(0)
        if pair_type is not None and pair_type.dim() == 2:
            pair_type = pair_type.unsqueeze(0)
        
        B, L = seq_tokens.shape
        
        # Sequence embedding
        s = self.aa_embed(seq_tokens)  # [B, L, s_dim]
        s = self.pos_enc(s)
        
        # Pair initialization
        z = self.init_pair_rep(s, pair_type)  # [B, L, L, z_dim]
        
        # Evoformer (expects [N, L, D] and [L, L, D])
        # Process batch sequentially since Evoformer doesn't handle batches well
        s_out_list = []
        z_out_list = []
        
        for b in range(B):
            # Evoformer expects single sample
            s_b = s[b:b+1]  # [1, L, s_dim]
            z_b = z[b]      # [L, L, z_dim]
            
            s_b_out, z_b_out = self.evoformer(s_b, z_b)
            s_out_list.append(s_b_out)
            z_out_list.append(z_b_out.unsqueeze(0))
        
        s = torch.cat(s_out_list, dim=0)  # [B, L, s_dim]
        z = torch.cat(z_out_list, dim=0)  # [B, L, L, z_dim]
        
        # Normalize
        s = self.s_norm(s)
        z = self.z_norm(z)
        
        # Task heads
        dist_logits = self.dist_head(z)  # [B, L, L, n_bins]
        contact_logits = self.contact_head(z)  # [B, L, L]
        energy = self.energy_head(z, interface_mask)  # [B]
        
        return {
            's': s,
            'z': z,
            'dist_logits': dist_logits,
            'contact_logits': contact_logits,
            'energy': energy,
        }
    
    def predict_contacts(
        self,
        seq_tokens: torch.Tensor,
        pair_type: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Predict binary contacts."""
        outputs = self.forward(seq_tokens, pair_type)
        probs = torch.sigmoid(outputs['contact_logits'])
        return (probs > threshold).long()
    
    def predict_distances(
        self,
        seq_tokens: torch.Tensor,
        pair_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict distance matrix (bin centers)."""
        outputs = self.forward(seq_tokens, pair_type)
        bins = outputs['dist_logits'].argmax(dim=-1)
        return self.dist_head.bins_to_distance(bins)
    
    def predict_energy(
        self,
        seq_tokens: torch.Tensor,
        pair_type: Optional[torch.Tensor] = None,
        interface_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Predict binding energy."""
        outputs = self.forward(seq_tokens, pair_type, interface_mask=interface_mask)
        return outputs['energy']


# =============================================================================
# Loss functions
# =============================================================================

def compute_distance_loss(
    pred_logits: torch.Tensor,
    target_dist: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dist_head: Optional[DistanceHead] = None,
) -> torch.Tensor:
    """
    Distance prediction loss (cross-entropy over bins).
    
    Args:
        pred_logits: [B, L, L, n_bins] predicted distance bin logits
        target_dist: [B, L, L] target distances
        mask: [B, L, L] valid pair mask
        dist_head: DistanceHead for bin conversion
    
    Returns:
        loss: scalar
    """
    if dist_head is None:
        dist_head = DistanceHead(pred_logits.size(-1))
        dist_head = dist_head.to(pred_logits.device)
    
    target_bins = dist_head.distance_to_bins(target_dist)  # [B, L, L]
    
    # Flatten
    pred_flat = pred_logits.reshape(-1, pred_logits.size(-1))  # [B*L*L, n_bins]
    target_flat = target_bins.reshape(-1)  # [B*L*L]
    
    if mask is not None:
        mask_flat = mask.reshape(-1).bool()
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]
    
    return F.cross_entropy(pred_flat, target_flat)


def compute_contact_loss(
    pred_logits: torch.Tensor,
    target_contacts: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    interface_weight: float = 10.0,
) -> torch.Tensor:
    """
    Contact prediction loss (binary cross-entropy with interface weighting).
    
    Args:
        pred_logits: [B, L, L] predicted contact logits
        target_contacts: [B, L, L] target contact map (binary)
        mask: [B, L, L] valid pair mask
        interface_weight: Extra weight for interface (contact) pairs
    
    Returns:
        loss: scalar
    """
    # Per-element loss
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, target_contacts, reduction='none'
    )
    
    # Interface weighting: give more weight to actual contacts
    weights = 1.0 + (interface_weight - 1.0) * target_contacts
    loss = loss * weights
    
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return loss.mean()


def compute_energy_loss(
    pred_energy: torch.Tensor,
    target_energy: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Energy prediction loss (MSE).
    
    Args:
        pred_energy: [B] predicted energies
        target_energy: [B] target EvoEF2 energies
        normalize: Whether to normalize targets by std (for stability)
    
    Returns:
        loss: scalar
    """
    if normalize:
        # Normalize by batch std for more stable training
        std = target_energy.std() + 1e-8
        return F.mse_loss(pred_energy / std, target_energy / std)
    else:
        return F.mse_loss(pred_energy, target_energy)


def compute_structure_loss(
    pred: Dict[str, torch.Tensor],
    target_dist: torch.Tensor,
    target_contacts: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dist_head: Optional[DistanceHead] = None,
    dist_weight: float = 1.0,
    contact_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Combined structure prediction loss for Phase 3A.
    
    Args:
        pred: Model output dict with 'dist_logits' and 'contact_logits'
        target_dist: [B, L, L] target distance matrix
        target_contacts: [B, L, L] target contact map
        mask: [B, L, L] valid pair mask
        dist_head: DistanceHead for bin conversion
        dist_weight: Weight for distance loss
        contact_weight: Weight for contact loss
    
    Returns:
        dict with 'loss', 'loss_dist', 'loss_contact'
    """
    loss_dist = compute_distance_loss(
        pred['dist_logits'], target_dist, mask, dist_head
    )
    loss_contact = compute_contact_loss(
        pred['contact_logits'], target_contacts, mask
    )
    
    total_loss = dist_weight * loss_dist + contact_weight * loss_contact
    
    return {
        'loss': total_loss,
        'loss_dist': loss_dist,
        'loss_contact': loss_contact,
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing TCRFoldProphet")
    print("=" * 60)
    
    # Test with dummy data
    B, L = 2, 50
    vocab_size = 22
    
    # Create model
    model = TCRFoldProphet(
        s_dim=128,  # Smaller for testing
        z_dim=64,
        n_layers=2,
    )
    print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dummy inputs
    seq_tokens = torch.randint(0, 20, (B, L))
    pair_type = torch.zeros(B, L, L, dtype=torch.long)
    pair_type[:, :L//2, :L//2] = 0  # intra-A
    pair_type[:, L//2:, L//2:] = 1  # intra-B
    pair_type[:, :L//2, L//2:] = 2  # inter-AB
    pair_type[:, L//2:, :L//2] = 2  # inter-BA
    
    # Forward pass
    print("\nRunning forward pass...")
    outputs = model(seq_tokens, pair_type)
    
    print(f"\nOutput shapes:")
    print(f"  s: {outputs['s'].shape}")
    print(f"  z: {outputs['z'].shape}")
    print(f"  dist_logits: {outputs['dist_logits'].shape}")
    print(f"  contact_logits: {outputs['contact_logits'].shape}")
    print(f"  energy: {outputs['energy'].shape}")
    
    # Test loss computation
    target_dist = torch.rand(B, L, L) * 20 + 2  # 2-22 Å
    target_contacts = (target_dist < 8).float()
    
    losses = compute_structure_loss(
        outputs, target_dist, target_contacts, dist_head=model.dist_head
    )
    print(f"\nLosses:")
    print(f"  loss_dist: {losses['loss_dist'].item():.4f}")
    print(f"  loss_contact: {losses['loss_contact'].item():.4f}")
    print(f"  total: {losses['loss'].item():.4f}")
    
    # Test energy loss
    target_energy = torch.randn(B) * 50 - 100
    loss_energy = compute_energy_loss(outputs['energy'], target_energy)
    print(f"  loss_energy: {loss_energy.item():.4f}")
    
    print("\n✅ TCRFoldProphet test passed!")

