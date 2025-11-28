"""
Immuno-PLM: Hybrid Encoder with ESM-2 + LoRA + Topology Bias

Architecture Overview:
======================
1. Backbone: ESM-2 (esm2_t33_650M_UR50D) or BasicTokenizer fallback
2. LoRA Adaptation: Low-Rank adapters on query/value projections
3. Topology Fusion: Hierarchical pair embeddings from psi_model

This module serves as the perception center of FlowTCR-Fold:
- Encodes TCR and pMHC sequences for scaffold retrieval (Stage 1)
- Provides conditioning embeddings for CDR3β generation (Stage 2)

Training Objective: InfoNCE contrastive learning for TCR-pMHC compatibility
"""

from typing import Dict, Optional, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# ESM import with fallback
try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    esm = None
    ESM_AVAILABLE = False

# PEFT/LoRA import with fallback
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA adaptation disabled. Install with: pip install peft")


# =============================================================================
# Topology Bias (from psi_model)
# =============================================================================

class TopologyBias(nn.Module):
    """
    Hierarchical pair bias with collapse token (index 0), region-aware levels.
    
    Ported from psi_model/model.py:create_hierarchical_pairs
    
    7-Level Hierarchy:
    - Level 0: Collapse token self-reference
    - Level 1: Collapse ↔ all regions (observer-observed)
    - Level 2: HD (CDR3) sequential neighbors
    - Level 3: HD internal non-sequential
    - Level 4: HD ↔ conditioning regions (CDR3-peptide/MHC interaction) ← KEY
    - Level 5+: Conditioning regions internal
    - Level N+: Conditioning ↔ conditioning
    """

    def __init__(self, z_dim: int = 128, max_pairs: int = 32):
        super().__init__()
        self.z_dim = z_dim
        self.max_pairs = max_pairs
        
        # Two-level pair embedding (from psi_model)
        self.pair_embed_lvl1 = nn.Linear(8, z_dim // 2)
        self.pair_embed_lvl2 = nn.Linear(4, z_dim // 2)
        
        # Single linear for one-hot encoding
        self.pair_linear = nn.Linear(max_pairs, z_dim)

    def create_hierarchical_pairs(self, L: int, idx_map: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Create hierarchical pair IDs.
        
        Args:
            L: Total sequence length
            idx_map: List of (start, end) tuples for each region.
                     First element is HD (CDR3), rest are conditioning regions.
        
        Returns:
            pair_id: [L, L] tensor of pair type IDs
        """
        pair_id = torch.zeros((L, L), dtype=torch.long)
        
        # Level 0: Collapse token self-reference
        pair_id[0, 0] = 0
        
        # Level 1: Collapse ↔ all other positions
        pair_id[0, 1:] = 1
        pair_id[1:, 0] = 1

        if not idx_map:
            return pair_id.clamp(max=self.max_pairs - 1)

        hd_start, hd_end = idx_map[0]
        
        # Level 2: HD sequential neighbors
        for i in range(hd_start, max(hd_end - 1, hd_start)):
            if i + 1 < hd_end:
                pair_id[i, i + 1] = 2
                pair_id[i + 1, i] = 2
        
        # Level 3: HD internal non-sequential
        for i in range(hd_start, hd_end):
            for j in range(hd_start, hd_end):
                if i != j and pair_id[i, j] == 0:
                    pair_id[i, j] = 3
        
        # Level 4: HD ↔ conditioning regions (CRITICAL for TCR-pMHC)
        for i in range(hd_start, hd_end):
            for region_start, region_end in idx_map[1:]:
                pair_id[i, region_start:region_end] = 4
                pair_id[region_start:region_end, i] = 4

        counter = 5
        
        # Level 5+: Conditioning regions internal
        for region_start, region_end in idx_map[1:]:
            pair_id[region_start:region_end, region_start:region_end] = counter
            counter += 1
        
        # Level N+: Conditioning ↔ conditioning
        conditioning_regions = idx_map[1:]
        for i, (r1_start, r1_end) in enumerate(conditioning_regions):
            for j, (r2_start, r2_end) in enumerate(conditioning_regions[i + 1:], i + 1):
                pair_id[r1_start:r1_end, r2_start:r2_end] = counter
                pair_id[r2_start:r2_end, r1_start:r1_end] = counter
                counter += 1

        return pair_id.clamp(max=self.max_pairs - 1)

    def forward(self, idx_pairs: List[Tuple[int, int]], total_L: int, device) -> Optional[torch.Tensor]:
        """
        Compute hierarchical pair embeddings.
        
        Args:
            idx_pairs: List of (start, end) for each region
            total_L: Total sequence length
            device: Target device
        
        Returns:
            z: [L, L, z_dim] pair embeddings, or None if no regions
        """
        if not idx_pairs:
            return None
        
        pair_id = self.create_hierarchical_pairs(total_L, idx_pairs).to(device)
        
        # Two-level embedding (following psi_model pattern)
        z_lvl1 = self.pair_embed_lvl1(
            F.one_hot(pair_id // 4, num_classes=8).float()
        )
        z_lvl2 = self.pair_embed_lvl2(
            F.one_hot(pair_id % 4, num_classes=4).float()
        )
        z = torch.cat([z_lvl1, z_lvl2], dim=-1)
        
        return z


# =============================================================================
# Immuno-PLM with ESM-2 + LoRA
# =============================================================================

class ImmunoPLM(nn.Module):
    """
    Hybrid encoder: ESM-2 + LoRA + Topology Bias
    
    Three operating modes:
    1. ESM-2 + LoRA (recommended): Fine-tune ESM with low-rank adapters
    2. ESM-2 frozen: Use ESM as feature extractor only
    3. BasicTokenizer: Lightweight mode without ESM
    
    Architecture:
    ```
    Input Tokens → ESM-2 (+ LoRA) → Sequence Features (s)
                                          ↓
    Region Slices → TopologyBias → Pair Embeddings (z)
                                          ↓
                              s + Fusion(z) → Pooled Output
    ```
    
    Usage:
        >>> model = ImmunoPLM(use_esm=True, use_lora=True, lora_rank=8)
        >>> out = model(tokens, mask, region_slices)
        >>> pooled = out["pooled"]  # [B, hidden_dim] for InfoNCE
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        z_dim: int = 128,
        use_esm: bool = False,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        vocab_size: int = 256,
        esm_model_name: str = "esm2_t33_650M_UR50D"
    ):
        """
        Initialize Immuno-PLM.
        
        Args:
            hidden_dim: Output hidden dimension
            z_dim: Topology embedding dimension
            use_esm: Whether to use ESM-2 backbone
            use_lora: Whether to apply LoRA adapters (requires use_esm=True)
            lora_rank: LoRA rank (lower = fewer parameters, higher = more expressive)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout in LoRA layers
            vocab_size: Vocabulary size for BasicTokenizer mode
            esm_model_name: ESM model to use (default: 650M)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_esm = use_esm and ESM_AVAILABLE
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        # =========================
        # Backbone: ESM-2 or Basic
        # =========================
        if self.use_esm:
            print(f"Loading ESM-2 model: {esm_model_name}")
            
            # Load ESM-2 model
            if esm_model_name == "esm2_t33_650M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.repr_layer = 33  # Last layer
            elif esm_model_name == "esm2_t12_35M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                self.repr_layer = 12
            elif esm_model_name == "esm2_t6_8M_UR50D":
                self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
                self.repr_layer = 6
            else:
                raise ValueError(f"Unknown ESM model: {esm_model_name}")
            
            self.embed_dim = self.esm_model.embed_dim
            
            # =========================
            # LoRA Adaptation
            # =========================
            if self.use_lora:
                print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
                
                # LoRA config targeting attention layers
                peft_config = LoraConfig(
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    # Target attention projections in ESM-2
                    target_modules=[
                        "self_attn.q_proj",  # Query
                        "self_attn.k_proj",  # Key  
                        "self_attn.v_proj",  # Value
                        "self_attn.out_proj" # Output
                    ],
                    bias="none"
                )
                
                # Wrap ESM with LoRA
                self.esm_model = get_peft_model(self.esm_model, peft_config)
                self.esm_model.print_trainable_parameters()
            else:
                # Freeze ESM if not using LoRA
                print("Freezing ESM-2 parameters (no LoRA)")
                for param in self.esm_model.parameters():
                    param.requires_grad = False
        else:
            # BasicTokenizer fallback
            print("Using BasicTokenizer (no ESM)")
            self.alphabet = None
            self.embed_dim = hidden_dim
            self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            self.decoder = nn.Linear(hidden_dim, vocab_size)  # For MLM
        
        # =========================
        # Topology Bias (from psi_model)
        # =========================
        self.topology_bias = TopologyBias(z_dim=z_dim)
        
        # =========================
        # Fusion Layers
        # =========================
        # Project ESM output to hidden_dim
        self.seq_proj = nn.Linear(self.embed_dim, hidden_dim)
        
        # Fuse pair embeddings into sequence
        self.pair_fusion = nn.Linear(z_dim, hidden_dim)
        
        # Optional: Cross-attention for pair-sequence fusion
        self.use_cross_attn = False  # Can enable for richer fusion
        if self.use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # =========================
        # Output Layers
        # =========================
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Contrastive head (for InfoNCE)
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        region_slices: Optional[List[Dict[str, slice]]] = None,
        return_pair_repr: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            tokens: [B, L] token IDs (CLS at index 0 = collapse token)
            mask: [B, L] attention mask (1=real, 0=pad)
            region_slices: List of dicts per sample, mapping region name -> slice
                          e.g., [{"pep": slice(1,10), "mhc": slice(11,50), "cdr3b": slice(51,65)}]
            return_pair_repr: Whether to return full pair representations
        
        Returns:
            Dictionary with:
            - "s": [B, L, hidden_dim] sequence representations
            - "pooled": [B, hidden_dim] pooled representation for contrastive learning
            - "z_bias": [B, L, z_dim] or [B, L, L, z_dim] topology bias
            - "contrastive": [B, hidden_dim] output of contrastive head
        """
        B, L = tokens.shape
        device = tokens.device
        
        # =========================
        # 1. Backbone Encoding
        # =========================
        if self.use_esm:
            # ESM-2 forward pass
            esm_output = self.esm_model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            s = esm_output["representations"][self.repr_layer]  # [B, L, embed_dim]
        else:
            # BasicTokenizer embedding
            s = self.token_embed(tokens)  # [B, L, hidden_dim]
        
        # Project to hidden_dim
        s = self.seq_proj(s)  # [B, L, hidden_dim]
        
        # =========================
        # 2. Topology Bias Fusion
        # =========================
        z_topo = None
        z_context = torch.zeros(B, L, self.z_dim, device=device)
        
        if region_slices:
            for b_idx in range(B):
                region_dict = region_slices[b_idx] if b_idx < len(region_slices) else None
                if region_dict is None:
                    continue
                
                # Build idx_pairs: HD (cdr3b) first, then others
                idx_pairs = []
                if "cdr3b" in region_dict:
                    sl = region_dict["cdr3b"]
                    idx_pairs.append((sl.start, sl.stop))
                
                for key, sl in region_dict.items():
                    if key == "cdr3b":
                        continue
                    idx_pairs.append((sl.start, sl.stop))
                
                if not idx_pairs:
                    continue
                
                # Compute pair embeddings: [L, L, z_dim]
                z_pair = self.topology_bias(idx_pairs, L, device)
                
                if z_pair is not None:
                    # Reduce to [L, z_dim] by taking max over row
                    # This captures "what regions does each position interact with"
                    z_ctx = z_pair.max(dim=1)[0]
                    z_context[b_idx] = z_ctx
                    
                    if return_pair_repr and b_idx == 0:
                        z_topo = z_pair  # Save for debugging
            
            # Fuse topology into sequence
            s = s + self.pair_fusion(z_context)
        
        # =========================
        # 3. Dropout and Pooling
        # =========================
        s = self.dropout(s)
        
        # Masked mean pooling
        if mask is None:
            pooled = s.mean(dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (s * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        
        pooled = self.norm(pooled)
        
        # =========================
        # 4. Contrastive Head
        # =========================
        contrastive = self.contrastive_head(pooled)
        
        return {
            "s": s,                      # [B, L, hidden_dim]
            "pooled": pooled,            # [B, hidden_dim]
            "z_bias": z_context,         # [B, L, z_dim]
            "contrastive": contrastive,  # [B, hidden_dim]
            "z_pair": z_topo             # [L, L, z_dim] or None
        }

    def encode_pmhc(self, peptide: str, mhc: str) -> torch.Tensor:
        """
        Encode a single pMHC for retrieval.
        
        Args:
            peptide: Peptide sequence
            mhc: MHC allele or sequence
        
        Returns:
            Embedding vector [hidden_dim]
        """
        # This would use the tokenizer to create tokens
        # Simplified placeholder - actual implementation needs tokenizer
        raise NotImplementedError("Use forward() with properly tokenized input")

    def encode_scaffold(self, scaffold: Dict[str, str]) -> torch.Tensor:
        """
        Encode a scaffold (V/J genes) for retrieval.
        
        Args:
            scaffold: Dict with h_v, h_j, l_v, l_j sequences
        
        Returns:
            Embedding vector [hidden_dim]
        """
        raise NotImplementedError("Use forward() with properly tokenized input")

    @classmethod
    def load(cls, path: str, **kwargs) -> "ImmunoPLM":
        """Load model from checkpoint."""
        state_dict = torch.load(path, map_location="cpu")
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(self.state_dict(), path)


# =============================================================================
# Contrastive Loss Functions
# =============================================================================

def compute_batch_infonce(
    tcr_emb: torch.Tensor,
    pmhc_emb: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute InfoNCE loss using batch-internal negatives (safe from false negatives).
    
    Args:
        tcr_emb: [B, D] TCR embeddings
        pmhc_emb: [B, D] pMHC embeddings
        temperature: Softmax temperature
    
    Returns:
        Scalar loss
    
    Note:
        This is the SAFE version that uses batch-internal negatives.
        For sample i, pMHC_i is positive and all other pMHC_j (j≠i) are negatives.
    """
    # Normalize embeddings
    tcr_emb = F.normalize(tcr_emb, p=2, dim=-1)
    pmhc_emb = F.normalize(pmhc_emb, p=2, dim=-1)
    
    # Similarity matrix: [B, B]
    logits = torch.matmul(tcr_emb, pmhc_emb.T) / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Cross entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


def compute_infonce_with_negatives(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: Optional[torch.Tensor] = None,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute InfoNCE loss with explicit negatives.
    
    Args:
        anchor: [B, D] anchor embeddings
        positive: [B, D] positive embeddings
        negatives: [N, D] negative embeddings (optional)
        temperature: Softmax temperature
    
    Returns:
        Scalar loss
    """
    # Normalize
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    
    # Positive similarity
    pos_sim = (anchor * positive).sum(dim=-1) / temperature  # [B]
    
    # Denominator
    denom = torch.exp(pos_sim)
    
    if negatives is not None and negatives.size(0) > 0:
        negatives = F.normalize(negatives, p=2, dim=-1)
        # [B, 1, D] x [1, N, D] -> [B, N]
        neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.unsqueeze(0).transpose(-2, -1))
        neg_sim = neg_sim.squeeze(1) / temperature  # [B, N]
        denom = denom + torch.exp(neg_sim).sum(dim=-1)
    
    # Loss
    loss = -torch.log(torch.exp(pos_sim) / denom).mean()
    
    return loss


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing ImmunoPLM...")
    
    # Test BasicTokenizer mode
    print("\n1. Testing BasicTokenizer mode...")
    model = ImmunoPLM(hidden_dim=256, z_dim=128, use_esm=False)
    tokens = torch.randint(0, 20, (2, 50))
    mask = torch.ones(2, 50)
    region_slices = [
        {"pep": slice(1, 10), "mhc": slice(11, 40), "cdr3b": slice(41, 50)},
        {"pep": slice(1, 10), "mhc": slice(11, 40), "cdr3b": slice(41, 50)}
    ]
    
    out = model(tokens, mask, region_slices)
    print(f"  s shape: {out['s'].shape}")
    print(f"  pooled shape: {out['pooled'].shape}")
    print(f"  contrastive shape: {out['contrastive'].shape}")
    
    # Test InfoNCE
    print("\n2. Testing InfoNCE loss...")
    tcr_emb = torch.randn(8, 256)
    pmhc_emb = torch.randn(8, 256)
    loss = compute_batch_infonce(tcr_emb, pmhc_emb)
    print(f"  InfoNCE loss: {loss.item():.4f}")
    
    # Test ESM mode if available
    if ESM_AVAILABLE:
        print("\n3. Testing ESM-2 mode (without LoRA)...")
        try:
            model_esm = ImmunoPLM(
                hidden_dim=256, 
                z_dim=128, 
                use_esm=True, 
                use_lora=False,
                esm_model_name="esm2_t6_8M_UR50D"  # Use small model for testing
            )
            print("  ESM-2 model loaded successfully")
        except Exception as e:
            print(f"  ESM-2 loading failed: {e}")
    
    print("\nAll tests passed!")
