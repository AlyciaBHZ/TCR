"""
FlowTCR-Gen Encoder: Adapts psi_model components for topology-aware conditioning.

Key components:
- CollapseAwareEmbedding: Collapse Token (ψ) + Hierarchical Pair IDs
- SequenceProfileEvoformer: Evoformer with sequence profile attention

The encoder produces rich conditioning embeddings for the flow matching model.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(tensor: Optional[torch.Tensor] = None) -> torch.device:
    """Get device from tensor or default to CUDA if available."""
    if tensor is not None:
        return tensor.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_d_positional_encoding(idx: torch.Tensor, dim: int, max_len: int = 2048) -> torch.Tensor:
    """Sinusoidal positional encoding."""
    if len(idx) == 0:
        return torch.zeros((0, dim), device=idx.device)
    
    idx = idx.long()
    device = idx.device
    max_idx = min(max_len, idx.max().item() + 1)
    
    emb = torch.zeros((max_idx, dim), device=device)
    positions = torch.arange(max_idx, device=device).float()
    
    for i in range(dim):
        if i % 2 == 0:
            emb[:, i] = torch.sin(positions / (10000 ** (i / dim)))
        else:
            emb[:, i] = torch.cos(positions / (10000 ** (i / dim)))
    
    return emb[idx]


class CollapseAwareEmbedding(nn.Module):
    """
    Embedding layer with Collapse Token (ψ) and Hierarchical Pair IDs.
    
    Key features:
    1. Collapse Token: Learnable global observer token
    2. Hierarchical Pair IDs: 7-level topology encoding for attention bias
    3. Region-specific adaptive weights
    4. x_t injection: Inject flow intermediate state via soft embedding
    """

    def __init__(
        self,
        s_in_dim: int = 21,
        s_dim: int = 256,
        z_dim: int = 64,
        max_len: int = 512,
        use_collapse: bool = True,  # Ablation switch
        use_hier_pairs: bool = True,  # Ablation switch
    ):
        super().__init__()
        self.s_in_dim = s_in_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.max_len = max_len
        self.use_collapse = use_collapse
        self.use_hier_pairs = use_hier_pairs

        # Sequence projection (for one-hot or soft distributions)
        self.seq_proj = nn.Linear(s_in_dim, s_dim)
        
        # Pair embeddings (2-level hierarchy)
        self.pair_embed_lvl1 = nn.Linear(8, z_dim // 2)
        self.pair_embed_lvl2 = nn.Linear(4, z_dim // 2)
        
        # Positional embedding
        self.pos_embed = nn.Linear(64, s_dim)

        # Collapse token (learnable global observer)
        if use_collapse:
            self.collapse_token = nn.Parameter(torch.randn(1, s_dim) * 0.1)
            self.collapse_weight = nn.Parameter(torch.ones(1))
        
        # Region-specific adaptive weights
        self.region_weights = nn.ParameterDict({
            'cdr3': nn.Parameter(torch.ones(2)),     # [seq_weight, pos_weight]
            'pep': nn.Parameter(torch.ones(2)),
            'mhc': nn.Parameter(torch.ones(2)),
            'hv': nn.Parameter(torch.ones(2)),
            'hj': nn.Parameter(torch.ones(2)),
            'lv': nn.Parameter(torch.ones(2)),
            'lj': nn.Parameter(torch.ones(2)),
        })

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for linear layers."""
        for module in [self.seq_proj, self.pair_embed_lvl1, self.pair_embed_lvl2, self.pos_embed]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def create_hierarchical_pairs(self, L: int, idx_map: Dict[str, Tuple[int, int]], device: torch.device) -> torch.Tensor:
        """
        Create hierarchical pair IDs for topology-aware attention.
        
        7-level hierarchy:
        - Level 0: ψ ↔ ψ (collapse self)
        - Level 1: ψ ↔ all regions
        - Level 2: CDR3 sequential neighbors
        - Level 3: CDR3 internal (non-sequential)
        - Level 4: CDR3 ↔ conditioning regions
        - Level 5: Conditioning region internal
        - Level 6+: Cross conditioning regions
        """
        if not self.use_hier_pairs:
            return torch.zeros((L, L), dtype=torch.long, device=device)
        
        pair_id = torch.zeros((L, L), dtype=torch.long, device=device)
        
        # Find region boundaries
        collapse_end = 1 if self.use_collapse else 0
        cdr3_range = idx_map.get('cdr3', (collapse_end, collapse_end))
        cdr3_start, cdr3_end = cdr3_range
        
        # Level 0: Collapse self-reference
        if self.use_collapse:
            pair_id[0, 0] = 0
            # Level 1: Collapse ↔ all other regions
            pair_id[0, 1:] = 1
            pair_id[1:, 0] = 1
        
        # Level 2: CDR3 sequential neighbors
        if cdr3_end > cdr3_start:
            for i in range(cdr3_start, cdr3_end - 1):
                pair_id[i, i + 1] = 2
                pair_id[i + 1, i] = 2
        
        # Level 3: CDR3 internal (non-sequential)
        for i in range(cdr3_start, cdr3_end):
            for j in range(cdr3_start, cdr3_end):
                if i != j and pair_id[i, j] == 0:
                    pair_id[i, j] = 3
        
        # Level 4: CDR3 ↔ conditioning regions
        cond_regions = [k for k in idx_map.keys() if k != 'cdr3']
        for i in range(cdr3_start, cdr3_end):
            for region in cond_regions:
                r_start, r_end = idx_map[region]
                pair_id[i, r_start:r_end] = 4
                pair_id[r_start:r_end, i] = 4
        
        # Level 5+: Conditioning regions internal and cross
        counter = 5
        for region in cond_regions:
            r_start, r_end = idx_map[region]
            pair_id[r_start:r_end, r_start:r_end] = counter
            counter += 1
        
        # Cross conditioning regions
        for i, region1 in enumerate(cond_regions):
            r1_start, r1_end = idx_map[region1]
            for region2 in cond_regions[i + 1:]:
                r2_start, r2_end = idx_map[region2]
                pair_id[r1_start:r1_end, r2_start:r2_end] = counter
                pair_id[r2_start:r2_end, r1_start:r1_end] = counter
                counter += 1
        
        return pair_id.clamp(max=31)

    def forward_single(
        self,
        cdr3_xt: torch.Tensor,           # [L_cdr3, vocab]
        cdr3_mask: torch.Tensor,         # [L_cdr3]
        conditioning_seqs: Dict[str, torch.Tensor],  # region -> [L_region, vocab]
        conditioning_masks: Dict[str, torch.Tensor], # region -> [L_region]
        conditioning_info: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[int, int]]]:
        """
        Forward pass for a single sample.
        """
        s_list = []
        idx_map: Dict[str, Tuple[int, int]] = {}
        offset = 0
        
        # 1. Add collapse token
        if self.use_collapse:
            collapse_emb = self.collapse_weight * self.collapse_token  # [1, s_dim]
            s_list.append(collapse_emb)
            offset = 1
        
        # 2. Add CDR3 (only valid positions, masked by cdr3_mask)
        L_cdr3 = int(cdr3_mask.sum().item())
        if L_cdr3 > 0:
            cdr3_valid = cdr3_xt[:L_cdr3]  # [L_cdr3_valid, vocab]
            cdr3_idx = torch.arange(L_cdr3, device=device)
            
            seq_emb = self.seq_proj(cdr3_valid)
            pos_emb = self.pos_embed(one_d_positional_encoding(cdr3_idx, 64))
            
            region_w = self.region_weights['cdr3']
            cdr3_emb = region_w[0] * seq_emb + region_w[1] * pos_emb
            
            s_list.append(cdr3_emb)
            idx_map['cdr3'] = (offset, offset + L_cdr3)
            offset += L_cdr3
        else:
            idx_map['cdr3'] = (offset, offset)
        
        # 3. Add conditioning regions (only valid positions)
        for region in conditioning_info:
            if region not in conditioning_seqs:
                continue
            
            seq = conditioning_seqs[region]
            mask = conditioning_masks.get(region)
            
            if mask is None or mask.sum() == 0:
                continue
            
            L_region = int(mask.sum().item())
            seq_valid = seq[:L_region]
            idx = torch.arange(L_region, device=device)
            
            seq_emb = self.seq_proj(seq_valid)
            pos_emb = self.pos_embed(one_d_positional_encoding(idx, 64))
            
            if region in self.region_weights:
                region_w = self.region_weights[region]
                region_emb = region_w[0] * seq_emb + region_w[1] * pos_emb
            else:
                region_emb = seq_emb + pos_emb
            
            s_list.append(region_emb)
            idx_map[region] = (offset, offset + L_region)
            offset += L_region
        
        # Concatenate all embeddings
        if len(s_list) == 0:
            s = torch.zeros(1, self.s_dim, device=device)
        else:
            s = torch.cat(s_list, dim=0)  # [L_total, s_dim]
        
        L_total = s.shape[0]
        
        # Create pair embeddings
        pair_id = self.create_hierarchical_pairs(L_total, idx_map, device)
        z = torch.cat([
            self.pair_embed_lvl1(F.one_hot(pair_id // 4, 8).float()),
            self.pair_embed_lvl2(F.one_hot(pair_id % 4, 4).float())
        ], dim=-1)  # [L, L, z_dim]
        
        return s, z, idx_map

    def forward(
        self,
        cdr3_xt: torch.Tensor,           # [B, L_cdr3, vocab]
        cdr3_mask: torch.Tensor,         # [B, L_cdr3]
        conditioning_seqs: Dict[str, torch.Tensor],  # region -> [B, L_region, vocab]
        conditioning_masks: Dict[str, torch.Tensor], # region -> [B, L_region]
        conditioning_info: List[str],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Tuple[int, int]]]]:
        """
        Batched forward pass with per-sample conditioning.
        Returns lists because sequence lengths vary per sample.
        """
        device = cdr3_xt.device
        B = cdr3_xt.shape[0]
        
        s_list: List[torch.Tensor] = []
        z_list: List[torch.Tensor] = []
        idx_maps: List[Dict[str, Tuple[int, int]]] = []
        
        for i in range(B):
            sample_cond_seqs = {k: v[i] for k, v in conditioning_seqs.items()}
            sample_cond_masks = {k: v[i] for k, v in conditioning_masks.items()}
            
            s, z, idx_map = self.forward_single(
                cdr3_xt=cdr3_xt[i],
                cdr3_mask=cdr3_mask[i],
                conditioning_seqs=sample_cond_seqs,
                conditioning_masks=sample_cond_masks,
                conditioning_info=conditioning_info,
                device=device,
            )
            
            s_list.append(s)
            z_list.append(z)
            idx_maps.append(idx_map)
        
        return s_list, z_list, idx_maps


class SequenceProfileAttention(nn.Module):
    """
    Attention block with sequence profile guidance.
    
    Uses position entropy to guide collapse token attention:
    - Low entropy positions (confident predictions) get higher attention
    - Supports ablation via use_collapse flag
    """

    def __init__(self, s_dim: int, z_dim: int, num_heads: int = 4, use_collapse: bool = True):
        super().__init__()
        self.s_dim = s_dim
        self.num_heads = num_heads
        self.head_dim = s_dim // num_heads
        self.use_collapse = use_collapse
        
        # Q, K, V projections
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_proj = nn.Linear(s_dim, s_dim)
        self.out_proj = nn.Linear(s_dim, s_dim)
        
        # Sequence profile prediction (for entropy guidance)
        self.profile_head = nn.Linear(s_dim, 20)
        
        # Collapse position bias
        if use_collapse:
            self.collapse_position_bias = nn.Parameter(torch.zeros(512))
            self.entropy_weight = nn.Parameter(torch.ones(1) * 0.1)
            nn.init.normal_(self.collapse_position_bias, mean=0.0, std=0.3)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim * 4),
            nn.ReLU(),
            nn.Linear(s_dim * 4, s_dim)
        )

    def compute_profile_entropy(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sequence profile entropy per position."""
        profile_logits = self.profile_head(hidden)  # [L, 20]
        profile_probs = F.softmax(profile_logits, dim=-1)
        position_entropy = -(profile_probs * torch.log(profile_probs + 1e-8)).sum(dim=-1)  # [L]
        return position_entropy, profile_probs

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Forward attention with optional profile guidance.
        
        Args:
            s: [L, s_dim] sequence representation
            z: [L, L, z_dim] pair representation (unused in this lightweight version)
        """
        L, D = s.shape
        s_ln = F.layer_norm(s, (D,))
        
        # Compute profile entropy
        position_entropy, profile_probs = self.compute_profile_entropy(s_ln)
        
        # Compute Q, K, V
        q = self.q_proj(s_ln).view(L, self.num_heads, self.head_dim).transpose(0, 1)  # [H, L, head_dim]
        k = self.k_proj(s_ln).view(L, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(s_ln).view(L, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [H, L, L]
        
        # Apply entropy-guided bias for collapse token
        entropy_guidance = None
        if self.use_collapse and L > 1:
            # Low entropy = high attention weight
            entropy_guidance = -position_entropy[:L] * self.entropy_weight
            # Extend position bias if sequence exceeds preset length
            bias_len = self.collapse_position_bias.shape[0]
            if L > bias_len:
                extra = self.collapse_position_bias.new_zeros(L - bias_len)
                position_bias = torch.cat([self.collapse_position_bias, extra], dim=0)
            else:
                position_bias = self.collapse_position_bias[:L]
            total_bias = entropy_guidance + position_bias
            scores[:, 0, :L] = scores[:, 0, :L] + total_bias.unsqueeze(0)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # [H, L, L]
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [H, L, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(L, D)  # [L, D]
        s_out = self.out_proj(attn_output)
        
        # Residual + FFN
        s = s + s_out
        s = s + self.ffn(s)
        
        profile_info = {
            'position_entropy': position_entropy,
            'profile_probs': profile_probs,
            'entropy_guidance': entropy_guidance,
        }
        
        return s, z, attn_weights[0], profile_info  # Return first head's weights


class SequenceProfileEvoformer(nn.Module):
    """
    Evoformer stack with sequence profile attention.
    
    A lightweight alternative to full Evoformer that focuses on
    collapse token dynamics and sequence profile prediction.
    """

    def __init__(self, s_dim: int, z_dim: int, n_layers: int = 6, use_collapse: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            SequenceProfileAttention(s_dim, z_dim, use_collapse=use_collapse)
            for _ in range(n_layers)
        ])
        self.log_attn: List[torch.Tensor] = []
        self.log_profile_info: List[Dict] = []

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through all layers."""
        self.log_attn = []
        self.log_profile_info = []
        
        for layer in self.layers:
            s, z, attn, profile_info = layer(s, z)
            self.log_attn.append(attn)
            self.log_profile_info.append(profile_info)
        
        return s, z


class FlowTCRGenEncoder(nn.Module):
    """
    Main encoder for FlowTCR-Gen (per-sample conditioning).
    Combines CollapseAwareEmbedding and SequenceProfileEvoformer.
    """

    def __init__(
        self,
        s_dim: int = 256,
        z_dim: int = 64,
        n_layers: int = 6,
        vocab_size: int = 21,
        max_len: int = 512,
        use_collapse: bool = True,
        use_hier_pairs: bool = True,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.vocab_size = vocab_size
        self.use_collapse = use_collapse
        self.use_hier_pairs = use_hier_pairs
        
        # Embedding layer with collapse token
        self.embedding = CollapseAwareEmbedding(
            s_in_dim=vocab_size,
            s_dim=s_dim,
            z_dim=z_dim,
            max_len=max_len,
            use_collapse=use_collapse,
            use_hier_pairs=use_hier_pairs,
        )
        
        # Evoformer backbone
        self.backbone = SequenceProfileEvoformer(
            s_dim=s_dim,
            z_dim=z_dim,
            n_layers=n_layers,
            use_collapse=use_collapse,
        )
        
        # Time embedding for flow matching
        self.time_embed = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.SiLU(),
            nn.Linear(s_dim, s_dim),
        )
        
        # Collapse scalar extractor (for model score hook)
        if use_collapse:
            self.collapse_proj = nn.Linear(s_dim, 1)

    def get_sinusoidal_time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal time embedding."""
        device = t.device
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1))
        )
        args = t.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(
        self,
        cdr3_xt: torch.Tensor,
        cdr3_mask: torch.Tensor,
        t: torch.Tensor,
        pep_one_hot: torch.Tensor,
        pep_mask: torch.Tensor,
        mhc_one_hot: torch.Tensor,
        mhc_mask: torch.Tensor,
        scaffold_one_hot: Dict[str, torch.Tensor],
        scaffold_mask: Dict[str, torch.Tensor],
        conditioning_info: Optional[List[str]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Tuple[int, int]]]]:
        """
        Encode all inputs for flow matching with per-sample conditioning.
        Returns lists because sequence lengths vary per sample.
        """
        if conditioning_info is None:
            conditioning_info = ['pep', 'mhc', 'hv', 'hj', 'lv', 'lj']
        
        device = cdr3_xt.device
        B = cdr3_xt.shape[0]
        
        # Build conditioning dicts (batched)
        conditioning_seqs: Dict[str, torch.Tensor] = {}
        conditioning_masks: Dict[str, torch.Tensor] = {}
        
        if 'pep' in conditioning_info:
            conditioning_seqs['pep'] = pep_one_hot
            conditioning_masks['pep'] = pep_mask
        
        if 'mhc' in conditioning_info:
            conditioning_seqs['mhc'] = mhc_one_hot
            conditioning_masks['mhc'] = mhc_mask
        
        for key in ['hv', 'hj', 'lv', 'lj']:
            if key in conditioning_info and key in scaffold_one_hot:
                conditioning_seqs[key] = scaffold_one_hot[key]
                conditioning_masks[key] = scaffold_mask[key]
        
        # Embedding (returns lists for variable-length outputs)
        s_list, z_list, idx_maps = self.embedding(
            cdr3_xt=cdr3_xt,
            cdr3_mask=cdr3_mask,
            conditioning_seqs=conditioning_seqs,
            conditioning_masks=conditioning_masks,
            conditioning_info=list(conditioning_seqs.keys()),
        )
        
        # Add time embedding and run Evoformer for each sample
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        processed_s_list: List[torch.Tensor] = []
        processed_z_list: List[torch.Tensor] = []
        
        for i in range(B):
            s = s_list[i]  # [L_i, s_dim]
            z = z_list[i]  # [L_i, L_i, z_dim]
            
            # Add time embedding
            time_emb = self.get_sinusoidal_time_embedding(t[i], self.s_dim)  # [1, s_dim]
            time_proj = self.time_embed(time_emb)  # [1, s_dim]
            s = s + time_proj  # Broadcast to all positions
            
            # Evoformer
            s, z = self.backbone(s, z)
            
            processed_s_list.append(s)
            processed_z_list.append(z)
        
        return processed_s_list, processed_z_list, idx_maps

    def get_collapse_scalar(self, s: torch.Tensor) -> torch.Tensor:
        """Extract collapse scalar for model score hook."""
        if not self.use_collapse:
            return torch.tensor(0.0, device=s.device)
        collapse_repr = s[0]  # [s_dim]
        return self.collapse_proj(collapse_repr).squeeze()

    def get_cdr3_representation(self, s: torch.Tensor, idx_map: Dict[str, Tuple[int, int]]) -> torch.Tensor:
        """Extract CDR3 region representation."""
        cdr3_start, cdr3_end = idx_map['cdr3']
        return s[cdr3_start:cdr3_end]  # [L_cdr3, s_dim]
