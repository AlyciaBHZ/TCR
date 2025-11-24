# psiCLM Collapse-Aware Evoformer Integration (Standalone)
# This file serves as the updated version of model.py with no dependency on src/.

# --------------------------------------------
# psihē Theory Principle: psi = psi(psi)
# Collapse-aware language modeling: model structure should reflect and preserve the semantic origin of data through a central point of subjective collapse
# Academic Justification: Anchor tokens, position-dependent pair encodings, and explicit attention tracing are widely accepted tools in explainable protein modeling, and collapse-inspired conditioning strengthens directional signal in generative models.
# --------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

# Utility

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_d(idx_, d, max_len=2056):
    device = idx_.device if isinstance(idx_, torch.Tensor) else get_device()
    idx = idx_[None].to(device)
    K = torch.arange(d//2).to(idx)
    sin = torch.sin(idx[..., None] * math.pi / (max_len**(2*K[None]/d)))
    cos = torch.cos(idx[..., None] * math.pi / (max_len**(2*K[None]/d)))
    return torch.cat([sin, cos], axis=-1)[0]

def nll_loss_withmask(pred, native, mask):
    """
    计算masked NLL loss with detailed debugging
    pred: (L, vocab_size) log probabilities
    native: (L, vocab_size) one-hot targets  
    mask: (L, 1) binary mask
    """
    pred = pred.to(mask.device)
    native = native.to(mask.device)
    
    # 确保mask维度正确
    if mask.dim() == 1:
        mask = mask[:, None]
    
    # # 详细debug信息
    # if random.random() < 0.001:  # 1/1000概率打印debug
    #     print(f"Loss debug:")
    #     print(f"  pred shape: {pred.shape}, range: [{pred.min():.4f}, {pred.max():.4f}]")
    #     print(f"  native shape: {native.shape}, sum: {native.sum():.4f}")
    #     print(f"  mask shape: {mask.shape}, sum: {mask.sum():.4f}")
    #     print(f"  pred is log_softmax: {torch.allclose(pred.exp().sum(dim=-1), torch.ones(pred.shape[0], device=pred.device), atol=1e-3)}")
    
    # 计算逐点loss
    pointwise_loss = -(pred * native * mask).sum(dim=-1)  # (L,)
    total_loss = pointwise_loss.sum()
    mask_count = mask.sum()
    
    # 数值稳定性：避免除零
    if mask_count == 0:
        print(f"  CRITICAL ERROR: mask_count = 0! This should never happen with diverse masking!")
        print(f"  pred shape: {pred.shape}, native shape: {native.shape}, mask shape: {mask.shape}")
        print(f"  mask values: {mask.flatten()}")
        # 返回一个合理的loss而不是0
        return torch.tensor(5.0, device=pred.device, requires_grad=True)
    
    # 计算最终loss
    loss = total_loss / mask_count
    
    # 添加数值检查
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  CRITICAL ERROR: Invalid loss! loss={loss}, total_loss={total_loss}, mask_count={mask_count}")
        return torch.tensor(5.0, device=pred.device, requires_grad=True)
    
    # 临时debug信息 - 暂时关闭
    # if random.random() < 0.01:  # 1%概率打印
    #     print(f"Loss debug: total_loss={total_loss:.4f}, mask_count={mask_count:.0f}, final_loss={loss:.4f}")

    return loss

# Modules

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        return self.linear(x)

class CollapseAwareEmbedding(nn.Module):
    """
    Embeds each region (hd, mhc, pep, etc.) into s and z space.

    psi-Update: Adds a learnable collapse token representing the observer center psi = psi(psi)
    psi-Justification: Collapse occurs relative to observer structure; must exist as global anchor

    Academic Justification:
    - Analogous to [CLS] token or start-of-sequence in transformers
    - Enhances conditioning effectiveness and interpretability

    Structural Differences from AlphaFold2 MSA Embedding:
    - AlphaFold2's MSA input encodes many aligned sequences (N x L), while this embedding is per-region and per-condition.
    - Instead of extracting features from MSA rows/columns, psiCLM inserts a global "collapse anchor" as the first token.
    - Pairwise encoding uses hierarchical indices (intra/inter-region) rather than contact-based or outer product features.
    - Designed for generation and conditional sampling, not structure prediction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.s_in_dim = cfg['s_in_dim']
        self.z_in_dim = cfg['z_in_dim']
        self.s_dim = cfg['s_dim']
        self.z_dim = cfg['z_dim']

        self.seq_proj = Linear(self.s_in_dim + 1, self.s_dim)
        self.pair_embed_lvl1 = Linear(8, self.z_dim // 2)
        self.pair_embed_lvl2 = Linear(4, self.z_dim // 2)
        self.pos_embed_s = Linear(64, self.s_dim)

        self.collapse_token = nn.Parameter(torch.randn(1, self.s_dim))
        
        #  默认启用区域特定的自适应权重
        self.region_weights = nn.ParameterDict({
            'hd': nn.Parameter(torch.ones(2)),      # [seq_weight, pos_weight] for HD
            'mhc': nn.Parameter(torch.ones(2)),     # [seq_weight, pos_weight] for MHC  
            'pep': nn.Parameter(torch.ones(2)),     # [seq_weight, pos_weight] for PEP
            'lv': nn.Parameter(torch.ones(2)),      # [seq_weight, pos_weight] for LV
            'lj': nn.Parameter(torch.ones(2)),      # [seq_weight, pos_weight] for LJ
            'hv': nn.Parameter(torch.ones(2)),      # [seq_weight, pos_weight] for HV
            'hj': nn.Parameter(torch.ones(2)),      # [seq_weight, pos_weight] for HJ
        })
        self.collapse_weight = nn.Parameter(torch.ones(1))

    def forward(self, in_dict, conditioning_info):
        device = get_device()
        s_list, idx_map = [], []
        
        # Collapse token with learnable weight
        collapse_emb = self.collapse_weight * self.collapse_token
        s_list.append(collapse_emb)
        
        offset = 1
        for k in ['hd'] + [k for k in ['mhc','pep','lv','lj','hv','hj'] if k in conditioning_info]:
            if k in in_dict and in_dict[k].shape[0] > 0:
                aa = in_dict[k].to(device)
                
                # 关键修复：对HD使用真正的training mask，对conditioning序列使用零mask
                if k == 'hd':
                    mask = in_dict['mask'].to(device)  # 使用真正的training mask
                else:
                    mask = torch.zeros(aa.shape[0], device=device)  # conditioning序列不被mask
                
                aa = torch.cat([mask[:, None], aa], dim=-1)
                
                # 分别计算序列和位置编码
                seq_emb = self.seq_proj(aa)
                pos_emb = self.pos_embed_s(one_d(in_dict[f'{k}_idx'].to(device), 64))
                
                # 暂时禁用位置编码，减少信息泄露
                if k == 'hd':
                    # 对HD序列，暂时只用序列信息，不用位置信息
                    s = seq_emb
                else:
                    # 对conditioning序列，正常使用位置信息
                    if k in self.region_weights:
                        region_seq_w, region_pos_w = self.region_weights[k]
                        s = region_seq_w * seq_emb + region_pos_w * pos_emb
                    else:
                        s = seq_emb + pos_emb
                
                s_list.append(s)
                idx_map.append((offset, offset + s.shape[0]))
                offset += s.shape[0]
                
        s_out = torch.cat(s_list, dim=0)
        L = s_out.shape[0]

        # 使用改进的层次化pair embedding
        pair_id = self.create_hierarchical_pairs(L, idx_map, device)
        z = torch.cat([
            self.pair_embed_lvl1(F.one_hot(pair_id//4, 8).float()),
            self.pair_embed_lvl2(F.one_hot(pair_id%4, 4).float())
        ], dim=-1)

        return s_out, z

    def create_hierarchical_pairs(self, L, idx_map, device):
        """
        创建更合理的层次化pair embedding
        Hierarchy:
        0: collapse ↔ collapse (self-reference psi=psi(psi))
        1: collapse ↔ any region (observer-observed)
        2: HD internal sequential (i, i+1 neighbors)  
        3: HD internal non-sequential
        4: HD ↔ conditioning (target-context)
        5-N: conditioning internal (intra-region)
        N+1-M: conditioning ↔ conditioning (inter-conditioning)
        """
        pair_id = torch.zeros((L, L), dtype=torch.long, device=device)
        
        # 找到各区域边界
        collapse_end = 1
        hd_start, hd_end = idx_map[0] if idx_map else (1, 1)
        
        # Level 0: Collapse self-reference (psi=psi(psi))
        pair_id[0, 0] = 0
        
        # Level 1: Collapse ↔ all other regions
        pair_id[0, 1:] = 1
        pair_id[1:, 0] = 1
        
        # Level 2: HD sequential neighbors (重要的序列邻近性)
        if hd_end > hd_start:
            for i in range(hd_start, hd_end-1):
                pair_id[i, i+1] = 2
                pair_id[i+1, i] = 2
        
        # Level 3: HD internal non-sequential
        for i in range(hd_start, hd_end):
            for j in range(hd_start, hd_end):
                if i != j and pair_id[i, j] == 0:  # 非邻近的HD内部关系
                    pair_id[i, j] = 3
        
        # Level 4: HD ↔ conditioning (最重要的target-context关系)
        for i in range(hd_start, hd_end):
            for region_start, region_end in idx_map[1:]:  # 跳过HD区域
                pair_id[i, region_start:region_end] = 4
                pair_id[region_start:region_end, i] = 4
        
        # Level 5+: Conditioning regions internal
        counter = 5
        for region_start, region_end in idx_map[1:]:
            pair_id[region_start:region_end, region_start:region_end] = counter
            counter += 1
        
        # Level N+: Conditioning ↔ conditioning
        conditioning_regions = idx_map[1:]
        for i, (r1_start, r1_end) in enumerate(conditioning_regions):
            for j, (r2_start, r2_end) in enumerate(conditioning_regions[i+1:], i+1):
                pair_id[r1_start:r1_end, r2_start:r2_end] = counter
                pair_id[r2_start:r2_end, r1_start:r1_end] = counter
                counter += 1
        
        return pair_id.clamp(max=31)

class EvoBlockWithLog(nn.Module):
    """
    Evoformer-like block with MultiheadAttention and FFN
    psi-Update: Explicitly logs attention maps to trace collapse pathway
    Academic Justification: Enables interpretability and verification of conditioning effects
    """
    def __init__(self, s_dim, z_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(s_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim*4),
            nn.ReLU(),
            nn.Linear(s_dim*4, s_dim)
        )

    def forward(self, s, z, attn_mask=None):
        s_ln = F.layer_norm(s, s.shape[-1:])
        
        #  确保attention mask与输入tensor在同一设备
        if attn_mask is not None:
            attn_mask = attn_mask.to(s.device)
        
        s_out, attn_map = self.attn(s_ln, s_ln, s_ln, 
                                   attn_mask=attn_mask, 
                                   need_weights=True)
        s = s + s_out
        s = s + self.ffn(s)
        return s, z, attn_map

class CollapseEvoformer(nn.Module):
    """
    Stack of EvoBlocks with full attention logging for interpretability
    psi-Update: Track each collapse trace across Evoformer layers
    Academic Justification: Similar to AlphaFold attention map extraction
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([EvoBlockWithLog(cfg['s_dim'], cfg['z_dim']) for _ in range(cfg['N_elayers'])])
        self.log_attn = []

    def forward(self, s, z, attn_mask=None):
        self.log_attn = []
        for layer in self.layers:
            s, z, a = layer(s, z, attn_mask)
            self.log_attn.append(a)
        return s, z

class AdaptiveWeightEmbedding(nn.Module):
    """更智能的权重学习：基于注意力动态调节"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.s_in_dim = cfg['s_in_dim']
        self.z_in_dim = cfg['z_in_dim']
        self.s_dim = cfg['s_dim']
        self.z_dim = cfg['z_dim']

        self.seq_proj = Linear(self.s_in_dim + 1, self.s_dim)
        self.pair_embed_lvl1 = Linear(8, self.z_dim // 2)
        self.pair_embed_lvl2 = Linear(4, self.z_dim // 2)
        self.pos_embed_s = Linear(64, self.s_dim)

        self.collapse_token = nn.Parameter(torch.randn(1, self.s_dim))
        
        # 添加可学习的权重平衡
        self.seq_weight = nn.Parameter(torch.ones(1))      # 序列内容权重
        self.pos_weight = nn.Parameter(torch.ones(1))      # 位置信息权重
        self.collapse_weight = nn.Parameter(torch.ones(1)) # collapse token权重
        
        # 注意力引导的权重网络
        self.weight_attention = nn.MultiheadAttention(
            embed_dim=cfg['s_dim'], 
            num_heads=1, 
            batch_first=True
        )
        self.weight_proj = nn.Linear(cfg['s_dim'], 2)  # 输出[seq_weight, pos_weight]

    def compute_adaptive_weights(self, seq_emb, pos_emb):
        """基于序列内容动态计算权重"""
        # 将seq和pos embedding合并
        combined = torch.stack([seq_emb, pos_emb], dim=1)  # (L, 2, s_dim)
        
        # 使用self-attention计算重要性
        attn_out, _ = self.weight_attention(combined, combined, combined)
        
        # 投影到权重空间
        weights = torch.softmax(self.weight_proj(attn_out.mean(dim=1)), dim=-1)  # (L, 2)
        
        return weights[:, 0:1], weights[:, 1:2]  # seq_weight, pos_weight
        
    def forward(self, in_dict, conditioning_info):
        device = get_device()
        s_list, idx_map = [], []
        
        # Collapse token with learnable weight
        collapse_emb = self.collapse_weight * self.collapse_token
        s_list.append(collapse_emb)
        
        offset = 1
        for k in ['hd'] + [k for k in ['mhc','pep','lv','lj','hv','hj'] if k in conditioning_info]:
            if k in in_dict and in_dict[k].shape[0] > 0:
                aa = in_dict[k].to(device)
                
                # 关键修复：对HD使用真正的training mask，对conditioning序列使用零mask
                if k == 'hd':
                    mask = in_dict['mask'].to(device)  # 使用真正的training mask
                else:
                    mask = torch.zeros(aa.shape[0], device=device)  # conditioning序列不被mask
                
                aa = torch.cat([mask[:, None], aa], dim=-1)
                
                # 分别计算序列和位置编码
                seq_emb = self.seq_proj(aa)
                pos_emb = self.pos_embed_s(one_d(in_dict[f'{k}_idx'].to(device), 64))
                
                # 动态权重计算
                seq_w, pos_w = self.compute_adaptive_weights(seq_emb, pos_emb)
                s = seq_w * seq_emb + pos_w * pos_emb
                
                s_list.append(s)
                idx_map.append((offset, offset + s.shape[0]))
                offset += s.shape[0]
                
        s_out = torch.cat(s_list, dim=0)
        L = s_out.shape[0]

        # 使用改进的层次化pair embedding
        pair_id = self.create_hierarchical_pairs(L, idx_map, device)
        z = torch.cat([
            self.pair_embed_lvl1(F.one_hot(pair_id//4, 8).float()),
            self.pair_embed_lvl2(F.one_hot(pair_id%4, 4).float())
        ], dim=-1)

        return s_out, z

    def create_hierarchical_pairs(self, L, idx_map, device):
        """
        创建更合理的层次化pair embedding
        Hierarchy:
        0: collapse ↔ collapse (self-reference psi=psi(psi))
        1: collapse ↔ any region (observer-observed)
        2: HD internal sequential (i, i+1 neighbors)  
        3: HD internal non-sequential
        4: HD ↔ conditioning (target-context)
        5-N: conditioning internal (intra-region)
        N+1-M: conditioning ↔ conditioning (inter-conditioning)
        """
        pair_id = torch.zeros((L, L), dtype=torch.long, device=device)
        
        # 找到各区域边界
        collapse_end = 1
        hd_start, hd_end = idx_map[0] if idx_map else (1, 1)
        
        # Level 0: Collapse self-reference (psi=psi(psi))
        pair_id[0, 0] = 0
        
        # Level 1: Collapse ↔ all other regions
        pair_id[0, 1:] = 1
        pair_id[1:, 0] = 1
        
        # Level 2: HD sequential neighbors (重要的序列邻近性)
        if hd_end > hd_start:
            for i in range(hd_start, hd_end-1):
                pair_id[i, i+1] = 2
                pair_id[i+1, i] = 2
        
        # Level 3: HD internal non-sequential
        for i in range(hd_start, hd_end):
            for j in range(hd_start, hd_end):
                if i != j and pair_id[i, j] == 0:  # 非邻近的HD内部关系
                    pair_id[i, j] = 3
        
        # Level 4: HD ↔ conditioning (最重要的target-context关系)
        for i in range(hd_start, hd_end):
            for region_start, region_end in idx_map[1:]:  # 跳过HD区域
                pair_id[i, region_start:region_end] = 4
                pair_id[region_start:region_end, i] = 4
        
        # Level 5+: Conditioning regions internal
        counter = 5
        for region_start, region_end in idx_map[1:]:
            pair_id[region_start:region_end, region_start:region_end] = counter
            counter += 1
        
        # Level N+: Conditioning ↔ conditioning
        conditioning_regions = idx_map[1:]
        for i, (r1_start, r1_end) in enumerate(conditioning_regions):
            for j, (r2_start, r2_end) in enumerate(conditioning_regions[i+1:], i+1):
                pair_id[r1_start:r1_end, r2_start:r2_end] = counter
                pair_id[r2_start:r2_end, r1_start:r1_end] = counter
                counter += 1
        
        return pair_id.clamp(max=31)

class psiCLM(nn.Module):
    """
    Main model: Collapse-aware language model for protein sequence generation
    整合所有loss计算逻辑
    """
    def __init__(self, cfg):
        super().__init__()
        self.embedding = CollapseAwareEmbedding(cfg)
        self.backbone = CollapseEvoformer(cfg)
        self.head = Linear(cfg['s_dim'], cfg['s_in_dim'])
        self.cfg = cfg

    def forward(self, in_dict, computeloss, conditioning_info=None):
        """
        简化版forward: 只使用input masking，不使用attention mask
        保留attention tracing用于观察collapse patterns
        """
        if conditioning_info is None:
            conditioning_info = []
            
        device = get_device()
        for key in in_dict:
            if isinstance(in_dict[key], torch.Tensor):
                in_dict[key] = in_dict[key].to(device)
        
        s, z = self.embedding(in_dict, conditioning_info)
        
        # 移除所有attention mask，让attention自由流动
        s, z = self.backbone(s[None, ...], z, attn_mask=None)
        
        L1 = in_dict['hd'].shape[0]
        pred_aa = self.head(s[0][1:L1+1])  # 跳过collapse token
        
        if not computeloss:
            return torch.softmax(pred_aa, dim=-1), self.backbone.log_attn
        else:
            # 标准预测loss (仅使用input masking)
            pred_aa = torch.log_softmax(pred_aa, dim=-1)
            nll_loss = self.compute_nll_loss(pred_aa, in_dict)
            return nll_loss

    def compute_nll_loss(self, pred_aa, in_dict):
        """
        计算NLL loss
        """
        return nll_loss_withmask(pred_aa, in_dict['hd'], in_dict['mask'][:, None])

    def compute_conditioning_contrast_loss(self, pred_aa, in_dict, conditioning_info):
        if len(conditioning_info) == 0:
            return torch.tensor(0.0, device=pred_aa.device)
        
        # 保存HD长度
        hd_len = in_dict['hd'].shape[0]
        
        # 更高效的方法：直接在embedding层面打乱
        original_embeddings = {}
        
        # 保存原始embedding并创建打乱版本
        for key in conditioning_info:
            if key in in_dict and in_dict[key].shape[0] > 0:
                original_embeddings[key] = in_dict[key].clone()
                perm = torch.randperm(in_dict[key].shape[0])
                in_dict[key] = in_dict[key][perm]
        
        # 只重新计算embedding部分
        with torch.no_grad():
            s_corrupted, z_corrupted = self.embedding(in_dict, conditioning_info)
            s_corrupted, z_corrupted = self.backbone(s_corrupted[None, ...], z_corrupted, None)
            corrupted_pred = torch.softmax(self.head(s_corrupted[0][1:hd_len+1]), dim=-1)
        
        # 恢复原始数据
        for key, original_data in original_embeddings.items():
            in_dict[key] = original_data
        
        # 计算对比loss
        contrast_loss = -torch.mean(
            torch.sum(torch.exp(pred_aa) * torch.log(corrupted_pred + 1e-8), dim=-1)
        )
        
        return contrast_loss

    def compute_composite_loss(self, in_dict, conditioning_info=None): # optional with entropy

        if conditioning_info is None:
            conditioning_info = []
            
        pred_logits, attn_traces = self(in_dict, computeloss=False, conditioning_info=conditioning_info)
        
        # 基础NLL损失
        pred_aa = torch.log_softmax(pred_logits, dim=-1)
        nll_loss = nll_loss_withmask(pred_aa, in_dict['hd'], in_dict['mask'][:, None])
        
        collapse_entropy = self._compute_collapse_entropy(attn_traces)
        
        lambda_nll = 1.0
        lambda_collapse = 1e-5
        
        total_loss = lambda_nll * nll_loss + lambda_collapse * collapse_entropy
        
        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'collapse_entropy': collapse_entropy
        }
    
    def _compute_collapse_entropy(self, attn_traces):
        """计算collapse token的注意力熵"""
        if not attn_traces:
            return torch.tensor(0.0, device=get_device())
        
        # 使用最后一层的collapse attention
        collapse_attn = attn_traces[-1][0, 0, :]  # shape: (L,)
        
        # 计算熵
        probs = F.softmax(collapse_attn, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        return entropy

def train(model, optimizer, start):
    mask_ratios = []
    
    for i, idx in enumerate(batch_idxs):
        sample = train_set[idx]
        mask_ratio = sample['mask'].sum().item() / len(sample['mask'])
        mask_ratios.append(mask_ratio)
        
        if i % 100 == 0:
            avg_mask_ratio = sum(mask_ratios) / len(mask_ratios)
            print(f"Average mask ratio: {avg_mask_ratio:.3f}")

def analyze_mask_distribution():
    """分析训练数据中mask的分布"""
    import data_clp as data
    import numpy as np
    
    # 加载训练数据
    train_set = data.CollapseProteinDataset('../data/trn.csv')
    
    mask_ratios = []
    sequence_lengths = []
    
    for i in range(min(1000, len(train_set))):
        sample = train_set[i]
        mask_ratio = sample['mask'].sum().item() / len(sample['mask'])
        seq_len = sample['hd'].shape[0]
        
        mask_ratios.append(mask_ratio)
        sequence_lengths.append(seq_len)
    
    print(f"Mask ratio - Mean: {np.mean(mask_ratios):.3f}, Std: {np.std(mask_ratios):.3f}")
    print(f"Seq length - Mean: {np.mean(sequence_lengths):.1f}, Max: {max(sequence_lengths)}")
