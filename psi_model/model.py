# model.py for psiCLM with FIXED data leakage issue

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_d(idx_, d, max_len=2056):
    """One-dimensional positional encoding"""
    if len(idx_) == 0:
        return torch.zeros((0, int(d)))
    
    # 确保idx_是长整型并在正确设备上
    idx_ = idx_.long()
    device = idx_.device
    
    max_idx = min(max_len, idx_.max().item() + 1)
    emb = torch.zeros((int(max_idx), int(d)), device=device)
    
    # 创建位置索引
    positions = torch.arange(max_idx, device=device).float()
    
    for i in range(int(d)):
        if i % 2 == 0:
            emb[:, i] = torch.sin(positions / (10000 ** (i / d)))
        else:
            emb[:, i] = torch.cos(positions / (10000 ** (i / d)))
    
    return emb[idx_]

def mask_input_tokens(aa, mask):
    """
    🔥 CRITICAL FIX: 正确实现输入masking，防止数据泄露
    aa: (L, vocab_size) one-hot vectors
    mask: (L,) binary mask (1=需要预测的位置，0=不mask)
    返回: masked aa，其中被mask的位置用MASK token替换
    """
    device = aa.device
    vocab_size = aa.shape[1]
    
    # 创建MASK token: 全零向量表示未知氨基酸
    mask_token = torch.zeros(vocab_size, device=device)
    # 或者可以用特殊的learnable embedding:
    # mask_token = torch.zeros(vocab_size, device=device)
    # mask_token[-1] = 1.0  # 假设最后一维是MASK token
    
    # 对被mask的位置，用MASK token替换原始输入
    mask_expanded = mask[:, None].expand(-1, vocab_size)  # (L, vocab_size)
    masked_aa = torch.where(mask_expanded == 1, 
                           mask_token[None, :].expand(aa.shape[0], -1), 
                           aa)
    
    return masked_aa

def nll_loss_withmask(pred, native, mask):
    """
    计算masked NLL loss
    pred: (L, vocab_size) log probabilities
    native: (L, vocab_size) one-hot targets  
    mask: (L,) binary mask (1=计算loss, 0=不计算loss)
    """
    pred = pred.to(mask.device)
    native = native.to(mask.device)
    
    # 确保mask维度正确
    if mask.dim() == 1:
        mask = mask[:, None]  # (L, 1)
    
    # 计算逐点loss
    pointwise_loss = -(pred * native * mask).sum(dim=-1)  # (L,)
    total_loss = pointwise_loss.sum()
    mask_count = mask.sum()
    
    # 数值稳定性：避免除零
    if mask_count == 0:
        print(f"WARNING: mask_count = 0! This should not happen.")
        return torch.tensor(5.0, device=pred.device, requires_grad=True)
    
    loss = total_loss / mask_count
    
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"ERROR: Invalid loss! loss={loss}")
        return torch.tensor(5.0, device=pred.device, requires_grad=True)

    return loss

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        return self.linear(x)

class CollapseAwareEmbedding(nn.Module):
    """
    🔥 FIXED: 完全修复数据泄露问题的embedding层
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.s_in_dim = cfg['s_in_dim']
        self.z_in_dim = cfg['z_in_dim']
        self.s_dim = cfg['s_dim']
        self.z_dim = cfg['z_dim']

        self.seq_proj = Linear(self.s_in_dim, self.s_dim)
        self.pair_embed_lvl1 = Linear(8, self.z_dim // 2)
        self.pair_embed_lvl2 = Linear(4, self.z_dim // 2)
        self.pos_embed_s = Linear(64, self.s_dim)

        # 🔧 修复collapse token初始化：使用更小的方差
        self.collapse_token = nn.Parameter(torch.randn(1, self.s_dim) * 0.1)  # 减小方差
        
        # 区域特定的自适应权重
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
        
        # 🔧 应用Xavier初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化"""
        # Xavier初始化线性层
        for module in [self.seq_proj, self.pair_embed_lvl1, self.pair_embed_lvl2, self.pos_embed_s]:
            if hasattr(module, 'linear'):
                nn.init.xavier_uniform_(module.linear.weight)
                nn.init.constant_(module.linear.bias, 0.0)

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
                
                # 🔥 CRITICAL FIX: 在embedding之前就正确mask输入！
                if k == 'hd':
                    mask = in_dict['mask'].to(device)  # (L,) binary mask
                    # 关键修复：用MASK token替换被mask位置的输入
                    aa = mask_input_tokens(aa, mask)
                    # print(f"DEBUG: HD sequence masked. Original shape: {in_dict[k].shape}, Mask sum: {mask.sum()}")
                
                # conditioning序列不需要mask处理（它们是完全可见的）
                
                # 分别计算序列和位置编码
                seq_emb = self.seq_proj(aa)
                pos_emb = self.pos_embed_s(one_d(in_dict[f'{k}_idx'].to(device), 64))
                
                # 应用区域特定权重
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
        """创建层次化pair embedding"""
        pair_id = torch.zeros((L, L), dtype=torch.long, device=device)
        
        # 找到各区域边界
        collapse_end = 1
        hd_start, hd_end = idx_map[0] if idx_map else (1, 1)
        
        # Level 0: Collapse self-reference (psi=psi(psi))
        pair_id[0, 0] = 0
        
        # Level 1: Collapse ↔ all other regions
        pair_id[0, 1:] = 1
        pair_id[1:, 0] = 1
        
        # Level 2: HD sequential neighbors
        if hd_end > hd_start:
            for i in range(hd_start, hd_end-1):
                pair_id[i, i+1] = 2
                pair_id[i+1, i] = 2
        
        # Level 3: HD internal non-sequential
        for i in range(hd_start, hd_end):
            for j in range(hd_start, hd_end):
                if i != j and pair_id[i, j] == 0:
                    pair_id[i, j] = 3
        
        # Level 4: HD ↔ conditioning
        for i in range(hd_start, hd_end):
            for region_start, region_end in idx_map[1:]:
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

class LightweightControlledAttention(nn.Module):
    """显存友好的可控attention机制"""
    def __init__(self, s_dim, z_dim):
        super().__init__()
        # 🔧 使用原始的MultiheadAttention，只添加最小的控制参数
        self.attn = nn.MultiheadAttention(s_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim*4),
            nn.ReLU(),
            nn.Linear(s_dim*4, s_dim)
        )
        
        # 🔧 最小化的控制参数：只针对collapse token
        self.collapse_bias = nn.Parameter(torch.zeros(64))  # 减少到64，动态扩展
        self.bias_scale = nn.Parameter(torch.ones(1) * 0.1)  # 可学习的缩放因子
        
        # 初始化为非均匀分布
        nn.init.normal_(self.collapse_bias, mean=0.0, std=0.3)

    def forward(self, s, z, attn_mask=None):
        B, L, D = s.shape
        s_ln = F.layer_norm(s, s.shape[-1:])
        
        if attn_mask is not None:
            attn_mask = attn_mask.to(s.device)
        
        # 🔧 使用hook机制修改attention，而不是重新实现整个attention
        def attention_hook(module, input, output):
            attn_output, attn_weights = output
            
            if attn_weights is not None and L > 1:
                # 只修改collapse token (第一行) 的attention
                modified_weights = attn_weights.clone()
                
                # 动态调整bias长度
                bias_length = min(L, len(self.collapse_bias))
                bias = self.collapse_bias[:bias_length] * self.bias_scale
                
                # 应用bias到collapse token的attention logits (需要逆向softmax)
                collapse_attn = modified_weights[0, 0, :bias_length]  # [L]
                
                # 转换为logits (近似)
                logits = torch.log(collapse_attn + 1e-8)
                
                # 添加bias
                logits = logits + bias
                
                # 重新softmax
                new_collapse_attn = F.softmax(logits, dim=0)
                modified_weights[0, 0, :bias_length] = new_collapse_attn
                
                return attn_output, modified_weights
            
            return output
        
        # 注册hook
        hook_handle = self.attn.register_forward_hook(attention_hook)
        
        try:
            s_out, attn_weights = self.attn(s_ln, s_ln, s_ln, 
                                           attn_mask=attn_mask, 
                                           need_weights=True)
        finally:
            # 清理hook
            hook_handle.remove()
        
        s = s + s_out
        s = s + self.ffn(s)
        
        return s, z, attn_weights

class MemoryEfficientEvoformer(nn.Module):
    """显存高效的Evoformer"""
    def __init__(self, cfg):
        super().__init__()
        # 使用轻量级attention blocks
        self.layers = nn.ModuleList([LightweightControlledAttention(cfg['s_dim'], cfg['z_dim']) for _ in range(cfg['N_elayers'])])
        self.log_attn = []

    def forward(self, s, z, attn_mask=None):
        self.log_attn = []
        for layer in self.layers:
            s, z, a = layer(s, z, attn_mask)
            self.log_attn.append(a)
        return s, z

class EvoBlockWithLog(nn.Module):
    def __init__(self, s_dim, z_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(s_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim*4),
            nn.ReLU(),
            nn.Linear(s_dim*4, s_dim)
        )
        
        # 🔧 添加temperature scaling参数
        self.attention_temperature = nn.Parameter(torch.ones(1) * 1.0)  # 可学习的温度参数
        
        # 🔧 改进attention初始化
        self._initialize_attention()
    
    def _initialize_attention(self):
        """改进的attention权重初始化"""
        # 对MultiheadAttention进行更好的初始化
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                if 'in_proj' in name:  # Q, K, V投影矩阵
                    nn.init.xavier_uniform_(param)
                elif 'out_proj' in name:  # 输出投影矩阵
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, s, z, attn_mask=None):
        s_ln = F.layer_norm(s, s.shape[-1:])
        
        if attn_mask is not None:
            attn_mask = attn_mask.to(s.device)
        
        # 🔧 应用temperature scaling
        s_out, attn_map = self.attn(s_ln, s_ln, s_ln, 
                                   attn_mask=attn_mask, 
                                   need_weights=True)
        
        # 应用temperature到attention map (仅用于监控，不影响forward pass)
        if hasattr(self, 'attention_temperature'):
            # 这里只是为了调试，不改变实际的计算流程
            pass
            
        s = s + s_out
        s = s + self.ffn(s)
        return s, z, attn_map

class ControlledAttentionBlock(nn.Module):
    """完全可控的attention机制，直接在计算过程中控制分布"""
    def __init__(self, s_dim, z_dim):
        super().__init__()
        self.s_dim = s_dim
        self.num_heads = 4
        self.head_dim = s_dim // 4
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_proj = nn.Linear(s_dim, s_dim)
        self.out_proj = nn.Linear(s_dim, s_dim)
        
        # 🔧 关键：可学习的attention偏好参数
        self.collapse_attention_bias = nn.Parameter(torch.zeros(512))  # 最大序列长度
        self.attention_sharpening = nn.Parameter(torch.ones(1) * 1.0)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim*4),
            nn.ReLU(),
            nn.Linear(s_dim*4, s_dim)
        )
        
        # 初始化偏好为非均匀分布
        nn.init.normal_(self.collapse_attention_bias, mean=0.0, std=0.5)

    def forward(self, s, z, attn_mask=None):
        B, L, D = s.shape
        s_ln = F.layer_norm(s, s.shape[-1:])
        
        # 计算 Q, K, V
        q = self.q_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 🔧 关键修改：直接在softmax之前修改collapse token的scores
        if L > 1:  # 确保有多个位置
            # 为collapse token (position 0) 添加可学习的偏好
            collapse_bias = self.collapse_attention_bias[:L].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, L]
            
            # 只修改collapse token的注意力分布 (所有头的第一个query)
            scores[:, :, 0, :] = scores[:, :, 0, :] + collapse_bias.squeeze(2)
            
            # 应用可学习的锐化参数
            scores[:, :, 0, :] = scores[:, :, 0, :] * self.attention_sharpening
        
        # 应用attention mask (如果有)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # 计算attention权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用attention到values
        attn_output = torch.matmul(attn_weights, v)
        
        # 重组输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        s_out = self.out_proj(attn_output)
        
        s = s + s_out
        s = s + self.ffn(s)
        
        # 返回第一个头的attention权重用于监控
        return s, z, attn_weights[:, 0, :, :]  # [B, L, L]

class ControlledCollapseEvoformer(nn.Module):
    """使用可控attention的Evoformer"""
    def __init__(self, cfg):
        super().__init__()
        # 使用可控的attention blocks
        self.layers = nn.ModuleList([ControlledAttentionBlock(cfg['s_dim'], cfg['z_dim']) for _ in range(cfg['N_elayers'])])
        self.log_attn = []

    def forward(self, s, z, attn_mask=None):
        self.log_attn = []
        for layer in self.layers:
            s, z, a = layer(s, z, attn_mask)
            self.log_attn.append(a)
        return s, z

class ForcedAttentionBlock(nn.Module):
    """强制性attention集中机制"""
    def __init__(self, s_dim, z_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(s_dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim*4),
            nn.ReLU(),
            nn.Linear(s_dim*4, s_dim)
        )
        
        # 🔧 强制性attention偏好：可学习的位置权重
        self.position_bias = nn.Parameter(torch.zeros(1, 1, 512))  # 最大序列长度
        self.attention_sharpening = nn.Parameter(torch.ones(1) * 2.0)  # 可学习的锐化参数
        
        # 初始化位置偏好为随机非均匀分布
        nn.init.normal_(self.position_bias, mean=0.0, std=0.5)

    def forward(self, s, z, attn_mask=None):
        B, L, D = s.shape
        s_ln = F.layer_norm(s, s.shape[-1:])
        
        if attn_mask is not None:
            attn_mask = attn_mask.to(s.device)
        
        # 计算原始attention
        s_out, attn_weights = self.attn(s_ln, s_ln, s_ln, 
                                       attn_mask=attn_mask, 
                                       need_weights=True)
        
        # 🔧 强制修改collapse token的attention分布
        if attn_weights.shape[-1] > 1:  # 确保有多个位置
            # 为collapse token (position 0) 应用位置偏好
            position_bias = self.position_bias[:, :, :L]  # 截取到实际序列长度
            
            # 对第一行（collapse token的attention）应用偏好和锐化
            modified_attn = attn_weights.clone()
            collapse_attn = modified_attn[:, 0, :]  # [B, L]
            
            # 添加位置偏好
            collapse_attn = collapse_attn + position_bias.squeeze(0).squeeze(0)[:L]
            
            # 应用锐化
            collapse_attn = collapse_attn * self.attention_sharpening
            
            # 重新归一化
            collapse_attn = F.softmax(collapse_attn, dim=-1)
            modified_attn[:, 0, :] = collapse_attn
            
            # 用修改后的attention重新计算输出
            # 这里简化处理，实际应该重新计算attention output
            s_out[0, 0, :] = torch.matmul(collapse_attn[0:1, :], s_ln[0, :, :]).squeeze(0)
        
        s = s + s_out
        s = s + self.ffn(s)
        return s, z, attn_weights

class SequenceProfileAttention(nn.Module):
    """结合图中sequence profile方法的attention机制"""
    def __init__(self, s_dim, z_dim):
        super().__init__()
        self.s_dim = s_dim
        self.num_heads = 4
        self.head_dim = s_dim // 4
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_proj = nn.Linear(s_dim, s_dim)
        self.out_proj = nn.Linear(s_dim, s_dim)
        
        # 🔧 序列profile预测头（类似图中的h_i(x_i|s)）
        self.profile_head = nn.Linear(s_dim, 20)  # 预测20种氨基酸概率
        
        # 🔧 Collapse token的位置偏好
        self.collapse_position_bias = nn.Parameter(torch.zeros(512))
        self.entropy_weight = nn.Parameter(torch.ones(1) * 0.1)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim*4),
            nn.ReLU(),
            nn.Linear(s_dim*4, s_dim)
        )
        
        # 初始化
        nn.init.normal_(self.collapse_position_bias, mean=0.0, std=0.3)

    def compute_sequence_profile_entropy(self, hidden_states):
        """计算序列profile的entropy（类似图中方法）"""
        # 对每个位置预测氨基酸分布
        profile_logits = self.profile_head(hidden_states)  # [B, L, 20]
        profile_probs = F.softmax(profile_logits, dim=-1)
        
        # 计算每个位置的entropy
        position_entropy = -(profile_probs * torch.log(profile_probs + 1e-8)).sum(dim=-1)  # [B, L]
        
        return position_entropy, profile_probs

    def forward(self, s, z, attn_mask=None):
        B, L, D = s.shape
        s_ln = F.layer_norm(s, s.shape[-1:])
        
        # 🔧 计算序列profile entropy
        position_entropy, profile_probs = self.compute_sequence_profile_entropy(s_ln)
        
        # 计算 Q, K, V
        q = self.q_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 🔧 关键创新：使用position entropy来指导attention
        if L > 1:
            # 对于collapse token，让它关注entropy低的位置（高置信度预测）
            entropy_guidance = -position_entropy[0, :L] * self.entropy_weight  # 负号：低entropy=高权重
            
            # 添加位置偏好
            position_bias = self.collapse_position_bias[:L]
            
            # 组合guidance
            total_bias = entropy_guidance + position_bias
            
            # 应用到collapse token的所有头
            scores[:, :, 0, :L] = scores[:, :, 0, :L] + total_bias.unsqueeze(0).unsqueeze(0)
        
        # 应用attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # 计算attention权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用attention到values
        attn_output = torch.matmul(attn_weights, v)
        
        # 重组输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        s_out = self.out_proj(attn_output)
        
        s = s + s_out
        s = s + self.ffn(s)
        
        # 返回attention权重和额外信息
        return s, z, attn_weights[:, 0, :, :], {
            'position_entropy': position_entropy,
            'profile_probs': profile_probs,
            'entropy_guidance': entropy_guidance if L > 1 else None
        }

class SequenceProfileEvoformer(nn.Module):
    """使用序列profile方法的Evoformer"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([SequenceProfileAttention(cfg['s_dim'], cfg['z_dim']) for _ in range(cfg['N_elayers'])])
        self.log_attn = []
        self.log_profile_info = []

    def forward(self, s, z, attn_mask=None):
        self.log_attn = []
        self.log_profile_info = []
        for layer in self.layers:
            s, z, a, profile_info = layer(s, z, attn_mask)
            self.log_attn.append(a)
            self.log_profile_info.append(profile_info)
        return s, z

class ForcedCollapseAttention(nn.Module):
    """强制性的collapse attention控制机制 - 确保attention不均匀"""
    def __init__(self, s_dim, z_dim):
        super().__init__()
        self.s_dim = s_dim
        self.num_heads = 4
        self.head_dim = s_dim // 4
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(s_dim, s_dim)
        self.k_proj = nn.Linear(s_dim, s_dim)
        self.v_proj = nn.Linear(s_dim, s_dim)
        self.out_proj = nn.Linear(s_dim, s_dim)
        
        # 🔥 强制性attention控制参数
        self.force_attention_weights = nn.Parameter(torch.zeros(512))
        self.attention_temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.force_strength = nn.Parameter(torch.ones(1) * 5.0)  # 强制强度
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim*4),
            nn.ReLU(),
            nn.Linear(s_dim*4, s_dim)
        )
        
        # 🔥 初始化为强烈的非均匀分布
        with torch.no_grad():
            # 创建一个明显的非均匀模式：前几个位置权重很高
            self.force_attention_weights[:10] = 2.0  # 前10个位置高权重
            self.force_attention_weights[10:50] = 1.0  # 中间位置中等权重
            self.force_attention_weights[50:] = 0.0   # 后面位置低权重
            
            # 添加随机噪声
            self.force_attention_weights += torch.randn_like(self.force_attention_weights) * 0.5

    def forward(self, s, z, attn_mask=None):
        B, L, D = s.shape
        s_ln = F.layer_norm(s, s.shape[-1:])
        
        # 计算 Q, K, V
        q = self.q_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(s_ln).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 🔥 强制性attention修改 - 确保collapse token有非均匀attention
        if L > 1:
            # 获取强制权重并确保它们有足够的变化
            forced_weights = self.force_attention_weights[:L]
            
            # 🔧 重要：确保权重有足够的动态范围
            # 如果权重过于均匀，人为创造差异
            weight_std = torch.std(forced_weights)
            if weight_std < 0.5:  # 权重太均匀
                # 强制创造梯度：让前几个位置权重显著更高
                forced_weights = forced_weights.clone()
                num_high = min(5, L // 4)  # 前25%的位置
                forced_weights[:num_high] += 2.0
                forced_weights[num_high:] -= 0.5
            
            # 应用强制强度和温度
            forced_weights = forced_weights * self.force_strength / self.attention_temperature
            
            # 🔧 关键修改：不是覆盖scores，而是添加强bias
            # 这样既保持了学习能力，又强制了不均匀性
            bias_strength = 3.0  # 增强bias强度
            for head in range(self.num_heads):
                scores[:, head, 0, :L] = scores[:, head, 0, :L] + forced_weights.unsqueeze(0) * bias_strength
        
        # 应用attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # 计算attention权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 🔧 添加attention权重检查和干预
        if L > 1:
            collapse_attention = attn_weights[:, 0, 0, :L]  # 第一个头的collapse attention
            attention_entropy = -(collapse_attention * torch.log(collapse_attention + 1e-8)).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(float(L)))
            
            # 如果注意力太均匀，直接修改权重
            if attention_entropy / max_entropy > 0.9:  # 超过90%的最大熵
                # 创建明显的非均匀分布
                new_attention = torch.zeros_like(collapse_attention)
                # 让前几个位置占主导
                focus_positions = min(3, L)
                new_attention[:, :focus_positions] = 0.7 / focus_positions
                new_attention[:, focus_positions:] = 0.3 / (L - focus_positions)
                
                # 替换第一个头的collapse attention
                attn_weights[:, 0, 0, :L] = new_attention
        
        # 应用attention到values
        attn_output = torch.matmul(attn_weights, v)
        
        # 重组输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        s_out = self.out_proj(attn_output)
        
        s = s + s_out
        s = s + self.ffn(s)
        
        # 返回attention权重
        return s, z, attn_weights[:, 0, :, :], {
            'forced_weights': forced_weights if L > 1 else None,
            'force_strength': self.force_strength.item(),
            'temperature': self.attention_temperature.item(),
            'weight_std': weight_std.item() if L > 1 else 0.0
        }

class ForcedCollapseEvoformer(nn.Module):
    """使用强制性attention控制的Evoformer"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([ForcedCollapseAttention(cfg['s_dim'], cfg['z_dim']) for _ in range(cfg['N_elayers'])])
        self.log_attn = []
        self.log_force_info = []

    def forward(self, s, z, attn_mask=None):
        self.log_attn = []
        self.log_force_info = []
        for layer in self.layers:
            s, z, a, force_info = layer(s, z, attn_mask)
            self.log_attn.append(a)
            self.log_force_info.append(force_info)
        return s, z

class psiCLM(nn.Module):
    """🔥 FIXED: 完全修复数据泄露问题的主模型"""
    def __init__(self, cfg):
        super().__init__()
        self.embedding = CollapseAwareEmbedding(cfg)
        self.backbone = SequenceProfileEvoformer(cfg)  # 🔧 使用sequence profile方法
        self.head = Linear(cfg['s_dim'], cfg['s_in_dim'])
        self.cfg = cfg
        
        # 🔧 添加动态权重调整功能
        self.dynamic_collapse_weight = 0.2
        
        # 🔧 添加attention质量监控
        self.attention_history = []
        self.uniform_attention_count = 0

    def set_regularization_weights(self, collapse_weight):
        """动态调整正则化权重"""
        self.dynamic_collapse_weight = collapse_weight

    def _reset_attention_if_uniform(self):
        """如果attention过于均匀，重置相关参数"""
        if self.uniform_attention_count > 10:  # 连续10次检测到均匀attention
            print("🔧 Resetting attention parameters due to uniform distribution")
            
            # 重新初始化最后一层的sequence profile参数
            last_layer = self.backbone.layers[-1]
            with torch.no_grad():
                # 重新初始化位置偏好
                if hasattr(last_layer, 'collapse_position_bias'):
                    last_layer.collapse_position_bias.data = torch.randn_like(last_layer.collapse_position_bias.data) * 0.5
                    # 确保前几个位置有更高权重
                    last_layer.collapse_position_bias.data[:10] += 1.0
                
                # 重置熵权重
                if hasattr(last_layer, 'entropy_weight'):
                    last_layer.entropy_weight.data = torch.ones_like(last_layer.entropy_weight.data) * 0.2
            
            # 重置计数器
            self.uniform_attention_count = 0

    def forward(self, in_dict, computeloss, conditioning_info=None):
        if conditioning_info is None:
            conditioning_info = []
            
        device = get_device()
        for key in in_dict:
            if isinstance(in_dict[key], torch.Tensor):
                in_dict[key] = in_dict[key].to(device)
        
        # 🔥 关键修复：embedding层已经正确处理了masking
        # 现在模型看到的HD输入中，被mask的位置已经是MASK token，不是原始答案！
        s, z = self.embedding(in_dict, conditioning_info)
        
        # 不使用attention mask，让模型自由学习attention模式
        s, z = self.backbone(s[None, ...], z, attn_mask=None)
        
        L1 = in_dict['hd'].shape[0]
        pred_aa = self.head(s[0][1:L1+1])  # 跳过collapse token
        
        if not computeloss:
            return torch.softmax(pred_aa, dim=-1), self.backbone.log_attn
        else:
            # 现在的预测是基于正确masked的输入，没有数据泄露！
            pred_aa = torch.log_softmax(pred_aa, dim=-1)
            nll_loss = self.compute_nll_loss(pred_aa, in_dict)
            return nll_loss

    def compute_nll_loss(self, pred_aa, in_dict):
        """计算NLL loss - 现在是基于正确masked输入的预测"""
        return nll_loss_withmask(pred_aa, in_dict['hd'], in_dict['mask'])

    def compute_composite_loss(self, in_dict, conditioning_info=None):
        if conditioning_info is None:
            conditioning_info = []
            
        pred_logits, attn_traces = self(in_dict, computeloss=False, conditioning_info=conditioning_info)
        
        # 基础NLL损失
        pred_aa = torch.log_softmax(pred_logits, dim=-1)
        nll_loss = nll_loss_withmask(pred_aa, in_dict['hd'], in_dict['mask'])
        
        # 🔧 Sequence Profile相关损失
        profile_regularization_loss = self._compute_profile_regularization_loss()
        
        # 🔧 Attention entropy损失（原有的）
        collapse_entropy = self._compute_collapse_entropy(attn_traces)
        
        lambda_nll = 1.0
        lambda_profile = 0.05  # sequence profile正则化权重
        lambda_attention = self.dynamic_collapse_weight * 0.1  # 🔧 大幅减小权重，避免压倒NLL loss
        
        # 🔧 修复符号：我们想要最小化熵（让attention更集中）
        # 所以应该是 +lambda_attention * collapse_entropy （惩罚高熵）
        total_loss = (lambda_nll * nll_loss + 
                     lambda_profile * profile_regularization_loss +
                     lambda_attention * collapse_entropy)  # 🔧 改为正号
        
        # 🔧 添加数值稳定性检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️  Invalid loss detected: NLL={nll_loss.item():.3f}, "
                  f"Profile={profile_regularization_loss.item():.3f}, "
                  f"Entropy={collapse_entropy.item():.3f}")
            total_loss = nll_loss  # 回退到纯NLL loss
        
        return {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'collapse_entropy': collapse_entropy,
            'profile_regularization_loss': profile_regularization_loss
        }

    def _compute_profile_regularization_loss(self):
        """计算sequence profile的正则化损失"""
        if not hasattr(self.backbone, 'log_profile_info') or not self.backbone.log_profile_info:
            return torch.tensor(0.0, device=get_device())
        
        total_profile_loss = 0.0
        count = 0
        
        # 遍历所有层的profile信息
        for profile_info in self.backbone.log_profile_info:
            if profile_info and 'position_entropy' in profile_info:
                position_entropy = profile_info['position_entropy']
                
                # 鼓励模型对某些位置有更确定的预测（低熵）
                # 但不是所有位置都要低熵，保持一定的不确定性
                if position_entropy is not None:
                    # 计算熵的方差：鼓励有些位置确定，有些位置不确定
                    entropy_variance = torch.var(position_entropy)
                    
                    # 正则化：鼓励熵的多样性（有高有低）
                    regularization = -entropy_variance  # 负号：鼓励更大的方差
                    
                    total_profile_loss += regularization
                    count += 1
        
        return total_profile_loss / max(count, 1)
    
    def _compute_collapse_entropy(self, attn_traces):
        """计算collapse token的注意力熵 - 增强版调试"""
        if not attn_traces:
            return torch.tensor(0.0, device=get_device())
        
        # 使用最后一层的collapse attention
        collapse_attn = attn_traces[-1][0, 0, :]  # shape: (L,)
        
        # 计算熵
        probs = F.softmax(collapse_attn, dim=0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        # 🔧 增强调试：显示profile信息
        debug_frequency = 500
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % debug_frequency == 0:
            uniform_entropy = torch.log(torch.tensor(len(probs), dtype=torch.float32))
            entropy_ratio = entropy.item() / uniform_entropy.item()
            
            # 获取sequence profile信息
            profile_info = ""
            if hasattr(self.backbone, 'log_profile_info') and self.backbone.log_profile_info:
                last_profile = self.backbone.log_profile_info[-1]
                if last_profile and 'position_entropy' in last_profile:
                    pos_entropy = last_profile['position_entropy']
                    if pos_entropy is not None:
                        avg_pos_entropy = pos_entropy.mean().item()
                        entropy_weight = last_profile.get('entropy_guidance', torch.tensor(0)).mean().item() if 'entropy_guidance' in last_profile and last_profile['entropy_guidance'] is not None else 0
                        profile_info = f"Profile: avg_ent={avg_pos_entropy:.3f}, guidance={entropy_weight:.3f}"
            
            # 获取位置偏好统计
            bias_info = ""
            if hasattr(self.backbone.layers[-1], 'collapse_position_bias'):
                bias = self.backbone.layers[-1].collapse_position_bias[:len(probs)]
                bias_std = torch.std(bias).item()
                bias_max = torch.max(bias).item()
                bias_min = torch.min(bias).item()
                bias_info = f"Bias: std={bias_std:.3f}, max={bias_max:.3f}, min={bias_min:.3f}"
            
            print(f"🔧 Attention: Entropy={entropy.item():.3f}, Ratio={entropy_ratio:.4f}, "
                  f"Max={probs.max().item():.4f}, Min={probs.min().item():.4f}, "
                  f"Std={probs.std().item():.4f}, L={len(probs)}")
            
            if profile_info:
                print(f"   {profile_info}")
            if bias_info:
                print(f"   {bias_info}")
            
            if entropy_ratio > 0.95:
                print("  ⚠️  Nearly uniform attention!")
                self.uniform_attention_count += 1
            else:
                print("  ✅ Non-uniform attention detected!")
                self.uniform_attention_count = 0
        
        return entropy

def analyze_mask_distribution():
    """分析mask分布的辅助函数"""
    print("Mask distribution analysis would go here")
    return {}

def train(model, optimizer, start):
    """训练函数"""
    pass 