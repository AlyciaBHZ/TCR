# data_loader.py for psiCLM with attention-based MHC/LV/HV subsampling and flexible field mapping

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
import random

AA_DICT = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
    'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
    'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    '-': 20, '*': 20
}

AA_DIM = 21
MAX_MHC_LEN = 96

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value = nn.Linear(input_dim, hidden_dim * num_heads)
        self.output_proj = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, x):
        seq_len, _ = x.size()
        query = self.query(x).view(seq_len, self.num_heads, -1).transpose(0, 1)
        key = self.key(x).view(seq_len, self.num_heads, -1).transpose(0, 1)
        value = self.value(x).view(seq_len, self.num_heads, -1).transpose(0, 1)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, -1)
        output = self.output_proj(attn_output)
        return output, attn_weights

class AttentionSubsampling(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=16, output_dim=21, num_heads=4, num_layers=2):
        super(AttentionSubsampling, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttention(input_dim, hidden_dim, output_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x, output_len):
        for layer in self.layers:
            x, _ = layer(x)
        return x[:output_len, :]

        return result


def dummy(seq):
    if pd.isna(seq):
        return torch.zeros((0, AA_DIM))
    arr = np.array([AA_DICT.get(residue, 20) for residue in seq])
    return torch.eye(AA_DIM)[arr]

def mask_gen(length, mask_ratio=None): 
    """
    生成更diverse的mask pattern
    现在可以使用更高的mask ratio，因为使用纯input masking
    """
    mask = torch.zeros(length)
    
    if mask_ratio is None:
        # Dynamic mask ratio: 10% - 85% based on sequence length
        if length <= 8:
            # 短序列：较低mask ratio，保证有足够信息
            base_ratio = random.uniform(0.1, 0.4)
        else:
            base_ratio = random.uniform(0.15, 0.8)
        # 添加一些随机性，让训练更robust
        mask_ratio = base_ratio + random.uniform(-0.05, 0.05)
        mask_ratio = max(0.1, min(0.85, mask_ratio))  # 限制在10%-85%
    
    # 计算要mask的位置数量
    num_mask = max(1, min(int(length * mask_ratio), length - 1))  # 确保不全mask
    indices = torch.randperm(length)[:num_mask]
    mask[indices] = 1
    
    # 添加debug信息
    actual_ratio = num_mask / length
    # 临时debug：检查mask生成 - 暂时关闭
    # if random.random() < 0.01:  # 1%概率打印，检查mask是否正常
    #     print(f"Mask gen: length={length}, num_mask={num_mask}, ratio={actual_ratio:.3f}, mask_sum={mask.sum().item()}")
    
    return mask

def create_idx(length):
    return torch.arange(1, length + 1, dtype=torch.float32)

class CollapseProteinDataset(Dataset):
    def __init__(self, csv_path, parse_mode='line-positional'):
        self.df = pd.read_csv(csv_path)
        self.fields = ['pep', 'mhc', 'lv', 'lj', 'hv', 'hd', 'hj']
        self.att_pool = AttentionSubsampling()
        self.mode = parse_mode
        print(f"Loaded {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        result = {}

        if self.mode == 'line-positional':
            # Positional column mapping
            tokens = [str(x).strip() if pd.notna(x) else '' for x in row.tolist()]
            mapping = dict(zip(self.fields, tokens))
        else:
            raise ValueError("Unsupported parse_mode")

        for field in self.fields:
            seq = mapping.get(field, '')
            emb = dummy(seq)

            if field in ['mhc', 'lv', 'hv'] and emb.shape[0] > MAX_MHC_LEN:
                emb = self.att_pool(emb, MAX_MHC_LEN)

            result[field] = emb
            result[f"{field}_idx"] = create_idx(emb.shape[0]) if emb.shape[0] > 0 else torch.tensor([])

        hd_len = result['hd'].shape[0]
        result['mask'] = mask_gen(hd_len) if hd_len > 0 else torch.tensor([])
        return result
