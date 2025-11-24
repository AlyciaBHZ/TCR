
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from random import randint
import os
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, hidden_dim * num_heads)
        self.key = nn.Linear(input_dim, hidden_dim * num_heads)
        self.value = nn.Linear(input_dim, hidden_dim * num_heads)
        self.output_proj = nn.Linear(hidden_dim * num_heads, output_dim)
    
    def forward(self, x):
        seq_len, _ = x.size() # x(365,22)
        
        # Project inputs to query, key, value vectors and split by heads
        query = self.query(x) # x(365,22) -> query(365,64)
        query = query.view(seq_len, self.num_heads, -1).transpose(0, 1)  # [num_heads, seq_len, hidden_dim] 
        key = self.key(x).view(seq_len, self.num_heads, -1).transpose(0, 1)      # [num_heads, seq_len, hidden_dim]
        value = self.value(x).view(seq_len, self.num_heads, -1).transpose(0, 1)  # [num_heads, seq_len, hidden_dim]
        
        # Compute scaled dot-product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)  # [num_heads, seq_len, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)  # [num_heads, seq_len, hidden_dim]
        
        # Combine heads and project the output
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, -1)  # [seq_len, hidden_dim * num_heads]
        output = self.output_proj(attn_output)  # [seq_len, output_dim]
        
        return output, attn_weights

class AttentionSubsampling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2):
        super(AttentionSubsampling, self).__init__()
        self.layers = nn.ModuleList([MultiHeadAttention(input_dim, hidden_dim,output_dim ,num_heads) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x,out_dim):
        seq_len, _ = x.size()

        # Pass through each attention layer
        for layer in self.layers:
            x, _ = layer(x)

        # Take the first token's output as the representation of the entire sequence
        sel_token = x[:out_dim,:]  # [batch_size, hidden_dim]

        # # Project the output to the desired dimension
        # output = self.output_proj(sel_token)  # [batch_size, output_dim]

        return sel_token

aadic = {
    'A': 0,
    'B': 20,
    'C': 4,
    'D': 3,
    'E': 6,

    'F': 13,
    'G': 7,
    'H': 8,
    'I': 9,
    'J': 20,

    'K': 11,
    'L': 10,
    'M': 12,
    'N': 2,
    'O': 20,

    'P': 14,
    'Q': 5,
    'R': 1,
    'S': 15,
    'T': 16,
    'U': 20,

    'V': 19,
    'W': 17,
    'X': 20,
    'Y': 18,
    'Z': 20,
    '-': 21,
    '*': 21,
}

def dummy(seq,adic):
    # print("seq",seq)
    if pd.isna(seq):
        # if seq is nan, return all zeros by the avglen -> should consider and test if avglen*20 or 1*20
        return torch.tensor([])
        
    seqnpy=np.zeros(len(seq),dtype=int) + adic['-']
    seq1=np.array(list(seq))  
    keys = list(adic.keys())
    for akey in keys:
        seqnpy[seq1==akey] = adic[akey]
    return torch.tensor(np.eye(adic['-']+1)[seqnpy], dtype=torch.float32) 

def create_idx_tensor(sequence):
    if pd.notna(sequence) and len(sequence) > 0:
        return torch.arange(len(sequence), dtype=torch.float32) + 1
    else:
        return torch.tensor([])  # Create a zero tensor of the expected length

def mask_gen(L, additional_mask_positions=None):
    Themask = np.zeros(L)  # Initialize the mask with zeros
    
    if additional_mask_positions is not None:
        # Ensure the positions are within the valid range
        # print('additional_mask_positions',additional_mask_positions)
        for pos in additional_mask_positions:
            # print('pos',pos)
            if pos < 0 or pos > L:
                raise ValueError(f"Position {pos} is out of bounds for sequence length {L}.")
            Themask[pos] = 1  # Apply mask to the specified positions
        return Themask

    sel_num = random.randint(1, int(L * 0.9))  # Select between 1 to 90% of the sequence length
    idex = torch.randperm(L)
    random.shuffle(idex)  # Shuffle the indices
    maskids = idex[:sel_num]  # Select the first sel_num indices
    Themask[maskids] = 1  # Set the selected indices in the mask to 1
 
    return Themask

class DummyPeptideLM(Dataset):
    def __init__(self,seqfile):
        self.maxlength=256
        self.lines=open(seqfile).readlines()
        self.length = len(self.lines)
        print('length',self.length)

    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 
        seq = self.lines[idx].strip()
        seqnpy = dummy(seq,aadic)
        L = len(seq)
        idx = torch.arange(L) + 1
        thedata ={'pep':[],'pep_idx':[],'aa':seqnpy,'idx':idx}
        thedata['mask'] = mask_gen(len(thedata['idx']))
        return thedata

class Load_Dataset(Dataset):
    def __init__(self, csv_file):
        self.maxlength = 512
        self.df = pd.read_csv(csv_file)
        self.length = len(self.df)
        self.mask_dict = {}

        print('Test set length:', self.length)
        self.mhc_att = AttentionSubsampling(input_dim=22, hidden_dim=16, output_dim=22)
        self.genev_att = AttentionSubsampling(input_dim=22, hidden_dim=16, output_dim=22)

    def __len__(self):
        return self.length

    def __getitem__(self, idx, masklist=None, sequence=None):
        
        # idx is used for handling batch processing, each time only return one line of data!!
        if torch.is_tensor(idx): 
            idx = idx.tolist()
        line = self.df.iloc[idx].values.flatten()
        
        assert len(line) == 7, f"Expected 6 segments per line, got {len(line)}: {line}"

        pep, mhc, lv, lj, hv,hd, hj = [str(item).strip() if pd.notnull(item) else '' for item in line]
        
        # for evaluation of the iteratively masking
        if sequence is not None:
            hd = sequence

        # Convert sequences to numerical representations using dummy function
        pepnpy = dummy(pep, aadic)
        mhcnpy = dummy(mhc, aadic)
        lvnpy = dummy(lv, aadic)
        ljnpy = dummy(lj, aadic)
        hvnpy = dummy(hv, aadic)
        hdnpy = dummy(hd, aadic)
        hjnpy = dummy(hj, aadic)

        if mhcnpy.nelement() != 0:
            # print(mhcnpy.shape)
            mhcnpy = self.mhc_att(mhcnpy,96)
            # print(mhcnpy.shape[0])
        if hvnpy.nelement() != 0:
            hvnpy = self.genev_att(hvnpy,48)
        if lvnpy.nelement() != 0:
            lvnpy = self.genev_att(lvnpy,48)
        
        # print('hdnpy',hdnpy.shape)
        # print("mhc",mhcnpy.shape)
        # print("lv",lvnpy.shape)
        # print("lj",ljnpy.shape)
        # print("hv",hvnpy.shape)
        # print("hj",hjnpy.shape)

        # Generate indices for each segment, handling potential NaN values
        pep_idx = torch.arange(len(pep), dtype=torch.float32) + 1
        mhc_idx = torch.arange(mhcnpy.shape[0], dtype=torch.float32) + 1
        lv_idx = torch.arange(lvnpy.shape[0], dtype=torch.float32) + 1
        lj_idx = torch.arange(len(lj), dtype=torch.float32) + 1
        hv_idx = torch.arange(hvnpy.shape[0], dtype=torch.float32) + 1
        hd_idx = torch.arange(len(hd), dtype=torch.float32) + 1  # hd should not be NaN
        hj_idx = torch.arange(len(hj), dtype=torch.float32) + 1

        # Generate or retrieve mask
        if masklist is not None:
            mask = torch.tensor(mask_gen(len(hd),masklist), dtype=torch.float32)
        else:
            mask = torch.tensor(mask_gen(len(hd)), dtype=torch.float32) 

        thedata = {
            'pep': pepnpy,
            'pep_idx': pep_idx,
            'mhc': mhcnpy,
            'mhc_idx': mhc_idx,
            'lv': lvnpy,
            'lv_idx': lv_idx,
            'lj': ljnpy,
            'lj_idx': lj_idx,
            'hv': hvnpy,
            'hv_idx': hv_idx,
            'hd': hdnpy,
            'hd_idx': hd_idx,
            'hj': hjnpy,
            'hj_idx': hj_idx,
            'mask': mask
        }

        return thedata
