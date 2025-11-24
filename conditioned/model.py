from numpy import select
import torch
from torch import nn
from torch.nn import functional as F
from src import Evoformer, EvoPair, EvoMSA
import math, sys
from torch.utils.checkpoint import checkpoint
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
    def forward(self, x):
        x = self.linear(x)
        return x

def one_d(idx_, d, max_len=2056*1):
    device = get_device()
    idx = idx_[None].to(device)
    K = torch.arange(d//2).to(idx)
    sin_e = torch.sin(idx[..., None] * math.pi / (max_len**(2*K[None]/d)))
    cos_e = torch.cos(idx[..., None] * math.pi / (max_len**(2*K[None]/d)))
    return torch.cat([sin_e, cos_e], axis=-1)[0]

def nll_loss_withmask(pred, native, mask):
    device = get_device()
    pred = pred.to(device)
    native = native.to(device)
    mask = mask.to(device)
    return (-(pred * native * mask).sum()) / (mask.sum())

class Embedding(nn.Module):
    def __init__(self, cfg):
        super(Embedding, self).__init__()
        self.s_in_dim = cfg['s_in_dim']  # Sequence input dimension
        self.z_in_dim = cfg['z_in_dim']  # Pairwise input dimension
        self.s_dim = cfg['s_dim']
        self.z_dim = cfg['z_dim']
        self.qlinear = Linear(self.s_in_dim + 1, self.z_dim)
        self.klinear = Linear(self.s_in_dim + 1, self.z_dim)
        self.slinear = Linear(self.s_in_dim + 1, self.s_dim)
        self.zlinear = Linear(self.z_in_dim + 1, self.z_dim)
        self.poslinears = Linear(64, self.s_dim)
        self.poslinearz = Linear(64, self.z_dim)
        self.pair_linear = Linear(28, self.z_dim)  # 28 pair types as per your specification

    def emb_seq(self, aa, mask, idx):
        device = get_device()
        L = aa.shape[0]
        aamask = mask[:, None].to(device)
        aa = aa.to(device)

        s = torch.cat([aamask, (1 - aamask) * aa], dim=-1).to(device)

        sq = self.qlinear(s)
        sk = self.klinear(s)
        z = sq[None, :, :] + sk[:, None, :]

        seq_idx = idx[None].to(device)
        relative_pos = seq_idx[:, :, None] - seq_idx[:, None, :]
        relative_pos = relative_pos.reshape([1, -1])

        relative_pos = one_d(relative_pos, 64)
        z = z + self.poslinearz(relative_pos.reshape([1, L, L, -1])[0])
        s = self.slinear(s) + self.poslinears(one_d(idx, 64))

        return s, z

    def forward(self, in_dict, conditioning_info):
        device = get_device()
        s_hd, z_hd = self.emb_seq(in_dict['hd'], in_dict['mask'], in_dict['hd_idx'])
        L = s_hd.shape[0]
        slist = [s_hd]
        zlist = [z_hd]

        # Include sequences based on conditioning_info
        for key in ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']:
            if key in conditioning_info and key in in_dict and in_dict[key].shape[0] > 0:
                seq = in_dict[key]
                idx = in_dict[f'{key}_idx']
                mask = torch.zeros(seq.shape[0])  # Assuming no mask for conditioning sequences
                s, z = self.emb_seq(seq, mask, idx)
                slist.append(s)
                zlist.append(z)
                # print(f'Including {key} sequence with shape {seq.shape}')
            else:
                continue  # Skip if key not in conditioning_info or sequence is empty

        s_great = torch.cat(slist, dim=0)
        total_L = s_great.shape[0]

        z_great = torch.zeros((total_L, total_L, self.z_dim), device=device)

        start_idx = 0
        for z in zlist:
            end_idx = start_idx + z.shape[0]
            z_great[start_idx:end_idx, start_idx:end_idx, :] = z
            start_idx = end_idx

        # Pairwise interaction logic
        z_pair = torch.zeros([total_L, total_L]).long().to(device)

        # Assign pair types for peptide conditioned on MHC and CDRs
        hd_length = in_dict['hd'].shape[0]
        indices = {'hd': (0, hd_length)}
        current_start = hd_length

        for key in ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']:
            if key in conditioning_info and key in in_dict and in_dict[key].shape[0] > 0:
                seq_len = in_dict[key].shape[0]
                indices[key] = (current_start, current_start + seq_len)
                current_start += seq_len
            else:
                indices[key] = (current_start, current_start)  # No length added

        # Intra-region pair assignments
        pair_counter = 0
        region_keys = ['hd'] + [key for key in ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'] if key in conditioning_info and in_dict[key].shape[0] > 0]
        region_indices = [indices[key] for key in region_keys]

        # Assign intra-region pair values
        for start, end in region_indices:
            if end > start:
                z_pair[start:end, start:end] = pair_counter
                pair_counter += 1

        # Inter-region pair assignments
        for i, (start_i, end_i) in enumerate(region_indices):
            for j, (start_j, end_j) in enumerate(region_indices):
                if i < j:
                    if end_i > start_i and end_j > start_j:
                        z_pair[start_i:end_i, start_j:end_j] = pair_counter
                        z_pair[start_j:end_j, start_i:end_i] = pair_counter  # Mirror assignment
                        pair_counter += 1

        # Convert pair indices to one-hot encoding and apply the pair linear transformation
        max_pairs = 28  # Ensure max pairs match the expected input size for pair_linear
        z_pair = z_pair.clamp(max=max_pairs - 1)  # Clamp in case pair_counter exceeds max_pairs
        z_pair = self.pair_linear(F.one_hot(z_pair, max_pairs).float())

        return s_great, z_pair + z_great

class Embedding2nd(nn.Module):
    def __init__(self, cfg):
        super(Embedding2nd, self).__init__()
        self.s_in_dim = cfg['s_in_dim']
        self.z_in_dim = cfg['z_in_dim']
        self.s_dim = cfg['s_dim']
        self.z_dim = cfg['z_dim']
        self.N_elayers = cfg['N_elayers']
        self.emb = Embedding(cfg)
        self.evmodel = Evoformer.Evoformer(self.s_dim, self.z_dim, self.N_elayers)
        self.seq_head = Linear(self.s_dim, self.s_in_dim)
        self.ss_head = Linear(self.z_dim, self.z_in_dim)

    def forward(self, in_dict, computeloss, conditioning_info=None):
        if conditioning_info is None:
            conditioning_info = []  # Default to no conditioning if not provided

        # Use existing embedding and evoformer model for conditioned learning
        s, z = self.embedding(in_dict, conditioning_info)
        L1 = in_dict['hd'].shape[0]
        pred_aa = self.seq_head(s[:L1])
        # else:
        #     # Non-conditioned mode: only use the sequence of interest (e.g., `hd`) without conditioning on other sequences
        #     s, z = self.non_conditioned_embedding(in_dict['hd'], in_dict['mask'], in_dict['hd_idx'])
        #     pred_aa = self.seq_head(s)

        if not computeloss:
            return torch.softmax(pred_aa, dim=-1)
        else:
            pred_aa = torch.log_softmax(pred_aa, dim=-1)
            aaloss = nll_loss_withmask(pred_aa, in_dict['hd'], in_dict['mask'][:, None])
            return aaloss

    def embedding(self, in_dict, conditioning_info):
        s, z = self.emb(in_dict, conditioning_info)
        s, z = self.evmodel(s[None, ...], z)
        return s[0], z

if __name__ == "__main__":
    device = get_device()
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = {}
    cfg['s_in_dim'] = 22
    cfg['z_in_dim'] = 2
    cfg['s_dim'] = 32
    cfg['z_dim'] = 16
    cfg['N_elayers'] = 2

    in_dict = {}

    import data
    tdata = data.Test_Dataset('../data/final_data/tst.csv', None)
    in_dict = tdata.__getitem__(5)

    import torch.optim as opt
    model = Embedding2nd(cfg)
    optimizer = opt.Adam(model.parameters(), lr=0.001)
    conditioning_info = ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']  # Example conditioning info

    for jj in range(10):
        optimizer.zero_grad()
        oneloss = model(in_dict, True, conditioning_info=conditioning_info)
        print(oneloss)
        oneloss.backward()
        optimizer.step()
