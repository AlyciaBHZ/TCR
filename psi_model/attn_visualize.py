# visualize_attention_labeled.py
# Visualize psiCLM attention with dual-view per layer (collapse and hd), with region labels

import torch
import matplotlib.pyplot as plt
from model import psiCLM
from data_clp import CollapseProteinDataset
import os


def visualize_attention(model_ckpt_path, sample_idx, csv_path, cfg, conditioning_info):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = psiCLM(cfg).to(device)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model.eval()

    dataset = CollapseProteinDataset(csv_path)
    sample = dataset[sample_idx]
    sample = {k: v.to(device) for k, v in sample.items()}

    with torch.no_grad():
        _, attn_traces = model(sample, computeloss=False, conditioning_info=conditioning_info)

    # Token region boundaries
    boundaries = {}
    pointer = 1  # skip collapse token
    for field in ['hd'] + conditioning_info:
        seq = sample[field]
        if seq is not None and seq.ndim == 2:
            L = seq.shape[0]
            boundaries[field] = (pointer, pointer + L)
            pointer += L

    def plot_attn_row(attn_rows, title, ax, boundaries):
        im = ax.imshow(attn_rows, aspect='auto', cmap='viridis')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Token Index", fontsize=10)
        
        for name, (start, end) in boundaries.items():
            ax.axvline(start, color='white', linestyle='--', linewidth=1, alpha=0.8)
            ax.axvline(end, color='white', linestyle='--', linewidth=1, alpha=0.8)
            
            mid_point = (start + end) / 2
            ax.text(mid_point, -0.5, name.upper(), 
                    ha='center', va='top', color='white', 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        return im

    for layer_idx, attn in enumerate(attn_traces):
        attn = attn[0].detach().cpu()  # shape: (L, L)
        collapse_row = attn[0:1]  # collapse token attention
        hd_start, hd_end = boundaries['hd']
        hd_rows = attn[hd_start:hd_end]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        plot_attn_row(collapse_row, f"First Token → All Tokens Attention (Layer {layer_idx})", ax1, boundaries)
        
        plot_attn_row(hd_rows, f" HD Tokens → All Tokens Attention (Layer {layer_idx})", ax2, boundaries)
        
        plt.tight_layout()
        fname = f"./attn_visualize/attn_dual_layer{layer_idx}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved {fname}")
        plt.close()


if __name__ == "__main__":
    cfg = {
        's_in_dim': 21,
        'z_in_dim': 2,
        's_dim': 128,
        'z_dim': 64,
        'N_elayers': 8
    }
    
    # First, let's inspect what keys are available in the dataset
    dataset = CollapseProteinDataset("../data/trn.csv")
    sample = dataset[0]
    print("Available keys in sample:", list(sample.keys()))
    
    # Now run visualization with correct conditioning_info
    # You'll need to update this list based on the printed keys
    visualize_attention(
        model_ckpt_path="saved_model/condition_1/model_epoch_650",
        sample_idx=0,
        csv_path="../data/tst.csv",
        cfg=cfg,
        conditioning_info=['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']  # Update this based on printed keys
    )
