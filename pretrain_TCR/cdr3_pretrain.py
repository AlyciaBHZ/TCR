#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDR3β Pure Language Model Pretraining
Learning CDR3β sequence patterns without any conditioning information
This is crucial for:
1. Interpretability analysis
2. Visualization of intrinsic CDR3β patterns  
3. Scientific analysis of TCR intrinsic encoding capabilities
"""

import torch
import torch.optim as opt
import numpy as np
import pandas as pd
import math
import sys
import os
import argparse
import random
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import psiCLM, get_device, nll_loss_withmask
from data_clp import AA_DICT, AA_DIM, dummy, create_idx, mask_gen

# Config
BATCH_SIZE = 4096
TEST_STEP = 25
VISION_STEP = 10
ACCUMULATION_STEP = 4  # Gradient accumulation steps

class CDR3OnlyDataset:
    """Dataset containing only CDR3β sequences for pure language modeling"""
    
    def __init__(self, data_path, train_split=0.9, is_train=True, random_seed=42):
        print(f"Loading CDR3β-only data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Total records: {len(df):,}")
        
        # Clean data - only CDR3β sequences
        df = df.dropna(subset=["cdr3_b"])
        df = df[(df["cdr3_b"].str.len() >= 6) & (df["cdr3_b"].str.len() <= 25)]
        df = df.drop_duplicates(subset=["cdr3_b"])
        print(f"After cleaning: {len(df):,}")
        
        # Split train/test
        if len(df) > 1000:
            train_df, test_df = train_test_split(df, train_size=train_split, random_state=random_seed)
        else:
            train_df = df
            test_df = df.sample(min(100, len(df)), random_state=random_seed)
        
        self.df = train_df if is_train else test_df
        print(f"CDR3β-only dataset size: {len(self.df):,}")
        
        if is_train:
            cdr3_lens = df["cdr3_b"].str.len()
            print(f"CDR3β length stats: min={cdr3_lens.min()}, max={cdr3_lens.max()}, mean={cdr3_lens.mean():.1f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Only CDR3β sequence
        hd = dummy(row["cdr3_b"])
        hd_idx = create_idx(hd.shape[0]) if hd.shape[0] > 0 else torch.tensor([])
        mask = mask_gen(hd.shape[0], 0.15) if hd.shape[0] > 0 else torch.tensor([])
        
        return {
            "hd": hd,
            "hd_idx": hd_idx, 
            "mask": mask
        }

def cdr3_test(model, test_dataset):
    """Test function for CDR3β-only model"""
    model.eval()
    device = get_device()
    
    with torch.no_grad():
        losses = []
        n_test = min(200, len(test_dataset))
        
        for i in range(n_test):
            try:
                sample = test_dataset[i]
                sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                
                # No conditioning information - pure language modeling
                loss = model(sample, computeloss=True, conditioning_info=[])
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    losses.append(loss.item())
            except Exception as e:
                continue
    
    if not losses:
        return float("inf"), float("inf")
    
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss) if avg_loss < 10 else float("inf")
    return avg_loss, ppl

def pretrain_cdr3_model():
    """Main function for CDR3β-only pretraining"""
    print("Starting CDR3β Pure Language Model Pretraining...")
    print("=" * 60)
    print("This model learns intrinsic CDR3β patterns without conditioning")
    print("Key applications:")
    print("  - Interpretability analysis")
    print("  - Visualization of CDR3β structure")
    print("  - Scientific analysis of TCR encoding")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Smaller model for CDR3β-only (no need for complex conditioning)
    cfg = {
        "s_in_dim": 21, "z_in_dim": 2, "s_dim": 128, "z_dim": 64, "N_elayers": 4  # Reduced layers
    }
    
    model = psiCLM(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Slightly higher learning rate for simpler task
    optimizer = opt.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    save_dir = os.path.join(os.path.dirname(__file__), "saved_model", "cdr3_pretrain")
    os.makedirs(save_dir, exist_ok=True)
    
    data_path = os.path.join(os.path.dirname(__file__), "data", "tcrdb", "processed_tcrdb.csv")
    
    train_dataset = CDR3OnlyDataset(data_path, is_train=True)
    test_dataset = CDR3OnlyDataset(data_path, is_train=False)
    
    log_file = os.path.join(save_dir, "cdr3_training.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("CDR3β Pure Language Model Pretraining Log\n")
        f.write("No conditioning information - intrinsic sequence modeling\n")
        f.write(f"Training samples: {len(train_dataset):,}\n")
        f.write(f"Testing samples: {len(test_dataset):,}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Accumulation steps: {ACCUMULATION_STEP}\n")
        f.write(f"Test frequency: every {TEST_STEP} epochs\n")
        f.write("=" * 70 + "\n")
        f.write("Epoch,Train_Loss,Test_Loss,Test_PPL,Status,Notes\n")
    
    best_ppl = None
    better_count = 0
    current_epoch = 0
    
    print("\n" + "=" * 80)
    print("CDR3β PURE LANGUAGE MODEL - PROGRESS MONITOR")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Test Loss':>9} | {'Test PPL':>8} | {'Status':>8} | Notes")
    print("-" * 80)
    
    while True:
        model.train()
        
        batch_size = min(BATCH_SIZE, len(train_dataset))
        batch_idxs = np.random.choice(len(train_dataset), batch_size, replace=False)
        
        total_loss = 0
        successful_samples = 0
        
        optimizer.zero_grad()
        
        for i, idx in enumerate(batch_idxs):
            try:
                sample = train_dataset[idx]
                sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                
                # Pure language modeling - no conditioning
                loss = model(sample, computeloss=True, conditioning_info=[])
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Scale loss by accumulation steps
                loss = loss / ACCUMULATION_STEP
                loss.backward()
                total_loss += loss.item() * ACCUMULATION_STEP  # Scale back for logging
                successful_samples += 1
                
                # Perform optimizer step every ACCUMULATION_STEP samples
                if (i + 1) % ACCUMULATION_STEP == 0 or (i + 1) == len(batch_idxs):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
            except Exception as e:
                if current_epoch % TEST_STEP == 0:
                    print(f"Warning: Error at sample {idx}: {str(e)[:50]}...")
                continue
        
        if successful_samples > 0:
            avg_loss = total_loss / successful_samples
        else:
            avg_loss = float("inf")
            if current_epoch % TEST_STEP == 0:
                print(f"Warning: No successful samples in epoch {current_epoch}")
        
        # Test and save checkpoints every TEST_STEP epochs
        if current_epoch % TEST_STEP == 0:
            test_loss, test_ppl = cdr3_test(model, test_dataset)
            
            status = "OK"
            notes = ""
            
            # Always save checkpoint every TEST_STEP epochs
            checkpoint_path = os.path.join(save_dir, f"cdr3_model_epoch_{current_epoch}")
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), checkpoint_path + ".opt")
            
            if not (math.isnan(test_ppl) or math.isinf(test_ppl)):
                if best_ppl is None or test_ppl < best_ppl:
                    best_ppl = test_ppl
                    better_count = 0
                    status = "IMPROVED"
                    notes = f"PPL improved: {test_ppl:.2f}"
                else:
                    better_count += TEST_STEP
                    improvement = ((test_ppl - best_ppl) / best_ppl) * 100
                    notes = f"PPL: {test_ppl:.2f} (+{improvement:.1f}%), patience: {better_count}/128"
            else:
                notes = "Invalid metrics"
                status = "ERROR"
            
            print(f"{current_epoch:>6} | {avg_loss:>10.4f} | {test_loss:>9.4f} | {test_ppl:>8.2f} | {status:>8} | {notes}")
            
            # Log to file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{current_epoch},{avg_loss:.6f},{test_loss:.6f},{test_ppl:.4f},{status},{notes}\n")
            
            # Early stopping check
            if better_count >= 128:
                print("-" * 80)
                print("Early stopping: no improvement in perplexity for 128 test steps.")
                print(f"Best PPL achieved: {best_ppl:.2f}")
                print(f"Total epochs: {current_epoch}")
                print("CDR3β pure language model training completed!")
                break
        
        current_epoch += 1
    
    # Return the latest checkpoint path
    latest_epoch = current_epoch - (current_epoch % TEST_STEP)
    return os.path.join(save_dir, f"cdr3_model_epoch_{latest_epoch}")

if __name__ == "__main__":
    pretrain_cdr3_model() 