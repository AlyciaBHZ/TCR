#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDR3β-peptide-MHC Paired Data Finetuning
Based on pretrained CDR3β pure language model
Uses CollapseProteinDataset from data_clp.py directly
"""

import torch
import torch.optim as opt
import numpy as np
import pandas as pd
import math
import sys
import os
import argparse
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use explicit package imports from psi_model
from psi_model.model import psiCLM, get_device, nll_loss_withmask
from psi_model.data_clp import CollapseProteinDataset

# Config
BATCH_SIZE = 1024  # Smaller batch for complex paired data
TEST_STEP = 25
ACCUMULATION_STEP = 16  # More accumulation for stability

def paired_test(model, test_dataset):
    """Test function for paired data model using CollapseProteinDataset - following train.py style"""
    model.eval()
    device = get_device()
    
    # Conditioning fields from CollapseProteinDataset
    conditioning_info = ["pep", "mhc", "lv", "lj", "hv", "hj"]
    
    with torch.no_grad():
        losses = []
        n_test = min(300, len(test_dataset))  # Smaller test set due to complexity
        
        for i in range(n_test):
            try:
                sample = test_dataset[i]
                # Move tensors to device
                sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in sample.items()}
                
                # 关键修复：测试时创建全1的mask（不mask任何位置）- 参考train.py
                sample_copy = sample.copy()
                hd_len = sample_copy['hd'].shape[0]
                sample_copy['mask'] = torch.ones(hd_len, device=device)  # 全1 = 不mask任何位置
                
                # Use the same conditioning approach as train.py
                loss = model(sample_copy, computeloss=True, conditioning_info=conditioning_info)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    losses.append(loss.item())
            except Exception as e:
                continue
    
    if not losses:
        return float("inf"), float("inf")
    
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss) if avg_loss < 10 else float("inf")
    return avg_loss, ppl

def finetune_paired_model(pretrained_path):
    """Main function for paired data finetuning using CollapseProteinDataset"""
    print("Starting CDR3β-Peptide-MHC Paired Data Finetuning...")
    print("Using CollapseProteinDataset from data_clp.py")
    print("=" * 70)
    print(f"Pretrained model: {pretrained_path}")
    print("Training data: ../data/trn.csv")
    print("Testing data:  ../data/tst.csv")
    print("=" * 70)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load pretrained model with same architecture
    cfg = {
        "s_in_dim": 21, "z_in_dim": 2, "s_dim": 128, "z_dim": 64, "N_elayers": 4
    }
    
    model = psiCLM(cfg).to(device)
    
    # Lower learning rate for finetuning
    optimizer = opt.Adam(model.parameters(), lr=2e-5, weight_decay=1e-5)
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), "saved_model", "paired_finetune")
    os.makedirs(save_dir, exist_ok=True)
    
    # Resume from checkpoint if possible - 参考train.py的实现
    start_epoch = 0
    best_ppl = None
    better_count = 0
    
    # Check for existing checkpoints
    files = [f for f in os.listdir(save_dir) if f.startswith('paired_model_epoch_') and not f.endswith('.opt')]
    if files:
        epochs = [int(re.findall(r'\d+', f)[0]) for f in files]
        latest = max(epochs)
        checkpoint_path = os.path.join(save_dir, f'paired_model_epoch_{latest}')
        print(f"Found existing checkpoint: {checkpoint_path}")
        
        try:
            # Load model state
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            # Load optimizer state
            optimizer.load_state_dict(torch.load(checkpoint_path + '.opt', map_location=device))
            start_epoch = latest + TEST_STEP  # Continue from next test step
            print(f"Successfully resumed from epoch {latest}, continuing from epoch {start_epoch}")
            
            # Try to restore best_ppl from log file
            log_file = os.path.join(save_dir, "paired_training.log")
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                    # Find the best PPL from previous training
                    best_ppls = []
                    for line in lines:
                        if 'IMPROVED' in line and 'PPL improved:' in line:
                            ppl_str = line.split('PPL improved: ')[1].split()[0]
                            try:
                                best_ppls.append(float(ppl_str))
                            except:
                                continue
                    if best_ppls:
                        best_ppl = min(best_ppls)
                        print(f"Restored best PPL from log: {best_ppl:.2f}")
                except Exception as e:
                    print(f"Could not restore best PPL from log: {e}")
                    
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh training...")
            start_epoch = 0
    else:
        # Load pretrained weights for initial training
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print("Pretrained weights loaded successfully!")
        else:
            print(f"Warning: Pretrained model not found at {pretrained_path}")
            print("Starting from random initialization...")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Use pre-split datasets directly
    train_dataset = CollapseProteinDataset('../data/trn.csv', parse_mode='line-positional')
    test_dataset = CollapseProteinDataset('../data/tst.csv', parse_mode='line-positional')
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Testing samples: {len(test_dataset):,}")
    
    # Conditioning fields from CollapseProteinDataset
    conditioning_info = ["pep", "mhc", "lv", "lj", "hv", "hj"]
    print(f"Using conditioning info: {conditioning_info}")
    
    # Setup logging - append mode if resuming
    log_file = os.path.join(save_dir, "paired_training.log")
    log_mode = "a" if start_epoch > 0 else "w"
    
    if log_mode == "w":
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("CDR3β-Peptide-MHC Paired Data Finetuning Log\n")
            f.write("Using CollapseProteinDataset from data_clp.py\n")
            f.write(f"Pretrained model: {pretrained_path}\n")
            f.write(f"Training samples: {len(train_dataset):,}\n")
            f.write(f"Testing samples: {len(test_dataset):,}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Accumulation steps: {ACCUMULATION_STEP}\n")
            f.write(f"Test frequency: every {TEST_STEP} epochs\n")
            f.write(f"Conditioning info: {conditioning_info}\n")
            f.write("=" * 70 + "\n")
            f.write("Epoch,Train_Loss,Test_Loss,Test_PPL,Status,Notes\n")
    else:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n# Resumed training from epoch {start_epoch}\n")
    
    current_epoch = start_epoch
    
    print("\n" + "=" * 80)
    print("PAIRED DATA FINETUNING - PROGRESS MONITOR")
    if start_epoch > 0:
        print(f"RESUMING FROM EPOCH {start_epoch}")
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
                # Move tensors to device
                sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in sample.items()}
                
                # Use the same conditioning approach as train.py
                loss = model(sample, computeloss=True, conditioning_info=conditioning_info)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Scale loss by accumulation steps
                loss = loss / ACCUMULATION_STEP
                loss.backward()
                total_loss += loss.item() * ACCUMULATION_STEP
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
            test_loss, test_ppl = paired_test(model, test_dataset)
            
            status = "OK"
            notes = ""
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"paired_model_epoch_{current_epoch}")
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), checkpoint_path + ".opt")
            
            if not (math.isnan(test_ppl) or math.isinf(test_ppl)):
                if best_ppl is None or test_ppl < best_ppl:
                    best_ppl = test_ppl
                    better_count = 0
                    status = "IMPROVED"
                    notes = f"PPL improved: {test_ppl:.2f}"
                    
                    # Save best model
                    best_model_path = os.path.join(save_dir, "best_paired_model")
                    torch.save(model.state_dict(), best_model_path)
                else:
                    better_count += TEST_STEP
                    improvement = ((test_ppl - best_ppl) / best_ppl) * 100
                    notes = f"PPL: {test_ppl:.2f} (+{improvement:.1f}%), patience: {better_count}/150"
            else:
                notes = "Invalid metrics"
                status = "ERROR"
            
            print(f"{current_epoch:>6} | {avg_loss:>10.4f} | {test_loss:>9.4f} | {test_ppl:>8.2f} | {status:>8} | {notes}")
            
            # Log to file
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{current_epoch},{avg_loss:.6f},{test_loss:.6f},{test_ppl:.4f},{status},{notes}\n")
            
            # Early stopping check (longer patience for finetuning)
            if better_count >= 150:  # 增加patience
                print("-" * 80)
                print("Early stopping: no improvement in perplexity for 150 test steps.")
                print(f"Best PPL achieved: {best_ppl:.2f}")
                print(f"Total epochs: {current_epoch}")
                print("Paired data finetuning completed!")
                break
        
        current_epoch += 1
        
        # Maximum epochs limit for finetuning
        if current_epoch >= 500:  # 增加最大epochs
            print("-" * 80)
            print("Maximum epochs reached (500). Stopping training.")
            print(f"Final PPL: {test_ppl:.2f}")
            print("Paired data finetuning completed!")
            break
    
    # Return the best model path
    return os.path.join(save_dir, "best_paired_model")

if __name__ == "__main__":
    # Fixed pretrained model path
    pretrained_path = "saved_model/cdr3_pretrain/cdr3_model_epoch_500"
    
    if not os.path.exists(pretrained_path):
        print(f"Error: Pretrained model not found at {pretrained_path}")
        print("Available checkpoints:")
        checkpoint_dir = "saved_model/cdr3_pretrain/"
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.startswith('cdr3_model_epoch_')]
            for f in sorted(files):
                print(f"  {f}")
        sys.exit(1)
    
    best_model_path = finetune_paired_model(pretrained_path)
    print(f"\nBest model saved at: {best_model_path}") 
