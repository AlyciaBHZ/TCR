#!/usr/bin/env python
"""
FlowTCR-Gen Training Script (Stage 2)

Trains the topology-aware Dirichlet flow matching model for CDR3β generation.

Key features:
- Collapse Token + Hierarchical Pairs (from psi_model)
- Dirichlet Flow Matching on amino acid simplex
- CFG (Classifier-Free Guidance) support
- Model Score Hook for Stage 3 integration

Usage:
    # Default training (with ablation evaluation)
    python flowtcr_fold/FlowTCR_Gen/train.py

    # Ablation: No collapse token
    python flowtcr_fold/FlowTCR_Gen/train.py --ablation no_collapse

    # Ablation: No hierarchical pairs
    python flowtcr_fold/FlowTCR_Gen/train.py --ablation no_hier

    # Resume training
    python flowtcr_fold/FlowTCR_Gen/train.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flowtcr_fold.FlowTCR_Gen.model_flow import FlowTCRGen
from flowtcr_fold.FlowTCR_Gen.data import FlowTCRGenDataset, FlowTCRGenTokenizer, create_collate_fn
from flowtcr_fold.FlowTCR_Gen.metrics import FlowTCRGenEvaluator, compute_recovery_rate, compute_diversity


# =============================================================================
# Config (fixed paths/defaults per AGENTS.md)
# =============================================================================
DEFAULT_DATA_PATH = "flowtcr_fold/data/trn.jsonl"
DEFAULT_VAL_PATH = "flowtcr_fold/data/val.jsonl"
DEFAULT_OUT_DIR = "flowtcr_fold/FlowTCR_Gen/saved_model/stage2"

# Model config (fixed)
S_DIM = 256
Z_DIM = 64
N_LAYERS = 6
VOCAB_SIZE = 25  # 20 AA + 5 special

# Training config (fixed)
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5
CFG_DROP_PROB = 0.1
LAMBDA_ENTROPY = 0.01
LAMBDA_PROFILE = 0.01
PATIENCE = 20


def parse_args():
    p = argparse.ArgumentParser(description="FlowTCR-Gen Training")
    p.add_argument("--ablation", type=str, default=None, 
                   choices=["no_collapse", "no_hier", "no_cfg"],
                   help="Ablation mode: no_collapse, no_hier, no_cfg")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--resume_best", action="store_true", help="Resume from best checkpoint")
    p.add_argument("--eval_only", action="store_true", help="Evaluation only")
    p.add_argument("--cfg_weight", type=float, default=1.5, help="CFG weight for generation")
    return p.parse_args()


def get_output_dir(ablation: Optional[str]) -> Path:
    """Get output directory based on ablation mode."""
    if ablation is None:
        return Path(DEFAULT_OUT_DIR)
    return Path(DEFAULT_OUT_DIR) / f"ablation_{ablation}"


def create_model(ablation: Optional[str], device: torch.device) -> FlowTCRGen:
    """Create model with ablation settings."""
    use_collapse = True
    use_hier_pairs = True
    cfg_drop_prob = CFG_DROP_PROB
    
    if ablation == "no_collapse":
        use_collapse = False
    elif ablation == "no_hier":
        use_hier_pairs = False
    elif ablation == "no_cfg":
        cfg_drop_prob = 0.0
    
    model = FlowTCRGen(
        s_dim=S_DIM,
        z_dim=Z_DIM,
        n_layers=N_LAYERS,
        vocab_size=VOCAB_SIZE,
        cfg_drop_prob=cfg_drop_prob,
        use_collapse=use_collapse,
        use_hier_pairs=use_hier_pairs,
        lambda_entropy=LAMBDA_ENTROPY,
        lambda_profile=LAMBDA_PROFILE,
    )
    
    return model.to(device)


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """Create AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, num_epochs: int, warmup_epochs: int, steps_per_epoch: int):
    """Create learning rate scheduler with warmup."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: FlowTCRGen,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch with per-sample conditioning."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_entropy = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move tensors to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Handle nested dicts (scaffold_tokens and scaffold_mask)
        for key in ['scaffold_tokens', 'scaffold_mask']:
            if key in batch and isinstance(batch[key], dict):
                batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch[key].items()}
        
        # Forward
        optimizer.zero_grad()
        losses = model.training_step(batch)
        
        # Backward
        loss = losses['loss']
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_mse += losses['mse_loss'].item()
        total_entropy += losses['entropy_loss'].item()
        n_batches += 1
        
        # Log every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  [Epoch {epoch+1} Batch {batch_idx+1}] "
                  f"loss={loss.item():.4f} mse={losses['mse_loss'].item():.4f}")
    
    return {
        'loss': total_loss / n_batches,
        'mse_loss': total_mse / n_batches,
        'entropy_loss': total_entropy / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: FlowTCRGen,
    val_loader: DataLoader,
    tokenizer: FlowTCRGenTokenizer,
    device: torch.device,
    cfg_weight: float = 1.5,
    n_samples: int = 3,
) -> Dict[str, float]:
    """Evaluate model on validation set with per-sample conditioning."""
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    generated_seqs = []
    ground_truth_seqs = []
    
    for batch in val_loader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Handle nested dicts (scaffold_tokens and scaffold_mask)
        for key in ['scaffold_tokens', 'scaffold_mask']:
            if key in batch and isinstance(batch[key], dict):
                batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch[key].items()}
        
        # Compute validation loss
        losses = model.forward(
            cdr3_tokens=batch['cdr3_tokens'],
            cdr3_mask=batch['cdr3_mask'],
            pep_tokens=batch['pep_tokens'],
            pep_mask=batch['pep_mask'],
            mhc_tokens=batch['mhc_tokens'],
            mhc_mask=batch['mhc_mask'],
            scaffold_tokens=batch['scaffold_tokens'],
            scaffold_mask=batch['scaffold_mask'],
            training=False,
        )
        total_loss += losses['loss'].item()
        n_batches += 1
        
        # Generate samples for recovery/diversity metrics (per-sample conditioning)
        B = batch['cdr3_tokens'].shape[0]
        for i in range(min(B, n_samples)):
            cdr3_len = int(batch['cdr3_mask'][i].sum().item())
            
            # Extract per-sample conditioning with batch dim
            pep_tokens_i = batch['pep_tokens'][i:i+1]
            pep_mask_i = batch['pep_mask'][i:i+1]
            mhc_tokens_i = batch['mhc_tokens'][i:i+1]
            mhc_mask_i = batch['mhc_mask'][i:i+1]
            scaffold_tokens_i = {k: v[i:i+1] for k, v in batch['scaffold_tokens'].items()}
            scaffold_mask_i = {k: v[i:i+1] for k, v in batch['scaffold_mask'].items()}
            
            tokens = model.generate(
                cdr3_len=cdr3_len,
                pep_tokens=pep_tokens_i,
                pep_mask=pep_mask_i,
                mhc_tokens=mhc_tokens_i,
                mhc_mask=mhc_mask_i,
                scaffold_tokens=scaffold_tokens_i,
                scaffold_mask=scaffold_mask_i,
                n_steps=50,
                cfg_weight=cfg_weight,
            )
            
            gen_seq = tokenizer.decode(tokens.tolist())
            gt_seq = batch['cdr3_seqs'][i]
            
            generated_seqs.append(gen_seq)
            ground_truth_seqs.append(gt_seq)
    
    # Compute metrics
    metrics = {
        'val_loss': total_loss / n_batches,
    }
    
    if generated_seqs:
        recovery = compute_recovery_rate(generated_seqs, ground_truth_seqs)
        diversity = compute_diversity(generated_seqs)
        
        metrics['recovery_exact'] = recovery['exact_match']
        metrics['recovery_80'] = recovery['partial_match_80']
        metrics['diversity_ratio'] = diversity['unique_ratio']
    
    return metrics


def save_checkpoint(
    model: FlowTCRGen,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    out_dir: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Save latest
    torch.save(ckpt, out_dir / "checkpoints" / "latest.pt")
    
    # Save periodic checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save(ckpt, out_dir / "checkpoints" / f"epoch_{epoch+1}.pt")
    
    # Save best
    if is_best:
        torch.save(ckpt, out_dir / "best_model" / "model.pt")


def load_checkpoint(model: FlowTCRGen, optimizer, path: Path, device: torch.device):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt.get('metrics', {})


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 FlowTCR-Gen Training (Stage 2)")
    print(f"   Device: {device}")
    print(f"   Ablation: {args.ablation or 'None (full model)'}")
    
    # Setup output directory
    out_dir = get_output_dir(args.ablation)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "best_model").mkdir(parents=True, exist_ok=True)
    (out_dir / "other_results").mkdir(parents=True, exist_ok=True)
    
    # Create tokenizer and dataset
    tokenizer = FlowTCRGenTokenizer()
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # Check if data exists
    data_path = Path(DEFAULT_DATA_PATH)
    if not data_path.exists():
        print(f"⚠️  Data file not found: {data_path}")
        print("   Creating mock data for testing...")
        # Create mock data
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with data_path.open('w') as f:
            for i in range(100):
                sample = {
                    'peptide': 'GILGFVFTL',
                    'mhc': 'HLA-A*02:01',
                    'cdr3_b': 'CASSLGQFF',
                    'h_v': 'TRBV19*01',
                    'h_j': 'TRBJ2-7*01',
                }
                f.write(json.dumps(sample) + '\n')
    
    train_ds = FlowTCRGenDataset(str(data_path), tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True,
    )
    
    # Validation loader (use same data if no val file)
    val_path = Path(DEFAULT_VAL_PATH)
    if val_path.exists():
        val_ds = FlowTCRGenDataset(str(val_path), tokenizer=tokenizer)
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            collate_fn=create_collate_fn(tokenizer),
        )
    else:
        val_loader = train_loader  # Use train for validation (for testing)
    
    # Create model
    model = create_model(args.ablation, device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, LR, WEIGHT_DECAY)
    scheduler = create_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS, len(train_loader))
    
    # Resume if requested
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume or args.resume_best:
        ckpt_path = out_dir / "best_model" / "model.pt" if args.resume_best else out_dir / "checkpoints" / "latest.pt"
        if ckpt_path.exists():
            saved_epoch, metrics = load_checkpoint(model, optimizer, ckpt_path, device)
            start_epoch = saved_epoch + 1  # Resume from NEXT epoch
            best_val_loss = metrics.get('val_loss', float('inf'))
            # Advance scheduler to correct position
            for _ in range(start_epoch * len(train_loader)):
                scheduler.step()
            print(f"   Resumed from epoch {saved_epoch + 1}, continuing from epoch {start_epoch + 1}, val_loss={best_val_loss:.4f}")
        else:
            print(f"   No checkpoint found at {ckpt_path}")
    
    # Eval only mode
    if args.eval_only:
        print("\n📊 Evaluation only mode")
        metrics = evaluate(model, val_loader, tokenizer, device, args.cfg_weight)
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        return
    
    # Training loop
    print(f"\n🏃 Starting training from epoch {start_epoch + 1}")
    patience_counter = 0
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, tokenizer, device, args.cfg_weight)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"\n📈 Epoch {epoch + 1}/{EPOCHS} ({epoch_time:.1f}s)")
        print(f"   Train: loss={train_metrics['loss']:.4f}, mse={train_metrics['mse_loss']:.4f}")
        print(f"   Val: loss={val_metrics['val_loss']:.4f}, "
              f"recovery={val_metrics.get('recovery_exact', 0):.3f}, "
              f"diversity={val_metrics.get('diversity_ratio', 0):.3f}")
        
        # Check for improvement
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            print(f"   ⭐ New best model!")
        else:
            patience_counter += 1
        
        # Save checkpoint
        all_metrics = {**train_metrics, **val_metrics}
        save_checkpoint(model, optimizer, epoch, all_metrics, out_dir, is_best)
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation
    print("\n📊 Final Evaluation")
    final_metrics = evaluate(model, val_loader, tokenizer, device, args.cfg_weight, n_samples=10)
    for k, v in final_metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # Save final metrics
    with (out_dir / "other_results" / "final_metrics.json").open('w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n✅ Training complete! Best val_loss: {best_val_loss:.4f}")
    print(f"   Checkpoints: {out_dir / 'checkpoints'}")
    print(f"   Best model: {out_dir / 'best_model'}")


if __name__ == "__main__":
    main()
