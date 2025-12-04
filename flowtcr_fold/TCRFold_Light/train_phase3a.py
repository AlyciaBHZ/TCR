#!/usr/bin/env python
"""
Phase 3A: PPI Structure Pretraining for TCRFold-Prophet.

Training objective:
- Distance prediction (cross-entropy over 64 bins, 2-22Å)
- Contact prediction (binary cross-entropy with interface weighting)

Data: 76,407 merged PPI samples from ppi_merged/

Targets:
- Contact Precision > 50%
- Distance MAE < 2.0 Å

Usage:
    # Full training
    python train_phase3a.py --data_dir flowtcr_fold/data/ppi_merged --epochs 100
    
    # Quick test
    python train_phase3a.py --data_dir flowtcr_fold/data/ppi_merged --epochs 2 --max_samples 100

Outputs:
    checkpoints/stage3_phase_a/
    ├── best.pt           # Best validation loss checkpoint
    ├── last.pt           # Last epoch checkpoint
    ├── config.json       # Training config
    └── train.log         # Training log
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset, random_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from flowtcr_fold.TCRFold_Light.ppi_dataset import PPIDataset, collate_ppi_batch
from flowtcr_fold.TCRFold_Light.tcrfold_prophet import (
    TCRFoldProphet,
    compute_structure_loss,
    compute_distance_loss,
    compute_contact_loss,
)


def setup_logging(log_dir: Path, log_file: str = "train.log") -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("Phase3A")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_dir / log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    pred_logits: torch.Tensor,
    pred_contacts: torch.Tensor,
    target_dist: torch.Tensor,
    target_contacts: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Returns:
        dict with:
        - contact_precision: TP / (TP + FP)
        - contact_recall: TP / (TP + FN)
        - contact_f1: 2 * P * R / (P + R)
        - dist_mae: Mean absolute error of predicted distances
    """
    with torch.no_grad():
        # Contact metrics
        pred_contacts_binary = (torch.sigmoid(pred_contacts) > 0.5).float()
        
        if mask is not None:
            pred_contacts_binary = pred_contacts_binary * mask
            target_contacts_masked = target_contacts * mask
        else:
            target_contacts_masked = target_contacts
        
        tp = ((pred_contacts_binary == 1) & (target_contacts_masked == 1)).sum().item()
        fp = ((pred_contacts_binary == 1) & (target_contacts_masked == 0)).sum().item()
        fn = ((pred_contacts_binary == 0) & (target_contacts_masked == 1)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Distance metrics - convert logits to predicted distances
        # Use bin centers (2-22Å, 64 bins)
        n_bins = pred_logits.size(-1)
        bin_edges = torch.linspace(2.0, 22.0, n_bins + 1, device=pred_logits.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        pred_bins = pred_logits.argmax(dim=-1)
        pred_dist = bin_centers[pred_bins]
        
        # Clamp target distances to [2, 22]
        target_dist_clamped = target_dist.clamp(2.0, 22.0)
        
        if mask is not None:
            diff = (pred_dist - target_dist_clamped).abs() * mask
            dist_mae = diff.sum() / (mask.sum() + 1e-8)
        else:
            dist_mae = (pred_dist - target_dist_clamped).abs().mean()
        
        return {
            'contact_precision': precision,
            'contact_recall': recall,
            'contact_f1': f1,
            'dist_mae': dist_mae.item(),
        }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_loss_dist = 0.0
    total_loss_contact = 0.0
    total_samples = 0
    
    metrics_accum = {'contact_precision': 0, 'contact_recall': 0, 'contact_f1': 0, 'dist_mae': 0}
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        seq = batch['merged_seq'].to(device)
        pair_type = batch['pair_type'].to(device)
        mask = batch['merged_mask'].to(device)
        
        # Build target distance/contact from individual chains
        # The model outputs [B, L, L] where L = L_a + L_b
        B = seq.size(0)
        L = seq.size(1)
        
        # Reconstruct full distance matrix
        dist_a_b = batch['distance_map'].to(device)  # [B, L_a, L_b]
        L_a = dist_a_b.size(1)
        L_b = dist_a_b.size(2)
        
        # Full distance matrix [B, L, L]
        target_dist = torch.zeros(B, L, L, device=device)
        target_dist[:, :L_a, L_a:L_a+L_b] = dist_a_b
        target_dist[:, L_a:L_a+L_b, :L_a] = dist_a_b.transpose(1, 2)
        # Intra-chain distances (diagonal blocks) - set to large value since we don't have them
        target_dist[:, :L_a, :L_a] = 20.0  # Large distance for intra-A
        target_dist[:, L_a:L_a+L_b, L_a:L_a+L_b] = 20.0  # Large distance for intra-B
        
        # Contact map
        contact_a_b = batch['contact_map'].to(device)  # [B, L_a, L_b]
        target_contacts = torch.zeros(B, L, L, device=device)
        target_contacts[:, :L_a, L_a:L_a+L_b] = contact_a_b
        target_contacts[:, L_a:L_a+L_b, :L_a] = contact_a_b.transpose(1, 2)
        
        # Pair mask for loss (only inter-chain pairs)
        pair_mask = torch.zeros(B, L, L, device=device)
        pair_mask[:, :L_a, L_a:L_a+L_b] = 1.0
        pair_mask[:, L_a:L_a+L_b, :L_a] = 1.0
        pair_mask = pair_mask * mask.unsqueeze(-1) * mask.unsqueeze(1)
        
        # Forward pass
        outputs = model(seq, pair_type, mask=mask)
        
        # Compute loss (only on inter-chain pairs)
        losses = compute_structure_loss(
            outputs,
            target_dist,
            target_contacts,
            mask=pair_mask,
            dist_head=model.dist_head,
            dist_weight=1.0,
            contact_weight=0.5,
        )
        
        loss = losses['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate
        batch_size = B
        total_loss += loss.item() * batch_size
        total_loss_dist += losses['loss_dist'].item() * batch_size
        total_loss_contact += losses['loss_contact'].item() * batch_size
        total_samples += batch_size
        
        # Metrics
        with torch.no_grad():
            batch_metrics = compute_metrics(
                outputs['dist_logits'],
                outputs['contact_logits'],
                target_dist,
                target_contacts,
                mask=pair_mask,
            )
            for k, v in batch_metrics.items():
                metrics_accum[k] += v * batch_size
    
    return {
        'loss': total_loss / total_samples,
        'loss_dist': total_loss_dist / total_samples,
        'loss_contact': total_loss_contact / total_samples,
        **{k: v / total_samples for k, v in metrics_accum.items()},
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_loss_dist = 0.0
    total_loss_contact = 0.0
    total_samples = 0
    
    metrics_accum = {'contact_precision': 0, 'contact_recall': 0, 'contact_f1': 0, 'dist_mae': 0}
    
    for batch in loader:
        seq = batch['merged_seq'].to(device)
        pair_type = batch['pair_type'].to(device)
        mask = batch['merged_mask'].to(device)
        
        B = seq.size(0)
        L = seq.size(1)
        
        dist_a_b = batch['distance_map'].to(device)
        L_a = dist_a_b.size(1)
        L_b = dist_a_b.size(2)
        
        target_dist = torch.zeros(B, L, L, device=device)
        target_dist[:, :L_a, L_a:L_a+L_b] = dist_a_b
        target_dist[:, L_a:L_a+L_b, :L_a] = dist_a_b.transpose(1, 2)
        target_dist[:, :L_a, :L_a] = 20.0
        target_dist[:, L_a:L_a+L_b, L_a:L_a+L_b] = 20.0
        
        contact_a_b = batch['contact_map'].to(device)
        target_contacts = torch.zeros(B, L, L, device=device)
        target_contacts[:, :L_a, L_a:L_a+L_b] = contact_a_b
        target_contacts[:, L_a:L_a+L_b, :L_a] = contact_a_b.transpose(1, 2)
        
        pair_mask = torch.zeros(B, L, L, device=device)
        pair_mask[:, :L_a, L_a:L_a+L_b] = 1.0
        pair_mask[:, L_a:L_a+L_b, :L_a] = 1.0
        pair_mask = pair_mask * mask.unsqueeze(-1) * mask.unsqueeze(1)
        
        outputs = model(seq, pair_type, mask=mask)
        
        losses = compute_structure_loss(
            outputs,
            target_dist,
            target_contacts,
            mask=pair_mask,
            dist_head=model.dist_head,
        )
        
        batch_size = B
        total_loss += losses['loss'].item() * batch_size
        total_loss_dist += losses['loss_dist'].item() * batch_size
        total_loss_contact += losses['loss_contact'].item() * batch_size
        total_samples += batch_size
        
        batch_metrics = compute_metrics(
            outputs['dist_logits'],
            outputs['contact_logits'],
            target_dist,
            target_contacts,
            mask=pair_mask,
        )
        for k, v in batch_metrics.items():
            metrics_accum[k] += v * batch_size
    
    return {
        'loss': total_loss / total_samples,
        'loss_dist': total_loss_dist / total_samples,
        'loss_contact': total_loss_contact / total_samples,
        **{k: v / total_samples for k, v in metrics_accum.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3A: PPI Structure Pretraining")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="flowtcr_fold/data/ppi_merged",
                        help="Directory containing merged .npz files")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length (skip longer)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for debugging")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Model
    parser.add_argument("--s_dim", type=int, default=384,
                        help="Sequence representation dimension")
    parser.add_argument("--z_dim", type=int, default=128,
                        help="Pair representation dimension")
    parser.add_argument("--n_layers", type=int, default=8,
                        help="Number of Evoformer layers")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="checkpoints/stage3_phase_a",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(out_dir)
    logger.info("=" * 60)
    logger.info("Phase 3A: PPI Structure Pretraining")
    logger.info("=" * 60)
    logger.info(f"Config: {vars(args)}")
    
    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Dataset
    logger.info(f"\nLoading dataset from {args.data_dir}...")
    full_dataset = PPIDataset(
        data_dir=args.data_dir,
        max_length=args.max_length,
        energy_targets=['E_bind'],
        verbose=True,
    )
    
    if args.max_samples is not None and args.max_samples < len(full_dataset):
        indices = random.sample(range(len(full_dataset)), args.max_samples)
        full_dataset = Subset(full_dataset, indices)
        logger.info(f"Using {args.max_samples} samples for debugging")
    
    # Split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_ppi_batch,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_ppi_batch,
        pin_memory=True,
    )
    
    # Model
    logger.info(f"\nCreating model...")
    model = TCRFoldProphet(
        s_dim=args.s_dim,
        z_dim=args.z_dim,
        n_layers=args.n_layers,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # LR scheduler: warmup + cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info(f"\n{'='*60}")
    logger.info("Starting training...")
    logger.info(f"{'='*60}")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, args.grad_clip
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log
        logger.info(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Contact P: {val_metrics['contact_precision']:.3f} | "
            f"Dist MAE: {val_metrics['dist_mae']:.2f}Å | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, out_dir / "best.pt")
            logger.info(f"  → Saved best model (val_loss={best_val_loss:.4f})")
        
        # Save periodic
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, out_dir / f"epoch_{epoch+1}.pt")
    
    # Save final
    torch.save({
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, out_dir / "last.pt")
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

