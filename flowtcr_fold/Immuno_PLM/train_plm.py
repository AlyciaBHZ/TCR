"""
Training Entrypoint for Immuno-PLM (ESM-2 + LoRA + Topology Bias)

Supports three training modes:
1. ESM-2 + LoRA (recommended): Fine-tune ESM with low-rank adapters
2. ESM-2 frozen: Use ESM as feature extractor only
3. BasicTokenizer: Lightweight mode without ESM

Training Objective: Batch InfoNCE (safe from false negatives)

Usage:
    # BasicTokenizer mode (fast, for debugging)
    python train_plm.py --data data/trn.csv --epochs 100 --batch_size 64

    # ESM-2 + LoRA mode (recommended for production)
    python train_plm.py --data data/trn.csv --epochs 100 --batch_size 32 \\
        --use_esm --use_lora --lora_rank 8 --esm_model esm2_t33_650M_UR50D

    # ESM-2 frozen mode (faster, less memory)
    python train_plm.py --data data/trn.csv --epochs 100 --batch_size 32 \\
        --use_esm --esm_model esm2_t12_35M_UR50D
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from flowtcr_fold.data.dataset import FlowDataset
from flowtcr_fold.data.tokenizer import vocab_size, BasicTokenizer
from flowtcr_fold.Immuno_PLM.immuno_plm import ImmunoPLM, compute_batch_infonce
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


def parse_args():
    p = argparse.ArgumentParser(description="Train Immuno-PLM")
    
    # Data
    p.add_argument("--data", type=str, default="data/trn.csv",
                   help="Path to training CSV")
    p.add_argument("--val_data", type=str, default=None,
                   help="Path to validation CSV (optional)")
    
    # Model architecture
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Hidden dimension for projections")
    p.add_argument("--z_dim", type=int, default=128,
                   help="Topology embedding dimension")
    
    # ESM-2 configuration
    p.add_argument("--use_esm", action="store_true",
                   help="Use ESM-2 as backbone")
    p.add_argument("--esm_model", type=str, default="esm2_t33_650M_UR50D",
                   choices=["esm2_t33_650M_UR50D", "esm2_t12_35M_UR50D", "esm2_t6_8M_UR50D"],
                   help="ESM-2 model to use")
    
    # LoRA configuration
    p.add_argument("--use_lora", action="store_true",
                   help="Apply LoRA adapters to ESM-2")
    p.add_argument("--lora_rank", type=int, default=8,
                   help="LoRA rank (lower = fewer parameters)")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha scaling factor")
    p.add_argument("--lora_dropout", type=float, default=0.1,
                   help="LoRA dropout rate")
    
    # Training
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size (larger = more negatives for InfoNCE)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("--tau", type=float, default=0.07,
                   help="InfoNCE temperature (lower = sharper)")
    p.add_argument("--mlm_weight", type=float, default=0.0,
                   help="Weight for MLM loss (0 = disabled)")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm")
    
    # Output
    p.add_argument("--out_dir", type=str, default="checkpoints/plm",
                   help="Output directory for checkpoints")
    p.add_argument("--log_interval", type=int, default=10,
                   help="Log every N batches")
    
    return p.parse_args()


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch with padding.
    
    For Batch InfoNCE, we only need positive samples (no explicit negatives).
    """
    tokens_list = [item["tokens_pos"] for item in batch]
    masks_list = [item["mask_pos"] for item in batch]
    slices_list = [item["slices_pos"] for item in batch]
    
    # Pad tokens
    max_len = max(t.size(0) for t in tokens_list)
    B = len(tokens_list)
    
    tokens_padded = torch.zeros(B, max_len, dtype=torch.long)
    mask_padded = torch.zeros(B, max_len, dtype=torch.long)
    
    for i, (t, m) in enumerate(zip(tokens_list, masks_list)):
        tokens_padded[i, :t.size(0)] = t
        mask_padded[i, :m.size(0)] = m
    
    return {
        "tokens": tokens_padded,
        "mask": mask_padded,
        "slices": slices_list,
        "meta": [item.get("meta", {}) for item in batch]
    }


def compute_mlm_loss(
    model: ImmunoPLM,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
    mask_prob: float = 0.15
) -> torch.Tensor:
    """
    Compute MLM loss (only for BasicTokenizer mode).
    """
    if model.use_esm or not hasattr(model, "decoder"):
        return torch.tensor(0.0, device=tokens.device)
    
    if not isinstance(tokenizer, BasicTokenizer):
        return torch.tensor(0.0, device=tokens.device)
    
    device = tokens.device
    labels = tokens.clone()
    
    # Create mask
    mask_token = tokenizer.stoi.get("[MASK]", 0)
    special = set([tokenizer.stoi.get(t, 0) for t in ["[PAD]", "[CLS]", "[SEP]"]])
    
    probability_matrix = torch.full(labels.shape, mask_prob, device=device)
    for sp in special:
        probability_matrix[tokens == sp] = 0
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Ignore index
    
    masked_tokens = tokens.clone()
    masked_tokens[masked_indices] = mask_token
    
    # Forward pass
    out = model(masked_tokens, mask)
    logits = model.decoder(out["s"])  # [B, L, vocab_size]
    
    # Loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    return loss


def train_epoch(
    model: ImmunoPLM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    tokenizer=None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_nce = 0.0
    total_mlm = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        slices = batch["slices"]
        
        # Forward pass
        out = model(tokens, mask, region_slices=slices)
        
        # ==============================
        # Batch InfoNCE Loss
        # ==============================
        # Use contrastive embeddings for InfoNCE
        embeddings = out["contrastive"]  # [B, hidden_dim]
        
        # Batch InfoNCE: each sample is its own anchor,
        # and other samples in batch are negatives
        # We compute similarity matrix and use diagonal as positive
        loss_nce = compute_batch_infonce(embeddings, embeddings, temperature=args.tau)
        
        # ==============================
        # MLM Loss (optional)
        # ==============================
        loss_mlm = torch.tensor(0.0, device=device)
        if args.mlm_weight > 0 and tokenizer is not None:
            loss_mlm = compute_mlm_loss(model, tokens, mask, tokenizer)
        
        # ==============================
        # Total Loss
        # ==============================
        loss = loss_nce + args.mlm_weight * loss_mlm
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        total_nce += loss_nce.item()
        total_mlm += loss_mlm.item()
        num_batches += 1
        
        if (batch_idx + 1) % args.log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}: "
                  f"loss={loss.item():.4f}, nce={loss_nce.item():.4f}, mlm={loss_mlm.item():.4f}")
    
    return {
        "loss": total_loss / num_batches,
        "nce": total_nce / num_batches,
        "mlm": total_mlm / num_batches
    }


def evaluate(
    model: ImmunoPLM,
    loader: DataLoader,
    device: torch.device,
    args
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            slices = batch["slices"]
            
            out = model(tokens, mask, region_slices=slices)
            embeddings = out["contrastive"]
            
            loss = compute_batch_infonce(embeddings, embeddings, temperature=args.tau)
            
            total_loss += loss.item()
            num_batches += 1
    
    return {"loss": total_loss / num_batches}


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Immuno-PLM Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"ESM-2: {args.use_esm} (model: {args.esm_model})")
    print(f"LoRA: {args.use_lora} (rank: {args.lora_rank})")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.tau}")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    print("\nLoading dataset...")
    train_ds = FlowDataset(args.data, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"Training samples: {len(train_ds)}")
    
    val_loader = None
    if args.val_data:
        val_ds = FlowDataset(args.val_data, split="val")
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        print(f"Validation samples: {len(val_ds)}")
    
    # Model
    print("\nInitializing model...")
    model = ImmunoPLM(
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        use_esm=args.use_esm,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        vocab_size=vocab_size(train_ds.tokenizer),
        esm_model_name=args.esm_model
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Early stopper
    stopper = EarlyStopper(patience=100)
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    best_loss = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, device, args, train_ds.tokenizer
        )
        print(f"  Train: loss={train_stats['loss']:.4f}, nce={train_stats['nce']:.4f}")
        
        # Validate
        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device, args)
            print(f"  Val: loss={val_stats['loss']:.4f}")
            current_loss = val_stats["loss"]
        else:
            current_loss = train_stats["loss"]
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), out_dir / "immuno_plm_best.pt")
            print(f"  New best model saved! (loss={best_loss:.4f})")
        
        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, str(out_dir), epoch + 1, tag="plm")
            print(f"  Checkpoint saved at epoch {epoch + 1}")
        
        # Early stopping
        if stopper.update(current_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), out_dir / "immuno_plm_final.pt")
    print(f"\nTraining complete! Models saved to {out_dir}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
