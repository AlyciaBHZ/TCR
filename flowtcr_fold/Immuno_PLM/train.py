"""
Stage 1: Immuno-PLM scaffold prior training (Plan v3.1)

Usage:
    python flowtcr_fold/Immuno_PLM/train.py                    # Normal training
    python flowtcr_fold/Immuno_PLM/train.py --ablation         # Ablation (peptide-off)
    python flowtcr_fold/Immuno_PLM/train.py --resume           # Resume from last checkpoint
    python flowtcr_fold/Immuno_PLM/train.py --resume_best      # Resume from best model
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from flowtcr_fold.common.utils import EarlyStopper, save_checkpoint
from flowtcr_fold.data.tokenizer import get_tokenizer, vocab_size

from flowtcr_fold.Immuno_PLM.data import ScaffoldRetrievalDataset, collate_fn_factory
from flowtcr_fold.Immuno_PLM.model import ScaffoldRetriever
from flowtcr_fold.Immuno_PLM.train_utils import (
    evaluate,
    evaluate_frequency_baseline_recall,
    train_epoch,
    FrequencyBaseline,
)


# =============================================================================
# Configuration (modify here instead of CLI arguments)
# =============================================================================
class Config:
    # Data paths (fixed)
    data_dir = Path("flowtcr_fold/data")
    train_data = data_dir / "trn.jsonl"
    val_data = data_dir / "val.jsonl"
    
    # Output base dir
    output_base = Path("flowtcr_fold/Immuno_PLM/saved_model")
    
    # Model config (ESM + LoRA is default)
    hidden_dim = 256
    esm_model = "esm2_t33_650M_UR50D"  # Options: "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D"
    lora_rank = 16
    lora_alpha = 32
    lora_dropout = 0.1
    max_len = 512
    
    # Training config
    batch_size = 32
    lr = 1e-4
    epochs = 100
    patience = 20
    grad_clip = 1.0
    k_list = [1, 5, 10, 20]
    
    # Loss weights
    tau = 0.07          # InfoNCE temperature
    lambda_pmhc = 0.3   # pMHC-group InfoNCE weight
    lambda_bce = 0.2    # Multi-label BCE weight
    lambda_pep = 0.1    # Peptide-only InfoNCE weight (for samples without MHC)

    # Logging
    ckpt_interval = 10
    log_interval = 25


def parse_args():
    p = argparse.ArgumentParser(description="Train Immuno-PLM scaffold prior (Stage 1)")
    p.add_argument("--ablation", action="store_true", help="Run ablation study (peptide-off)")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--resume_best", action="store_true", help="Resume from best model")
    return p.parse_args()


def get_output_dir(ablation: bool) -> Path:
    """Get output directory based on ablation mode."""
    if ablation:
        return Config.output_base / "ablation_peptide_off"
    return Config.output_base / "stage1"


def prepare_dirs(out_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": out_dir,
        "checkpoints": out_dir / "checkpoints",
        "best": out_dir / "best_model",
        "other": out_dir / "other_results",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the directory."""
    ckpts = list(ckpt_dir.glob("scaffold_epoch_*.pt"))
    if not ckpts:
        return None
    # Sort by epoch number
    ckpts.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return ckpts[-1]


def load_checkpoint(model, optimizer, ckpt_path: Path, device) -> int:
    """Load model and optimizer state, return the epoch number."""
    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    
    # Try to load optimizer state
    opt_path = ckpt_path.with_suffix(".opt")
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"Loaded optimizer state from {opt_path}")
    
    # Extract epoch from filename (e.g., scaffold_epoch_10.pt -> 10)
    try:
        epoch = int(ckpt_path.stem.split("_")[-1])
    except:
        epoch = 0
    return epoch


def main():
    args = parse_args()
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine output directory
    out_dir = get_output_dir(args.ablation)
    out_paths = prepare_dirs(out_dir)
    mask_peptide = args.ablation  # If ablation, mask peptide input
    
    print("=" * 70)
    print("Stage 1: Immuno-PLM Scaffold Prior Training")
    print("=" * 70)
    print(f"Mode: {'ABLATION (peptide-off)' if args.ablation else 'NORMAL'}")
    print(f"Data: {cfg.train_data}")
    print(f"Model: ESM={cfg.esm_model} + LoRA(rank={cfg.lora_rank})")
    print(f"Loss weights: λ_pmhc={cfg.lambda_pmhc}, λ_bce={cfg.lambda_bce}, λ_pep={cfg.lambda_pep}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    print("=" * 70)

    # Tokenizer
    tokenizer = get_tokenizer()

    # Dataset / Loader
    train_ds = ScaffoldRetrievalDataset(
        str(cfg.train_data), tokenizer=tokenizer, max_len=cfg.max_len, build_vocab=True
    )
    
    # Log pos_weight stats for debugging
    print("\n[pos_weight stats]")
    for gtype in ["h_v", "h_j", "l_v", "l_j"]:
        pw = train_ds.pos_weight[gtype]
        print(f"  {gtype}: min={pw.min():.2f}, max={pw.max():.2f}, mean={pw.mean():.2f}")

    collate_train = collate_fn_factory(tokenizer, cfg.max_len, train_ds, mask_peptide=mask_peptide)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_train, num_workers=0
    )

    val_loader: Optional[DataLoader] = None
    val_ds = None
    if cfg.val_data.exists():
        val_ds = ScaffoldRetrievalDataset(
            str(cfg.val_data),
            tokenizer=tokenizer,
            max_len=cfg.max_len,
            gene_vocab=train_ds.gene_vocab,
            allele_vocab=train_ds.allele_vocab,
            build_vocab=False,
        )
        collate_val = collate_fn_factory(tokenizer, cfg.max_len, val_ds, mask_peptide=mask_peptide)
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_val, num_workers=0
        )

    # Model
    model = ScaffoldRetriever(
        hidden_dim=cfg.hidden_dim,
        num_hv=len(train_ds.gene_vocab["h_v"]),
        num_hj=len(train_ds.gene_vocab["h_j"]),
        num_lv=len(train_ds.gene_vocab["l_v"]),
        num_lj=len(train_ds.gene_vocab["l_j"]),
        num_alleles=len(train_ds.allele_vocab),
        use_esm=True,
        use_lora=True,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        vocab_size=vocab_size(tokenizer),
        esm_model_name=cfg.esm_model,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {trainable_params:,}/{total_params:,} trainable ({trainable_params/total_params:.2%})")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=0.01
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume_best:
        best_ckpt = out_paths["best"] / "model_best.pt"
        if best_ckpt.exists():
            start_epoch = load_checkpoint(model, optimizer, best_ckpt, device)
            print(f"Resumed from best model (will start from epoch {start_epoch + 1})")
        else:
            print(f"No best checkpoint found at {best_ckpt}, starting fresh")
    elif args.resume:
        latest_ckpt = find_latest_checkpoint(out_paths["checkpoints"])
        if latest_ckpt:
            start_epoch = load_checkpoint(model, optimizer, latest_ckpt, device)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found in {out_paths['checkpoints']}, starting fresh")

    stopper = EarlyStopper(patience=cfg.patience)
    best_loss = float("inf")

    # Save vocab/meta
    with open(out_paths["root"] / "gene_vocab.json", "w") as f:
        json.dump(train_ds.gene_vocab, f, indent=2)
    with open(out_paths["root"] / "allele_vocab.json", "w") as f:
        json.dump(train_ds.allele_vocab, f, indent=2)
    with open(out_paths["root"] / "config.json", "w") as f:
        config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()}
        config_dict["ablation"] = args.ablation
        json.dump(config_dict, f, indent=2)

    # Build frequency baseline
    print("\nBuilding frequency baseline...")
    freq_baseline = FrequencyBaseline(train_ds.samples, train_ds.gene_vocab)
    baseline_train = evaluate_frequency_baseline_recall(
        freq_baseline, train_ds.samples, train_ds.gene_vocab, k_list=cfg.k_list
    )
    print("Frequency Baseline R@K (train):")
    for gtype in ["h_v", "h_j", "l_v", "l_j"]:
        r = baseline_train.get(gtype, {})
        print(f"  {gtype}: R@1={r.get(1,0):.3f}, R@5={r.get(5,0):.3f}, R@10={r.get(10,0):.3f}, R@20={r.get(20,0):.3f}")

    # Create args-like object for train_epoch/evaluate
    class Args:
        tau = cfg.tau
        lambda_pmhc = cfg.lambda_pmhc
        lambda_bce = cfg.lambda_bce
        lambda_pep = cfg.lambda_pep
        grad_clip = cfg.grad_clip
    train_args = Args()

    def format_recall(stats: Dict[str, float], gtype: str) -> str:
        parts = []
        for k in cfg.k_list:
            key = f"recall@{k}_{gtype}"
            if key in stats:
                parts.append(f"R@{k}={stats[key]:.3f}")
        return " ".join(parts) if parts else "R@N/A"

    print(f"\nStarting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        
        train_stats = train_epoch(model, train_loader, optimizer, device, train_args, pos_weight=train_ds.pos_weight)
        print(
            f"  Train: loss={train_stats['loss']:.4f} | "
            f"NCE(mhc)={train_stats['nce_mhc']:.3f} NCE(pmhc)={train_stats['nce_pmhc']:.3f} "
            f"NCE(pep)={train_stats['nce_pep']:.3f} | BCE={train_stats['bce']:.3f}"
        )

        current_loss = train_stats["loss"]
        if val_loader is not None and val_ds is not None:
            val_stats = evaluate(
                model, val_loader, device, train_args, 
                pos_weight=train_ds.pos_weight, 
                frequency_baseline=freq_baseline,
                k_list=cfg.k_list,
            )
            
            # Baseline comparison
            baseline_val = evaluate_frequency_baseline_recall(
                freq_baseline, val_ds.samples, train_ds.gene_vocab, k_list=[10]
            )
            # Baseline deltas for all gene types
            deltas = {}
            for gtype in ["h_v", "h_j", "l_v", "l_j"]:
                short = gtype.replace("_", "")
                b_val = baseline_val.get(gtype, {}).get(10, 0.0)
                m_val = val_stats.get(f"recall@10_{short}", 0.0)
                deltas[gtype] = {"baseline": b_val, "model": m_val, "delta": m_val - b_val}

            print(
                f"  Val: loss={val_stats['loss']:.4f} | "
                f"NCE(mhc)={val_stats['nce_mhc']:.3f} NCE(pmhc)={val_stats['nce_pmhc']:.3f} "
                f"NCE(pep)={val_stats['nce_pep']:.3f} | BCE={val_stats['bce']:.3f}"
            )
            print(f"       HV: {format_recall(val_stats, 'hv')} | HJ: {format_recall(val_stats, 'hj')}")
            print(f"       LV: {format_recall(val_stats, 'lv')} | LJ: {format_recall(val_stats, 'lj')}")
            print(
                f"       Baseline R@10: HV={deltas['h_v']['baseline']:.3f} HJ={deltas['h_j']['baseline']:.3f} "
                f"LV={deltas['l_v']['baseline']:.3f} LJ={deltas['l_j']['baseline']:.3f}"
            )
            print(
                f"       Δ vs Baseline: HV={deltas['h_v']['delta']:+.3f} HJ={deltas['h_j']['delta']:+.3f} "
                f"LV={deltas['l_v']['delta']:+.3f} LJ={deltas['l_j']['delta']:+.3f}"
            )
            if "kl_hv" in val_stats:
                print(f"       KL_HV={val_stats['kl_hv']:.4f} | KL_HJ={val_stats['kl_hj']:.4f}")

            current_loss = val_stats["loss"]

        # Save best
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), out_paths["best"] / "model_best.pt")
            torch.save(optimizer.state_dict(), out_paths["best"] / "optimizer_best.pt")
            print(f"  ✓ New best model saved (loss={best_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % cfg.ckpt_interval == 0:
            save_checkpoint(model, optimizer, str(out_paths["checkpoints"]), epoch + 1, tag="scaffold")
            print(f"  Checkpoint saved at epoch {epoch + 1}")

        # Early stopping
        if stopper.update(current_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Final save
    torch.save(model.state_dict(), out_paths["root"] / "model_final.pt")
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Artifacts: {out_paths['root']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
