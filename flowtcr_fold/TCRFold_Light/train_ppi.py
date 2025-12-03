"""
TCRFold-Light Phase 3A: PPI 结构预训练
======================================

使用统一的 PPIDataset 进行训练，解决之前的接口对齐问题:
1. 使用预处理的 .npz 文件（而非重新解析 PDB）
2. 读取预计算的 JSONL 能量缓存（而非重新调用 EvoEF2）
3. 统一的链分割策略
4. 完整的日志和验证

Usage:
    python flowtcr_fold/TCRFold_Light/train_ppi.py \
        --npz_dir flowtcr_fold/data/pdb_structures/processed \
        --energy_cache flowtcr_fold/data/energy_cache.jsonl \
        --epochs 100 \
        --batch_size 4

    # 带验证集
    python flowtcr_fold/TCRFold_Light/train_ppi.py \
        --npz_dir ... --val_split 0.1
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from flowtcr_fold.TCRFold_Light.tcrfold_light import TCRFoldLight
from flowtcr_fold.TCRFold_Light.ppi_dataset import (
    PPIDataset, 
    collate_ppi_batch, 
    merge_chains_for_evoformer
)
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


# ============================================================================
# 日志设置
# ============================================================================

def setup_logging(out_dir: str, name: str = "train_ppi") -> logging.Logger:
    """设置日志，输出到文件和控制台。"""
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(out_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除旧 handlers
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"日志文件: {log_file}")
    return logger


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3A: PPI 结构预训练")
    
    # 数据
    p.add_argument("--npz_dir", type=str, required=True,
                   help="预处理 .npz 文件目录 (from preprocess_ppi_pairs.py)")
    p.add_argument("--energy_cache", type=str, default=None,
                   help="JSONL 能量缓存 (from compute_evoef2_batch.py)")
    p.add_argument("--val_split", type=float, default=0.1,
                   help="验证集比例 (default: 0.1)")
    
    # 模型
    p.add_argument("--s_dim", type=int, default=256,
                   help="Single representation dimension")
    p.add_argument("--z_dim", type=int, default=64,
                   help="Pair representation dimension")
    p.add_argument("--n_layers", type=int, default=8,
                   help="Number of Evoformer layers")
    
    # 训练
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    
    # Loss 权重
    p.add_argument("--lambda_dist", type=float, default=1.0)
    p.add_argument("--lambda_contact", type=float, default=1.0)
    p.add_argument("--lambda_energy", type=float, default=0.1)
    p.add_argument("--interface_weight", type=float, default=10.0,
                   help="接口残基的额外权重")
    
    # 输出
    p.add_argument("--out_dir", type=str, default="checkpoints/phase3a_ppi")
    p.add_argument("--save_every", type=int, default=10,
                   help="每 N 个 epoch 保存 checkpoint")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience")
    
    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    
    return p.parse_args()


# ============================================================================
# Loss 函数
# ============================================================================

def compute_ppi_loss(
    model_output: dict,
    batch: dict,
    lambda_dist: float = 1.0,
    lambda_contact: float = 1.0,
    lambda_energy: float = 0.1,
    interface_weight: float = 10.0
) -> dict:
    """
    计算 PPI 预训练 loss。
    
    Args:
        model_output: 模型输出 {'distance': [B,L,L,1], 'contact': [B,L,L,1], 'energy': [B]}
        batch: 数据批次 (from merge_chains_for_evoformer)
        lambda_*: loss 权重
        interface_weight: 接口残基额外权重
    
    Returns:
        {'total': loss, 'dist': dist_loss, 'contact': contact_loss, 'energy': energy_loss}
    """
    device = batch["binding_energy"].device
    B, L = batch["mask"].shape
    
    mask = batch["mask"].float()  # [B, L]
    mask_pair = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, L, L]
    
    # 真实值
    true_dist = batch["distance_map"]  # [B, L, L]
    true_contact = batch["contact_map"]  # [B, L, L]
    true_energy = batch["binding_energy"]  # [B]
    
    # 预测值
    pred_dist = model_output["distance"].squeeze(-1)  # [B, L, L]
    pred_contact = torch.sigmoid(model_output["contact"].squeeze(-1))  # [B, L, L]
    pred_energy = model_output["energy"]  # [B]
    
    # === Distance Loss (MSE) ===
    # 只计算接口部分 (pair_type == 1)
    inter_mask = (batch["pair_type"] == 1).float() * mask_pair
    
    dist_loss = torch.sum((pred_dist - true_dist) ** 2 * inter_mask)
    dist_loss = dist_loss / (inter_mask.sum() + 1e-8)
    
    # === Contact Loss (BCE with interface weighting) ===
    # 接口权重
    weight = torch.ones_like(true_contact)
    weight = weight + (interface_weight - 1) * inter_mask  # 接口位置权重更高
    
    contact_loss = nn.functional.binary_cross_entropy(
        pred_contact * mask_pair,
        true_contact * mask_pair,
        weight=weight * mask_pair,
        reduction='sum'
    ) / (mask_pair.sum() + 1e-8)
    
    # === Energy Loss (MSE) ===
    # 只在有能量标签时计算
    has_energy = (true_energy != 0).float()
    energy_loss = torch.sum((pred_energy - true_energy) ** 2 * has_energy)
    energy_loss = energy_loss / (has_energy.sum() + 1e-8)
    
    # === Total ===
    total_loss = (
        lambda_dist * dist_loss +
        lambda_contact * contact_loss +
        lambda_energy * energy_loss
    )
    
    return {
        "total": total_loss,
        "dist": dist_loss,
        "contact": contact_loss,
        "energy": energy_loss
    }


# ============================================================================
# 训练/验证循环
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    logger: logging.Logger
) -> dict:
    """训练一个 epoch。"""
    model.train()
    
    total_loss = 0.0
    total_dist = 0.0
    total_contact = 0.0
    total_energy = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        # 合并双链为单序列 (for Evoformer)
        merged = merge_chains_for_evoformer(batch)
        
        # 移动到设备
        for k, v in merged.items():
            if isinstance(v, torch.Tensor):
                merged[k] = v.to(device)
        
        # 构建模型输入
        # PPIDataset 返回序列索引，需要转换为 placeholder s/z
        B, L = merged["mask"].shape
        s = torch.zeros(B, L, args.s_dim, device=device)
        z = torch.zeros(B, L, L, args.z_dim, device=device)
        
        # 注入距离信息到 z
        dist_normalized = merged["distance_map"].unsqueeze(-1) / 20.0  # 归一化
        z[:, :, :, 0:1] = dist_normalized.clamp(0, 1)
        
        # Forward
        out = model(s, z)
        
        # Loss
        loss_dict = compute_ppi_loss(
            out, merged,
            lambda_dist=args.lambda_dist,
            lambda_contact=args.lambda_contact,
            lambda_energy=args.lambda_energy,
            interface_weight=args.interface_weight
        )
        
        loss = loss_dict["total"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # 累计
        total_loss += loss.item()
        total_dist += loss_dict["dist"].item()
        total_contact += loss_dict["contact"].item()
        total_energy += loss_dict["energy"].item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "dist": total_dist / n_batches,
        "contact": total_contact / n_batches,
        "energy": total_energy / n_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args
) -> dict:
    """验证。"""
    model.eval()
    
    total_loss = 0.0
    total_dist = 0.0
    total_contact = 0.0
    total_energy = 0.0
    n_batches = 0
    
    for batch in loader:
        merged = merge_chains_for_evoformer(batch)
        
        for k, v in merged.items():
            if isinstance(v, torch.Tensor):
                merged[k] = v.to(device)
        
        B, L = merged["mask"].shape
        s = torch.zeros(B, L, args.s_dim, device=device)
        z = torch.zeros(B, L, L, args.z_dim, device=device)
        
        dist_normalized = merged["distance_map"].unsqueeze(-1) / 20.0
        z[:, :, :, 0:1] = dist_normalized.clamp(0, 1)
        
        out = model(s, z)
        
        loss_dict = compute_ppi_loss(
            out, merged,
            lambda_dist=args.lambda_dist,
            lambda_contact=args.lambda_contact,
            lambda_energy=args.lambda_energy,
            interface_weight=args.interface_weight
        )
        
        total_loss += loss_dict["total"].item()
        total_dist += loss_dict["dist"].item()
        total_contact += loss_dict["contact"].item()
        total_energy += loss_dict["energy"].item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "dist": total_dist / n_batches,
        "contact": total_contact / n_batches,
        "energy": total_energy / n_batches
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.out_dir)
    
    logger.info("=" * 60)
    logger.info("Phase 3A: PPI 结构预训练")
    logger.info("=" * 60)
    logger.info(f"配置:")
    logger.info(f"  - NPZ 目录: {args.npz_dir}")
    logger.info(f"  - 能量缓存: {args.energy_cache}")
    logger.info(f"  - 输出目录: {args.out_dir}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.lr}")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  - Device: {device}")
    
    # 数据集
    logger.info("\n加载数据集...")
    
    dataset = PPIDataset(
        npz_dir=args.npz_dir,
        energy_cache=args.energy_cache,
        verbose=True
    )
    
    if len(dataset) == 0:
        logger.error("数据集为空！请先运行 preprocess_ppi_pairs.py")
        return
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 划分训练/验证
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ppi_batch,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ppi_batch,
        num_workers=args.num_workers
    ) if val_size > 0 else None
    
    # 模型
    logger.info("\n初始化模型...")
    model = TCRFoldLight(
        s_dim=args.s_dim,
        z_dim=args.z_dim,
        n_layers=args.n_layers
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {n_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Early stopping
    stopper = EarlyStopper(patience=args.patience)
    
    # 训练循环
    logger.info("\n开始训练...")
    logger.info("-" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_stats = train_epoch(model, train_loader, optimizer, device, args, logger)
        
        log_msg = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_stats['loss']:.4f} "
            f"(dist: {train_stats['dist']:.4f}, "
            f"contact: {train_stats['contact']:.4f}, "
            f"energy: {train_stats['energy']:.4f})"
        )
        
        # 验证
        if val_loader is not None:
            val_stats = validate(model, val_loader, device, args)
            log_msg += (
                f" | Val Loss: {val_stats['loss']:.4f}"
            )
            
            # 保存最佳模型
            if val_stats['loss'] < best_val_loss:
                best_val_loss = val_stats['loss']
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                log_msg += " *"
            
            # Early stopping
            if stopper.update(val_stats['loss']):
                logger.info(log_msg)
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(log_msg)
        
        # 定期保存
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, args.out_dir, epoch, tag="phase3a")
            logger.info(f"  Checkpoint saved at epoch {epoch}")
    
    # 保存最终模型
    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"\n最终模型保存到: {final_path}")
    
    if val_loader is not None:
        logger.info(f"最佳验证 Loss: {best_val_loss:.4f}")
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()



