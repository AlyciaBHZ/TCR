"""
TCRFold-Light Training with EvoEF2 Energy Supervision
======================================================

Integrates EvoEF2 energy computation into the training loop for:
1. PPI pretraining with real structure/energy labels
2. Energy surrogate head supervision
3. Interface-weighted loss

This replaces the placeholder random tensors in train_ppi_impl.py.

Usage:
    python flowtcr_fold/TCRFold_Light/train_with_energy.py \
        --pdb_dir data/pdb_structures \
        --epochs 100 \
        --batch_size 4 \
        --lr 1e-4
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flowtcr_fold.TCRFold_Light.tcrfold_light import TCRFoldLight
from flowtcr_fold.physics.energy_dataset import EnergyStructureDataset, collate_energy_batch
from flowtcr_fold.common.utils import save_checkpoint, EarlyStopper


def parse_args():
    p = argparse.ArgumentParser(description="Train TCRFold-Light with EvoEF2 energy supervision")
    p.add_argument("--pdb_dir", type=str, default="data/pdb_structures",
                   help="Directory containing PDB files")
    p.add_argument("--cache_dir", type=str, default="data/energy_cache",
                   help="Cache directory for computed energies")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints/tcrfold_energy")
    p.add_argument("--recompute_energy", action="store_true",
                   help="Force recompute cached energies")
    p.add_argument("--interface_weight", type=float, default=10.0,
                   help="Weight for interface residue loss (as per USER_MANUAL)")
    return p.parse_args()


def compute_interface_mask(contact_map: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """
    Identify interface residues based on inter-chain contacts.

    Args:
        contact_map: [B, L, L] binary contact map
        threshold: Not used (already binarized in dataset)

    Returns:
        [B, L] binary mask (1 = interface residue)
    """
    # Simple heuristic: residue is at interface if it has >5 contacts
    interface_mask = (contact_map.sum(dim=-1) > 5).float()
    return interface_mask


def compute_physics_loss(
    pred_dist: torch.Tensor,
    pred_contact: torch.Tensor,
    pred_energy: torch.Tensor,
    true_dist: torch.Tensor,
    true_contact: torch.Tensor,
    true_energy: torch.Tensor,
    mask: torch.Tensor,
    interface_weight: float = 10.0
) -> dict:
    """
    Compute composite physics loss as per USER_MANUAL:
    - L_dist: Distance map MSE
    - L_contact: Contact BCE (interface-weighted x10)
    - L_energy: Energy MSE

    This replaces the placeholder loss in train_ppi_impl.py.

    Args:
        pred_dist: [B, L, L, 1] predicted distances
        pred_contact: [B, L, L, 1] predicted contacts (sigmoid)
        pred_energy: [B] predicted binding energy
        true_dist: [B, L, L] ground truth distances
        true_contact: [B, L, L] ground truth contacts
        true_energy: [B] ground truth energy from EvoEF2
        mask: [B, L] valid residue mask
        interface_weight: Weight for interface residues

    Returns:
        dict with 'total', 'dist', 'contact', 'energy'
    """
    B, L = mask.shape

    # Expand mask for pairwise operations
    mask_pair = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, L, L]

    # === Distance Loss ===
    pred_dist = pred_dist.squeeze(-1)  # [B, L, L]
    loss_dist = torch.nn.functional.mse_loss(
        pred_dist * mask_pair,
        true_dist * mask_pair,
        reduction='sum'
    ) / (mask_pair.sum() + 1e-8)

    # === Contact Loss (Interface-Weighted) ===
    pred_contact = pred_contact.squeeze(-1)  # [B, L, L]

    # Compute interface mask (residues with many contacts)
    interface_mask = compute_interface_mask(true_contact)  # [B, L]
    interface_pair = interface_mask.unsqueeze(2) * interface_mask.unsqueeze(1)  # [B, L, L]

    # Separate interface and non-interface loss
    interface_loss = torch.nn.functional.binary_cross_entropy(
        pred_contact * interface_pair * mask_pair,
        true_contact * interface_pair * mask_pair,
        reduction='sum'
    ) / (interface_pair.sum() + 1e-8)

    non_interface_loss = torch.nn.functional.binary_cross_entropy(
        pred_contact * (1 - interface_pair) * mask_pair,
        true_contact * (1 - interface_pair) * mask_pair,
        reduction='sum'
    ) / ((1 - interface_pair).sum() + 1e-8)

    # Weighted contact loss (x10 for interface)
    loss_contact = interface_weight * interface_loss + non_interface_loss

    # === Energy Loss ===
    loss_energy = torch.nn.functional.mse_loss(pred_energy, true_energy)

    # === Total Loss ===
    total_loss = loss_dist + loss_contact + loss_energy

    return {
        'total': total_loss,
        'dist': loss_dist,
        'contact': loss_contact,
        'energy': loss_energy
    }


def train_epoch(model, loader, optimizer, device, interface_weight):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_dist = 0.0
    total_contact = 0.0
    total_energy = 0.0

    for batch_idx, batch in enumerate(loader):
        # Move to device
        s = batch['s'].to(device)
        z = batch['z'].to(device)
        true_dist = batch['distance_map'].to(device)
        true_contact = batch['contact_map'].to(device)
        true_energy = batch['energy'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        out = model(s, z)

        # Compute loss
        loss_dict = compute_physics_loss(
            pred_dist=out['distance'],
            pred_contact=out['contact'],
            pred_energy=out['energy'],
            true_dist=true_dist,
            true_contact=true_contact,
            true_energy=true_energy,
            mask=mask,
            interface_weight=interface_weight
        )

        loss = loss_dict['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate stats
        total_loss += loss.item()
        total_dist += loss_dict['dist'].item()
        total_contact += loss_dict['contact'].item()
        total_energy += loss_dict['energy'].item()

    n = len(loader)
    return {
        'loss': total_loss / n,
        'dist': total_dist / n,
        'contact': total_contact / n,
        'energy': total_energy / n
    }


def main():
    args = parse_args()

    # Check if PDB directory exists
    if not os.path.exists(args.pdb_dir):
        print(f"Error: PDB directory not found: {args.pdb_dir}")
        print("\nPlease prepare PDB files first:")
        print("  1. Download PDB files (e.g., from RCSB)")
        print("  2. Filter for protein-protein complexes")
        print("  3. Place in", args.pdb_dir)
        print("\nFor TCR-pMHC structures, check:")
        print("  - STCRDab: http://opig.stats.ox.ac.uk/webapps/stcrdab/")
        print("  - TCR3d: https://tcr3d.ibbr.umd.edu/")
        return

    print("=" * 60)
    print("TCRFold-Light Training with EvoEF2 Energy Supervision")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    print(f"\nLoading dataset from {args.pdb_dir}...")
    dataset = EnergyStructureDataset(
        pdb_dir=args.pdb_dir,
        cache_dir=args.cache_dir,
        recompute=args.recompute_energy,
        verbose=True
    )

    if len(dataset) == 0:
        print("Error: No valid structures found in dataset.")
        return

    print(f"Dataset size: {len(dataset)}")

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_energy_batch,
        num_workers=0  # Set >0 if multiprocessing works
    )

    # Model
    print(f"\nInitializing TCRFold-Light...")
    model = TCRFoldLight(s_dim=512, z_dim=128, n_layers=12).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Early stopper
    stopper = EarlyStopper(patience=100)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        stats = train_epoch(model, loader, optimizer, device, args.interface_weight)

        print(f"Epoch {epoch+1:3d} | "
              f"Loss: {stats['loss']:.4f} | "
              f"Dist: {stats['dist']:.4f} | "
              f"Contact: {stats['contact']:.4f} | "
              f"Energy: {stats['energy']:.4f}")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, args.out_dir, epoch + 1, tag="tcrfold_energy")
            print(f"  Checkpoint saved at epoch {epoch+1}")

        # Early stopping
        if stopper.update(stats['loss']):
            print(f"Early stopping at epoch {epoch+1} (no improvement for {stopper.patience} epochs)")
            break

    # Save final model
    os.makedirs(args.out_dir, exist_ok=True)
    final_path = os.path.join(args.out_dir, "tcrfold_light_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
