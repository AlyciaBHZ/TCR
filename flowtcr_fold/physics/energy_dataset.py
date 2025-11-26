"""
Energy-supervised Dataset for TCRFold-Light
============================================

Dataset that pairs structures with EvoEF2 energy labels for training
the energy surrogate head in TCRFold-Light.

Usage in training:
    >>> from flowtcr_fold.physics.energy_dataset import EnergyStructureDataset
    >>> dataset = EnergyStructureDataset(
    ...     pdb_dir="data/structures",
    ...     cache_dir="data/energy_cache"
    ... )
    >>> dataloader = DataLoader(dataset, batch_size=4)
    >>> for batch in dataloader:
    ...     s, z, energy_label = batch
    ...     loss = model.energy_head_loss(s, z, energy_label)
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, PDBIO

from flowtcr_fold.physics.evoef_runner import EvoEF2Runner, BindingResult


class EnergyStructureDataset(Dataset):
    """
    Dataset combining PDB structures with EvoEF2 energy labels.

    For each structure:
    1. Extract coordinates, distance matrix, contact map
    2. Compute binding energy with EvoEF2
    3. Cache results for fast loading

    This provides supervision for the energy surrogate head.
    """

    def __init__(
        self,
        pdb_dir: str,
        cache_dir: str,
        chain_splits: Optional[Dict[str, str]] = None,
        max_length: int = 256,
        recompute: bool = False,
        verbose: bool = False
    ):
        """
        Args:
            pdb_dir: Directory containing PDB files
            cache_dir: Where to cache computed energies/features
            chain_splits: Dict mapping PDB names to chain splits (e.g., {"1abc": "AB,C"})
            max_length: Maximum sequence length
            recompute: Force recompute cached energies
            verbose: Print progress
        """
        self.pdb_dir = Path(pdb_dir)
        self.cache_dir = Path(cache_dir)
        self.chain_splits = chain_splits or {}
        self.max_length = max_length
        self.verbose = verbose

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize EvoEF2 runner
        self.evoef = EvoEF2Runner(verbose=False)

        # Scan for PDB files
        self.pdb_files = self._scan_pdb_files()

        # Compute or load cached energies
        self.energy_cache_file = self.cache_dir / "energy_cache.json"
        if recompute or not self.energy_cache_file.exists():
            self.energy_cache = self._compute_all_energies()
        else:
            with open(self.energy_cache_file, 'r') as f:
                self.energy_cache = json.load(f)

        # Filter valid samples
        self.samples = self._build_sample_list()

        if self.verbose:
            print(f"EnergyStructureDataset: {len(self.samples)} structures loaded")

    def _scan_pdb_files(self) -> List[str]:
        """Scan for PDB files."""
        pdb_files = []
        for ext in ['*.pdb', '*.ent']:
            pdb_files.extend(self.pdb_dir.glob(ext))
        return [str(p) for p in sorted(pdb_files)]

    def _compute_all_energies(self) -> Dict[str, float]:
        """Compute binding energies for all PDB files."""
        print(f"Computing energies for {len(self.pdb_files)} structures...")

        energy_cache = {}

        for i, pdb_path in enumerate(self.pdb_files):
            pdb_name = Path(pdb_path).stem

            try:
                # Get chain split if specified
                split = self.chain_splits.get(pdb_name, None)

                # Compute binding energy
                result = self.evoef.compute_binding(pdb_path, split=split)

                energy_cache[pdb_name] = {
                    'binding_energy': result.binding_energy,
                    'complex_energy': result.complex_energy,
                    'pdb_path': pdb_path
                }

                if self.verbose and (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(self.pdb_files)}")

            except Exception as e:
                print(f"Warning: Failed to compute energy for {pdb_name}: {e}")
                energy_cache[pdb_name] = {
                    'binding_energy': 0.0,
                    'complex_energy': 0.0,
                    'pdb_path': pdb_path,
                    'error': str(e)
                }

        # Save cache
        with open(self.energy_cache_file, 'w') as f:
            json.dump(energy_cache, f, indent=2)

        print(f"Energy cache saved to {self.energy_cache_file}")

        return energy_cache

    def _build_sample_list(self) -> List[Dict]:
        """Build list of valid samples."""
        samples = []

        for pdb_name, energy_data in self.energy_cache.items():
            if 'error' in energy_data:
                continue

            pdb_path = energy_data['pdb_path']
            if not os.path.exists(pdb_path):
                continue

            samples.append({
                'pdb_name': pdb_name,
                'pdb_path': pdb_path,
                'binding_energy': energy_data['binding_energy'],
                'complex_energy': energy_data['complex_energy']
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get structure features + energy label.

        Returns:
            {
                's': [L, s_dim] sequence features (placeholder),
                'z': [L, L, z_dim] pairwise features (placeholder),
                'distance_map': [L, L] Cβ distance matrix,
                'contact_map': [L, L] binary contact map (< 8Å),
                'energy': scalar binding energy,
                'mask': [L] valid residue mask
            }
        """
        sample = self.samples[idx]

        # Load structure and compute geometric features
        structure_data = self._load_structure(sample['pdb_path'])

        # Add energy label
        structure_data['energy'] = torch.tensor(sample['binding_energy'], dtype=torch.float32)

        return structure_data

    def _load_structure(self, pdb_path: str) -> Dict[str, torch.Tensor]:
        """
        Load PDB and extract geometric features.

        For simplicity, we provide placeholders for s/z and compute distance/contact maps.
        In full implementation, this would extract residue embeddings from the encoder.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        # Extract Cα/Cβ coordinates
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] != ' ':  # Skip HETATM
                        continue

                    # Try Cβ first (not for Gly), then Cα
                    if 'CB' in residue:
                        coords.append(residue['CB'].get_coord())
                    elif 'CA' in residue:
                        coords.append(residue['CA'].get_coord())

        if not coords:
            raise ValueError(f"No valid atoms found in {pdb_path}")

        coords = np.array(coords)
        L = len(coords)

        # Truncate if too long
        if L > self.max_length:
            coords = coords[:self.max_length]
            L = self.max_length

        # Compute distance matrix
        dist_mat = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                dist_mat[i, j] = np.linalg.norm(coords[i] - coords[j])

        # Contact map (< 8Å)
        contact_map = (dist_mat < 8.0).astype(np.float32)

        # Placeholders for s/z (to be filled by encoder in training)
        s_dim = 512
        z_dim = 128
        s = torch.zeros(L, s_dim)
        z = torch.zeros(L, L, z_dim)

        # Mask (all valid in this simple implementation)
        mask = torch.ones(L, dtype=torch.long)

        return {
            's': s,
            'z': z,
            'distance_map': torch.from_numpy(dist_mat).float(),
            'contact_map': torch.from_numpy(contact_map).float(),
            'mask': mask
        }


# =============================================================================
# Collate Function for DataLoader
# =============================================================================

def collate_energy_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch with padding for variable-length structures.
    """
    max_len = max(item['mask'].size(0) for item in batch)
    B = len(batch)

    s_dim = batch[0]['s'].size(1)
    z_dim = batch[0]['z'].size(2)

    # Initialize padded tensors
    s_batch = torch.zeros(B, max_len, s_dim)
    z_batch = torch.zeros(B, max_len, max_len, z_dim)
    dist_batch = torch.zeros(B, max_len, max_len)
    contact_batch = torch.zeros(B, max_len, max_len)
    mask_batch = torch.zeros(B, max_len, dtype=torch.long)
    energy_batch = torch.zeros(B)

    for i, item in enumerate(batch):
        L = item['mask'].size(0)
        s_batch[i, :L] = item['s']
        z_batch[i, :L, :L] = item['z']
        dist_batch[i, :L, :L] = item['distance_map']
        contact_batch[i, :L, :L] = item['contact_map']
        mask_batch[i, :L] = item['mask']
        energy_batch[i] = item['energy']

    return {
        's': s_batch,
        'z': z_batch,
        'distance_map': dist_batch,
        'contact_map': contact_batch,
        'mask': mask_batch,
        'energy': energy_batch
    }


# =============================================================================
# Example Training Integration
# =============================================================================

def train_energy_head_example():
    """
    Example: Training TCRFold-Light's energy head with EvoEF2 supervision.
    """
    from torch.utils.data import DataLoader
    # from flowtcr_fold.TCRFold_Light.tcrfold_light import TCRFoldLight

    # Create dataset
    dataset = EnergyStructureDataset(
        pdb_dir="data/tcr_structures",
        cache_dir="data/energy_cache",
        verbose=True
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_energy_batch
    )

    # model = TCRFoldLight(s_dim=512, z_dim=128)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):
        for batch in loader:
            s = batch['s']  # [B, L, 512]
            z = batch['z']  # [B, L, L, 128]
            energy_label = batch['energy']  # [B]

            # Forward pass (placeholder)
            # out = model(s, z)
            # energy_pred = out['energy']  # [B]

            # Loss
            # loss = torch.nn.functional.mse_loss(energy_pred, energy_label)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            pass


if __name__ == "__main__":
    # Test dataset creation
    import sys

    print("Testing EnergyStructureDataset...")

    test_dir = "data/test_structures"
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Creating dummy...")
        os.makedirs(test_dir, exist_ok=True)

    try:
        dataset = EnergyStructureDataset(
            pdb_dir=test_dir,
            cache_dir="data/test_cache",
            verbose=True,
            recompute=True
        )

        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"  s shape: {sample['s'].shape}")
            print(f"  z shape: {sample['z'].shape}")
            print(f"  distance_map shape: {sample['distance_map'].shape}")
            print(f"  energy: {sample['energy'].item():.2f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
