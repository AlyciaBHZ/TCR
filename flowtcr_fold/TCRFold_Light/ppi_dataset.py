"""
PPIDataset: Unified dataset for PPI structure and energy data.
Loads Tier 1+2+3 features for Stage 3 training.

Tier 2 (Structure & Interface):
- seq_a, seq_b, ca_a, ca_b
- contact_map, distance_map
- n_interface_contacts, n_interface_res_a/b, interface_res_mask_a/b

Tier 1 (Global Energies):
- E_complex, E_receptor, E_ligand, E_bind

Tier 2 Derived (Normalized Energies):
- E_bind_per_contact, E_bind_per_residue, E_complex_per_len

Tier 3 (Energy Terms):
- energy_terms.{complex, receptor, ligand}.{vdw, elec, desolv, hbond, ...}

Usage:
    from flowtcr_fold.TCRFold_Light.ppi_dataset import PPIDataset, collate_ppi_batch
    
    dataset = PPIDataset(
        data_dir="flowtcr_fold/data/ppi_merged",
        max_length=512,
        energy_targets=['E_bind', 'E_complex']
    )
    
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_ppi_batch)
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


# Standard amino acid mapping
AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,  # Unknown
}
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}
PAD_IDX = 21  # Padding index


class PPIDataset(Dataset):
    """
    Dataset for PPI structures with Tier 1+2+3 features.
    
    Loads merged .npz files containing:
    - Tier 2: Structure (seq, coords, contact_map, interface stats)
    - Tier 1: Global energies (E_complex, E_receptor, E_ligand, E_bind)
    - Tier 2 derived: Normalized energies
    - Tier 3: Energy term breakdown (optional)
    
    Args:
        data_dir: Directory containing merged .npz files
        max_length: Maximum sequence length (longer sequences are skipped)
        energy_targets: List of energy keys to include as labels
        include_tier3: Whether to load Tier 3 energy term breakdown
        verbose: Print loading statistics
    """
    
    def __init__(
        self,
        data_dir: str,
        max_length: int = 512,
        energy_targets: List[str] = None,
        include_tier3: bool = False,
        verbose: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.energy_targets = energy_targets or ['E_bind', 'E_complex', 'E_receptor', 'E_ligand']
        self.include_tier3 = include_tier3
        self.verbose = verbose
        
        # Load all samples
        self.samples = self._load_samples()
        
        if self.verbose:
            print(f"[PPIDataset] Loaded {len(self.samples)} samples from {self.data_dir}")
    
    def _load_samples(self) -> List[Dict]:
        """Load and validate all .npz files."""
        npz_files = sorted(self.data_dir.glob("*.npz"))
        
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")
        
        samples = []
        skipped_length = 0
        skipped_missing = 0
        
        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                
                # Required fields
                seq_a = str(data['seq_a'])
                seq_b = str(data['seq_b'])
                
                # Skip if too long
                if len(seq_a) > self.max_length or len(seq_b) > self.max_length:
                    skipped_length += 1
                    continue
                
                # Build sample dict
                sample = {
                    'path': str(npz_path),
                    'sample_key': npz_path.stem,
                    'pdb_id': str(data['pdb_id']),
                    'chain_a': str(data['chain_a']),
                    'chain_b': str(data['chain_b']),
                    
                    # Tier 2: Sequences
                    'seq_a': seq_a,
                    'seq_b': seq_b,
                    'len_a': len(seq_a),
                    'len_b': len(seq_b),
                    
                    # Tier 2: Coordinates
                    'ca_a': data['ca_a'].astype(np.float32),
                    'ca_b': data['ca_b'].astype(np.float32),
                    
                    # Tier 2: Contact map
                    'contact_map': data['contact_map'].astype(np.float32),
                    
                    # Tier 2: Interface statistics
                    'n_interface_contacts': int(data['n_interface_contacts']),
                    'n_interface_res_a': int(data['n_interface_res_a']),
                    'n_interface_res_b': int(data['n_interface_res_b']),
                }
                
                # Optional: Distance map
                if 'distance_map' in data:
                    sample['distance_map'] = data['distance_map'].astype(np.float32)
                
                # Optional: Interface residue masks
                if 'interface_res_mask_a' in data:
                    sample['interface_res_mask_a'] = data['interface_res_mask_a'].astype(np.float32)
                if 'interface_res_mask_b' in data:
                    sample['interface_res_mask_b'] = data['interface_res_mask_b'].astype(np.float32)
                
                # Tier 1: Global energies
                for key in ['E_complex', 'E_receptor', 'E_ligand', 'E_bind']:
                    if key in data:
                        sample[key] = float(data[key])
                    else:
                        sample[key] = 0.0
                
                # Tier 2 derived: Normalized energies
                for key in ['E_bind_per_contact', 'E_bind_per_residue', 'E_complex_per_len']:
                    if key in data:
                        sample[key] = float(data[key])
                    else:
                        sample[key] = 0.0
                
                # Tier 3: Energy terms (optional)
                if self.include_tier3 and 'energy_terms_json' in data:
                    try:
                        sample['energy_terms'] = json.loads(str(data['energy_terms_json']))
                    except json.JSONDecodeError:
                        sample['energy_terms'] = {}
                
                samples.append(sample)
                
            except (KeyError, ValueError) as e:
                skipped_missing += 1
                if self.verbose:
                    print(f"[WARN] Skipping {npz_path.name}: {e}")
        
        if self.verbose:
            print(f"[PPIDataset] Skipped {skipped_length} (too long), {skipped_missing} (missing data)")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Convert sequences to indices
        seq_a_idx = torch.tensor([AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in sample['seq_a']], dtype=torch.long)
        seq_b_idx = torch.tensor([AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in sample['seq_b']], dtype=torch.long)
        
        # Coordinates
        ca_a = torch.from_numpy(sample['ca_a'])
        ca_b = torch.from_numpy(sample['ca_b'])
        
        # Contact map
        contact_map = torch.from_numpy(sample['contact_map'])
        
        # Distance map (compute if not available)
        if 'distance_map' in sample:
            distance_map = torch.from_numpy(sample['distance_map'])
        else:
            diff = ca_a[:, None, :] - ca_b[None, :, :]
            distance_map = torch.sqrt((diff ** 2).sum(-1))
        
        # Masks (1 = valid, 0 = padding)
        mask_a = torch.ones(len(seq_a_idx), dtype=torch.float32)
        mask_b = torch.ones(len(seq_b_idx), dtype=torch.float32)
        
        # Interface residue masks
        if 'interface_res_mask_a' in sample:
            interface_mask_a = torch.from_numpy(sample['interface_res_mask_a'])
        else:
            interface_mask_a = (contact_map.sum(dim=1) > 0).float()
        
        if 'interface_res_mask_b' in sample:
            interface_mask_b = torch.from_numpy(sample['interface_res_mask_b'])
        else:
            interface_mask_b = (contact_map.sum(dim=0) > 0).float()
        
        # Energy labels
        energy_dict = {}
        for key in self.energy_targets:
            if key in sample:
                energy_dict[key] = torch.tensor(sample[key], dtype=torch.float32)
        
        return {
            'sample_key': sample['sample_key'],
            
            # Sequences
            'seq_a': seq_a_idx,
            'seq_b': seq_b_idx,
            
            # Coordinates
            'ca_a': ca_a,
            'ca_b': ca_b,
            
            # Maps
            'contact_map': contact_map,
            'distance_map': distance_map,
            
            # Masks
            'mask_a': mask_a,
            'mask_b': mask_b,
            'interface_mask_a': interface_mask_a,
            'interface_mask_b': interface_mask_b,
            
            # Interface stats
            'n_interface_contacts': torch.tensor(sample['n_interface_contacts'], dtype=torch.long),
            'n_interface_res_a': torch.tensor(sample['n_interface_res_a'], dtype=torch.long),
            'n_interface_res_b': torch.tensor(sample['n_interface_res_b'], dtype=torch.long),
            
            # Energy labels (Tier 1 + Tier 2 derived)
            **energy_dict,
        }


def collate_ppi_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch with padding for variable-length structures.
    
    Also prepares merged inputs for Evoformer-style models:
    - merged_seq: [B, L_a + L_b] concatenated sequences
    - merged_mask: [B, L_a + L_b] attention mask
    - pair_type: [B, L_a + L_b, L_a + L_b] pair type IDs
        - 0: intra-A
        - 1: intra-B  
        - 2: inter-AB
    """
    B = len(batch)
    max_len_a = max(item['seq_a'].size(0) for item in batch)
    max_len_b = max(item['seq_b'].size(0) for item in batch)
    max_len_merged = max_len_a + max_len_b
    
    # Initialize padded tensors
    seq_a_batch = torch.full((B, max_len_a), PAD_IDX, dtype=torch.long)
    seq_b_batch = torch.full((B, max_len_b), PAD_IDX, dtype=torch.long)
    ca_a_batch = torch.zeros(B, max_len_a, 3)
    ca_b_batch = torch.zeros(B, max_len_b, 3)
    contact_map_batch = torch.zeros(B, max_len_a, max_len_b)
    distance_map_batch = torch.zeros(B, max_len_a, max_len_b)
    mask_a_batch = torch.zeros(B, max_len_a)
    mask_b_batch = torch.zeros(B, max_len_b)
    interface_mask_a_batch = torch.zeros(B, max_len_a)
    interface_mask_b_batch = torch.zeros(B, max_len_b)
    
    # Interface stats
    n_contacts_batch = torch.zeros(B, dtype=torch.long)
    n_int_res_a_batch = torch.zeros(B, dtype=torch.long)
    n_int_res_b_batch = torch.zeros(B, dtype=torch.long)
    
    # Merged tensors for Evoformer
    merged_seq_batch = torch.full((B, max_len_merged), PAD_IDX, dtype=torch.long)
    merged_mask_batch = torch.zeros(B, max_len_merged)
    pair_type_batch = torch.zeros(B, max_len_merged, max_len_merged, dtype=torch.long)
    
    # Energy labels
    energy_keys = ['E_bind', 'E_complex', 'E_receptor', 'E_ligand',
                   'E_bind_per_contact', 'E_bind_per_residue', 'E_complex_per_len']
    energy_batches = {k: torch.zeros(B) for k in energy_keys}
    
    sample_keys = []
    
    for i, item in enumerate(batch):
        L_a = item['seq_a'].size(0)
        L_b = item['seq_b'].size(0)
        
        # Fill individual chain tensors
        seq_a_batch[i, :L_a] = item['seq_a']
        seq_b_batch[i, :L_b] = item['seq_b']
        ca_a_batch[i, :L_a] = item['ca_a']
        ca_b_batch[i, :L_b] = item['ca_b']
        contact_map_batch[i, :L_a, :L_b] = item['contact_map']
        distance_map_batch[i, :L_a, :L_b] = item['distance_map']
        mask_a_batch[i, :L_a] = item['mask_a']
        mask_b_batch[i, :L_b] = item['mask_b']
        interface_mask_a_batch[i, :L_a] = item['interface_mask_a']
        interface_mask_b_batch[i, :L_b] = item['interface_mask_b']
        
        n_contacts_batch[i] = item['n_interface_contacts']
        n_int_res_a_batch[i] = item['n_interface_res_a']
        n_int_res_b_batch[i] = item['n_interface_res_b']
        
        sample_keys.append(item['sample_key'])
        
        # Energy labels
        for k in energy_keys:
            if k in item:
                energy_batches[k][i] = item[k]
        
        # Build merged tensors
        merged_seq_batch[i, :L_a] = item['seq_a']
        merged_seq_batch[i, L_a:L_a + L_b] = item['seq_b']
        merged_mask_batch[i, :L_a + L_b] = 1.0
        
        # Pair type: 0=intra-A, 1=intra-B, 2=inter-AB
        pair_type_batch[i, :L_a, :L_a] = 0  # intra-A
        pair_type_batch[i, L_a:L_a + L_b, L_a:L_a + L_b] = 1  # intra-B
        pair_type_batch[i, :L_a, L_a:L_a + L_b] = 2  # inter-AB
        pair_type_batch[i, L_a:L_a + L_b, :L_a] = 2  # inter-BA (symmetric)
    
    return {
        'sample_keys': sample_keys,
        
        # Individual chain data
        'seq_a': seq_a_batch,
        'seq_b': seq_b_batch,
        'ca_a': ca_a_batch,
        'ca_b': ca_b_batch,
        'contact_map': contact_map_batch,
        'distance_map': distance_map_batch,
        'mask_a': mask_a_batch,
        'mask_b': mask_b_batch,
        'interface_mask_a': interface_mask_a_batch,
        'interface_mask_b': interface_mask_b_batch,
        
        # Interface stats
        'n_interface_contacts': n_contacts_batch,
        'n_interface_res_a': n_int_res_a_batch,
        'n_interface_res_b': n_int_res_b_batch,
        
        # Merged for Evoformer
        'merged_seq': merged_seq_batch,
        'merged_mask': merged_mask_batch,
        'pair_type': pair_type_batch,
        
        # Energy labels
        **energy_batches,
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Test PPIDataset loading.")
    ap.add_argument("--data_dir", type=str, default="flowtcr_fold/data/ppi_merged",
                    help="Directory containing merged .npz files.")
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    args = ap.parse_args()
    
    print("=" * 60)
    print("Testing PPIDataset")
    print("=" * 60)
    
    try:
        dataset = PPIDataset(
            data_dir=args.data_dir,
            max_length=512,
            energy_targets=['E_bind', 'E_complex', 'E_receptor', 'E_ligand',
                          'E_bind_per_contact', 'E_bind_per_residue'],
            verbose=True
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test single sample
            sample = dataset[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"  seq_a shape: {sample['seq_a'].shape}")
            print(f"  seq_b shape: {sample['seq_b'].shape}")
            print(f"  ca_a shape: {sample['ca_a'].shape}")
            print(f"  contact_map shape: {sample['contact_map'].shape}")
            print(f"  E_bind: {sample.get('E_bind', 'N/A')}")
            print(f"  E_bind_per_contact: {sample.get('E_bind_per_contact', 'N/A')}")
            print(f"  n_interface_contacts: {sample['n_interface_contacts'].item()}")
            
            # Test DataLoader
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_ppi_batch
            )
            
            batch = next(iter(loader))
            print(f"\nBatch keys: {list(batch.keys())}")
            print(f"  seq_a batch: {batch['seq_a'].shape}")
            print(f"  contact_map batch: {batch['contact_map'].shape}")
            print(f"  merged_seq: {batch['merged_seq'].shape}")
            print(f"  pair_type: {batch['pair_type'].shape}")
            print(f"  E_bind: {batch['E_bind']}")
            
        print("\n✅ PPIDataset test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
