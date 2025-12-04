"""
Preprocess downloaded PDB complexes into pairwise PPI samples.
Outputs Tier 2 structure features for Stage 3 training.

Tier 2 Features (per sample):
- pdb_id, chain_id_a, chain_id_b
- seq_a, seq_b (strings)
- ca_a, ca_b (float32 arrays of shape [L, 3])
- contact_map (int8 matrix [L_a, L_b], 8Å cutoff)
- Interface statistics:
  - n_interface_contacts
  - n_interface_res_a, n_interface_res_b
  - interface_res_mask_a, interface_res_mask_b (boolean arrays)

Usage:
    python preprocess_ppi_pairs.py \
        --pdb_dir flowtcr_fold/data/pdb_structures/raw \
        --out_dir flowtcr_fold/data/pdb_structures/processed
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, is_aa


AA_THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def extract_chain(chain, min_len: int) -> Optional[Tuple[str, np.ndarray]]:
    """Return (sequence, CA coords) if valid; otherwise None."""
    seq: List[str] = []
    coords: List[np.ndarray] = []

    for res in chain:
        if not is_aa(res, standard=True):
            continue
        aa = AA_THREE_TO_ONE.get(res.get_resname())
        if aa is None:
            continue
        try:
            ca_atom = res["CA"]
        except KeyError:
            return None  # Missing CA atom
        seq.append(aa)
        coords.append(ca_atom.get_coord())

    if len(seq) < min_len:
        return None

    return "".join(seq), np.asarray(coords, dtype=np.float32)


def compute_contact_map(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """
    Compute binary contact map using CA-CA distance.
    
    Args:
        coords_a: [L_a, 3] CA coordinates of chain A
        coords_b: [L_b, 3] CA coordinates of chain B
        cutoff: Distance cutoff in Angstroms (default 8.0)
    
    Returns:
        [L_a, L_b] binary contact map (int8)
    """
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return (dist2 <= cutoff * cutoff).astype(np.int8)


def compute_interface_stats(contact_map: np.ndarray) -> Dict:
    """
    Compute interface statistics from contact map.
    
    Args:
        contact_map: [L_a, L_b] binary contact map
    
    Returns:
        Dictionary with interface statistics:
        - n_interface_contacts: total number of contacts
        - interface_res_mask_a: [L_a] boolean mask of interface residues in chain A
        - interface_res_mask_b: [L_b] boolean mask of interface residues in chain B
        - n_interface_res_a: number of interface residues in chain A
        - n_interface_res_b: number of interface residues in chain B
    """
    n_contacts = int(contact_map.sum())
    
    # Interface residues: residues that have at least one contact
    interface_res_mask_a = (contact_map.sum(axis=1) > 0)  # [L_a]
    interface_res_mask_b = (contact_map.sum(axis=0) > 0)  # [L_b]
    
    n_interface_res_a = int(interface_res_mask_a.sum())
    n_interface_res_b = int(interface_res_mask_b.sum())
    
    return {
        'n_interface_contacts': n_contacts,
        'interface_res_mask_a': interface_res_mask_a.astype(np.int8),
        'interface_res_mask_b': interface_res_mask_b.astype(np.int8),
        'n_interface_res_a': n_interface_res_a,
        'n_interface_res_b': n_interface_res_b,
    }


def compute_distance_map(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """
    Compute distance map between two chains.
    
    Args:
        coords_a: [L_a, 3] CA coordinates
        coords_b: [L_b, 3] CA coordinates
    
    Returns:
        [L_a, L_b] distance matrix (float32)
    """
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)


def process_one(
    pdb_path: Path,
    out_dir: Path,
    cutoff: float,
    min_len: int,
    min_contacts: int,
    min_contact_ratio: float,
    existing_npz: set = None,
) -> Tuple[int, int]:
    """
    Process one PDB/CIF file and save Tier 2 structure features.
    
    Returns:
        (saved_count, skipped_count)
    """
    if existing_npz is None:
        existing_npz = set()
    
    # Choose parser based on file extension
    suffix = pdb_path.suffix.lower()
    if suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as exc:
        print(f"[WARN] failed to parse {pdb_path.name}: {exc}", file=sys.stderr)
        return 0, 0

    model = next(structure.get_models(), None)
    if model is None:
        return 0, 0

    # Extract all valid chains
    chains = []
    for chain in model.get_chains():
        chain_id = chain.get_id()
        extracted = extract_chain(chain, min_len=min_len)
        if extracted is None:
            continue
        seq, coords = extracted
        chains.append((chain_id, seq, coords))

    if len(chains) < 2:
        return 0, 0

    saved = 0
    skipped = 0
    
    for (id_a, seq_a, ca_a), (id_b, seq_b, ca_b) in combinations(chains, 2):
        out_name = f"{pdb_path.stem}_{id_a}{id_b}.npz"
        
        # Skip existing files
        if out_name in existing_npz or (out_dir / out_name).exists():
            skipped += 1
            continue
        
        # Compute contact map (Tier 2 core)
        cm = compute_contact_map(ca_a, ca_b, cutoff=cutoff)
        
        # Compute interface statistics (Tier 2)
        interface_stats = compute_interface_stats(cm)
        n_contacts = interface_stats['n_interface_contacts']
        
        # Filter by contacts
        ratio = n_contacts / float(len(seq_a) * len(seq_b))
        if n_contacts < min_contacts or ratio < min_contact_ratio:
            continue

        # Compute distance map (for model training)
        dist_map = compute_distance_map(ca_a, ca_b)

        # Save Tier 2 structure features
        out_path = out_dir / out_name
        np.savez_compressed(
            out_path,
            # Basic identifiers
            pdb_id=pdb_path.stem,
            chain_id_a=id_a,
            chain_id_b=id_b,
            # Sequences
            seq_a=seq_a,
            seq_b=seq_b,
            # Coordinates
            ca_a=ca_a,
            ca_b=ca_b,
            # Contact/distance maps
            contact_map=cm,
            distance_map=dist_map,
            # Interface statistics (Tier 2)
            n_interface_contacts=n_contacts,
            n_interface_res_a=interface_stats['n_interface_res_a'],
            n_interface_res_b=interface_stats['n_interface_res_b'],
            interface_res_mask_a=interface_stats['interface_res_mask_a'],
            interface_res_mask_b=interface_stats['interface_res_mask_b'],
            # Placeholder for interface SASA (can be filled later)
            interface_sasa=np.float32(-1.0),
        )
        saved += 1

    return saved, skipped


def main():
    ap = argparse.ArgumentParser(description="Preprocess PDB complexes into PPI pair samples (Tier 2).")
    ap.add_argument("--pdb_dir", required=True, help="Directory containing raw .pdb/.cif files.")
    ap.add_argument("--out_dir", required=True, help="Directory to write processed .npz files.")
    ap.add_argument("--cutoff", type=float, default=8.0, help="Contact cutoff in Å (default: 8.0).")
    ap.add_argument("--min_len", type=int, default=30, help="Minimum chain length (AA).")
    ap.add_argument("--min_contacts", type=int, default=10, help="Minimum CA-CA contacts to keep a pair.")
    ap.add_argument("--min_contact_ratio", type=float, default=0.0,
                    help="Minimum contact ratio (#contacts / (La*Lb)).")
    ap.add_argument("--force", action="store_true", help="Force reprocess existing .npz files.")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Support both PDB and CIF formats
    pdb_files = sorted(list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif")))
    if not pdb_files:
        print(f"No .pdb or .cif files found in {pdb_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(pdb_files)} PDB/CIF files to process")

    # Pre-collect existing .npz files for fast skipping
    existing_npz = set()
    if not args.force:
        existing_npz = {f.name for f in out_dir.glob("*.npz")}
        print(f"[INFO] Found {len(existing_npz)} existing .npz files (will skip)")

    total_pairs = 0
    total_skipped = 0
    processed_count = 0
    
    for i, pdb_path in enumerate(pdb_files, 1):
        saved, skipped = process_one(
            pdb_path,
            out_dir,
            cutoff=args.cutoff,
            min_len=args.min_len,
            min_contacts=args.min_contacts,
            min_contact_ratio=args.min_contact_ratio,
            existing_npz=existing_npz,
        )
        
        if saved > 0:
            total_pairs += saved
        
        total_skipped += skipped
        processed_count += 1
        
        # Progress every 500 files
        if processed_count % 500 == 0:
            print(f"[INFO] Progress: {processed_count}/{len(pdb_files)} files, "
                  f"{total_pairs} new pairs, {total_skipped} skipped", flush=True)

    print(f"\n[DONE] Processed {processed_count} files.")
    print(f"  - New pairs saved: {total_pairs}")
    print(f"  - Pairs skipped: {total_skipped}")
    print(f"  - Total .npz files: {len(list(out_dir.glob('*.npz')))}")


if __name__ == "__main__":
    main()
