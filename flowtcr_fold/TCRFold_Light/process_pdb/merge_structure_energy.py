"""
Merge Tier 2 structure caches (.npz) with EvoEF2 energy JSONL.

Outputs merged .npz files containing:
- Structure: seq_a/seq_b, ca_a/ca_b, contact_map, distance_map, interface stats
- Energies: E_complex, E_receptor, E_ligand, E_bind, binding_energy
- Derived: E_bind_per_contact, E_bind_per_residue, E_complex_per_len, E_bind_per_area (if SASA available)
- Energy terms (complex/receptor/ligand) if present in cache

Usage:
    python merge_structure_energy.py \
        --npz_dir flowtcr_fold/data/pdb_structures/processed \
        --energy_json flowtcr_fold/data/energy_cache.jsonl \
        --out_dir flowtcr_fold/data/pdb_structures/merged
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_energy_cache(path: Path) -> Dict[str, dict]:
    """Load energy JSONL into dict keyed by sample_key."""
    cache: Dict[str, dict] = {}
    if not path.exists():
        raise FileNotFoundError(f"Energy cache not found: {path}")
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = entry.get("sample_key")
                if not key:
                    # Fallback to pdb_chainAchainB
                    pdb_id = entry.get("pdb_id")
                    chain_a = entry.get("chain_a")
                    chain_b = entry.get("chain_b")
                    if pdb_id and chain_a and chain_b:
                        key = f"{pdb_id}_{chain_a}{chain_b}"
                if not key:
                    continue
                cache[key] = entry
            except json.JSONDecodeError:
                continue
    return cache


def derive_interface_stats(cm: np.ndarray, interface_mask_a=None, interface_mask_b=None) -> Tuple[int, int, int]:
    """Ensure interface stats exist; compute from contact map if needed."""
    n_contacts = int(cm.sum())
    if interface_mask_a is None or interface_mask_b is None:
        interface_mask_a = (cm.sum(axis=1) > 0)
        interface_mask_b = (cm.sum(axis=0) > 0)
    n_int_a = int(interface_mask_a.sum())
    n_int_b = int(interface_mask_b.sum())
    return n_contacts, n_int_a, n_int_b


def main():
    ap = argparse.ArgumentParser(description="Merge structure .npz with EvoEF2 energy cache.")
    ap.add_argument("--npz_dir", required=True, help="Directory of Tier2 .npz files (from preprocess_ppi_pairs.py).")
    ap.add_argument("--energy_json", required=True, help="EvoEF2 energy JSONL (from compute_evoef2_batch.py).")
    ap.add_argument("--out_dir", required=True, help="Output directory for merged .npz.")
    ap.add_argument("--skip_missing_energy", action="store_true", help="Skip samples without energy entry.")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    energy_cache = load_energy_cache(Path(args.energy_json))
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    total = 0
    skipped_no_energy = 0
    written = 0

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        sample_key = npz_path.stem  # e.g., 1ABC_AB

        energy_entry = energy_cache.get(sample_key)
        if energy_entry is None:
            if args.skip_missing_energy:
                skipped_no_energy += 1
                continue
            # fallback to zeros
            energy_entry = {
                "E_complex": 0.0,
                "E_receptor": 0.0,
                "E_ligand": 0.0,
                "E_bind": 0.0,
                "binding_energy": 0.0,
                "energy_terms": {},
            }

        # Load structure basics
        cm = data["contact_map"]
        interface_mask_a = data.get("interface_res_mask_a") if isinstance(data, dict) else None
        interface_mask_b = data.get("interface_res_mask_b") if isinstance(data, dict) else None
        n_contacts, n_int_a, n_int_b = derive_interface_stats(cm, interface_mask_a, interface_mask_b)
        interface_sasa = data["interface_sasa"] if "interface_sasa" in data else np.float32(-1.0)

        len_a = len(data["seq_a"])
        len_b = len(data["seq_b"])
        total_len = max(1, len_a + len_b)

        # Energies
        e_bind = float(energy_entry.get("E_bind", energy_entry.get("binding_energy", 0.0)))
        e_complex = float(energy_entry.get("E_complex", 0.0))
        e_receptor = float(energy_entry.get("E_receptor", 0.0))
        e_ligand = float(energy_entry.get("E_ligand", 0.0))

        # Derived metrics
        e_bind_per_contact = e_bind / max(1, n_contacts)
        e_bind_per_residue = e_bind / max(1, n_int_a + n_int_b)
        e_complex_per_len = e_complex / float(total_len)
        if interface_sasa is not None and float(interface_sasa) > 0:
            e_bind_per_area = e_bind / float(interface_sasa)
        else:
            e_bind_per_area = np.nan

        out_path = out_dir / npz_path.name
        np.savez_compressed(
            out_path,
            # identifiers
            pdb_id=data["pdb_id"],
            chain_id_a=data["chain_id_a"],
            chain_id_b=data["chain_id_b"],
            sample_key=sample_key,
            # sequences
            seq_a=data["seq_a"],
            seq_b=data["seq_b"],
            # coords
            ca_a=data["ca_a"],
            ca_b=data["ca_b"],
            # maps
            contact_map=cm,
            distance_map=data["distance_map"] if "distance_map" in data else np.sqrt(((data["ca_a"][:, None, :] - data["ca_b"][None, :, :]) ** 2).sum(-1)).astype(np.float32),
            # interface stats
            n_interface_contacts=n_contacts,
            n_interface_res_a=n_int_a,
            n_interface_res_b=n_int_b,
            interface_res_mask_a=data["interface_res_mask_a"] if "interface_res_mask_a" in data else (cm.sum(axis=1) > 0).astype(np.int8),
            interface_res_mask_b=data["interface_res_mask_b"] if "interface_res_mask_b" in data else (cm.sum(axis=0) > 0).astype(np.int8),
            interface_sasa=interface_sasa,
            # energies (tier 1)
            E_complex=e_complex,
            E_receptor=e_receptor,
            E_ligand=e_ligand,
            E_bind=e_bind,
            binding_energy=e_bind,
            # derived (tier 2)
            E_bind_per_contact=e_bind_per_contact,
            E_bind_per_residue=e_bind_per_residue,
            E_bind_per_area=e_bind_per_area,
            E_complex_per_len=e_complex_per_len,
            # energy terms (tier 3)
            energy_terms=energy_entry.get("energy_terms", {}),
        )
        written += 1
        total += 1

    print(f"[DONE] merged {written} samples. Skipped missing energy: {skipped_no_energy}")


if __name__ == "__main__":
    main()
