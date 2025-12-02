"""
Preprocess downloaded PDB complexes into pairwise PPI samples.

Filters:
- At least two protein chains
- Chain length >= min_len (default 30 AA)
- At least min_contacts CA-CA pairs within cutoff (default 10 contacts at 8Å)

Outputs compressed .npz files containing:
- pdb_id, chain_id_a, chain_id_b
- seq_a, seq_b (strings)
- ca_a, ca_b (float32 arrays of shape [L, 3])
- contact_map (int8 matrix [L_a, L_b])
- num_contacts (int)
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser, is_aa


AA_THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
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
        ca_atom = res.get("CA")
        if ca_atom is None:
            return None
        seq.append(aa)
        coords.append(ca_atom.get_coord())

    if len(seq) < min_len:
        return None

    return "".join(seq), np.asarray(coords, dtype=np.float32)


def contact_map(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float) -> np.ndarray:
    """Compute binary contact map using CA-CA distance."""
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return (dist2 <= cutoff * cutoff).astype(np.int8)


def process_one(
    pdb_path: Path,
    out_dir: Path,
    cutoff: float,
    min_len: int,
    min_contacts: int,
    min_contact_ratio: float,
) -> int:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as exc:
        print(f"[WARN] failed to parse {pdb_path.name}: {exc}", file=sys.stderr)
        return 0

    model = next(structure.get_models(), None)
    if model is None:
        return 0

    chains = []
    for chain in model.get_chains():
        chain_id = chain.get_id()
        extracted = extract_chain(chain, min_len=min_len)
        if extracted is None:
            continue
        seq, coords = extracted
        chains.append((chain_id, seq, coords))

    if len(chains) < 2:
        return 0

    saved = 0
    for (id_a, seq_a, ca_a), (id_b, seq_b, ca_b) in combinations(chains, 2):
        cm = contact_map(ca_a, ca_b, cutoff=cutoff)
        num_contacts = int(cm.sum())
        ratio = num_contacts / float(len(seq_a) * len(seq_b))

        if num_contacts < min_contacts or ratio < min_contact_ratio:
            continue

        out_name = f"{pdb_path.stem}_{id_a}{id_b}.npz"
        out_path = out_dir / out_name
        np.savez_compressed(
            out_path,
            pdb_id=pdb_path.stem,
            chain_id_a=id_a,
            chain_id_b=id_b,
            seq_a=seq_a,
            seq_b=seq_b,
            ca_a=ca_a,
            ca_b=ca_b,
            contact_map=cm,
            num_contacts=num_contacts,
        )
        saved += 1

    return saved


def main():
    ap = argparse.ArgumentParser(description="Preprocess PDB complexes into PPI pair samples.")
    ap.add_argument("--pdb_dir", required=True, help="Directory containing raw .pdb files.")
    ap.add_argument("--out_dir", required=True, help="Directory to write processed .npz files.")
    ap.add_argument("--cutoff", type=float, default=8.0, help="Contact cutoff in Å (default: 8.0).")
    ap.add_argument("--min_len", type=int, default=30, help="Minimum chain length (AA).")
    ap.add_argument("--min_contacts", type=int, default=10, help="Minimum CA-CA contacts to keep a pair.")
    ap.add_argument(
        "--min_contact_ratio",
        type=float,
        default=0.0,
        help="Minimum contact ratio (#contacts / (La*Lb)); set >0 to filter weak interfaces.",
    )
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"No .pdb files found in {pdb_dir}", file=sys.stderr)
        sys.exit(1)

    total_pairs = 0
    for pdb_path in pdb_files:
        saved = process_one(
            pdb_path,
            out_dir,
            cutoff=args.cutoff,
            min_len=args.min_len,
            min_contacts=args.min_contacts,
            min_contact_ratio=args.min_contact_ratio,
        )
        if saved:
            print(f"[OK]   {pdb_path.name}: saved {saved} pairs")
            total_pairs += saved

    print(f"Done. Total pairs saved: {total_pairs}")


if __name__ == "__main__":
    main()
