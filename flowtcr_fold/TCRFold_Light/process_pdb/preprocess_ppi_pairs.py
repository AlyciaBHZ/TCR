"""
Preprocess downloaded PDB complexes into pairwise PPI samples.

Filters:
- At least two protein chains
- Chain length >= min_len (default 30 AA)
- At least min_contacts CA-CA pairs within cutoff (default 10 contacts at 8Å)

Outputs compressed .npz files containing (Stage3 Phase0 processed PPI pairs):
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
from Bio.PDB import PDBParser, MMCIFParser, is_aa


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
        try:
            ca_atom = res["CA"]
        except KeyError:
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
    existing_npz: set = None,
) -> Tuple[int, int]:
    """
    Process one PDB/CIF file.
    
    Returns:
        (saved_count, skipped_count)
    """
    if existing_npz is None:
        existing_npz = set()
    
    # 根据文件扩展名选择解析器
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
        
        # 跳过已存在的文件
        if out_name in existing_npz or (out_dir / out_name).exists():
            skipped += 1
            continue
        
        cm = contact_map(ca_a, ca_b, cutoff=cutoff)
        num_contacts = int(cm.sum())
        ratio = num_contacts / float(len(seq_a) * len(seq_b))

        if num_contacts < min_contacts or ratio < min_contact_ratio:
            continue

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

    return saved, skipped


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
    ap.add_argument("--force", action="store_true", help="Force reprocess existing .npz files.")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 支持 PDB 和 CIF 格式
    pdb_files = sorted(list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif")))
    if not pdb_files:
        print(f"No .pdb or .cif files found in {pdb_dir}", file=sys.stderr)
        sys.exit(1)

    # 预先收集已存在的 .npz 文件名 (用于快速跳过)
    existing_npz = set()
    if not args.force:
        existing_npz = {f.name for f in out_dir.glob("*.npz")}
        print(f"Found {len(existing_npz)} existing .npz files (will skip)")

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
            print(f"[OK]   {pdb_path.name}: saved {saved} pairs")
            total_pairs += saved
        
        total_skipped += skipped
        processed_count += 1
        
        # 每处理 500 个文件打印进度
        if processed_count % 500 == 0:
            print(f"[INFO] Progress: {processed_count}/{len(pdb_files)} files processed, "
                  f"{total_pairs} pairs saved, {total_skipped} skipped")

    print(f"\nDone. Processed {processed_count} files.")
    print(f"  - New pairs saved: {total_pairs}")
    print(f"  - Pairs skipped (already exist): {total_skipped}")
    print(f"  - Total .npz files: {len(list(out_dir.glob('*.npz')))}")


if __name__ == "__main__":
    main()
