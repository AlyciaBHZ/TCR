"""
Batch EvoEF2 energy computation for PDB complexes.

Usage:
    python scripts/batch_evoef2.py \
        --pdb_dir flowtcr_fold/data/pdb_structures/raw \
        --output flowtcr_fold/data/energy_cache.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from Bio.PDB import PDBParser

from flowtcr_fold.physics.evoef_runner import EvoEF2Runner


def guess_split(pdb_path: Path) -> Optional[str]:
    """Guess chain split as first chain vs the rest (e.g., A vs BC...)."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as exc:
        print(f"[WARN] failed to parse {pdb_path.name}: {exc}", file=sys.stderr)
        return None

    model = next(structure.get_models(), None)
    if model is None:
        return None

    chain_ids = [c.get_id() for c in model.get_chains()]
    if len(chain_ids) < 2:
        return None

    return f"{chain_ids[0]},{''.join(chain_ids[1:])}"


def main():
    ap = argparse.ArgumentParser(description="Compute EvoEF2 binding energies for a directory of PDBs.")
    ap.add_argument("--pdb_dir", required=True, help="Directory of .pdb files (repaired or raw).")
    ap.add_argument("--output", required=True, help="Output JSONL path for binding energies.")
    ap.add_argument("--split", help="Optional chain split string (e.g., 'A,BC'). Default: first vs rest.")
    ap.add_argument("--repair", action="store_true", help="Repair structures before energy computation.")
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of files (0 = all).")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if args.max_files:
        pdb_files = pdb_files[: args.max_files]

    if not pdb_files:
        print(f"No .pdb files found in {pdb_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        runner = EvoEF2Runner()
    except FileNotFoundError as exc:
        print(f"[ERR] EvoEF2 not found: {exc}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        for pdb_path in pdb_files:
            split = args.split or guess_split(pdb_path)
            if not split:
                print(f"[SKIP] {pdb_path.name}: cannot determine chain split", file=sys.stderr)
                continue

            target_path = str(pdb_path)
            if args.repair:
                try:
                    target_path = runner.repair_structure(target_path)
                except Exception as exc:
                    print(f"[WARN] repair failed for {pdb_path.name}: {exc}", file=sys.stderr)
                    continue

            try:
                result = runner.compute_binding(target_path, split=split)
            except Exception as exc:
                print(f"[WARN] energy failed for {pdb_path.name}: {exc}", file=sys.stderr)
                continue

            record = {
                "pdb_id": pdb_path.stem,
                "pdb_path": str(target_path),
                "split": split,
                "binding_energy": result.binding_energy,
                "complex_energy": result.complex_energy,
                "receptor_energy": result.receptor_energy,
                "ligand_energy": result.ligand_energy,
            }
            fout.write(json.dumps(record) + "\n")
            print(f"[OK] {pdb_path.name}: ΔΔG={result.binding_energy:.2f}")


if __name__ == "__main__":
    main()
