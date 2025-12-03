"""
Batch EvoEF2 energy computation for PDB complexes.

Usage (Stage3 Phase0):
    python flowtcr_fold/TCRFold_Light/process_pdb/compute_evoef2_batch.py \
        --pdb_dir flowtcr_fold/data/pdb_structures/raw \
        --output flowtcr_fold/data/energy_cache.jsonl

    # 追加模式（跳过已计算的条目）
    python ... --append
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Set

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


def load_existing_ids(jsonl_path: Path) -> Set[str]:
    """Load already computed PDB IDs from existing JSONL file."""
    existing = set()
    if not jsonl_path.exists():
        return existing
    
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                pdb_id = entry.get("pdb_id", "")
                if pdb_id:
                    existing.add(pdb_id)
            except json.JSONDecodeError:
                continue
    
    return existing


def main():
    ap = argparse.ArgumentParser(description="Compute EvoEF2 binding energies for a directory of PDBs.")
    ap.add_argument("--pdb_dir", required=True, help="Directory of .pdb files (repaired or raw).")
    ap.add_argument("--output", required=True, help="Output JSONL path for binding energies.")
    ap.add_argument("--split", help="Optional chain split string (e.g., 'A,BC'). Default: first vs rest.")
    ap.add_argument("--repair", action="store_true", help="Repair structures before energy computation.")
    ap.add_argument("--max_files", type=int, default=0, help="Limit number of files (0 = all).")
    ap.add_argument("--append", action="store_true", help="Append mode: skip already computed entries.")
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

    # 追加模式：加载已有条目
    existing_ids = set()
    if args.append:
        existing_ids = load_existing_ids(out_path)
        print(f"[INFO] Append mode: found {len(existing_ids)} existing entries, will skip")

    # 打开文件模式
    file_mode = "a" if args.append else "w"
    
    ok_count = 0
    skip_count = 0
    warn_count = 0
    
    with out_path.open(file_mode) as fout:
        for i, pdb_path in enumerate(pdb_files, 1):
            pdb_id = pdb_path.stem
            
            # 跳过已计算的
            if pdb_id in existing_ids:
                skip_count += 1
                continue
            
            split = args.split or guess_split(pdb_path)
            if not split:
                print(f"[SKIP] {pdb_path.name}: cannot determine chain split", file=sys.stderr)
                skip_count += 1
                continue

            target_path = str(pdb_path)
            if args.repair:
                try:
                    target_path = runner.repair_structure(target_path)
                except Exception as exc:
                    print(f"[WARN] repair failed for {pdb_path.name}: {exc}", file=sys.stderr)
                    warn_count += 1
                    continue

            try:
                result = runner.compute_binding(target_path, split=split)
            except Exception as exc:
                print(f"[WARN] energy failed for {pdb_path.name}: {exc}", file=sys.stderr)
                warn_count += 1
                continue

            # 解析 split 为 chain_a / chain_b (兼容 PPIDataset)
            split_parts = split.split(",")
            chain_a = split_parts[0] if split_parts else "A"
            chain_b = split_parts[1] if len(split_parts) > 1 else "B"
            
            record = {
                "pdb_id": pdb_id,
                "pdb_path": str(target_path),
                "split": split,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "binding_energy": result.binding_energy,
                "complex_energy": result.complex_energy,
                "receptor_energy": result.receptor_energy,
                "ligand_energy": result.ligand_energy,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()  # 及时刷新，避免丢失
            
            ok_count += 1
            print(f"[OK] {pdb_path.name}: ΔΔG={result.binding_energy:.2f} (chains {chain_a},{chain_b})")
            
            # 每 100 个打印进度
            if ok_count % 100 == 0:
                print(f"[INFO] Progress: {i}/{len(pdb_files)} files, "
                      f"{ok_count} computed, {skip_count} skipped, {warn_count} warnings")
    
    print(f"\n[DONE] Computed: {ok_count}, Skipped: {skip_count}, Warnings: {warn_count}")


if __name__ == "__main__":
    main()
