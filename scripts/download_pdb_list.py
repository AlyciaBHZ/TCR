"""
Download PDB/mmCIF files from an ID list.

Usage:
    python scripts/download_pdb_list.py \
        --id_file flowtcr_fold/data/pdb/batch1.txt \
        --id_file flowtcr_fold/data/pdb/batch2.txt \
        --out_dir flowtcr_fold/data/pdb_structures/raw

The ID files can be comma-separated or newline-separated. Existing files
are skipped to allow safe retries.
"""

import argparse
import concurrent.futures as futures
import re
import sys
from pathlib import Path
from typing import Iterable, List, Set
from urllib import error, request


def load_ids(paths: Iterable[str]) -> List[str]:
    """Load PDB IDs from one or more files (comma or newline separated)."""
    ids: Set[str] = set()
    for path in paths:
        text = Path(path).read_text()
        for token in re.split(r"[,\s]+", text):
            token = token.strip()
            if not token:
                continue
            ids.add(token.upper())
    return sorted(ids)


def download_one(pdb_id: str, out_dir: Path, fmt: str, timeout: int = 30) -> str:
    """Download a single structure; returns status string."""
    suffix = ".cif" if fmt == "cif" else ".pdb"
    url = f"https://files.rcsb.org/download/{pdb_id}{suffix}"
    out_path = out_dir / f"{pdb_id}{suffix}"

    if out_path.exists():
        return f"[SKIP] {pdb_id} already exists"

    try:
        with request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return f"[WARN] {pdb_id} HTTP {resp.status}"
            out_path.write_bytes(resp.read())
        return f"[OK]   {pdb_id}"
    except error.HTTPError as exc:
        return f"[WARN] {pdb_id} HTTPError {exc.code}"
    except Exception as exc:  # pragma: no cover - network errors
        return f"[ERR]  {pdb_id} {exc}"


def main():
    ap = argparse.ArgumentParser(description="Download PDB/mmCIF files from ID list.")
    ap.add_argument("--id_file", action="append", required=True,
                    help="Path to ID list (comma or newline separated). Can be provided multiple times.")
    ap.add_argument("--out_dir", required=True, help="Output directory for downloaded structures.")
    ap.add_argument("--format", choices=["pdb", "cif"], default="pdb", help="Download format (default: pdb).")
    ap.add_argument("--num_workers", type=int, default=8, help="Parallel download workers (default: 8).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = load_ids(args.id_file)
    if not pdb_ids:
        print("No IDs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {len(pdb_ids)} IDs to {out_dir} (format={args.format})")

    with futures.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        for msg in ex.map(lambda pid: download_one(pid, out_dir, args.format), pdb_ids):
            print(msg)


if __name__ == "__main__":
    main()
