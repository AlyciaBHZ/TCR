"""
Download PDB/mmCIF files from an ID list.

Usage (Stage 3 Phase0):
    python flowtcr_fold/TCRFold_Light/process_pdb/download_from_id_list.py \
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
    """Download a single structure; returns status string.
    
    If PDB format fails with 404, automatically tries CIF format.
    """
    # 检查是否已存在 (任意格式)
    for ext in [".pdb", ".cif"]:
        if (out_dir / f"{pdb_id}{ext}").exists():
            return f"[SKIP] {pdb_id} already exists"
    
    # 尝试下载的格式顺序
    formats_to_try = [fmt]
    if fmt == "pdb":
        formats_to_try.append("cif")  # PDB 失败时尝试 CIF
    
    for try_fmt in formats_to_try:
        suffix = ".cif" if try_fmt == "cif" else ".pdb"
        url = f"https://files.rcsb.org/download/{pdb_id}{suffix}"
        out_path = out_dir / f"{pdb_id}{suffix}"
        
        try:
            with request.urlopen(url, timeout=timeout) as resp:
                if resp.status == 200:
                    out_path.write_bytes(resp.read())
                    if try_fmt != fmt:
                        return f"[OK]   {pdb_id} (fallback to {try_fmt})"
                    return f"[OK]   {pdb_id}"
        except error.HTTPError as exc:
            if exc.code == 404 and try_fmt == "pdb" and "cif" in formats_to_try:
                continue  # 尝试 CIF 格式
            return f"[WARN] {pdb_id} HTTPError {exc.code}"
        except Exception as exc:  # pragma: no cover - network errors
            return f"[ERR]  {pdb_id} {exc}"
    
    return f"[WARN] {pdb_id} not available in any format"


def get_missing_ids(pdb_ids: List[str], out_dir: Path, fmt: str) -> List[str]:
    """Filter to only IDs that haven't been downloaded yet (checks both .pdb and .cif)."""
    missing = []
    for pdb_id in pdb_ids:
        # 检查两种格式是否都不存在
        has_pdb = (out_dir / f"{pdb_id}.pdb").exists()
        has_cif = (out_dir / f"{pdb_id}.cif").exists()
        if not has_pdb and not has_cif:
            missing.append(pdb_id)
    return missing


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

    # 先筛选出缺失的 ID
    missing_ids = get_missing_ids(pdb_ids, out_dir, args.format)
    
    print(f"总 ID 数: {len(pdb_ids)}")
    print(f"已下载: {len(pdb_ids) - len(missing_ids)}")
    print(f"待下载: {len(missing_ids)}")
    
    if not missing_ids:
        print("✅ 所有文件已下载完成！")
        return
    
    print(f"\n开始下载 {len(missing_ids)} 个文件到 {out_dir} (format={args.format})...")

    ok_count = 0
    err_count = 0
    
    with futures.ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        for i, msg in enumerate(ex.map(lambda pid: download_one(pid, out_dir, args.format), missing_ids), 1):
            if "[OK]" in msg:
                ok_count += 1
            elif "[ERR]" in msg or "[WARN]" in msg:
                err_count += 1
                print(msg)  # 只打印错误
            
            # 每 100 个打印进度
            if i % 100 == 0:
                print(f"进度: {i}/{len(missing_ids)} (成功: {ok_count}, 失败: {err_count})")
    
    print(f"\n✅ 下载完成！成功: {ok_count}, 失败: {err_count}")


if __name__ == "__main__":
    main()
