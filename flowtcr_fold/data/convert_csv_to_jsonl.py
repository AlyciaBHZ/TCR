"""
Utility to convert paired CSV data to JSONL for FlowTCR.

Expected CSV columns (minimal):
  peptide,mhc,cdr3_b,h_v,h_j
Optional columns:
  l_v,l_j,cdr3_a

Usage:
  python flowtcr_fold/data/convert_csv_to_jsonl.py --input data/trn.csv --output data/trn.jsonl
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to output JSONL")
    ap.add_argument("--min_len", type=int, default=3, help="Minimum length for sequences")
    return ap.parse_args()


def clean_row(row, min_len):
    pep = row.get("peptide", "").strip()
    mhc = row.get("mhc", "").strip()
    cdr3b = row.get("cdr3_b", "").strip() or row.get("cdr3", "").strip()
    if len(pep) < min_len or len(mhc) < min_len or len(cdr3b) < min_len:
        return None
    return {
        "peptide": pep,
        "mhc": mhc,
        "cdr3b": cdr3b,
        "h_v": row.get("h_v") or row.get("v_gene"),
        "h_j": row.get("h_j") or row.get("j_gene"),
        "l_v": row.get("l_v"),
        "l_j": row.get("l_j"),
        "cdr3a": row.get("cdr3_a") or row.get("cdr3_alpha"),
    }


def main():
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with inp.open() as f_in, outp.open("w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            cleaned = clean_row(row, args.min_len)
            if cleaned is None:
                continue
            f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
    print(f"wrote {kept} records to {outp}")


if __name__ == "__main__":
    main()
