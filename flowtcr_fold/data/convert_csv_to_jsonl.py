"""
Utility to convert paired CSV data to JSONL for FlowTCR.

Pass-through CSV→JSONL converter for FlowTCR data.
Keeps all columns present in the CSV; no filtering or length checks.

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
    return ap.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with inp.open(newline="", encoding="utf-8") as f_in, outp.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            # Preserve all columns as-is
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} records to {outp}")


if __name__ == "__main__":
    main()
