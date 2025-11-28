"""
Reverse-map gene sequences to gene names using IMGT gene map.

Inputs:
  --input:  CSV with columns peptide, mhc, h_v, h_j, l_v, l_j, cdr3_b (gene fields may be sequences)
  --gene_map: CSV with columns Gene, Sequence (imgt_process/tcr.csv)
  --output: CSV to write (adds *_name columns)
  --jsonl: optional JSONL output

Rules:
  - If field already looks like a gene name (contains '*' or length <= 20), keep as name.
  - If field looks like an AA sequence (all in ACDEFGHIKLMNPQRSTVWY and length > 8),
    try exact match against gene_map Sequence -> Gene. If unique, fill *_name; otherwise leave blank.
  - Sequences are preserved in original columns; new columns *_name hold mapped names.
"""

import argparse
import pandas as pd
from pathlib import Path

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def looks_like_seq(val: str, min_len: int = 8) -> bool:
    return len(val) >= min_len and all(c in AA_SET for c in val)


def looks_like_name(val: str) -> bool:
    return "*" in val or len(val) <= 20


def build_seq_to_gene(gene_map_path: str) -> dict:
    df = pd.read_csv(gene_map_path)
    seq2gene = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("Gene")) or pd.isna(row.get("Sequence")):
            continue
        gene = str(row["Gene"]).strip()
        seq = str(row["Sequence"]).strip()
        if not gene or not seq:
            continue
        seq2gene.setdefault(seq, set()).add(gene)
    return seq2gene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gene_map", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    seq2gene = build_seq_to_gene(args.gene_map)
    df = pd.read_csv(args.input)

    def map_field(val: str) -> str:
        if pd.isna(val):
            return ""
        sval = str(val).strip()
        if not sval:
            return ""
        if looks_like_name(sval):
            return sval
        if looks_like_seq(sval):
            genes = seq2gene.get(sval, set())
            return list(genes)[0] if len(genes) == 1 else ""
        return ""

    for key in ["h_v", "h_j", "l_v", "l_j"]:
        df[f"{key}_name"] = df.get(key, "").apply(map_field)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    if args.jsonl:
        df.to_json(args.jsonl, orient="records", lines=True)
        print(f"Wrote JSONL to {args.jsonl}")


if __name__ == "__main__":
    main()
