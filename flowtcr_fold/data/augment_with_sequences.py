"""
Augment TCR-pMHC CSV with gene and MHC sequences.

Inputs:
- --input: CSV with columns peptide, mhc, h_v, h_j, l_v, l_j, cdr3_b (some may already be sequences)
- --gene_map: CSV with columns Gene, Sequence (from processing/imgt_process/tcr.csv)
- --mhc_map: CSV with columns Name, Sequence (from processing/mhc_process/mhc.csv)
- --output: target CSV path
- --jsonl: optional JSONL output path

Rules:
- Gene fields may already be sequences or names:
  * If value looks like AA sequence, map sequence -> gene name (unique match). If multiple/none, name left empty; sequence kept in *_seq.
  * If value looks like a name, sequence is filled from gene_map if available.
  * Final CSV stores gene names in h_v/h_j/l_v/l_j, sequences in *_seq.
- MHC: if looks like sequence (long & AA-like), keep as-is and duplicate to mhc_seq; otherwise try lookup in mhc_map.Name; if not found, keep original string.
"""

import argparse
import pandas as pd
from pathlib import Path


def load_map(path: str, key: str = "Gene") -> dict:
    df = pd.read_csv(path)
    return {str(row[key]).strip(): str(row["Sequence"]).strip() for _, row in df.iterrows() if pd.notna(row[key])}

def is_aa_sequence(val: str, min_len: int = 20) -> bool:
    aas = set("ACDEFGHIKLMNPQRSTVWY")
    return len(val) >= min_len and all(c in aas for c in val)


def map_mhc(name: str, mhc_dict: dict) -> str:
    if not name:
        return ""
    name_strip = name.strip()
    # treat as sequence if already long and AA-like
    if len(name_strip) > 50 and is_aa_sequence(name_strip, min_len=30):
        return name_strip
    key = name_strip.replace("HLA-", "").upper()
    # direct
    if key in mhc_dict:
        return mhc_dict[key]
    # try add/remove :01
    parts = key.split(":")
    if len(parts) == 1:
        trial = key + ":01"
        return mhc_dict.get(trial, "")
    if len(parts) >= 2:
        trial = parts[0] + ":" + parts[1] + ":01"
        if trial in mhc_dict:
            return mhc_dict[trial]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--gene_map", required=True)
    ap.add_argument("--mhc_map", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--jsonl", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    gene_dict = load_map(args.gene_map, key="Gene")
    mhc_dict = load_map(args.mhc_map, key="Name")

    # reverse map seq -> gene (may be multi-mapped)
    seq2gene = {}
    for g, seq in gene_dict.items():
        seq2gene.setdefault(seq, set()).add(g)

    def map_gene(val):
        """
        Return (name, seq)
        """
        if pd.isna(val) or not str(val).strip():
            return "", ""
        sval = str(val).strip()
        if is_aa_sequence(sval, min_len=8):
            genes = seq2gene.get(sval, set())
            name = list(genes)[0] if len(genes) == 1 else ""
            return name, sval
        else:
            seq = gene_dict.get(sval, "")
            return sval, seq

    # map all gene fields
    for key in ["h_v", "h_j", "l_v", "l_j"]:
        names = []
        seqs = []
        for v in df.get(key, ""):
            name, seq = map_gene(v)
            names.append(name)
            seqs.append(seq)
        df[key] = names
        df[f"{key}_seq"] = seqs

    df["mhc_seq"] = df.get("mhc", "").apply(lambda x: map_mhc(str(x), mhc_dict))
    df["mhc_seq"] = df.get("mhc", "").apply(lambda x: map_mhc(str(x), mhc_dict))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")

    if args.jsonl:
        df.to_json(args.jsonl, orient="records", lines=True)
        print(f"Wrote JSONL to {args.jsonl}")


if __name__ == "__main__":
    main()
