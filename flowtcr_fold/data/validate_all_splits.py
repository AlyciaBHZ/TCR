"""
Validation script for gene name <-> sequence mapping in ALL data splits.
Checks trn_seq.csv, val_seq.csv, and tst_seq.csv.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
GENE_MAP = BASE_DIR / "data" / "collected data" / "processing" / "imgt_process" / "tcr.csv"
DATA_DIR = BASE_DIR / "flowtcr_fold" / "data"

def validate_split(data_file, gene_to_seq, seq_to_genes):
    """Validate a single data split."""
    data = pd.read_csv(data_file)
    split_name = data_file.stem

    print(f"\n{'='*80}")
    print(f"VALIDATING: {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Total rows: {len(data)}")

    # Forward mapping validation
    print(f"\n--- Forward Mapping (gene name -> sequence) ---")
    forward_errors = 0
    gene_fields = ['h_v', 'h_j', 'l_v', 'l_j']

    for field in gene_fields:
        field_seq = f"{field}_seq"
        valid = 0
        total = 0

        for idx, row in data.iterrows():
            if pd.notna(row[field]) and str(row[field]).strip():
                total += 1
                gene_name = str(row[field]).strip()
                expected_seq = gene_to_seq.get(gene_name, "")
                actual_seq = str(row[field_seq]).strip() if pd.notna(row[field_seq]) else ""

                if expected_seq == actual_seq:
                    valid += 1
                else:
                    forward_errors += 1
                    if forward_errors <= 3:
                        print(f"\nERROR in row {idx}, field {field}:")
                        print(f"  Gene: {gene_name}")
                        print(f"  Expected seq: {expected_seq[:60]}...")
                        print(f"  Actual seq: {actual_seq[:60]}...")

        if total > 0:
            print(f"{field}: {valid}/{total} correct ({100*valid/total:.1f}%)")
        else:
            print(f"{field}: No data")

    # Reverse mapping validation
    print(f"\n--- Reverse Mapping (sequence -> gene name) ---")
    reverse_errors = 0

    for field in gene_fields:
        field_seq = f"{field}_seq"
        correct_empty = 0
        correct_filled = 0
        incorrect = 0
        total_seqs = 0

        for idx, row in data.iterrows():
            if pd.notna(row[field_seq]) and str(row[field_seq]).strip():
                total_seqs += 1
                seq = str(row[field_seq]).strip()
                actual_name = str(row[field]).strip() if pd.notna(row[field]) else ""
                possible_genes = seq_to_genes.get(seq, set())

                if len(possible_genes) == 1:
                    expected_name = list(possible_genes)[0]
                    if actual_name == expected_name:
                        correct_filled += 1
                    else:
                        incorrect += 1
                        reverse_errors += 1
                        if reverse_errors <= 3:
                            print(f"\nERROR in row {idx}, field {field_seq}:")
                            print(f"  Sequence: {seq[:60]}...")
                            print(f"  Expected gene: {expected_name}")
                            print(f"  Actual gene: {actual_name}")
                else:
                    if actual_name == "":
                        correct_empty += 1
                    else:
                        incorrect += 1
                        reverse_errors += 1
                        if reverse_errors <= 3:
                            print(f"\nERROR in row {idx}, field {field_seq}:")
                            print(f"  Sequence: {seq[:60]}...")
                            print(f"  Possible genes: {possible_genes}")
                            print(f"  Actual gene: {actual_name} (should be empty)")

        if total_seqs > 0:
            print(f"\n{field_seq}:")
            print(f"  Correctly filled (unique match): {correct_filled}")
            print(f"  Correctly empty (multi/no match): {correct_empty}")
            print(f"  Incorrect: {incorrect}")
            print(f"  Accuracy: {100*(correct_filled + correct_empty)/total_seqs:.1f}%")

    # Data completeness
    print(f"\n--- Data Completeness ---")
    for field in gene_fields:
        field_seq = f"{field}_seq"
        name_count = data[field].notna().sum()
        seq_count = data[field_seq].notna().sum()
        print(f"{field:4s}: names {name_count:6d} ({100*name_count/len(data):5.1f}%), seqs {seq_count:6d} ({100*seq_count/len(data):5.1f}%)")

    # Check for problematic rows
    print(f"\n--- Problematic Rows (name but no sequence) ---")
    total_problematic = 0
    for field in gene_fields:
        field_seq = f"{field}_seq"
        problematic = data[
            (data[field].notna()) &
            (data[field].astype(str).str.strip() != '') &
            ((data[field_seq].isna()) | (data[field_seq].astype(str).str.strip() == ''))
        ]
        if len(problematic) > 0:
            total_problematic += len(problematic)
            print(f"{field}: {len(problematic)} rows")

    if total_problematic == 0:
        print("None found!")

    return {
        'split': split_name,
        'total_rows': len(data),
        'forward_errors': forward_errors,
        'reverse_errors': reverse_errors,
        'problematic_rows': total_problematic
    }


def main():
    print("="*80)
    print("VALIDATING ALL DATA SPLITS")
    print("="*80)

    # Load gene map
    gene_map = pd.read_csv(GENE_MAP)

    # Build maps
    gene_to_seq = {}
    seq_to_genes = {}
    for _, row in gene_map.iterrows():
        if pd.notna(row.get("Gene")) and pd.notna(row.get("Sequence")):
            gene = str(row["Gene"]).strip()
            seq = str(row["Sequence"]).strip()
            gene_to_seq[gene] = seq
            seq_to_genes.setdefault(seq, set()).add(gene)

    print(f"\nGene map statistics:")
    print(f"  Total gene entries: {len(gene_map)}")
    print(f"  Unique genes: {len(gene_to_seq)}")
    print(f"  Unique sequences: {len(seq_to_genes)}")
    print(f"  Sequences with multiple genes: {sum(1 for genes in seq_to_genes.values() if len(genes) > 1)} ({100*sum(1 for genes in seq_to_genes.values() if len(genes) > 1)/len(seq_to_genes):.1f}%)")

    # Validate all splits
    results = []
    for split_file in ['trn.csv', 'val.csv', 'tst.csv']:
        file_path = DATA_DIR / split_file
        if file_path.exists():
            result = validate_split(file_path, gene_to_seq, seq_to_genes)
            results.append(result)
        else:
            print(f"\n[WARNING] File not found: {split_file}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Split':<10} {'Rows':>8} {'F.Errors':>10} {'R.Errors':>10} {'Problematic':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['split']:<10} {r['total_rows']:>8} {r['forward_errors']:>10} {r['reverse_errors']:>10} {r['problematic_rows']:>12}")

    total_errors = sum(r['forward_errors'] + r['reverse_errors'] for r in results)
    total_problematic = sum(r['problematic_rows'] for r in results)

    if total_errors == 0:
        print("\n[PASS] ALL SPLITS PASSED VALIDATION!")
        print("The gene name <-> sequence mapping is correct across all data splits.")
    else:
        print(f"\n[WARNING] Found {total_errors} total mapping errors")

    if total_problematic > 0:
        print(f"\n[NOTE] {total_problematic} rows have gene names but no sequences")
        print("This is due to sequences not found in IMGT gene map (data quality issue)")

if __name__ == "__main__":
    main()
