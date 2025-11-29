"""
Validation script for gene name <-> sequence mapping in flowtcr_fold dataset.
Checks that the reverse mapping was done correctly.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
GENE_MAP = BASE_DIR / "data" / "collected data" / "processing" / "imgt_process" / "tcr.csv"
DATA_FILE = BASE_DIR / "flowtcr_fold" / "data" / "trn.csv"

def main():
    print("=" * 80)
    print("VALIDATING GENE NAME <-> SEQUENCE MAPPING")
    print("=" * 80)

    # Load data
    gene_map = pd.read_csv(GENE_MAP)
    data = pd.read_csv(DATA_FILE)

    # Build forward and reverse maps
    gene_to_seq = {}
    seq_to_genes = {}
    for _, row in gene_map.iterrows():
        if pd.notna(row['Gene']) and pd.notna(row['Sequence']):
            gene = str(row['Gene']).strip()
            seq = str(row['Sequence']).strip()
            gene_to_seq[gene] = seq
            seq_to_genes.setdefault(seq, set()).add(gene)

    print(f"\nGene map statistics:")
    print(f"  Total gene entries: {len(gene_map)}")
    print(f"  Unique genes: {len(gene_to_seq)}")
    print(f"  Unique sequences: {len(seq_to_genes)}")
    print(f"  Sequences with multiple genes: {sum(1 for genes in seq_to_genes.values() if len(genes) > 1)} ({100*sum(1 for genes in seq_to_genes.values() if len(genes) > 1)/len(seq_to_genes):.1f}%)")

    # Validate forward mapping (gene name -> sequence)
    print(f"\n{'='*80}")
    print("FORWARD MAPPING VALIDATION (gene name -> sequence)")
    print("=" * 80)

    errors = 0
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
                    errors += 1
                    if errors <= 3:  # Show first 3 errors
                        print(f"\nERROR in row {idx}, field {field}:")
                        print(f"  Gene: {gene_name}")
                        print(f"  Expected seq: {expected_seq[:60]}...")
                        print(f"  Actual seq: {actual_seq[:60]}...")

        print(f"\n{field}: {valid}/{total} correct ({100*valid/total:.1f}%)" if total > 0 else f"\n{field}: No data")

    # Validate reverse mapping (sequence -> gene name)
    print(f"\n{'='*80}")
    print("REVERSE MAPPING VALIDATION (sequence -> gene name)")
    print("=" * 80)

    for field in gene_fields:
        field_seq = f"{field}_seq"
        correct_empty = 0  # Correctly empty (multi-match or no match)
        correct_filled = 0  # Correctly filled (unique match)
        incorrect = 0
        total_seqs = 0

        for idx, row in data.iterrows():
            if pd.notna(row[field_seq]) and str(row[field_seq]).strip():
                total_seqs += 1
                seq = str(row[field_seq]).strip()
                actual_name = str(row[field]).strip() if pd.notna(row[field]) else ""
                possible_genes = seq_to_genes.get(seq, set())

                if len(possible_genes) == 1:
                    # Unique match - should be filled
                    expected_name = list(possible_genes)[0]
                    if actual_name == expected_name:
                        correct_filled += 1
                    else:
                        incorrect += 1
                        if incorrect <= 3:
                            print(f"\nERROR in row {idx}, field {field_seq}:")
                            print(f"  Sequence: {seq[:60]}...")
                            print(f"  Expected gene: {expected_name}")
                            print(f"  Actual gene: {actual_name}")
                else:
                    # Multi-match or no match - should be empty
                    if actual_name == "":
                        correct_empty += 1
                    else:
                        incorrect += 1
                        if incorrect <= 3:
                            print(f"\nERROR in row {idx}, field {field_seq}:")
                            print(f"  Sequence: {seq[:60]}...")
                            print(f"  Possible genes: {possible_genes}")
                            print(f"  Actual gene: {actual_name} (should be empty)")

        if total_seqs > 0:
            print(f"\n{field_seq}:")
            print(f"  Total sequences: {total_seqs}")
            print(f"  Correctly filled (unique match): {correct_filled}")
            print(f"  Correctly empty (multi/no match): {correct_empty}")
            print(f"  Incorrect: {incorrect}")
            print(f"  Accuracy: {100*(correct_filled + correct_empty)/total_seqs:.1f}%")
        else:
            print(f"\n{field_seq}: No data")

    # Data completeness
    print(f"\n{'='*80}")
    print("DATA COMPLETENESS")
    print("=" * 80)

    total = len(data)
    for field in gene_fields:
        field_seq = f"{field}_seq"
        name_count = data[field].notna().sum()
        seq_count = data[field_seq].notna().sum()
        print(f"\n{field}:")
        print(f"  Gene names: {name_count:6d} ({100*name_count/total:5.1f}%)")
        print(f"  Sequences:  {seq_count:6d} ({100*seq_count/total:5.1f}%)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    if errors == 0 and incorrect == 0:
        print("\n[PASS] ALL VALIDATIONS PASSED!")
        print("The gene name <-> sequence mapping is correct.")
    else:
        print(f"\n[WARNING] Found {errors} forward mapping errors and {incorrect} reverse mapping errors")
        print("Please review the errors above.")

    print("\nNotes:")
    print("- Some sequences have no gene name because they map to multiple genes")
    print("- This is expected behavior to avoid ambiguity")
    print("- Gene sequences with unique matches are correctly mapped to gene names")

if __name__ == "__main__":
    main()
