"""
Check the 60-row discrepancy between forward and reverse mapping counts for h_v field.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
GENE_MAP = BASE_DIR / "data" / "collected data" / "processing" / "imgt_process" / "tcr.csv"
DATA_FILE = BASE_DIR / "flowtcr_fold" / "data" / "trn_seq.csv"

def main():
    print("Investigating 60-row discrepancy in h_v field...")
    print("=" * 80)

    # Load data
    gene_map = pd.read_csv(GENE_MAP)
    data = pd.read_csv(DATA_FILE)

    # Build maps
    gene_to_seq = {}
    seq_to_genes = {}
    for _, row in gene_map.iterrows():
        if pd.notna(row.get("Gene")) and pd.notna(row.get("Sequence")):
            gene = str(row["Gene"]).strip()
            seq = str(row["Sequence"]).strip()
            gene_to_seq[gene] = seq
            seq_to_genes.setdefault(seq, set()).add(gene)

    # Find rows counted in forward mapping (has h_v gene name)
    forward_rows = data[data['h_v'].notna() & (data['h_v'].astype(str).str.strip() != '')]
    print(f"\nForward mapping rows (has h_v gene name): {len(forward_rows)}")

    # Find rows counted as "correctly filled" in reverse mapping
    # (has h_v_seq AND seq maps to unique gene AND h_v matches that gene)
    reverse_filled = []
    for idx, row in data.iterrows():
        if pd.notna(row['h_v_seq']) and str(row['h_v_seq']).strip():
            seq = str(row['h_v_seq']).strip()
            possible_genes = seq_to_genes.get(seq, set())

            if len(possible_genes) == 1:
                expected_gene = list(possible_genes)[0]
                actual_gene = str(row['h_v']).strip() if pd.notna(row['h_v']) else ""
                if actual_gene == expected_gene:
                    reverse_filled.append(idx)

    print(f"Reverse mapping 'correctly filled' rows: {len(reverse_filled)}")
    print(f"\nDiscrepancy: {len(forward_rows)} - {len(reverse_filled)} = {len(forward_rows) - len(reverse_filled)}")

    # Find the rows in forward but not in reverse "correctly filled"
    forward_indices = set(forward_rows.index)
    reverse_indices = set(reverse_filled)
    discrepancy_indices = forward_indices - reverse_indices

    print(f"\nAnalyzing {len(discrepancy_indices)} rows with discrepancy...")
    print("=" * 80)

    if len(discrepancy_indices) == 0:
        print("\nNo discrepancy found! Counts match.")
        return

    # Categorize discrepancy rows
    categories = {
        'no_seq': 0,           # Has gene name but no sequence
        'seq_not_in_map': 0,   # Sequence not in IMGT gene map
        'multi_match': 0,      # Sequence matches multiple genes
        'name_mismatch': 0,    # Gene name doesn't match expected
        'other': 0
    }

    examples = {key: [] for key in categories}

    for idx in list(discrepancy_indices)[:100]:  # Check first 100
        row = data.loc[idx]
        gene_name = str(row['h_v']).strip()
        seq = str(row['h_v_seq']).strip() if pd.notna(row['h_v_seq']) else ""

        if not seq:
            categories['no_seq'] += 1
            if len(examples['no_seq']) < 3:
                examples['no_seq'].append({
                    'idx': idx,
                    'gene': gene_name,
                    'seq': 'EMPTY'
                })
        else:
            possible_genes = seq_to_genes.get(seq, set())

            if len(possible_genes) == 0:
                categories['seq_not_in_map'] += 1
                if len(examples['seq_not_in_map']) < 3:
                    examples['seq_not_in_map'].append({
                        'idx': idx,
                        'gene': gene_name,
                        'seq': seq[:50] + '...'
                    })
            elif len(possible_genes) > 1:
                categories['multi_match'] += 1
                if len(examples['multi_match']) < 3:
                    examples['multi_match'].append({
                        'idx': idx,
                        'gene': gene_name,
                        'possible': list(possible_genes),
                        'seq': seq[:50] + '...'
                    })
            else:
                expected_gene = list(possible_genes)[0]
                if gene_name != expected_gene:
                    categories['name_mismatch'] += 1
                    if len(examples['name_mismatch']) < 3:
                        examples['name_mismatch'].append({
                            'idx': idx,
                            'actual': gene_name,
                            'expected': expected_gene,
                            'seq': seq[:50] + '...'
                        })
                else:
                    categories['other'] += 1

    # Print categories
    print("\nCategory breakdown:")
    for cat, count in categories.items():
        if count > 0:
            print(f"  {cat}: {count}")

    # Print examples
    for cat, count in categories.items():
        if count > 0 and examples[cat]:
            print(f"\n--- Examples of '{cat}' ---")
            for ex in examples[cat]:
                print(f"  Row {ex.get('idx', '?')}:")
                for key, val in ex.items():
                    if key != 'idx':
                        print(f"    {key}: {val}")

    # Verify if this is expected behavior
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    if categories['no_seq'] == len(discrepancy_indices):
        print("\nAll discrepancy rows have gene names but NO sequences.")
        print("This is why they pass forward mapping (expected_seq = actual_seq = empty)")
        print("but are not counted in reverse mapping's 'correctly filled'.")
        print("\n[EXPECTED BEHAVIOR] This is normal and correct.")
    elif categories['multi_match'] > 0:
        print(f"\n{categories['multi_match']} rows have sequences that map to multiple genes.")
        print("They were assigned one of the valid gene names (forward mapping passes),")
        print("but reverse mapping expects them to be empty (to avoid ambiguity).")
        print("\n[POTENTIAL ISSUE] Consider whether these should have gene names.")
    else:
        print("\nOther patterns detected. Review examples above.")

if __name__ == "__main__":
    main()
