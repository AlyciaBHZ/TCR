"""
Explain why some sequences cannot be reverse-mapped to gene names.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
GENE_MAP = BASE_DIR / "data" / "collected data" / "processing" / "imgt_process" / "tcr.csv"

def main():
    print("="*80)
    print("WHY SOME SEQUENCES CANNOT BE REVERSE-MAPPED TO GENE NAMES")
    print("="*80)

    gene_map = pd.read_csv(GENE_MAP)

    # Build sequence to gene mapping
    seq_to_genes = {}
    for _, row in gene_map.iterrows():
        if pd.notna(row.get("Gene")) and pd.notna(row.get("Sequence")):
            gene = str(row["Gene"]).strip()
            seq = str(row["Sequence"]).strip()
            seq_to_genes.setdefault(seq, []).append(gene)

    # Statistics
    total_seqs = len(seq_to_genes)
    unique_seqs = sum(1 for genes in seq_to_genes.values() if len(genes) == 1)
    multi_seqs = sum(1 for genes in seq_to_genes.values() if len(genes) > 1)

    print(f"\nGene Map Statistics:")
    print(f"  Total sequences: {total_seqs}")
    print(f"  Unique mapping (1 seq -> 1 gene): {unique_seqs} ({100*unique_seqs/total_seqs:.1f}%)")
    print(f"  Multi-mapping (1 seq -> multiple genes): {multi_seqs} ({100*multi_seqs/total_seqs:.1f}%)")

    print(f"\n{'='*80}")
    print("MULTI-MAPPING EXAMPLES (Same sequence -> Multiple gene names):")
    print("="*80)

    # Show multi-mapping examples
    multi_examples = [(seq, genes) for seq, genes in seq_to_genes.items() if len(genes) > 1]

    # Sort by number of genes
    multi_examples.sort(key=lambda x: len(x[1]), reverse=True)

    print("\nMost extreme cases (one sequence -> most genes):")
    for i, (seq, genes) in enumerate(multi_examples[:5]):
        print(f"\nExample {i+1}: {len(genes)} genes map to the SAME sequence")
        print(f"  Sequence: {seq[:60]}...")
        print(f"  Genes: {', '.join(sorted(genes)[:10])}")
        if len(genes) > 10:
            print(f"         (and {len(genes)-10} more genes...)")

    # Analyze TRBV genes
    print(f"\n{'='*80}")
    print("TRBV (beta-chain V region) Multi-mapping Analysis:")
    print("="*80)

    trbv_multi = [(seq, genes) for seq, genes in seq_to_genes.items()
                  if len(genes) > 1 and any(g.startswith('TRBV') for g in genes)]

    print(f"\nTRBV sequences with multi-mapping: {len(trbv_multi)}")

    if trbv_multi:
        print("\nExamples:")
        for i, (seq, genes) in enumerate(trbv_multi[:3]):
            trbv_genes = [g for g in genes if g.startswith('TRBV')]
            print(f"\n{i+1}. Sequence (length {len(seq)}): {seq[:50]}...")
            print(f"   TRBV genes: {', '.join(sorted(trbv_genes))}")

    # Analyze TRAV genes
    print(f"\n{'='*80}")
    print("TRAV (alpha-chain V region) Multi-mapping Analysis:")
    print("="*80)

    trav_multi = [(seq, genes) for seq, genes in seq_to_genes.items()
                  if len(genes) > 1 and any(g.startswith('TRAV') for g in genes)]

    print(f"\nTRAV sequences with multi-mapping: {len(trav_multi)}")

    if trav_multi:
        print("\nExamples:")
        for i, (seq, genes) in enumerate(trav_multi[:3]):
            trav_genes = [g for g in genes if g.startswith('TRAV')]
            print(f"\n{i+1}. Sequence (length {len(seq)}): {seq[:50]}...")
            print(f"   TRAV genes: {', '.join(sorted(trav_genes))}")

    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print("="*80)
    print("\n1. THIS IS NOT AN ERROR: Multiple genes sharing the same sequence is biological reality")
    print("   - Allelic variants (e.g., TRBV2*01, TRBV2*02, TRBV2*03)")
    print("   - Conserved regions in gene families")
    print("   - Fine-grained IMGT nomenclature")
    print("\n2. WHY LEAVE EMPTY: When a sequence maps to multiple genes, we cannot determine which one")
    print("   - Randomly choosing one would introduce errors")
    print("   - Leaving empty is the safer choice")
    print("\n3. DATA IS STILL USABLE: Even if gene name is empty, sequence information exists")
    print("   - Models can use sequences for training")
    print("   - Gene names are just additional annotations")

if __name__ == "__main__":
    main()
