"""
Test script for negative sampling functionality in FlowDataset.
"""
import random
from flowtcr_fold.data.dataset import FlowDataset

def main():
    # Load dataset with 100% negative sampling rate for testing
    ds = FlowDataset('flowtcr_fold/data/trn.jsonl', split='train', negative_fraction=1.0)
    print(f"Loaded {len(ds)} samples")
    print(f"MHC groups: {len(ds.by_mhc)}")
    print()
    
    # Test negative sampling on random samples
    neg_type_counts = {}
    sample_indices = random.sample(range(len(ds)), min(100, len(ds)))
    
    for idx, i in enumerate(sample_indices):
        item = ds[i]
        neg_type = item['meta']['neg_type']
        neg_type_counts[neg_type] = neg_type_counts.get(neg_type, 0) + 1
        
        # Print first few examples of each type
        if idx < 5 and neg_type:
            print(f"Sample {i}: neg_type = {neg_type}")
            print(f"  Anchor peptide: {item['meta']['peptide']}")
            print(f"  Anchor MHC: {item['meta']['mhc'][:50]}...")
            print(f"  Anchor CDR3b: {item['meta']['cdr3b']}")
            if item['tokens_neg'] is not None:
                print(f"  Neg tokens shape: {item['tokens_neg'].shape}")
            print()
    
    print("=" * 50)
    print("Negative type distribution (100 samples):")
    for k, v in sorted(neg_type_counts.items(), key=lambda x: -x[1]):
        pct = v / sum(neg_type_counts.values()) * 100
        print(f"  {k or 'None'}: {v} ({pct:.1f}%)")
    
    # Also test the identity filtering
    print("\n" + "=" * 50)
    print("Testing sequence identity computation:")
    test_pairs = [
        ("CASSF", "CASSF"),  # identical
        ("CASSF", "CASST"),  # 80% identity
        ("ABCDEFG", "ABCDXYZ"),  # ~57% identity
    ]
    for a, b in test_pairs:
        identity = ds._seq_identity(a, b)
        print(f"  '{a}' vs '{b}': {identity:.2%} identity")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()

