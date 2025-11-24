#!/usr/bin/env python3
"""
Summarize Language Model Evaluation Results
Creates a clean summary table of average NLL and accuracy by model and masking ratio.
"""

import pandas as pd
import numpy as np
import os

def summarize_lm_results():
    """Create summary tables from language model evaluation results"""
    
    # Load the results - adjust path to current environment
    results_file = "./lm_evaluation_results.csv"
    if not os.path.exists(results_file):
        # Try alternative paths
        alt_paths = [
            "./nll/lm_evaluation_results.csv",
            "../lm_evaluation_results.csv",
            "lm_evaluation_results_proper/lm_evaluation_results.csv"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                results_file = alt_path
                break
        else:
            print(f"❌ Could not find results file. Tried:")
            print(f"   - ./lm_evaluation_results.csv")
            for path in alt_paths:
                print(f"   - {path}")
            return None
    
    print(f"Loading results from {results_file}...")
    df = pd.read_csv(results_file)
    
    print(f"Loaded {len(df)} evaluation records")
    print(f"Models: {df['model'].unique()}")
    print(f"Mask ratios: {sorted(df['mask_ratio'].unique())}")
    
    # Calculate summary statistics by model and mask ratio
    summary = df.groupby(['model', 'mask_ratio']).agg({
        'recovery_accuracy': ['mean', 'std', 'count'],
        'nll': ['mean', 'std'],
        'perplexity': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Create a pivot table for easier reading
    print("\n" + "="*80)
    print("RECOVERY ACCURACY BY MODEL AND MASK RATIO")
    print("="*80)
    
    acc_pivot = df.groupby(['model', 'mask_ratio'])['recovery_accuracy'].mean().unstack()
    acc_pivot = acc_pivot.round(4)
    print(acc_pivot.to_string())
    
    print("\n" + "="*80)
    print("AVERAGE NEGATIVE LOG-LIKELIHOOD BY MODEL AND MASK RATIO")
    print("="*80)
    
    nll_pivot = df.groupby(['model', 'mask_ratio'])['nll'].mean().unstack()
    nll_pivot = nll_pivot.round(4)
    print(nll_pivot.to_string())
    
    print("\n" + "="*80)
    print("AVERAGE PERPLEXITY BY MODEL AND MASK RATIO")
    print("="*80)
    
    perp_pivot = df.groupby(['model', 'mask_ratio'])['perplexity'].mean().unstack()
    perp_pivot = perp_pivot.round(4)
    print(perp_pivot.to_string())
    
    # Create detailed summary table
    print("\n" + "="*120)
    print("DETAILED SUMMARY TABLE")
    print("="*120)
    
    detailed_summary = []
    for model in sorted(df['model'].unique()):
        for mask_ratio in sorted(df['mask_ratio'].unique()):
            subset = df[(df['model'] == model) & (df['mask_ratio'] == mask_ratio)]
            if len(subset) > 0:
                detailed_summary.append({
                    'Model': model,
                    'Mask_Ratio': mask_ratio,
                    'Avg_Recovery_Acc': subset['recovery_accuracy'].mean(),
                    'Std_Recovery_Acc': subset['recovery_accuracy'].std(),
                    'Avg_NLL': subset['nll'].mean(),
                    'Std_NLL': subset['nll'].std(),
                    'Avg_Perplexity': subset['perplexity'].mean(),
                    'Std_Perplexity': subset['perplexity'].std(),
                    'Num_Samples': len(subset)
                })
    
    detailed_df = pd.DataFrame(detailed_summary)
    detailed_df = detailed_df.round(4)
    print(detailed_df.to_string(index=False))
    
    # Create output directory
    output_dir = "./nll"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary tables
    print(f"\nSaving summary tables...")
    
    # Save pivot tables
    acc_pivot.to_csv(os.path.join(output_dir, "recovery_accuracy_summary.csv"))
    nll_pivot.to_csv(os.path.join(output_dir, "nll_summary.csv"))
    perp_pivot.to_csv(os.path.join(output_dir, "perplexity_summary.csv"))
    
    # Save detailed summary
    detailed_df.to_csv(os.path.join(output_dir, "detailed_summary.csv"), index=False)
    
    print("✅ Summary tables saved:")
    print("   - ./nll/recovery_accuracy_summary.csv")
    print("   - ./nll/nll_summary.csv") 
    print("   - ./nll/perplexity_summary.csv")
    print("   - ./nll/detailed_summary.csv")
    
    # Calculate overall model rankings
    print("\n" + "="*80)
    print("MODEL RANKINGS BY AVERAGE PERFORMANCE")
    print("="*80)
    
    model_rankings = df.groupby('model').agg({
        'recovery_accuracy': 'mean',
        'nll': 'mean',
        'perplexity': 'mean'
    }).round(4)
    
    model_rankings = model_rankings.sort_values('recovery_accuracy', ascending=False)
    print("Ranked by Recovery Accuracy (higher is better):")
    print(model_rankings.to_string())
    
    model_rankings_nll = model_rankings.sort_values('nll', ascending=True)
    print("\nRanked by NLL (lower is better):")
    print(model_rankings_nll.to_string())
    
    return detailed_df, acc_pivot, nll_pivot, perp_pivot

if __name__ == "__main__":
    summarize_lm_results() 