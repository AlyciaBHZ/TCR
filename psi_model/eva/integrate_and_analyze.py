import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading CSV files...")
    
    # Load the two CSV files
    complete_df = pd.read_csv('complete_model_predictions.csv')
    composite_df = pd.read_csv('composite_model_predictions.csv')
    
    print(f"Complete model predictions shape: {complete_df.shape}")
    print(f"Composite model predictions shape: {composite_df.shape}")
    
    # Check column structures
    print("\nComplete model columns:", complete_df.columns.tolist())
    print("Composite model columns:", composite_df.columns.tolist())
    
    # Get the common columns for merging
    common_cols = ['pep', 'mhc', 'lv', 'lj', 'hv', 'hd', 'hj', 'padj', 'log2FoldChange', 'binary_label']
    
    # Extract model prediction columns
    complete_model_cols = [col for col in complete_df.columns if col not in common_cols]
    composite_model_cols = [col for col in composite_df.columns if col not in common_cols]
    
    print(f"\nComplete model prediction columns: {complete_model_cols}")
    print(f"Composite model prediction columns: {composite_model_cols}")
    
    # Merge the dataframes on common columns
    print("\nMerging dataframes...")
    integrated_df = complete_df.merge(
        composite_df[common_cols + composite_model_cols], 
        on=common_cols, 
        how='inner'
    )
    
    print(f"Integrated dataframe shape: {integrated_df.shape}")
    
    # Filter for padj > 0.05
    print("\nFiltering for padj > 0.05...")
    filtered_df = integrated_df[integrated_df['padj'] > 0.05].copy()
    print(f"Filtered dataframe shape: {filtered_df.shape}")
    
    # Extract all model prediction score columns (not NLL columns)
    score_columns = [col for col in integrated_df.columns 
                    if 'score' in col and 'nll' not in col.lower()]
    
    print(f"\nModel score columns for correlation analysis: {score_columns}")
    
    # Calculate Spearman correlations with log2FoldChange
    correlations = {}
    p_values = {}
    
    print("\nCalculating Spearman correlations...")
    for col in score_columns:
        if col in filtered_df.columns:
            # Remove any NaN values for correlation calculation
            valid_data = filtered_df[['log2FoldChange', col]].dropna()
            if len(valid_data) > 0:
                corr, p_val = spearmanr(valid_data['log2FoldChange'], valid_data[col])
                correlations[col] = corr
                p_values[col] = p_val
                print(f"{col}: r = {corr:.4f}, p = {p_val:.2e}")
    
    # Create correlation results dataframe
    correlation_results = pd.DataFrame({
        'Model': list(correlations.keys()),
        'Spearman_Correlation': list(correlations.values()),
        'P_Value': list(p_values.values())
    })
    
    # Sort by absolute correlation value
    correlation_results['Abs_Correlation'] = correlation_results['Spearman_Correlation'].abs()
    correlation_results = correlation_results.sort_values('Abs_Correlation', ascending=False)
    
    # Save results
    print("\nSaving results...")
    
    # Save integrated and filtered data
    integrated_df.to_csv('integrated_tcr_predictions.csv', index=False)
    filtered_df.to_csv('integrated_tcr_predictions_filtered.csv', index=False)
    
    # Save correlation results
    correlation_results.to_csv('model_correlations_with_log2FoldChange.csv', index=False)
    
    print(f"\nFiles saved:")
    print(f"- integrated_tcr_predictions.csv (shape: {integrated_df.shape})")
    print(f"- integrated_tcr_predictions_filtered.csv (shape: {filtered_df.shape})")
    print(f"- model_correlations_with_log2FoldChange.csv")
    
    # Display correlation summary
    print("\n=== CORRELATION SUMMARY ===")
    print(correlation_results[['Model', 'Spearman_Correlation', 'P_Value']].to_string(index=False))
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Bar plot of correlations
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(correlation_results)), 
                   correlation_results['Spearman_Correlation'],
                   color=['red' if x < 0 else 'blue' for x in correlation_results['Spearman_Correlation']])
    plt.xlabel('Model')
    plt.ylabel('Spearman Correlation with log2FoldChange')
    plt.title('Model Predictions vs log2FoldChange Correlations')
    plt.xticks(range(len(correlation_results)), 
               [col.replace('_score', '').replace('_', '\n') for col in correlation_results['Model']], 
               rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, correlation_results['Spearman_Correlation'])):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.01 if corr > 0 else -0.05),
                f'{corr:.3f}', ha='center', va='bottom' if corr > 0 else 'top')
    
    # Heatmap of correlations
    plt.subplot(2, 1, 2)
    corr_matrix = filtered_df[score_columns + ['log2FoldChange']].corr()
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix: Model Scores and log2FoldChange')
    
    plt.tight_layout()
    plt.savefig('tcr_correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: tcr_correlation_analysis.png")
    
    return integrated_df, filtered_df, correlation_results

if __name__ == "__main__":
    integrated_df, filtered_df, correlation_results = main() 