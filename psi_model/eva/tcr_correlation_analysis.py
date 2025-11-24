#!/usr/bin/env python3
"""
TCR Model Correlation Analysis
Compares Spearman correlations for padj and log2FoldChange across all models
in composite_model_predictions.csv and complete_model_predictions.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_datasets():
    """Load both datasets and return them with model information"""
    print("Loading datasets...")
    
    # Load complete model predictions
    complete_df = pd.read_csv('./complete_model_predictions.csv')
    print(f"Complete dataset: {len(complete_df)} samples")
    
    # Load composite model predictions  
    composite_df = pd.read_csv('./composite_model_predictions.csv')
    print(f"Composite dataset: {len(composite_df)} samples")
    
    return complete_df, composite_df

def extract_model_scores(df, dataset_name):
    """Extract model score columns from dataframe"""
    score_cols = [col for col in df.columns if col.endswith('_nll')]
    model_names = [col.replace('_score', '') for col in score_cols]
    
    print(f"\n{dataset_name} models found:")
    for model in model_names:
        print(f"  - {model}")
    
    return score_cols, model_names

def calculate_spearman_correlations(df, score_cols, metrics=['padj', 'log2FoldChange']):
    """Calculate Spearman correlations between model scores and target metrics"""
    correlations = {}
    
    for metric in metrics:
        correlations[metric] = {}
        for score_col in score_cols:
            model_name = score_col.replace('_score', '')
            
            # Calculate Spearman correlation
            valid_mask = ~(pd.isna(df[metric]) | pd.isna(df[score_col]))
            if valid_mask.sum() > 0:
                corr, p_val = spearmanr(df[metric][valid_mask], df[score_col][valid_mask])
                correlations[metric][model_name] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': valid_mask.sum()
                }
            else:
                correlations[metric][model_name] = {
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'n_samples': 0
                }
    
    return correlations

def create_correlation_summary_table(complete_corr, composite_corr):
    """Create a summary table of correlations"""
    
    # Create summary dataframes
    summary_data = []
    
    for metric in ['padj', 'log2FoldChange']:
        # Complete dataset models
        for model in complete_corr[metric].keys():
            summary_data.append({
                'Dataset': 'Complete',
                'Model': model,
                'Metric': metric,
                'Spearman_r': complete_corr[metric][model]['correlation'],
                'p_value': complete_corr[metric][model]['p_value'],
                'n_samples': complete_corr[metric][model]['n_samples']
            })
        
        # Composite dataset models  
        for model in composite_corr[metric].keys():
            summary_data.append({
                'Dataset': 'Composite',
                'Model': model,
                'Metric': metric,
                'Spearman_r': composite_corr[metric][model]['correlation'],
                'p_value': composite_corr[metric][model]['p_value'],
                'n_samples': composite_corr[metric][model]['n_samples']
            })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def plot_correlation_heatmap(summary_df):
    """Create correlation heatmap comparing all models across both datasets"""
    
    # Create pivot table for heatmap
    pivot_padj = summary_df[summary_df['Metric'] == 'padj'].pivot(
        index='Model', columns='Dataset', values='Spearman_r'
    )
    
    pivot_l2fc = summary_df[summary_df['Metric'] == 'log2FoldChange'].pivot(
        index='Model', columns='Dataset', values='Spearman_r'
    )
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot padj correlations
    sns.heatmap(pivot_padj, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', ax=ax1, cbar_kws={'label': 'Spearman r'})
    ax1.set_title('Spearman Correlation: Model Scores vs padj', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Model', fontsize=12)
    
    # Plot log2FoldChange correlations
    sns.heatmap(pivot_l2fc, annot=True, cmap='RdBu_r', center=0,
                fmt='.3f', ax=ax2, cbar_kws={'label': 'Spearman r'})
    ax2.set_title('Spearman Correlation: Model Scores vs log2FoldChange', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('./nll/model_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_binary_metrics_at_threshold(y_true, y_scores, threshold=0.0):
    """Calculate binary classification metrics at a specific threshold"""
    y_pred = (y_scores > threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1 - Specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    
    return {
        'threshold': threshold,
        'tpr': tpr,
        'fpr': fpr, 
        'precision': precision,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """Find optimal threshold based on specified metric"""
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
    best_score = 0
    best_threshold = 0
    
    for threshold in thresholds:
        metrics = calculate_binary_metrics_at_threshold(y_true, y_scores, threshold)
        
        if metric == 'f1':
            score = metrics['f1']
        elif metric == 'youden':
            score = metrics['tpr'] - metrics['fpr']  # Youden's J statistic
        else:
            score = metrics[metric]
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

def plot_binary_classification_performance(complete_df, composite_df):
    """Create combined precision-recall and ROC curves for binary classification"""
    
    # Extract model scores for each dataset
    complete_score_cols, complete_models = extract_model_scores(complete_df, "Complete")
    composite_score_cols, composite_models = extract_model_scores(composite_df, "Composite")
    
    # Create binary labels (padj < 1e-05)
    complete_binary = (complete_df['padj'] < 1e-05).astype(int)
    composite_binary = (composite_df['padj'] < 1e-05).astype(int)
    
    # Create combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for models - use different colors for complete vs composite
    complete_colors = plt.cm.Set1(np.linspace(0, 1, len(complete_models)))
    composite_colors = plt.cm.Set2(np.linspace(0, 1, len(composite_models)))
    
    # Random baseline
    np.random.seed(42)  # For reproducibility
    random_scores = np.random.rand(len(complete_binary))
    
    # PRECISION-RECALL CURVES
    # Add random baseline first
    precision_rand, recall_rand, _ = precision_recall_curve(complete_binary, random_scores)
    ap_rand = average_precision_score(complete_binary, random_scores)
    ax1.plot(recall_rand, precision_rand, 'k--', alpha=0.7, label=f'Random (AP={ap_rand:.3f})', linewidth=2)
    
    # Complete dataset models
    for i, (score_col, model) in enumerate(zip(complete_score_cols, complete_models)):
        valid_mask = ~pd.isna(complete_df[score_col])
        if valid_mask.sum() > 0:
            y_true = complete_binary[valid_mask]
            y_scores = complete_df[score_col][valid_mask]
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            ax1.plot(recall, precision, color=complete_colors[i], 
                    label=f'{model} (AP={ap:.3f})', linewidth=2, linestyle='-')
    
    # Composite dataset models
    for i, (score_col, model) in enumerate(zip(composite_score_cols, composite_models)):
        valid_mask = ~pd.isna(composite_df[score_col])
        if valid_mask.sum() > 0:
            y_true = composite_binary[valid_mask]
            y_scores = composite_df[score_col][valid_mask]
            
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            # Use shorter names for composite models
            short_name = model.replace('_condition_1', '').replace('_composite', '_comp')
            ax1.plot(recall, precision, color=composite_colors[i], 
                    label=f'{short_name} (AP={ap:.3f})', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curves\n(Solid: Complete Dataset, Dashed: Composite Dataset)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # ROC CURVES
    # Add random baseline
    fpr_rand, tpr_rand, _ = roc_curve(complete_binary, random_scores)
    auc_rand = auc(fpr_rand, tpr_rand)
    ax2.plot(fpr_rand, tpr_rand, 'k--', alpha=0.7, label=f'Random (AUC={auc_rand:.3f})', linewidth=2)
    
    # Complete dataset models
    for i, (score_col, model) in enumerate(zip(complete_score_cols, complete_models)):
        valid_mask = ~pd.isna(complete_df[score_col])
        if valid_mask.sum() > 0:
            y_true = complete_binary[valid_mask]
            y_scores = complete_df[score_col][valid_mask]
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = auc(fpr, tpr)
            
            ax2.plot(fpr, tpr, color=complete_colors[i], 
                    label=f'{model} (AUC={auc_score:.3f})', linewidth=2, linestyle='-')
    
    # Composite dataset models
    for i, (score_col, model) in enumerate(zip(composite_score_cols, composite_models)):
        valid_mask = ~pd.isna(composite_df[score_col])
        if valid_mask.sum() > 0:
            y_true = composite_binary[valid_mask]
            y_scores = composite_df[score_col][valid_mask]
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = auc(fpr, tpr)
            
            # Use shorter names for composite models
            short_name = model.replace('_condition_1', '').replace('_composite', '_comp')
            ax2.plot(fpr, tpr, color=composite_colors[i], 
                    label=f'{short_name} (AUC={auc_score:.3f})', linewidth=2, linestyle='--')
    
    # Add diagonal line for reference
    ax2.plot([0, 1], [0, 1], 'gray', alpha=0.5, linewidth=1)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curves\n(Solid: Complete Dataset, Dashed: Composite Dataset)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./nll/combined_binary_classification_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_threshold_based_analysis(complete_df, composite_df):
    """Create threshold-based binary classification analysis"""
    
    # Extract model scores for each dataset
    complete_score_cols, complete_models = extract_model_scores(complete_df, "Complete")
    composite_score_cols, composite_models = extract_model_scores(composite_df, "Composite")
    
    # Create binary labels (padj < 1e-05)
    complete_binary = (complete_df['padj'] < 1e-05).astype(int)
    composite_binary = (composite_df['padj'] < 1e-05).astype(int)
    
    # Analyze different thresholds
    threshold_results = []
    
    print("\n=== THRESHOLD-BASED BINARY CLASSIFICATION ANALYSIS ===")
    
    # Complete dataset analysis
    print(f"\nCOMPLETE DATASET ANALYSIS:")
    for score_col, model in zip(complete_score_cols, complete_models):
        valid_mask = ~pd.isna(complete_df[score_col])
        if valid_mask.sum() > 0:
            y_true = complete_binary[valid_mask]
            y_scores = complete_df[score_col][valid_mask]
            
            # Find optimal threshold
            opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_scores, 'f1')
            
            # Calculate metrics at optimal threshold
            metrics_opt = calculate_binary_metrics_at_threshold(y_true, y_scores, opt_threshold)
            
            # Calculate metrics at score = 0 threshold
            metrics_zero = calculate_binary_metrics_at_threshold(y_true, y_scores, 0.0)
            
            print(f"\n{model}:")
            print(f"  Optimal threshold (F1): {opt_threshold:.3f}")
            print(f"    F1: {metrics_opt['f1']:.3f}, TPR: {metrics_opt['tpr']:.3f}, FPR: {metrics_opt['fpr']:.3f}, Precision: {metrics_opt['precision']:.3f}")
            print(f"  Zero threshold:")
            print(f"    F1: {metrics_zero['f1']:.3f}, TPR: {metrics_zero['tpr']:.3f}, FPR: {metrics_zero['fpr']:.3f}, Precision: {metrics_zero['precision']:.3f}")
            
            threshold_results.append({
                'Dataset': 'Complete',
                'Model': model,
                'Threshold_Type': 'Optimal_F1',
                'Threshold': opt_threshold,
                'F1': metrics_opt['f1'],
                'TPR': metrics_opt['tpr'],
                'FPR': metrics_opt['fpr'],
                'Precision': metrics_opt['precision'],
                'TP': metrics_opt['tp'],
                'TN': metrics_opt['tn'],
                'FP': metrics_opt['fp'],
                'FN': metrics_opt['fn']
            })
            
            threshold_results.append({
                'Dataset': 'Complete',
                'Model': model,
                'Threshold_Type': 'Zero',
                'Threshold': 0.0,
                'F1': metrics_zero['f1'],
                'TPR': metrics_zero['tpr'],
                'FPR': metrics_zero['fpr'],
                'Precision': metrics_zero['precision'],
                'TP': metrics_zero['tp'],
                'TN': metrics_zero['tn'],
                'FP': metrics_zero['fp'],
                'FN': metrics_zero['fn']
            })
    
    # Composite dataset analysis
    print(f"\nCOMPOSITE DATASET ANALYSIS:")
    for score_col, model in zip(composite_score_cols, composite_models):
        valid_mask = ~pd.isna(composite_df[score_col])
        if valid_mask.sum() > 0:
            y_true = composite_binary[valid_mask]
            y_scores = composite_df[score_col][valid_mask]
            
            # Find optimal threshold
            opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_scores, 'f1')
            
            # Calculate metrics at optimal threshold
            metrics_opt = calculate_binary_metrics_at_threshold(y_true, y_scores, opt_threshold)
            
            # Calculate metrics at score = 0 threshold
            metrics_zero = calculate_binary_metrics_at_threshold(y_true, y_scores, 0.0)
            
            short_name = model.replace('_condition_1', '').replace('_composite', '_comp')
            print(f"\n{short_name}:")
            print(f"  Optimal threshold (F1): {opt_threshold:.3f}")
            print(f"    F1: {metrics_opt['f1']:.3f}, TPR: {metrics_opt['tpr']:.3f}, FPR: {metrics_opt['fpr']:.3f}, Precision: {metrics_opt['precision']:.3f}")
            print(f"  Zero threshold:")
            print(f"    F1: {metrics_zero['f1']:.3f}, TPR: {metrics_zero['tpr']:.3f}, FPR: {metrics_zero['fpr']:.3f}, Precision: {metrics_zero['precision']:.3f}")
            
            threshold_results.append({
                'Dataset': 'Composite',
                'Model': model,
                'Threshold_Type': 'Optimal_F1',
                'Threshold': opt_threshold,
                'F1': metrics_opt['f1'],
                'TPR': metrics_opt['tpr'],
                'FPR': metrics_opt['fpr'],
                'Precision': metrics_opt['precision'],
                'TP': metrics_opt['tp'],
                'TN': metrics_opt['tn'],
                'FP': metrics_opt['fp'],
                'FN': metrics_opt['fn']
            })
            
            threshold_results.append({
                'Dataset': 'Composite',
                'Model': model,
                'Threshold_Type': 'Zero',
                'Threshold': 0.0,
                'F1': metrics_zero['f1'],
                'TPR': metrics_zero['tpr'],
                'FPR': metrics_zero['fpr'],
                'Precision': metrics_zero['precision'],
                'TP': metrics_zero['tp'],
                'TN': metrics_zero['tn'],
                'FP': metrics_zero['fp'],
                'FN': metrics_zero['fn']
            })
    
    # Create DataFrame and save results
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv('./nll/threshold_analysis.csv', index=False)
    
    return threshold_df

def main():
    """Main analysis function"""
    print("=== TCR Model Correlation Analysis ===\n")
    
    # Load datasets
    complete_df, composite_df = load_datasets()
    
    # Extract model information
    complete_score_cols, complete_models = extract_model_scores(complete_df, "Complete Dataset")
    composite_score_cols, composite_models = extract_model_scores(composite_df, "Composite Dataset")
    
    # Calculate Spearman correlations
    print("\nCalculating Spearman correlations...")
    complete_corr = calculate_spearman_correlations(complete_df, complete_score_cols)
    composite_corr = calculate_spearman_correlations(composite_df, composite_score_cols)
    
    # Create summary table
    summary_df = create_correlation_summary_table(complete_corr, composite_corr)
    
    # Print summary table
    print("\n=== CORRELATION SUMMARY TABLE ===")
    print(summary_df.round(4))
    
    # Save summary table
    summary_df.to_csv('./nll/correlation_summary.csv', index=False)
    print(f"\nSummary table saved to: ./nll/correlation_summary.csv")
    
    # Create visualizations
    print("\nCreating correlation heatmap...")
    plot_correlation_heatmap(summary_df)
    
    print("Creating binary classification performance plots...")
    plot_binary_classification_performance(complete_df, composite_df)
    
    # Threshold-based binary classification analysis
    print("Performing threshold-based binary classification analysis...")
    threshold_df = create_threshold_based_analysis(complete_df, composite_df)
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    
    # Best correlations for padj
    padj_corr = summary_df[summary_df['Metric'] == 'padj'].sort_values('Spearman_r', key=abs, ascending=False)
    print(f"\nStrongest padj correlations:")
    for _, row in padj_corr.head(3).iterrows():
        print(f"  {row['Dataset']} - {row['Model']}: r = {row['Spearman_r']:.4f} (p = {row['p_value']:.2e})")
    
    # Best correlations for log2FoldChange
    l2fc_corr = summary_df[summary_df['Metric'] == 'log2FoldChange'].sort_values('Spearman_r', key=abs, ascending=False)
    print(f"\nStrongest log2FoldChange correlations:")
    for _, row in l2fc_corr.head(3).iterrows():
        print(f"  {row['Dataset']} - {row['Model']}: r = {row['Spearman_r']:.4f} (p = {row['p_value']:.2e})")
    
    # Binary classification performance
    complete_pos = (complete_df['padj'] < 1e-05).sum()
    composite_pos = (composite_df['padj'] < 1e-05).sum()
    print(f"\nBinary classification (padj < 1e-05):")
    print(f"  Complete dataset: {complete_pos}/{len(complete_df)} positive ({complete_pos/len(complete_df):.1%})")
    print(f"  Composite dataset: {composite_pos}/{len(composite_df)} positive ({composite_pos/len(composite_df):.1%})")
    
    print(f"\nAnalysis complete! Check generated files:")
    print(f"  - ./nll/model_correlation_heatmap.png")
    print(f"  - ./nll/combined_binary_classification_performance.png")
    print(f"  - ./nll/correlation_summary.csv")
    print(f"  - ./nll/threshold_analysis.csv")

if __name__ == "__main__":
    main() 