#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

print("Loading dataset...")
# Read the dataset
df = pd.read_csv('integrated_tcr_predictions.csv')

# Create binary labels based on padj < 1e-5 cutoff
if 'binary_label' not in df.columns:
    df['binary_label'] = (df['padj'] < 1e-5).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Unique peptides: {df['pep'].unique()}")
print(f"Binary label distribution:\n{df['binary_label'].value_counts()}")
print(f"Binary label distribution by peptide:\n{df.groupby('pep')['binary_label'].agg(['count', 'sum', 'mean'])}")

# Define model names
model_names = {
    'ori_model_score': 'tcrdist3',
    'update_model_score': 'stapler', 
    'optimized_composite_condition_1_score': 'TCRbridge'
}

# Colors for each model
colors = ['#ff7f0e', '#9467bd', '#2ca02c', '#d62728']

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

peptides = ['YLQPRTFLL', 'GLCTLVAML'] 

for i, peptide in enumerate(peptides):
    print(f"\nAnalyzing {peptide}...")
    peptide_data = df[df['pep'] == peptide].copy()
    n_samples = len(peptide_data)
    n_positive = peptide_data['binary_label'].sum()
    
    print(f"  Samples: {n_samples}, Positive: {n_positive} ({n_positive/n_samples*100:.1f}%)")
    
    # Precision-Recall plot (top row)
    ax_pr = axes[0, i]
    # ROC plot (bottom row) 
    ax_roc = axes[1, i]
    
    # Random baselines
    random_precision = n_positive / n_samples
    ax_pr.axhline(y=random_precision, color='red', linestyle='--', label=f'Random {random_precision:.2f} AP 0.50 AUC')
    ax_roc.plot([0, 1], [0, 1], 'r--', label='Random 0.50 AP 0.50 AUC')
    
    # Analyze each model
    for j, (score_col, model_name) in enumerate(model_names.items()):
        if score_col in peptide_data.columns:
            y_true = peptide_data['binary_label'].values
            y_scores = peptide_data[score_col].values
            
            # Handle missing values
            valid_mask = ~np.isnan(y_scores)
            if np.sum(valid_mask) < 2:
                continue
                
            y_true_clean = y_true[valid_mask]
            y_scores_clean = y_scores[valid_mask]
            
            # Calculate metrics
            fpr, tpr, _ = roc_curve(y_true_clean, y_scores_clean)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_true_clean, y_scores_clean)
            avg_precision = average_precision_score(y_true_clean, y_scores_clean)
            
            print(f"  {model_name}: AUROC = {roc_auc:.3f}, AP = {avg_precision:.3f}")
            
            # Plot curves
            color = colors[j % len(colors)]
            ax_pr.plot(recall, precision, color=color, linewidth=2,
                      label=f'{model_name} {avg_precision:.2f} AP {roc_auc:.2f} AUC')
            ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                       label=f'{model_name} {roc_auc:.2f} AP {avg_precision:.2f} AUC')
    
    # Format plots
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.0])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Prediction of {peptide} validating vs non-validating (n={n_samples})')
    ax_pr.legend(fontsize=8)
    ax_pr.grid(True, alpha=0.3)
    
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.0])
    ax_roc.set_xlabel('False pos. rate')
    ax_roc.set_ylabel('True pos. rate') 
    ax_roc.set_title(f'Prediction of {peptide} validating vs non-validating (n={n_samples})')
    ax_roc.legend(fontsize=8)
    ax_roc.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tcr_performance_simple.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as tcr_performance_simple.png")
plt.show()

print("\nAnalysis complete!") 