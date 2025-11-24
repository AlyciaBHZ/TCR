#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

# Read the dataset
df = pd.read_csv('integrated_tcr_predictions.csv')

# Create binary labels based on padj < 1e-5 cutoff
df['binary_label'] = (df['padj'] < 1e-5).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Unique peptides: {df['pep'].unique()}")
print(f"Binary label distribution:\n{df['binary_label'].value_counts()}")

# Check the distribution of binary labels by peptide
print("\nBinary label distribution by peptide:")
print(df.groupby('pep')['binary_label'].agg(['count', 'sum', 'mean']))

# Define the models to evaluate (excluding the binary_label column)
model_score_columns = [col for col in df.columns if col.endswith('_score')]
print(f"\nAvailable model score columns: {model_score_columns}")

# Create subplots for ROC and PR curves
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Color palette for different models
colors = plt.cm.tab10(np.linspace(0, 1, len(model_score_columns)))

# Process each peptide separately
peptides = df['pep'].unique()
peptides = [p for p in peptides if p != 'pep']  # Remove header if it exists

for i, peptide in enumerate(peptides):
    peptide_data = df[df['pep'] == peptide].copy()
    n_samples = len(peptide_data)
    n_positive = peptide_data['binary_label'].sum()
    
    print(f"\n--- {peptide} (n={n_samples}) ---")
    print(f"Positive samples: {n_positive} ({n_positive/n_samples*100:.1f}%)")
    
    # ROC curves
    ax_roc = axes[0, i]
    # PR curves  
    ax_pr = axes[1, i]
    
    # Plot random baseline for ROC
    ax_roc.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random', linewidth=2)
    
    # Plot random baseline for PR (proportion of positive samples)
    random_precision = n_positive / n_samples
    ax_pr.axhline(y=random_precision, color='r', linestyle='--', alpha=0.5, label='Random', linewidth=2)
    
    model_performances = []
    
    for j, score_col in enumerate(model_score_columns):
        if score_col in peptide_data.columns:
            y_true = peptide_data['binary_label'].values
            y_scores = peptide_data[score_col].values
            
            # Skip if all scores are NaN
            if np.all(np.isnan(y_scores)):
                continue
                
            # Handle NaN values
            valid_mask = ~np.isnan(y_scores)
            if np.sum(valid_mask) < 2:
                continue
                
            y_true_clean = y_true[valid_mask]
            y_scores_clean = y_scores[valid_mask]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_clean, y_scores_clean)
            roc_auc = auc(fpr, tpr)
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true_clean, y_scores_clean)
            avg_precision = average_precision_score(y_true_clean, y_scores_clean)
            
            # Store performance
            model_name = score_col.replace('_score', '')
            model_performances.append({
                'model': model_name,
                'auroc': roc_auc,
                'ap': avg_precision,
                'n_samples': len(y_true_clean)
            })
            
            # Plot ROC curve
            ax_roc.plot(fpr, tpr, color=colors[j], linewidth=2,
                       label=f'{model_name} {roc_auc:.2f} AP {avg_precision:.2f} AUC')
            
            # Plot PR curve
            ax_pr.plot(recall, precision, color=colors[j], linewidth=2,
                      label=f'{model_name} {avg_precision:.2f} AP {roc_auc:.2f} AUC')
            
            print(f"{model_name}: AUROC = {roc_auc:.3f}, AP = {avg_precision:.3f}")
    
    # Format ROC subplot
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False pos. rate')
    ax_roc.set_ylabel('True pos. rate')
    ax_roc.set_title(f'Prediction of {peptide} validating vs non-validating (n={n_samples})')
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, alpha=0.3)
    
    # Format PR subplot
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'Prediction of {peptide} validating vs non-validating (n={n_samples})')
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tcr_performance_by_peptide.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table
print("\n=== Performance Summary ===")
for i, peptide in enumerate(peptides):
    peptide_data = df[df['pep'] == peptide].copy()
    print(f"\n{peptide}:")
    for score_col in model_score_columns:
        if score_col in peptide_data.columns:
            y_true = peptide_data['binary_label'].values
            y_scores = peptide_data[score_col].values
            
            valid_mask = ~np.isnan(y_scores)
            if np.sum(valid_mask) < 2:
                continue
                
            y_true_clean = y_true[valid_mask]
            y_scores_clean = y_scores[valid_mask]
            
            fpr, tpr, _ = roc_curve(y_true_clean, y_scores_clean)
            roc_auc = auc(fpr, tpr)
            avg_precision = average_precision_score(y_true_clean, y_scores_clean)
            
            model_name = score_col.replace('_score', '')
            print(f"  {model_name:<30}: AUROC = {roc_auc:.3f}, AP = {avg_precision:.3f}") 