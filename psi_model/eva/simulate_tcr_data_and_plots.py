#!/usr/bin/env python3
"""
TCR Binding Data Simulation and Plot Generation
Creates simulated TCR binding data with specified model performance characteristics
and generates publication-quality plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_original_binding_data():
    """Load and filter original binding data"""
    print("Loading original binding data...")
    
    # Try to load from existing file
    try:
        df = pd.read_csv('./complete_model_predictions.csv')
        print(f"Loaded {len(df)} samples from complete_model_predictions.csv")
    except FileNotFoundError:
        print("complete_model_predictions.csv not found, creating synthetic base data...")
        df = create_synthetic_base_data()
    
    # Filter to binding samples (binary_label == 1)
    binding_samples = df[df['binary_label'] == 1].copy()
    print(f"Found {len(binding_samples)} binding samples")
    
    return binding_samples

def create_synthetic_base_data():
    """Create synthetic base TCR data for demonstration"""
    np.random.seed(42)
    
    # Synthetic TCR sequences and properties
    peptides = ['YLQPRTFLL', 'GLCTLVAML']
    mhc_alleles = ['HLA-A*02:01', 'HLA-A*01:01', 'HLA-B*07:02']
    
    data = []
    for i in range(500):
        # Generate synthetic TCR data
        peptide = np.random.choice(peptides)
        mhc = np.random.choice(mhc_alleles)
        
        # Generate synthetic experimental values
        padj = np.random.lognormal(-10, 2)  # Log-normal distribution for p-values
        log2fc = np.random.normal(0, 2)     # Normal distribution for fold change
        binary_label = 1 if padj < 1e-5 else 0
        
        data.append({
            'pep': peptide,
            'mhc': mhc,
            'lv': f'TRBV{np.random.randint(1, 30)}',
            'lj': f'TRBJ{np.random.randint(1, 3)}-{np.random.randint(1, 6)}',
            'hv': f'TRAV{np.random.randint(1, 40)}',
            'hd': ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), 
                         size=np.random.randint(8, 15))),
            'hj': f'TRAJ{np.random.randint(1, 60)}',
            'padj': padj,
            'log2FoldChange': log2fc,
            'binary_label': binary_label
        })
    
    return pd.DataFrame(data)

def generate_model_scores_with_target_auc(y_true, target_aucs):
    """
    Generate model scores that achieve specific target AUC values
    
    Args:
        y_true: Binary labels (0/1)
        target_aucs: Dict mapping model names to target AUC values
    
    Returns:
        Dict of model scores
    """
    model_scores = {}
    
    for model_name, target_auc in target_aucs.items():
        # Generate scores that achieve the target AUC
        scores = generate_scores_for_auc(y_true, target_auc)
        model_scores[f'{model_name}_nll'] = -scores  # NLL is negative log likelihood
        model_scores[f'{model_name}_score'] = scores
    
    return model_scores

def generate_scores_for_auc(y_true, target_auc, max_iter=1000):
    """Generate prediction scores that achieve a specific AUC value"""
    np.random.seed(hash(target_auc) % 2**32)  # Deterministic but different for each AUC
    
    best_scores = None
    best_auc_diff = float('inf')
    
    for _ in range(max_iter):
        # Generate random scores with some correlation to true labels
        base_scores = np.random.normal(0, 1, len(y_true))
        
        # Add signal based on true labels to get closer to target AUC
        signal_strength = (target_auc - 0.5) * 4  # Scale signal
        signal = y_true * signal_strength + np.random.normal(0, 0.5, len(y_true))
        scores = base_scores + signal
        
        # Calculate AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            current_auc = auc(fpr, tpr)
            auc_diff = abs(current_auc - target_auc)
            
            if auc_diff < best_auc_diff:
                best_auc_diff = auc_diff
                best_scores = scores.copy()
                
                # If we're close enough, stop
                if auc_diff < 0.01:
                    break
        except:
            continue
    
    return best_scores if best_scores is not None else np.random.normal(0, 1, len(y_true))

def double_binding_samples(binding_data):
    """Double the number of binding samples by creating variations"""
    print(f"Doubling binding samples from {len(binding_data)} to {len(binding_data) * 2}...")
    
    # Create copy with slight variations
    doubled_data = binding_data.copy()
    additional_samples = binding_data.copy()
    
    # Add small variations to experimental values for additional samples
    np.random.seed(42)
    noise_factor = 0.1
    
    additional_samples['padj'] *= np.random.lognormal(0, noise_factor, len(additional_samples))
    additional_samples['log2FoldChange'] += np.random.normal(0, noise_factor, len(additional_samples))
    
    # Combine original and additional samples
    final_data = pd.concat([doubled_data, additional_samples], ignore_index=True)
    
    print(f"Generated {len(final_data)} total samples")
    return final_data

def add_non_binding_samples(binding_data, n_non_binding=None):
    """Add non-binding samples to create a balanced dataset"""
    if n_non_binding is None:
        n_non_binding = len(binding_data) // 2  # Add half as many non-binding samples
    
    print(f"Adding {n_non_binding} non-binding samples...")
    
    # Create non-binding samples
    non_binding_samples = []
    np.random.seed(123)
    
    for i in range(n_non_binding):
        # Copy structure from random binding sample
        template = binding_data.sample(1).iloc[0].copy()
        
        # Modify to be non-binding
        template['padj'] = np.random.uniform(0.01, 1.0)  # Higher p-values
        template['log2FoldChange'] = np.random.normal(0, 1)  # Less extreme fold changes
        template['binary_label'] = 0
        
        non_binding_samples.append(template)
    
    non_binding_df = pd.DataFrame(non_binding_samples)
    combined_data = pd.concat([binding_data, non_binding_df], ignore_index=True)
    
    print(f"Total dataset: {len(combined_data)} samples ({len(binding_data)} binding, {len(non_binding_df)} non-binding)")
    return combined_data

def create_simulated_dataset():
    """Create the complete simulated dataset with all model scores"""
    print("=== Creating Simulated TCR Dataset ===")
    
    # Step 1: Load original binding data
    binding_data = load_original_binding_data()
    
    # Step 2: Double the binding samples
    doubled_binding = double_binding_samples(binding_data)
    
    # Step 3: Add some non-binding samples for contrast
    complete_data = add_non_binding_samples(doubled_binding)
    
    # Step 4: Define target AUC values for each model (descending from 0.75 to 0.60)
    target_aucs = {
        'staged_composite': 0.75,
        'composite': 0.70,
        'cdr3_pretrained': 0.67,
        'tcr_pretrained': 0.64,
        'update_model': 0.62,
        'ori_model': 0.60
    }
    
    print(f"\nGenerating model scores with target AUCs:")
    for model, auc in target_aucs.items():
        print(f"  {model}: {auc:.2f}")
    
    # Step 5: Generate model scores
    y_true = complete_data['binary_label'].values
    model_scores = generate_model_scores_with_target_auc(y_true, target_aucs)
    
    # Step 6: Add model scores to dataset
    for score_col, scores in model_scores.items():
        complete_data[score_col] = scores
    
    # Step 7: Verify AUC values
    print(f"\nVerifying generated AUC values:")
    for model_name in target_aucs.keys():
        score_col = f'{model_name}_score'
        if score_col in complete_data.columns:
            fpr, tpr, _ = roc_curve(y_true, complete_data[score_col])
            actual_auc = auc(fpr, tpr)
            print(f"  {model_name}: target={target_aucs[model_name]:.2f}, actual={actual_auc:.3f}")
    
    return complete_data

def plot_binary_classification_performance(df, output_prefix="simulated"):
    """Create comprehensive binary classification performance plots"""
    print(f"\nGenerating binary classification performance plots...")
    
    # Define models and their scores
    models = ['staged_composite', 'composite', 'cdr3_pretrained', 'tcr_pretrained', 'update_model', 'ori_model']
    score_columns = [f'{model}_score' for model in models]
    
    # Create binary labels
    y_true = df['binary_label'].values
    
    # Create combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # Random baseline
    np.random.seed(42)
    random_scores = np.random.rand(len(y_true))
    
    # PRECISION-RECALL CURVES
    precision_rand, recall_rand, _ = precision_recall_curve(y_true, random_scores)
    ap_rand = average_precision_score(y_true, random_scores)
    ax1.plot(recall_rand, precision_rand, 'k--', alpha=0.7, label=f'Random (AP={ap_rand:.3f})', linewidth=2)
    
    # MODEL CURVES
    performance_data = []
    for i, (model, score_col) in enumerate(zip(models, score_columns)):
        if score_col in df.columns:
            y_scores = df[score_col].values
            
            # Handle NaN values
            valid_mask = ~np.isnan(y_scores)
            if valid_mask.sum() > 0:
                y_true_clean = y_true[valid_mask]
                y_scores_clean = y_scores[valid_mask]
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_true_clean, y_scores_clean)
                ap = average_precision_score(y_true_clean, y_scores_clean)
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true_clean, y_scores_clean)
                auc_score = auc(fpr, tpr)
                
                # Store performance
                performance_data.append({
                    'model': model,
                    'auc': auc_score,
                    'ap': ap
                })
                
                # Plot Precision-Recall
                ax1.plot(recall, precision, color=colors[i], 
                        label=f'{model} (AP={ap:.3f})', linewidth=2)
                
                # Plot ROC
                ax2.plot(fpr, tpr, color=colors[i], 
                        label=f'{model} (AUC={auc_score:.3f})', linewidth=2)
    
    # Format Precision-Recall plot
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curves\nTCR Binding Prediction', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Format ROC plot
    ax2.plot([0, 1], [0, 1], 'gray', alpha=0.5, linewidth=1)  # Diagonal reference
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curves\nTCR Binding Prediction', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_binary_classification_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return performance_data

def plot_correlation_heatmap(df, output_prefix="simulated"):
    """Create correlation heatmap for all model scores"""
    print(f"\nGenerating correlation heatmap...")
    
    # Select score columns and experimental metrics
    score_cols = [col for col in df.columns if col.endswith('_score')]
    metrics = ['padj', 'log2FoldChange']
    
    # Calculate correlations
    correlation_data = []
    for metric in metrics:
        for score_col in score_cols:
            model_name = score_col.replace('_score', '')
            
            # Calculate Spearman correlation
            valid_mask = ~(pd.isna(df[metric]) | pd.isna(df[score_col]))
            if valid_mask.sum() > 0:
                corr, p_val = spearmanr(df[metric][valid_mask], df[score_col][valid_mask])
                correlation_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Spearman_r': corr,
                    'p_value': p_val
                })
    
    # Create pivot table for heatmap
    corr_df = pd.DataFrame(correlation_data)
    pivot_table = corr_df.pivot(index='Model', columns='Metric', values='Spearman_r')
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': 'Spearman Correlation'})
    plt.title('Model Score Correlations with Experimental Metrics', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Experimental Metric', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_df

def generate_summary_statistics(df, performance_data):
    """Generate comprehensive summary statistics"""
    print(f"\n=== SIMULATED DATASET SUMMARY ===")
    print(f"Total samples: {len(df)}")
    print(f"Binding samples: {df['binary_label'].sum()} ({df['binary_label'].mean():.1%})")
    print(f"Non-binding samples: {(df['binary_label'] == 0).sum()} ({(df['binary_label'] == 0).mean():.1%})")
    
    print(f"\n=== MODEL PERFORMANCE SUMMARY ===")
    print("Model\t\t\tAUC\tAP")
    print("-" * 40)
    for perf in sorted(performance_data, key=lambda x: x['auc'], reverse=True):
        print(f"{perf['model']:<20}\t{perf['auc']:.3f}\t{perf['ap']:.3f}")
    
    # Experimental metrics summary
    print(f"\n=== EXPERIMENTAL METRICS SUMMARY ===")
    print(f"padj range: {df['padj'].min():.2e} - {df['padj'].max():.2e}")
    print(f"log2FoldChange range: {df['log2FoldChange'].min():.2f} - {df['log2FoldChange'].max():.2f}")

def main():
    """Main simulation and analysis function"""
    print("🧬 TCR Binding Data Simulation and Analysis")
    print("=" * 60)
    
    # Create simulated dataset
    simulated_df = create_simulated_dataset()
    
    # Save the dataset
    output_file = "simulated_tcr_binding_data.csv"
    simulated_df.to_csv(output_file, index=False)
    print(f"\nSimulated dataset saved to: {output_file}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("📊 GENERATING PLOTS")
    print("=" * 60)
    
    # Binary classification performance plots
    performance_data = plot_binary_classification_performance(simulated_df)
    
    # Correlation heatmap
    correlation_df = plot_correlation_heatmap(simulated_df)
    
    # Save analysis results
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv("simulated_model_performance.csv", index=False)
    correlation_df.to_csv("simulated_correlation_analysis.csv", index=False)
    
    # Generate summary statistics
    generate_summary_statistics(simulated_df, performance_data)
    
    print(f"\n✅ Analysis complete! Generated files:")
    print(f"  📁 {output_file}")
    print(f"  📁 simulated_binary_classification_performance.png")
    print(f"  📁 simulated_correlation_heatmap.png")
    print(f"  📁 simulated_model_performance.csv")
    print(f"  📁 simulated_correlation_analysis.csv")

if __name__ == "__main__":
    main() 