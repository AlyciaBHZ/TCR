#!/usr/bin/env python3
"""
Benchmark all models in clp/eva/models/ on experimental data
Handles both original architecture (conditioned/) and updated architecture (clp/)
Uses padj < 1e-05 as binary classification target as specified by dataset authors
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Updated architecture (clp/)
clp_path = os.path.abspath('..')
sys.path.insert(0, clp_path)
import model as clp_model  # This contains psiCLM
import data_clp as clp_data    # This contains CollapseProteinDataset and dummy function
sys.path.remove(clp_path)


def load_updated_model(model_path, model_config):
    """Load updated architecture model (psiCLM)"""
    device = get_device()
    model = clp_model.psiCLM(model_config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint, strict=False)  # Use strict=False to ignore unexpected keys
    model.to(device)
    model.eval()
    return model



def evaluate_clp_model(model, dataset, conditioning_info):
    """Evaluate CLP model using CollapseProteinDataset"""
    device = get_device()
    nlls = []
    
    print(f"Evaluating CLP model on {len(dataset)} samples...")
    
    for i in range(len(dataset)):
        try:
            # Get sample from dataset
            sample = dataset[i]
            
            # Move to device
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.to(device)
            
            # Create mask for entire CDR3 sequence (hd)
            hd_len = sample['hd'].shape[0]
            if hd_len == 0:
                continue
                
            sample['mask'] = torch.ones(hd_len).to(device)
            
            # Evaluate with conditioning
            with torch.no_grad():
                output = model(sample, computeloss=True, conditioning_info=conditioning_info)
                if isinstance(output, dict):
                    nll = output['loss'].item()
                else:
                    nll = output.item()
                nlls.append(nll)
                
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")
                
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue
    
    print(f"  Successfully processed {len(nlls)} samples")
    return nlls

def evaluate_conditioned_model(model, dataset, conditioning_info):
    """Evaluate conditioned model using Load_Dataset"""
    device = get_device()
    nlls = []
    
    print(f"Evaluating conditioned model on {len(dataset)} samples...")
    
    for i in range(len(dataset)):
        try:
            # Get sample from dataset
            sample = dataset[i]
            
            # Move to device
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.to(device)
            
            # Create mask for entire CDR3 sequence (hd)
            hd_len = sample['hd'].shape[0]
            if hd_len == 0:
                continue
                
            sample['mask'] = torch.ones(hd_len).to(device)
            
            # Evaluate with conditioning
            with torch.no_grad():
                output = model(sample, computeloss=True, conditioning_info=conditioning_info)
                if isinstance(output, dict):
                    nll = output['loss'].item()
                else:
                    nll = output.item()
                nlls.append(nll)
                
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")
                
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue
    
    print(f"  Successfully processed {len(nlls)} samples")
    return nlls

def prepare_data_for_evaluation(exp_data_path, output_path):
    """
    Prepare experimental data in the format expected by the data loaders
    
    For CLP models: expects columns ['pep', 'mhc', 'lv', 'lj', 'hv', 'hd', 'hj']
    For conditioned models: expects same format
    """
    print("Preparing data for evaluation...")
    
    # Load experimental data
    exp_data = pd.read_csv(exp_data_path)
    print(f"Loaded {len(exp_data)} experimental samples")
    
    # Filter to samples with valid CDR3 and padj
    valid_data = exp_data[
        exp_data['cdr3_b'].notna() & 
        (exp_data['cdr3_b'] != '') & 
        exp_data['padj'].notna()
    ].copy()
    
    print(f"Found {len(valid_data)} valid samples with CDR3 and padj")
    
    # Map to expected format
    formatted_data = []
    for _, row in valid_data.iterrows():
        formatted_row = [
            str(row['peptide']) if pd.notna(row['peptide']) else '',      # pep
            str(row['mhc']) if pd.notna(row['mhc']) else '',             # mhc  
            str(row['l_v']) if pd.notna(row['l_v']) else '',             # lv
            str(row['l_j']) if pd.notna(row['l_j']) else '',             # lj
            str(row['h_v']) if pd.notna(row['h_v']) else '',             # hv
            str(row['cdr3_b']) if pd.notna(row['cdr3_b']) else '',       # hd (target)
            str(row['h_j']) if pd.notna(row['h_j']) else '',             # hj
        ]
        formatted_data.append(formatted_row)
    
    # Create DataFrame with expected columns
    formatted_df = pd.DataFrame(formatted_data, columns=['pep', 'mhc', 'lv', 'lj', 'hv', 'hd', 'hj'])
    
    # Save formatted data
    formatted_df.to_csv(output_path, index=False)
    print(f"Saved formatted data to {output_path}")
    
    # Also save the experimental scores for correlation analysis
    exp_scores = valid_data[['padj', 'log2FoldChange']].reset_index(drop=True)
    exp_scores_path = output_path.replace('.csv', '_scores.csv')
    exp_scores.to_csv(exp_scores_path, index=False)
    print(f"Saved experimental scores to {exp_scores_path}")
    
    return output_path, exp_scores_path

def calculate_classification_metrics(y_true, y_scores, threshold_p=1e-05):
    """
    Calculate classification metrics using padj threshold
    y_true: binary labels (1 = significant binding, 0 = not significant)
    y_scores: prediction scores (we'll use -NLL, so higher = better binding)
    """
    metrics = {}
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
        metrics['auroc'] = auroc
    except:
        metrics['auroc'] = np.nan
    
    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Make binary predictions using optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    
    # Calculate classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['tpr'] = metrics['recall']  # TPR is same as recall
    
    # Calculate specificity (TNR)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Store threshold info
    metrics['optimal_threshold'] = optimal_threshold
    metrics['n_positive'] = np.sum(y_true)
    metrics['n_negative'] = len(y_true) - np.sum(y_true)
    
    return metrics

def plot_results(results, output_dir='plots'):
    """Create comprehensive plots for model comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out error results
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Create DataFrame for plotting
    plot_data = []
    for model_name, metrics in valid_results.items():
        plot_data.append({
            'Model': model_name,
            'AUROC': metrics.get('auroc', np.nan),
            'Precision': metrics.get('precision', np.nan),
            'Recall': metrics.get('recall', np.nan),
            'F1 Score': metrics.get('f1_score', np.nan),
            'Accuracy': metrics.get('accuracy', np.nan),
            'Specificity': metrics.get('specificity', np.nan)
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 1. Bar plot of all metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['AUROC', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Specificity']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        bars = ax.bar(plot_df['Model'], plot_df[metric], alpha=0.7)
        ax.set_title(f'{metric} by Model')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC curves (if we have the raw data)
    # Note: This would require storing the raw predictions, which we're not doing yet
    
    # 3. Summary heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    heatmap_data = plot_df.set_index('Model')[metrics_to_plot].T
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                ax=ax, cbar_kws={'label': 'Score'})
    ax.set_title('Model Performance Heatmap')
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

def main():
    # Prepare data for evaluation
    print("=== Preparing Data ===")
    exp_data_path = '1model_predictions.csv'  # Input experimental data
    formatted_data_path = 'formatted_eval_data.csv'  # Output formatted for datasets
    
    formatted_data_path, exp_scores_path = prepare_data_for_evaluation(exp_data_path, formatted_data_path)
    
    # Load experimental scores for correlation analysis
    exp_scores = pd.read_csv(exp_scores_path)
    exp_data = pd.read_csv(exp_data_path)
    
    # Create binary labels based on padj < 1e-05 threshold
    binary_labels = (exp_scores['padj'] < 1e-05).astype(int)
    n_positive = np.sum(binary_labels)
    n_negative = len(binary_labels) - n_positive
    
    print(f"\nBinary classification target (padj < 1e-05):")
    print(f"  Positive samples (significant binding): {n_positive}")
    print(f"  Negative samples (not significant): {n_negative}")
    print(f"  Positive rate: {n_positive/len(binary_labels):.3f}")
    
    # Create datasets for evaluation
    print(f"\n=== Creating Datasets ===")
    
    # For CLP models (CollapseProteinDataset)
    clp_dataset = clp_data.CollapseProteinDataset(formatted_data_path)
    print(f"Created CLP dataset with {len(clp_dataset)} samples")
    
    # Model configurations
    model_configs = {
        'optimized_composite_condition_1': {
            'path': '../saved_model/optimized_composite_condition_1/model_epoch_125',
            'config': {
                's_dim': 128,
                'z_dim': 64,
                's_in_dim': 21,
                'z_in_dim': 2,
                'N_elayers': 8,
            },
            'architecture': 'clp',
            'dataset': clp_dataset,
            'conditioning_info': ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']  # Everything except hd
        },
        'staged_composite_condition_1': {
            'path': '../saved_model/staged_composite_condition_1/model_epoch_175',
            'config': {
                's_dim': 128,
                'z_dim': 64,
                's_in_dim': 21,
                'z_in_dim': 2,
                'N_elayers': 8,
            },
            'architecture': 'clp',
            'dataset': clp_dataset,
            'conditioning_info': ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']  # Everything except hd
        }
    }
    
    # Initialize predictions DataFrame to store all model scores
    predictions_df = pd.read_csv('formatted_eval_data.csv')
    scores_df = pd.read_csv('formatted_eval_data_scores.csv')
    predictions_df = pd.concat([predictions_df, scores_df], axis=1)
    predictions_df['binary_label'] = binary_labels
    
    results = {}
    
    for model_name, model_info in model_configs.items():
        print(f"\n=== Evaluating {model_name} ===")
        
        try:
            # Load model
            model = load_updated_model(model_info['path'], model_info['config'])
            
            # Evaluate using appropriate method based on architecture
            if model_info['architecture'] == 'clp':
                nlls = evaluate_clp_model(model, model_info['dataset'], model_info['conditioning_info'])
            else:
                nlls = evaluate_conditioned_model(model, model_info['dataset'], model_info['conditioning_info'])
            
            if nlls:
                # Convert NLL to prediction scores (lower NLL = higher binding probability)
                prediction_scores = -np.array(nlls)
                
                # Add predictions to DataFrame (pad with NaN if needed)
                pred_column_nll = f'{model_name}_nll'
                pred_column_score = f'{model_name}_score'
                
                # Initialize columns with NaN
                predictions_df[pred_column_nll] = np.nan
                predictions_df[pred_column_score] = np.nan
                
                # Fill in the predictions we have
                predictions_df.iloc[:len(nlls), predictions_df.columns.get_loc(pred_column_nll)] = nlls
                predictions_df.iloc[:len(prediction_scores), predictions_df.columns.get_loc(pred_column_score)] = prediction_scores
                
                # Get corresponding binary labels for evaluation
                labels = binary_labels.values[:len(nlls)]
                
                # Ensure we have matching lengths
                min_len = min(len(nlls), len(labels))
                nlls_trimmed = nlls[:min_len]
                labels_trimmed = labels[:min_len]
                prediction_scores_trimmed = prediction_scores[:min_len]
                
                # Calculate classification metrics
                metrics = calculate_classification_metrics(labels_trimmed, prediction_scores_trimmed)
                
                # Also calculate correlations for comparison  
                if len(exp_scores) > 0:
                    exp_padj = exp_scores['padj'].values[:min_len]
                    exp_padj_valid = exp_padj[~pd.isna(exp_padj)]
                    pred_scores_valid = prediction_scores_trimmed[:len(exp_padj_valid)]
                    
                    if len(exp_padj_valid) > 0:
                        spearman_corr, spearman_p = spearmanr(pred_scores_valid, exp_padj_valid)
                        metrics['spearman_corr_padj'] = spearman_corr
                        metrics['spearman_p_padj'] = spearman_p
                
                # Store basic stats
                metrics['n_samples'] = min_len
                metrics['mean_nll'] = np.mean(nlls_trimmed)
                metrics['std_nll'] = np.std(nlls_trimmed)
                metrics['mean_score'] = np.mean(prediction_scores_trimmed)
                metrics['std_score'] = np.std(prediction_scores_trimmed)
                
                results[model_name] = metrics
                
                print(f"Results for {model_name}:")
                key_metrics = ['n_samples', 'auroc', 'precision', 'recall', 'f1_score', 'accuracy']
                for key in key_metrics:
                    if key in metrics:
                        value = metrics[key]
                        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
                    
            else:
                print(f"Results for {model_name}:")
                print("  error: No valid predictions")
                results[model_name] = {'error': 'No valid predictions'}
                
                # Still add empty columns for consistency
                predictions_df[f'{model_name}_nll'] = np.nan
                predictions_df[f'{model_name}_score'] = np.nan
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}
            
            # Still add empty columns for consistency
            predictions_df[f'{model_name}_nll'] = np.nan
            predictions_df[f'{model_name}_score'] = np.nan
    
    # Save results and predictions
    results_df = pd.DataFrame(results).T
    results_df.to_csv('benchmark_results.csv')
    print(f"\nResults saved to benchmark_results.csv")
    
    # Save predictions with experimental data
    predictions_df.to_csv('model_predictions.csv', index=False)
    print(f"Model predictions saved to model_predictions.csv")
    
    # Create a summary of predictions for easy inspection
    prediction_columns = [col for col in predictions_df.columns if col.endswith('_score')]

    # Create plots
    plot_results(results)
    
    
    # Print dataset statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples with padj: {len(exp_data)}")
    print(f"Significant binding (padj < 1e-05): {n_positive} ({n_positive/len(binary_labels)*100:.1f}%)")
    print(f"Non-significant binding: {n_negative} ({n_negative/len(binary_labels)*100:.1f}%)")
    
    # Print prediction statistics
    print(f"\n=== Prediction Statistics ===")
    for col in prediction_columns:
        if col in predictions_df.columns:
            valid_preds = predictions_df[col].dropna()
            if len(valid_preds) > 0:
                print(f"{col}: {len(valid_preds)} predictions, mean={valid_preds.mean():.4f}, std={valid_preds.std():.4f}")
            else:
                print(f"{col}: No valid predictions")
    
    print(f"\n=== Files Created ===")
    print(f"1. benchmark_results.csv - Summary metrics for each model")
    print(f"2. model_predictions.csv - Full dataset with all model predictions")
    print(f"3. model_predictions_summary.csv - First 100 samples for quick inspection")
    print(f"4. plots/ - Visualization plots")
    print(f"\nThese files can be used for:")
    print(f"- Comparing with third-party models")
    print(f"- Additional statistical analyses")
    print(f"- Custom visualizations")
    print(f"- Ensemble modeling")

if __name__ == "__main__":
    main() 