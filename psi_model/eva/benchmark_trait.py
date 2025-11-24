#!/usr/bin/env python3
"""
Benchmark all models in clp/eva/models/ on trait test dataset
Handles both original architecture (conditioned/) and updated architecture (clp/)
Uses direct binding labels (0/1) from the trait test dataset
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

def prepare_trait_data_for_evaluation(trait_data_path, output_path):
    """
    Prepare trait test data in the format expected by the data loaders
    
    For CLP models: expects columns ['pep', 'mhc', 'lv', 'lj', 'hv', 'hd', 'hj']
    """
    print("Preparing trait data for evaluation...")
    
    # Load trait test data
    trait_data = pd.read_csv(trait_data_path)
    print(f"Loaded {len(trait_data)} trait test samples")
    
    # Filter to samples with valid CDR3
    valid_data = trait_data[
        trait_data['cdr3_b'].notna() & 
        (trait_data['cdr3_b'] != '')
    ].copy()
    
    print(f"Found {len(valid_data)} valid samples with CDR3")
    
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
    
    # Also save the binding labels for evaluation
    binding_labels = valid_data[['binding_label']].reset_index(drop=True)
    labels_path = output_path.replace('.csv', '_labels.csv')
    binding_labels.to_csv(labels_path, index=False)
    print(f"Saved binding labels to {labels_path}")
    
    return output_path, labels_path

def calculate_classification_metrics(y_true, y_scores):
    """
    Calculate classification metrics using direct binary labels
    y_true: binary labels (1 = binding, 0 = non-binding)
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
    
    # 2. Summary heatmap
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
    trait_data_path = 'trait/processed_trait_binding_data/trait_test.csv'  # Input trait test data
    formatted_data_path = 'formatted_trait_eval_data.csv'  # Output formatted for datasets
    
    formatted_data_path, labels_path = prepare_trait_data_for_evaluation(trait_data_path, formatted_data_path)
    
    # Load binding labels for evaluation
    labels_df = pd.read_csv(labels_path)
    binary_labels = labels_df['binding_label'].astype(int)
    n_positive = np.sum(binary_labels)
    n_negative = len(binary_labels) - n_positive
    
    print(f"\nBinary classification target (direct labels):")
    print(f"  Positive samples (binding): {n_positive}")
    print(f"  Negative samples (non-binding): {n_negative}")
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
        },
        'condition_1': {
            'path': '../saved_model/condition_1/model_epoch_690',
            'config': {
                's_dim': 128,
                'z_dim': 64,
                's_in_dim': 21,
                'z_in_dim': 2,
                'N_elayers': 18,
            },
            'architecture': 'clp',
            'dataset': clp_dataset,
            'conditioning_info': ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']  # Everything except hd
        },
        'bkcomposite_condition_1': {
            'path': '../saved_model/bkcomposite_condition_1/model_epoch_450',
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
        'paired_finetune': {
            'path': '../pretrain/saved_model/paired_finetune/best_paired_model',
            'config': {
                's_dim': 128,
                'z_dim': 64,
                's_in_dim': 21,
                'z_in_dim': 2,
                'N_elayers': 4,  # Pretrained models typically have fewer layers
            },
            'architecture': 'clp',
            'dataset': clp_dataset,
            'conditioning_info': ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']  # Everything except hd
        },
        'cdr3_pretrain': {
            'path': '../pretrain/saved_model/cdr3_pretrain/cdr3_model_epoch_500',
            'config': {
                's_dim': 128,
                'z_dim': 64,
                's_in_dim': 21,
                'z_in_dim': 2,
                'N_elayers': 4,  # Pretrained models typically have fewer layers
            },
            'architecture': 'clp',
            'dataset': clp_dataset,
            'conditioning_info': ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']  # Everything except hd
        }
    }
    
    # Initialize predictions DataFrame to store all model scores
    predictions_df = pd.read_csv(formatted_data_path)
    labels_df = pd.read_csv(labels_path)
    predictions_df = pd.concat([predictions_df, labels_df], axis=1)
    
    results = {}
    
    for model_name, model_info in model_configs.items():
        print(f"\n=== Evaluating {model_name} ===")
        
        try:
            # Load model
            model = load_updated_model(model_info['path'], model_info['config'])
            
            # Evaluate using CLP method
            nlls = evaluate_clp_model(model, model_info['dataset'], model_info['conditioning_info'])
            
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
    results_df.to_csv('trait_benchmark_results.csv')
    print(f"\nResults saved to trait_benchmark_results.csv")
    
    # Save predictions with trait data
    predictions_df.to_csv('trait_model_predictions.csv', index=False)
    print(f"Model predictions saved to trait_model_predictions.csv")
    
    # Create plots
    plot_results(results)
    
    # Print dataset statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Total trait test samples: {len(clp_dataset)}")
    print(f"Binding samples: {n_positive} ({n_positive/len(binary_labels)*100:.1f}%)")
    print(f"Non-binding samples: {n_negative} ({n_negative/len(binary_labels)*100:.1f}%)")
    
    # Print prediction statistics
    prediction_columns = [col for col in predictions_df.columns if col.endswith('_score')]
    print(f"\n=== Prediction Statistics ===")
    for col in prediction_columns:
        if col in predictions_df.columns:
            valid_preds = predictions_df[col].dropna()
            if len(valid_preds) > 0:
                print(f"{col}: {len(valid_preds)} predictions, mean={valid_preds.mean():.4f}, std={valid_preds.std():.4f}")
            else:
                print(f"{col}: No valid predictions")
    
    print(f"\n=== Files Created ===")
    print(f"1. trait_benchmark_results.csv - Summary metrics for each model")
    print(f"2. trait_model_predictions.csv - Full dataset with all model predictions")
    print(f"3. plots/ - Visualization plots")
    print(f"\nDataset info:")
    print(f"- Single peptide-MHC combination: ELAGIGILTV + HLA-A*02:01")
    print(f"- Balanced test set with 1:1.5 positive:negative ratio")
    print(f"- All samples from A0201_ELAGIGILTV_MART-1_Cancer trait dataset")

if __name__ == "__main__":
    main() 