# language_model_evaluation_proper.py
# Language Model Evaluation with Random Masking using proper data loaders

import argparse
import torch
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import random
from collections import defaultdict
warnings.filterwarnings('ignore')

# Try to import CLP modules - handle import errors gracefully
try:
    sys.path.insert(0, '..')
    from model import psiCLM
    from data_clp import CollapseProteinDataset, dummy, create_idx
    import data_clp
    CLP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  CLP modules not available: {e}")
    CLP_AVAILABLE = False

# Try to import conditioned modules - handle import errors gracefully  
try:
    sys.path.insert(0, '../../conditioned')
    from model import Embedding2nd
    from data import Load_Dataset
    import data as conditioned_data
    sys.path.remove('../../conditioned')
    CONDITIONED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Conditioned modules not available: {e}")
    CONDITIONED_AVAILABLE = False

# Model configurations - updated paths to be relative
MODELS_CONFIG = {
    'optimized_composite_condition_1': {
        'path': '../saved_model/optimized_composite_condition_1/model_epoch_100',
        'cfg': {
            's_dim': 128,
            'z_dim': 64,
            's_in_dim': 21,
            'z_in_dim': 21,
            'N_elayers': 8
        },
        'name': 'Optimized Composite',
        'type': 'clp'
    },
    'staged_composite_condition_1': {
        'path': '../saved_model/staged_composite_condition_1/model_epoch_100',
        'cfg': {
            's_dim': 128,
            'z_dim': 64,
            's_in_dim': 21,
            'z_in_dim': 21,
            'N_elayers': 6
        },
        'name': 'Staged Composite',
        'type': 'clp'
    }
}

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_config, device):
    """Load a model from configuration"""
    if not CLP_AVAILABLE and model_config['type'] == 'clp':
        print(f"❌ CLP not available for {model_config['name']}")
        return None
    if not CONDITIONED_AVAILABLE and model_config['type'] == 'conditioned':
        print(f"❌ Conditioned not available for {model_config['name']}")
        return None
        
    try:
        if model_config['type'] == 'clp':
            model = psiCLM(model_config['cfg']).to(device)
        else:  # conditioned
            model = Embedding2nd(model_config['cfg']).to(device)
            
        # Check if model file exists
        if not os.path.exists(model_config['path']):
            print(f"❌ Model file not found: {model_config['path']}")
            return None
            
        model.load_state_dict(torch.load(model_config['path'], map_location=device), strict=False)
        model.eval()
        print(f"✅ Loaded {model_config['name']} from {model_config['path']}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_config['name']}: {e}")
        return None

def create_random_mask(sequence_length, mask_ratio=0.3, min_mask=1, max_mask=None):
    """
    Create random mask for a sequence
    Args:
        sequence_length: Length of sequence to mask
        mask_ratio: Proportion of sequence to mask (0.0 to 1.0)
        min_mask: Minimum number of positions to mask
        max_mask: Maximum number of positions to mask (None for no limit)
    Returns:
        List of positions to mask
    """
    if sequence_length == 0:
        return []
    
    # Calculate number of positions to mask
    num_to_mask = max(min_mask, int(sequence_length * mask_ratio))
    if max_mask is not None:
        num_to_mask = min(num_to_mask, max_mask)
    num_to_mask = min(num_to_mask, sequence_length)  # Can't mask more than sequence length
    
    # Randomly select positions to mask
    positions = list(range(sequence_length))
    mask_positions = random.sample(positions, num_to_mask)
    
    return sorted(mask_positions)

def apply_mask_to_sample(sample, mask_positions):
    """
    Apply random mask to a sample's CDR3 sequence (hd)
    Args:
        sample: Sample dictionary from dataset
        mask_positions: List of positions to mask in CDR3 sequence
    Returns:
        Modified sample with mask applied
    """
    sample_copy = sample.copy()
    
    # Create mask tensor
    hd_len = sample_copy['hd'].shape[0]
    mask = torch.zeros(hd_len)
    for pos in mask_positions:
        if 0 <= pos < hd_len:
            mask[pos] = 1
    sample_copy['mask'] = mask
    
    return sample_copy

def evaluate_model_recovery(model, sample, conditioning_info, original_sequence, model_type):
    """
    Evaluate model's ability to recover masked positions
    Args:
        model: The model to evaluate
        sample: Input sample with masking
        conditioning_info: List of conditioning variables
        original_sequence: Original CDR3 sequence for comparison
        model_type: 'clp' or 'conditioned'
    Returns:
        Dictionary with evaluation metrics
    """
    device = get_device()
    
    # Move sample to device
    sample_device = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample_device[key] = value.to(device)
        else:
            sample_device[key] = value
    
    try:
        with torch.no_grad():
            # Get predictions
            model_output = model(sample_device, computeloss=False, conditioning_info=conditioning_info)
            
            # Handle different model output formats
            if isinstance(model_output, tuple):
                predicted_probs = model_output[0]  # First element is usually the logits/probs
            elif isinstance(model_output, dict):
                predicted_probs = model_output.get('logits', model_output.get('probs', model_output))
            else:
                predicted_probs = model_output
            
            predicted_tokens = torch.argmax(predicted_probs, dim=-1)
            
            # Get NLL
            nll_output = model(sample_device, computeloss=True, conditioning_info=conditioning_info)
            if isinstance(nll_output, tuple):
                nll = nll_output[0]  # First element is usually the loss
            elif isinstance(nll_output, dict):
                nll = nll_output.get('loss', nll_output)
            else:
                nll = nll_output
            
            # Calculate recovery accuracy
            mask = sample_device['mask'].bool()
            masked_positions = torch.where(mask)[0]
            
            if len(masked_positions) == 0:
                return {
                    'recovery_accuracy': 0.0,
                    'nll': float('inf'),
                    'perplexity': float('inf'),
                    'num_masked': 0,
                    'num_recovered': 0,
                    'success': False
                }
            
            # Convert original sequence to tokens for comparison
            if model_type == 'clp':
                original_data = dummy(original_sequence)
                # Handle different return formats from dummy function
                if isinstance(original_data, tuple):
                    original_tokens = original_data[0]  # Usually the one-hot tensor
                elif isinstance(original_data, dict):
                    original_tokens = original_data.get('hd', original_data)
                else:
                    original_tokens = original_data
            else:  # conditioned
                original_tokens = conditioned_data.dummy(original_sequence, conditioned_data.aadic)
            
            # Ensure original_tokens is a tensor
            if not isinstance(original_tokens, torch.Tensor):
                print(f"Warning: original_tokens is not a tensor: {type(original_tokens)}")
                return {
                    'recovery_accuracy': 0.0,
                    'nll': nll.item() if hasattr(nll, 'item') else float(nll),
                    'perplexity': torch.exp(nll).item() if hasattr(nll, 'item') else float('inf'),
                    'num_masked': len(masked_positions),
                    'num_recovered': 0,
                    'success': False
                }
            
            # Compare predictions with original at masked positions
            correct_predictions = 0
            for pos in masked_positions:
                pos_item = pos.item() if hasattr(pos, 'item') else pos
                if pos_item < original_tokens.shape[0] and pos_item < predicted_tokens.shape[0]:
                    # Get the true token at this position
                    if len(original_tokens.shape) > 1:  # One-hot encoded
                        true_token = torch.argmax(original_tokens[pos_item])
                    else:  # Already token indices
                        true_token = original_tokens[pos_item]
                    
                    predicted_token = predicted_tokens[pos_item]
                    
                    if predicted_token == true_token:
                        correct_predictions += 1
            
            recovery_accuracy = correct_predictions / len(masked_positions)
            
            # Safe NLL and perplexity calculation
            try:
                nll_value = nll.item() if hasattr(nll, 'item') else float(nll)
                perplexity = torch.exp(nll).item() if hasattr(nll, 'item') else float('inf')
            except:
                nll_value = float('inf')
                perplexity = float('inf')
            
            return {
                'recovery_accuracy': recovery_accuracy,
                'nll': nll_value,
                'perplexity': perplexity,
                'num_masked': len(masked_positions),
                'num_recovered': correct_predictions,
                'success': True
            }
            
    except Exception as e:
        print(f"❌ Error in model evaluation: {e}")
        return {
            'recovery_accuracy': 0.0,
            'nll': float('inf'),
            'perplexity': float('inf'),
            'num_masked': 0,
            'num_recovered': 0,
            'success': False
        }

def prepare_data_for_lm_evaluation(exp_data_path, output_path):
    """Prepare experimental data for language model evaluation"""
    print("Preparing data for language model evaluation...")
    
    # Try different data file locations
    data_files = [
        exp_data_path,
        './exp_data.csv',
        './formatted_eval_data.csv',
        './complete_model_predictions.csv'
    ]
    
    exp_data = None
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            exp_data = pd.read_csv(data_file)
            break
    
    if exp_data is None:
        print(f"❌ Could not find data file. Tried: {data_files}")
        return None
    
    print(f"Loaded {len(exp_data)} experimental samples")
    
    # Check available columns
    print(f"Available columns: {list(exp_data.columns)}")
    
    # Try to find CDR3 column with different possible names
    cdr3_cols = ['cdr3_b', 'cdr3', 'hd', 'CDR3', 'CDR3_seq']
    cdr3_col = None
    for col in cdr3_cols:
        if col in exp_data.columns:
            cdr3_col = col
            break
    
    if cdr3_col is None:
        print(f"❌ Could not find CDR3 column. Available columns: {list(exp_data.columns)}")
        return None
    
    # Filter to samples with valid CDR3
    valid_data = exp_data[
        exp_data[cdr3_col].notna() & 
        (exp_data[cdr3_col] != '') &
        (exp_data[cdr3_col].str.len() >= 5)  # Minimum length for meaningful masking
    ].copy()
    
    print(f"Found {len(valid_data)} valid samples with CDR3")
    
    # Map to expected format - handle missing columns gracefully
    formatted_data = []
    for _, row in valid_data.iterrows():
        formatted_row = [
            str(row.get('peptide', row.get('pep', ''))) if pd.notna(row.get('peptide', row.get('pep', ''))) else '',      # pep
            str(row.get('mhc', '')) if pd.notna(row.get('mhc', '')) else '',             # mhc  
            str(row.get('l_v', row.get('lv', ''))) if pd.notna(row.get('l_v', row.get('lv', ''))) else '',             # lv
            str(row.get('l_j', row.get('lj', ''))) if pd.notna(row.get('l_j', row.get('lj', ''))) else '',             # lj
            str(row.get('h_v', row.get('hv', ''))) if pd.notna(row.get('h_v', row.get('hv', ''))) else '',             # hv
            str(row[cdr3_col]) if pd.notna(row[cdr3_col]) else '',       # hd (target)
            str(row.get('h_j', row.get('hj', ''))) if pd.notna(row.get('h_j', row.get('hj', ''))) else '',             # hj
        ]
        formatted_data.append(formatted_row)
    
    # Create DataFrame with expected columns
    formatted_df = pd.DataFrame(formatted_data, columns=['pep', 'mhc', 'lv', 'lj', 'hv', 'hd', 'hj'])
    
    # Save formatted data
    formatted_df.to_csv(output_path, index=False)
    print(f"Saved formatted data to {output_path}")
    
    return output_path

def run_language_model_evaluation(models, data_path, conditioning_info, 
                                mask_ratios=[0.1, 0.3, 0.5], 
                                num_samples=500, 
                                trials_per_sample=3):
    """
    Run comprehensive language model evaluation
    Args:
        models: Dictionary of loaded models
        data_path: Path to formatted data
        conditioning_info: List of conditioning variables
        mask_ratios: List of masking ratios to test
        num_samples: Number of samples to evaluate
        trials_per_sample: Number of random masking trials per sample
    Returns:
        DataFrame with evaluation results
    """
    print(f"🔍 Running Language Model Evaluation...")
    print(f"📊 Mask ratios: {mask_ratios}")
    print(f"📊 Samples: {num_samples}, Trials per sample: {trials_per_sample}")
    
    results = []
    
    # Evaluate each model
    for model_key, model in models.items():
        if model is None:
            continue
            
        model_config = MODELS_CONFIG[model_key]
        print(f"\n🤖 Evaluating {model_config['name']}...")
        
        # Create appropriate dataset
        try:
            if model_config['type'] == 'clp' and CLP_AVAILABLE:
                dataset = CollapseProteinDataset(data_path)
            elif model_config['type'] == 'conditioned' and CONDITIONED_AVAILABLE:
                dataset = Load_Dataset(data_path)
            else:
                print(f"⚠️  Dataset loader not available for {model_config['name']}")
                continue
        except Exception as e:
            print(f"❌ Failed to create dataset for {model_config['name']}: {e}")
            continue
        
        # Limit samples if needed
        total_samples = min(len(dataset), num_samples)
        
        # Test each mask ratio
        for mask_ratio in mask_ratios:
            print(f"  📝 Testing mask ratio: {mask_ratio}")
            
            # Evaluate samples with different random masks
            for sample_idx in tqdm(range(total_samples), desc=f"Mask {mask_ratio}"):
                
                try:
                    # Get original sample
                    original_sample = dataset[sample_idx]
                    original_cdr3 = None
                    
                    # Extract original CDR3 sequence
                    if model_config['type'] == 'clp':
                        # For CLP, reconstruct sequence from one-hot
                        hd_tensor = original_sample['hd']
                        if hd_tensor.shape[0] > 0:
                            original_cdr3 = ''.join([list(data_clp.AA_DICT.keys())[torch.argmax(hd_tensor[i]).item()] 
                                                    for i in range(hd_tensor.shape[0])])
                    else:  # conditioned
                        # For conditioned, reconstruct from one-hot
                        hd_tensor = original_sample['hd']
                        if hd_tensor.shape[0] > 0:
                            aa_keys = list(conditioned_data.aadic.keys())
                            original_cdr3 = ''.join([aa_keys[torch.argmax(hd_tensor[i]).item()] 
                                                   for i in range(hd_tensor.shape[0])])
                    
                    if not original_cdr3 or len(original_cdr3) < 3:
                        continue
                    
                    cdr3_length = len(original_cdr3)
                    
                    # Run multiple trials with different random masks
                    for trial in range(trials_per_sample):
                        # Create random mask
                        mask_positions = create_random_mask(cdr3_length, mask_ratio=mask_ratio)
                        
                        if not mask_positions:  # Skip if no positions to mask
                            continue
                        
                        # Apply mask to sample
                        masked_sample = apply_mask_to_sample(original_sample, mask_positions)
                        
                        # Evaluate model
                        eval_result = evaluate_model_recovery(
                            model, masked_sample, conditioning_info, 
                            original_cdr3, model_config['type']
                        )
                        
                        # Store results
                        result = {
                            'model': model_config['name'],
                            'model_key': model_key,
                            'model_type': model_config['type'],
                            'sample_idx': sample_idx,
                            'trial': trial,
                            'mask_ratio': mask_ratio,
                            'sequence_length': cdr3_length,
                            'original_sequence': original_cdr3,
                            **eval_result
                        }
                        results.append(result)
                        
                except Exception as e:
                    print(f"  Error processing sample {sample_idx}: {e}")
                    continue
    
    return pd.DataFrame(results)

def create_language_model_plots(results_df, output_dir):
    """Create comprehensive plots for language model evaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful results
    success_results = results_df[results_df['success'] == True]
    
    if len(success_results) == 0:
        print("❌ No successful results to plot")
        return
    
    # 1. Recovery Accuracy by Model and Mask Ratio
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=success_results, x='mask_ratio', y='recovery_accuracy', hue='model')
    plt.title('Recovery Accuracy by Model and Mask Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Mask Ratio')
    plt.ylabel('Recovery Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_accuracy_by_mask_ratio.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Model Comparison Summary
    model_summary = success_results.groupby('model').agg({
        'recovery_accuracy': ['mean', 'std', 'count'],
        'nll': lambda x: np.mean(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.nan,
        'perplexity': lambda x: np.mean(x[np.isfinite(x) & (x < 1000)]) if np.any(np.isfinite(x) & (x < 1000)) else np.nan
    }).round(4)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Recovery accuracy
    models = model_summary.index
    acc_means = model_summary[('recovery_accuracy', 'mean')]
    acc_stds = model_summary[('recovery_accuracy', 'std')]
    
    axes[0].bar(models, acc_means, yerr=acc_stds, capsize=5, alpha=0.7)
    axes[0].set_title('Mean Recovery Accuracy by Model')
    axes[0].set_ylabel('Recovery Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # NLL - fix the array issue
    nll_means = model_summary['nll']
    # Convert to list to handle potential array issues
    nll_values = [float(val) if not np.isnan(val) else 0.0 for val in nll_means.values]
    axes[1].bar(models, nll_values, alpha=0.7)
    axes[1].set_title('Mean NLL by Model')
    axes[1].set_ylabel('NLL')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Perplexity - fix the array issue
    perp_means = model_summary['perplexity']
    # Convert to list to handle potential array issues
    perp_values = [float(val) if not np.isnan(val) else 0.0 for val in perp_means.values]
    axes[2].bar(models, perp_values, alpha=0.7)
    axes[2].set_title('Mean Perplexity by Model')
    axes[2].set_ylabel('Perplexity')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Language Model Evaluation with Random Masking')
    parser.add_argument('--data_path', type=str, default='./exp_data.csv',
                       help='Path to experimental data')
    parser.add_argument('--output_dir', type=str, default='./nll',
                       help='Output directory for results')
    parser.add_argument('--conditioning_info', type=str, nargs='+', 
                       default=['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'],
                       help='Conditioning information')
    parser.add_argument('--mask_ratios', type=float, nargs='+',
                       default=[0.1, 0.3, 0.5],
                       help='Mask ratios to test')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='Number of samples to evaluate')
    parser.add_argument('--trials_per_sample', type=int, default=2,
                       help='Number of random masking trials per sample')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--skip_model_loading', action='store_true',
                       help='Skip model loading and focus on data preparation')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = get_device()
    print(f"🚀 Using device: {device}")
    print(f"📊 Conditioning info: {args.conditioning_info}")
    
    # Prepare data
    formatted_data_path = prepare_data_for_lm_evaluation(args.data_path, './formatted_lm_data.csv')
    
    if formatted_data_path is None:
        print("❌ Failed to prepare data. Exiting.")
        return
    
    if args.skip_model_loading:
        print("⚠️  Skipping model loading as requested")
        print(f"✅ Data prepared successfully: {formatted_data_path}")
        return
    
    # Load all models
    print("\n🔧 Loading models...")
    models = {}
    for model_key, model_config in MODELS_CONFIG.items():
        models[model_key] = load_model(model_config, device)
    
    successful_models = sum(1 for model in models.values() if model is not None)
    print(f"✅ Successfully loaded {successful_models}/{len(MODELS_CONFIG)} models")
    
    if successful_models == 0:
        print("❌ No models loaded successfully. You can:")
        print("   1. Check model paths in MODELS_CONFIG")
        print("   2. Run with --skip_model_loading to just prepare data")
        print("   3. Use the prepared data with other evaluation scripts")
        return
    
    # Run evaluation
    results_df = run_language_model_evaluation(
        models=models,
        data_path=formatted_data_path,
        conditioning_info=args.conditioning_info,
        mask_ratios=args.mask_ratios,
        num_samples=args.num_samples,
        trials_per_sample=args.trials_per_sample
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(args.output_dir, 'lm_evaluation_results.csv'), index=False)
    
    # Create plots
    create_language_model_plots(results_df, args.output_dir)
    
    # Print summary
    success_results = results_df[results_df['success'] == True]
    
    print(f"\n📈 LANGUAGE MODEL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"📊 Total evaluations: {len(results_df)}")
    print(f"📊 Successful evaluations: {len(success_results)}")
    print(f"📊 Success rate: {len(success_results)/len(results_df)*100:.1f}%")
    
    if len(success_results) > 0:
        model_summary = success_results.groupby('model').agg({
            'recovery_accuracy': ['mean', 'std', 'count'],
            'nll': lambda x: np.mean(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else np.nan,
            'perplexity': lambda x: np.mean(x[np.isfinite(x) & (x < 1000)]) if np.any(np.isfinite(x) & (x < 1000)) else np.nan
        }).round(4)
        
        print(f"\n🏆 MODEL PERFORMANCE SUMMARY:")
        for model in model_summary.index:
            print(f"📋 {model}:")
            print(f"   Recovery Accuracy: {model_summary.loc[model, ('recovery_accuracy', 'mean')]:.4f} ± {model_summary.loc[model, ('recovery_accuracy', 'std')]:.4f}")
            
            # Safe formatting for NLL and Perplexity
            nll_val = model_summary.loc[model, 'nll']
            perp_val = model_summary.loc[model, 'perplexity']
            
            if isinstance(nll_val, (int, float)) and not np.isnan(nll_val):
                print(f"   Mean NLL: {float(nll_val):.4f}")
            else:
                print(f"   Mean NLL: N/A")
                
            if isinstance(perp_val, (int, float)) and not np.isnan(perp_val):
                print(f"   Mean Perplexity: {float(perp_val):.4f}")
            else:
                print(f"   Mean Perplexity: N/A")
                
            print(f"   Evaluations: {model_summary.loc[model, ('recovery_accuracy', 'count')]}")
    
    print(f"\n✅ Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 