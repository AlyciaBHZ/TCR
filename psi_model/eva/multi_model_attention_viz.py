# multi_model_attention_viz.py
# Visualize attention patterns across different model architectures for comparison

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import CLP modules - handle import errors gracefully
try:
    sys.path.insert(0, '..')
    from model import psiCLM
    from data_clp import CollapseProteinDataset, dummy
    import data_clp
    CLP_AVAILABLE = True
    print("✅ CLP model imports available")
except ImportError as e:
    print(f"⚠️ Warning: CLP model imports not available: {e}")
    CLP_AVAILABLE = False

# Try to import conditioned modules - handle import errors gracefully  
try:
    sys.path.insert(0, '../../conditioned')
    from model import Embedding2nd
    from data import Load_Dataset
    import data as conditioned_data
    sys.path.remove('../../conditioned')
    CONDITIONED_AVAILABLE = True
    print("✅ Conditioned model imports available")
except ImportError as e:
    print(f"⚠️ Warning: Conditioned model imports not available: {e}")
    CONDITIONED_AVAILABLE = False

import seaborn as sns

# Model configurations - updated to match reference script pattern
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
        'type': 'clp',
        'num_layers': 8
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
        'type': 'clp',
        'num_layers': 6
    },
    'bkcomposite_condition_1': {
        'path': '../saved_model/bkcomposite_condition_1/model_epoch_100',
        'cfg': {
            's_dim': 128,
            'z_dim': 64,
            's_in_dim': 21,
            'z_in_dim': 21,
            'N_elayers': 4
        },
        'name': 'Background Composite',
        'type': 'clp',
        'num_layers': 4
    }
}

def get_device():
    """Get available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_config, device):
    """Load a model from configuration with proper error handling"""
    if not CLP_AVAILABLE and model_config['type'] == 'clp':
        print(f"❌ CLP not available for {model_config['name']}")
        return None
    if not CONDITIONED_AVAILABLE and model_config['type'] == 'conditioned':
        print(f"❌ Conditioned not available for {model_config['name']}")
        return None
        
    try:
        # Check if model file exists
        if not os.path.exists(model_config['path']):
            print(f"❌ Model file not found: {model_config['path']}")
            return None
            
        if model_config['type'] == 'clp':
            model = psiCLM(model_config['cfg']).to(device)
        else:  # conditioned
            model = Embedding2nd(model_config['cfg']).to(device)
            
        model.load_state_dict(torch.load(model_config['path'], map_location=device), strict=False)
        model.eval()
        print(f"✅ Loaded {model_config['name']} from {model_config['path']}")
        return model
    except Exception as e:
        print(f"❌ Failed to load {model_config['name']}: {e}")
        return None

def prepare_data_for_attention_viz(data_path, output_path):
    """Prepare data for attention visualization using reference script pattern"""
    print("Preparing data for attention visualization...")
    
    # Try different data file locations - prioritize tst.csv
    data_files = [
        data_path,
        "../../data/tst.csv",
        "../../data/val.csv", 
        "./exp_data.csv",
        "./formatted_eval_data.csv",
        "./complete_model_predictions.csv"
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
        (exp_data[cdr3_col].str.len() >= 5)  # Minimum length
    ].copy()
    
    print(f"Found {len(valid_data)} valid samples with CDR3")
    
    # Map to expected format - handle missing columns gracefully
    formatted_data = []
    for _, row in valid_data.iterrows():
        formatted_row = [
            str(row.get('peptide', row.get('pep', ''))) if pd.notna(row.get('peptide', row.get('pep', ''))) else '',
            str(row.get('mhc', '')) if pd.notna(row.get('mhc', '')) else '',
            str(row.get('l_v', row.get('lv', ''))) if pd.notna(row.get('l_v', row.get('lv', ''))) else '',
            str(row.get('l_j', row.get('lj', ''))) if pd.notna(row.get('l_j', row.get('lj', ''))) else '',
            str(row.get('h_v', row.get('hv', ''))) if pd.notna(row.get('h_v', row.get('hv', ''))) else '',
            str(row[cdr3_col]) if pd.notna(row[cdr3_col]) else '',
            str(row.get('h_j', row.get('hj', ''))) if pd.notna(row.get('h_j', row.get('hj', ''))) else '',
        ]
        formatted_data.append(formatted_row)
    
    # Create DataFrame with expected columns
    formatted_df = pd.DataFrame(formatted_data, columns=['pep', 'mhc', 'lv', 'lj', 'hv', 'hd', 'hj'])
    
    # Save formatted data
    formatted_df.to_csv(output_path, index=False)
    print(f"Saved formatted data to {output_path}")
    
    return output_path

def prepare_sample_for_model(sample, model_type, device):
    """Prepare sample data for different model types"""
    sample_device = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample_device[key] = value.to(device)
        elif value is not None:
            sample_device[key] = torch.tensor(value).to(device)
        else:
            sample_device[key] = None
    return sample_device

def get_attention_from_model(model, sample, model_type, conditioning_info, device):
    """Extract attention patterns from different model types"""
    sample_device = prepare_sample_for_model(sample, model_type, device)
    
    with torch.no_grad():
        try:
            if model_type == 'clp':
                # CLP models
                output = model(sample_device, computeloss=False, conditioning_info=conditioning_info)
                if isinstance(output, tuple) and len(output) >= 2:
                    pred_logits, attn_traces = output
                    return attn_traces
                else:
                    print(f"⚠️ CLP model output format unexpected: {type(output)}")
                    return None
                    
            elif model_type == 'conditioned':
                # Conditioned models
                try:
                    output = model(sample_device, computeloss=False)
                    
                    if isinstance(output, tuple) and len(output) >= 2:
                        pred_logits, attn_traces = output
                        return attn_traces
                    elif hasattr(model, 'get_attention'):
                        return model.get_attention()
                    else:
                        print(f"⚠️ Conditioned model doesn't return attention traces")
                        return None
                except Exception as e:
                    print(f"⚠️ Conditioned model forward pass failed: {e}")
                    return None
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting attention from {model_type} model: {e}")
            return None

def get_token_boundaries(sample, conditioning_info):
    """Get token boundaries for different regions"""
    boundaries = {}
    pointer = 1  # skip collapse token
    
    # Order matters - follow the model's expected sequence order
    sequence_order = ['hd'] + conditioning_info
    
    for field in sequence_order:
        if field in sample and sample[field] is not None:
            seq = sample[field]
            if isinstance(seq, torch.Tensor):
                if seq.ndim == 2:
                    L = seq.shape[0]
                elif seq.ndim == 1:
                    L = seq.shape[0]
                else:
                    continue
            elif isinstance(seq, (list, np.ndarray)):
                L = len(seq)
            else:
                continue
                
            if L > 0:
                boundaries[field] = (pointer, pointer + L)
                pointer += L
    
    return boundaries

def create_attention_comparison_plot(attention_data, boundaries, sample_idx, layer_idx, output_dir):
    """Create side-by-side attention heatmaps with improved visibility"""
    successful_models = [(name, attn) for name, attn in attention_data.items() if attn is not None]
    n_models = len(successful_models)
    
    if n_models == 0:
        print(f"⚠️ No successful models for sample {sample_idx}, layer {layer_idx}")
        return
    
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 8))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    elif n_models == 0:
        return
    
    # Define attention types
    attention_types = [
        ("First Token → All", 0),  # Collapse token attending to all
        ("CDR3 → All", slice(boundaries.get('hd', (1, 2))[0], boundaries.get('hd', (1, 2))[1]))  # CDR3 region attending to all
    ]
    
    for model_idx, (model_name, attention_matrix) in enumerate(successful_models):
        # Convert to CPU numpy for processing
        if isinstance(attention_matrix, torch.Tensor):
            attention_matrix = attention_matrix.detach().cpu().numpy()
        
        # Handle different tensor shapes - extract 2D attention matrix
        original_shape = attention_matrix.shape
        if attention_matrix.ndim == 4:
            # (batch, num_heads, seq_len, seq_len)
            attention_matrix = attention_matrix[0, 0]  # First batch, first head
        elif attention_matrix.ndim == 3:
            # (num_heads, seq_len, seq_len) or (batch, seq_len, seq_len)
            attention_matrix = attention_matrix[0]
        elif attention_matrix.ndim != 2:
            print(f"⚠️ Warning: Unexpected attention matrix shape {original_shape} for {model_name}")
            continue
        
        print(f"📊 Processing {model_name}: {original_shape} → {attention_matrix.shape}")
        
        for plot_idx, (attn_type, row_selection) in enumerate(attention_types):
            ax = axes[plot_idx, model_idx]
            
            try:
                if isinstance(row_selection, slice):
                    # Average attention across CDR3 region
                    if row_selection.start < attention_matrix.shape[0] and row_selection.stop <= attention_matrix.shape[0]:
                        attn_weights = attention_matrix[row_selection, :].mean(axis=0)
                        attn_to_plot = attn_weights.reshape(1, -1)
                    else:
                        print(f"⚠️ CDR3 region {row_selection} out of bounds for {model_name}")
                        continue
                else:
                    # Single row (collapse token)
                    if row_selection < attention_matrix.shape[0]:
                        attn_to_plot = attention_matrix[row_selection:row_selection+1, :]
                    else:
                        print(f"⚠️ Collapse token index {row_selection} out of bounds for {model_name}")
                        continue
                
                # Improved color scaling
                vmin, vmax = np.percentile(attn_to_plot.flatten(), [1, 99])
                if vmax - vmin < 1e-8:
                    vmax = vmin + 1e-8
                
                # Use log scale for very small values
                plot_data = attn_to_plot.copy()
                use_log_scale = np.max(plot_data) < 0.01
                
                if use_log_scale:
                    plot_data = np.log10(plot_data + 1e-10)
                    vmin, vmax = np.percentile(plot_data.flatten(), [1, 99])
                    title_suffix = " (log scale)"
                else:
                    title_suffix = ""
                
                # Create heatmap
                im = ax.imshow(plot_data, cmap='plasma', aspect='auto', 
                              vmin=vmin, vmax=vmax, interpolation='nearest')
                
                # Add region boundaries
                y_min, y_max = ax.get_ylim()
                boundary_colors = ['cyan', 'yellow', 'lime', 'orange', 'red', 'purple']
                for i, (region, (start, end)) in enumerate(boundaries.items()):
                    color = boundary_colors[i % len(boundary_colors)]
                    ax.axvline(x=start-0.5, color=color, linestyle='--', alpha=0.8, linewidth=1.5)
                    ax.axvline(x=end-0.5, color=color, linestyle='--', alpha=0.8, linewidth=1.5)
                    
                    # Add region labels
                    if plot_idx == 0:
                        ax.text((start + end - 1) / 2, y_max + 0.15, region.upper(), 
                               ha='center', va='bottom', fontsize=8, fontweight='bold',
                               color=color, bbox=dict(boxstyle='round,pad=0.2', 
                                                    facecolor='black', alpha=0.5))
                
                # Add statistics
                max_attn = np.max(attn_to_plot)
                mean_attn = np.mean(attn_to_plot)
                
                ax.set_title(f'{model_name}\n{attn_type}{title_suffix}\n'
                            f'Max: {max_attn:.1e}, Mean: {mean_attn:.1e}', 
                            fontsize=9, fontweight='bold')
                ax.set_xlabel('Token Index', fontsize=8)
                ax.set_ylabel('Attention Weight', fontsize=8)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.ax.tick_params(labelsize=7)
                if use_log_scale:
                    cbar.set_label('log10(Attention)', fontsize=7)
                else:
                    cbar.set_label('Attention', fontsize=7)
                
                ax.tick_params(axis='both', which='major', labelsize=7)
                
            except Exception as e:
                print(f"❌ Error plotting {attn_type} for {model_name}: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name} - Error')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'attention_comparison_sample{sample_idx}_layer{layer_idx}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📊 Saved {filename}")
    
    return filename

def compute_attention_entropy(attn_matrix):
    """Compute entropy of attention distribution"""
    if attn_matrix is None:
        return None
    
    # Convert to CPU numpy if it's a tensor
    if isinstance(attn_matrix, torch.Tensor):
        attn_matrix = attn_matrix.detach().cpu().numpy()
    
    # Handle different shapes
    if attn_matrix.ndim > 2:
        # Take first batch/head
        while attn_matrix.ndim > 2:
            attn_matrix = attn_matrix[0]
    
    if attn_matrix.ndim != 2:
        return None
    
    # Compute entropy for each row
    entropies = []
    for i in range(attn_matrix.shape[0]):
        row = attn_matrix[i]
        # Normalize to probabilities
        probs = row / (np.sum(row) + 1e-10)
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
    
    return {
        'mean_entropy': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'collapse_entropy': entropies[0] if len(entropies) > 0 else 0,
        'max_entropy': np.max(entropies),
        'min_entropy': np.min(entropies)
    }

def analyze_attention_patterns(models, dataset, conditioning_info, device, output_dir, sample_ids=None):
    """Analyze attention patterns across different models and layers"""
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = {
        'model': [],
        'sample_idx': [],
        'layer': [],
        'collapse_entropy': [],
        'mean_entropy': [],
        'max_entropy': []
    }
    
    # Use specific sample IDs if provided, otherwise use first few samples
    if sample_ids is None:
        sample_ids = list(range(min(3, len(dataset))))
    else:
        # Ensure sample IDs are within dataset bounds
        sample_ids = [idx for idx in sample_ids if 0 <= idx < len(dataset)]
    
    print(f"📊 Analyzing samples: {sample_ids}")
    
    for sample_idx in sample_ids:
        sample = dataset[sample_idx]
        boundaries = get_token_boundaries(sample, conditioning_info)
        
        print(f"\n🔍 Analyzing sample {sample_idx}")
        print(f"Boundaries: {boundaries}")
        
        # Analyze across different layers (limit to reasonable number)
        max_layers = 8  # Reasonable limit for visualization
        
        for layer_idx in range(max_layers):
            print(f"\n  Layer {layer_idx}:")
            attention_data = {}
            
            for model_key, model_config in MODELS_CONFIG.items():
                model = models.get(model_key)
                if model is None:
                    print(f"    ❌ {model_key}: Model not loaded")
                    continue
                
                # Check if layer exists
                if layer_idx >= model_config['num_layers']:
                    print(f"    ⚠️ {model_key}: Layer {layer_idx} exceeds model layers ({model_config['num_layers']})")
                    continue
                
                # Get attention
                attn_traces = get_attention_from_model(
                    model, sample, model_config['type'], conditioning_info, device
                )
                
                if attn_traces and layer_idx < len(attn_traces):
                    attn = attn_traces[layer_idx]
                    attention_data[model_config['name']] = attn
                    print(f"    ✅ {model_config['name']}: Got attention shape {attn.shape}")
                    
                    # Compute entropy metrics
                    entropy_metrics = compute_attention_entropy(attn)
                    if entropy_metrics:
                        results_summary['model'].append(model_config['name'])
                        results_summary['sample_idx'].append(sample_idx)
                        results_summary['layer'].append(layer_idx)
                        results_summary['collapse_entropy'].append(entropy_metrics['collapse_entropy'])
                        results_summary['mean_entropy'].append(entropy_metrics['mean_entropy'])
                        results_summary['max_entropy'].append(entropy_metrics['max_entropy'])
                else:
                    print(f"    ❌ {model_config['name']}: No attention for layer {layer_idx}")
            
            # Create comparison plot
            if attention_data:
                create_attention_comparison_plot(attention_data, boundaries, sample_idx, layer_idx, output_dir)
    
    # Save summary
    if results_summary['model']:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(os.path.join(output_dir, 'attention_analysis_summary.csv'), index=False)
        print(f"\n📊 Saved attention analysis summary with {len(summary_df)} records")
        return summary_df
    else:
        print("\n⚠️ No attention data collected")
        return None

def main():
    parser = argparse.ArgumentParser(description='Multi-Model Attention Visualization')
    parser.add_argument('--data_path', type=str, default='../../data/tst.csv',
                       help='Path to experimental data')
    parser.add_argument('--output_dir', type=str, default='./nll/attention_analysis',
                       help='Output directory for results')
    parser.add_argument('--conditioning_info', type=str, nargs='+', 
                       default=['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'],
                       help='Conditioning information')
    parser.add_argument('--sample_ids', type=int, nargs='+',
                       default=[0, 2, 15],
                       help='Specific sample IDs to analyze')
    parser.add_argument('--skip_model_loading', action='store_true',
                       help='Skip model loading and focus on data preparation')
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"🚀 Using device: {device}")
    print(f"📊 Conditioning info: {args.conditioning_info}")
    
    # Prepare data
    formatted_data_path = prepare_data_for_attention_viz(args.data_path, './formatted_attention_data.csv')
    
    if formatted_data_path is None:
        print("❌ Failed to prepare data. Exiting.")
        return
    
    if args.skip_model_loading:
        print("⚠️ Skipping model loading as requested")
        print(f"✅ Data prepared successfully: {formatted_data_path}")
        return
    
    # Create dataset
    try:
        if CLP_AVAILABLE:
            dataset = CollapseProteinDataset(formatted_data_path)
            print(f"✅ Loaded dataset with {len(dataset)} samples")
        else:
            print("❌ CLP not available for dataset loading")
            return
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return
    
    # Load all models
    print("\n🔧 Loading models...")
    models = {}
    for model_key, model_config in MODELS_CONFIG.items():
        models[model_key] = load_model(model_config, device)
    
    successful_models = sum(1 for model in models.values() if model is not None)
    print(f"✅ Successfully loaded {successful_models}/{len(MODELS_CONFIG)} models")
    
    if successful_models == 0:
        print("❌ No models loaded successfully!")
        return
    
    # Analyze attention patterns
    summary_df = analyze_attention_patterns(
        models, dataset, args.conditioning_info, device, args.output_dir, sample_ids=args.sample_ids
    )
    
    print(f"\n✅ Analysis complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main() 