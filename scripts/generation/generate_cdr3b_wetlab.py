"""
CDR3b Generation Script for Wet-Lab Targets
Based on proposal.md Approach 1 implementation
Usage: python generate_cdr3b_wetlab.py --targets wetlab_targets.csv --condition 1 --n_samples 20 --output results/wetlab_cdr3b_candidates.csv
"""

import argparse
import torch
import pandas as pd
import numpy as np
import os
import re
import random
from torch.nn.functional import softmax
import sys

# Add the conditioned directory to path for imports
sys.path.append('conditioned')
sys.path.append('conditioned/src')

from conditioned.model import Embedding2nd
import conditioned.data as data

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Amino acid dictionaries
aa_to_idx = {
    'A': 0, 'B': 20, 'C': 4, 'D': 3, 'E': 6, 'F': 13, 'G': 7, 'H': 8, 'I': 9, 'J': 20,
    'K': 11, 'L': 10, 'M': 12, 'N': 2, 'O': 20, 'P': 14, 'Q': 5, 'R': 1, 'S': 15, 'T': 16,
    'U': 20, 'V': 19, 'W': 17, 'X': 20, 'Y': 18, 'Z': 20, '-': 21, '*': 21
}

idx_to_aa = {v: k for k, v in aa_to_idx.items() if k not in ['B', 'J', 'O', 'U', 'X', 'Z', '-', '*']}

# Define conditioning schemes
condition_sets = {
    1: ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'],  # All features - best performer
    2: ['pep', 'lv', 'lj', 'hv', 'hj'],         # No mhc
    3: ['mhc', 'lv', 'lj', 'hv', 'hj'],         # No pep
    4: ['lv', 'lj', 'hv', 'hj'],                # No pep and mhc
    5: ['mhc', 'pep'],                          # Only mhc and pep
    6: [],                                      # No conditioning
    7: ['pep']                                  # Peptide only
}

def load_model(condition_number, epoch=None):
    """Load the trained model for specified condition"""
    device = get_device()

    # Model configuration matching CLAUDE.md
    cfg = {
        's_in_dim': 22,
        'z_in_dim': 2,
        's_dim': 512,
        'z_dim': 128,
        'N_elayers': 18
    }

    model = Embedding2nd(cfg)
    model.to(device)

    # Set up model path
    model_path = os.path.join('conditioned', 'saved_model', f'condition_{condition_number}')

    if epoch:
        model_checkpoint = os.path.join(model_path, f'model_epoch_{epoch}')
    else:
        # Load the latest model (best performer is epoch 3450 for condition 1)
        if condition_number == 1 and os.path.exists(os.path.join(model_path, 'model_epoch_3450')):
            model_checkpoint = os.path.join(model_path, 'model_epoch_3450')
        else:
            # Find latest epoch
            if os.path.isdir(model_path):
                model_files = [f for f in os.listdir(model_path) if f.startswith('model_epoch_') and not f.endswith('.opt')]
                if model_files:
                    epochs = [int(re.findall(r'\\d+', f)[0]) for f in model_files]
                    latest_epoch = max(epochs)
                    model_checkpoint = os.path.join(model_path, f'model_epoch_{latest_epoch}')
                else:
                    raise FileNotFoundError(f"No model files found in {model_path}")
            else:
                raise FileNotFoundError(f"Model path {model_path} does not exist")

    print(f"Loading model from: {model_checkpoint}")
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    return model

def indices_to_sequence(indices, idx_to_aa):
    """Convert predicted indices to amino acid sequence"""
    return ''.join([idx_to_aa.get(idx.item(), 'X') for idx in indices])

def evaluate_model(model, sample, conditioning_info):
    """Evaluate model and return predictions"""
    device = get_device()

    # Move sample to device
    for key in sample:
        if torch.is_tensor(sample[key]):
            sample[key] = sample[key].to(device)

    with torch.no_grad():
        # Get loss (NLL)
        nll = model(sample, computeloss=True, conditioning_info=conditioning_info)

        # Get probabilities
        predicted_aa_prob = model(sample, computeloss=False, conditioning_info=conditioning_info)

        # Get predicted amino acids
        predicted_aa = torch.argmax(predicted_aa_prob, dim=-1)

    return nll.item(), predicted_aa, predicted_aa_prob

def generate_random_cdr3b(length):
    """Generate a random CDR3b sequence of specified length"""
    # Common CDR3b amino acids (excluding problematic ones)
    common_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    return ''.join(random.choices(common_aas, k=length))

def iterative_generate_cdr3b(target_row, model, conditioning_info, cdr3b_length=12, temperature=0.8, max_attempts=5):
    """
    Generate CDR3b sequence using iterative unmasking approach
    """
    device = get_device()
    results = []

    for attempt in range(max_attempts):
        try:
            # Start with random sequence
            initial_sequence = generate_random_cdr3b(cdr3b_length)

            # Create temporary CSV for this target
            temp_data = target_row.copy()
            temp_data['hd'] = initial_sequence

            temp_df = pd.DataFrame([temp_data])
            temp_csv = f'temp_target_{attempt}.csv'
            temp_df.to_csv(temp_csv, index=False)

            # Load dataset
            dataset = data.Load_Dataset(temp_csv)

            # Create full mask for CDR3b
            mask_positions = list(range(cdr3b_length))
            current_sequence = initial_sequence

            # Iteratively unmask positions
            for iteration in range(cdr3b_length):
                if not mask_positions:
                    break

                # Choose random position to unmask
                unmask_pos = random.choice(mask_positions)
                mask_positions.remove(unmask_pos)

                # Get sample with current mask
                sample = dataset.__getitem__(0, sequence=current_sequence, masklist=mask_positions)

                # Get model predictions
                nll, predicted_aa, predicted_aa_prob = evaluate_model(model, sample, conditioning_info)

                # Apply temperature sampling to the position being unmasked
                masked_positions = sample['mask'].bool().to(device)
                if masked_positions.sum() > 0:
                    masked_probs = predicted_aa_prob[masked_positions]

                    # Find the relative index of our unmask position
                    masked_positions_indices = masked_positions.nonzero(as_tuple=True)[0]
                    relative_index = (masked_positions_indices == unmask_pos).nonzero(as_tuple=True)[0]

                    if len(relative_index) > 0:
                        pos_probs = masked_probs[relative_index[0]]

                        # Apply temperature
                        if temperature != 1.0:
                            pos_probs = pos_probs / temperature

                        # Sample from distribution
                        pos_probs = softmax(pos_probs, dim=-1)
                        sampled_idx = torch.multinomial(pos_probs, 1).item()

                        # Update sequence
                        new_aa = idx_to_aa.get(sampled_idx, 'A')
                        current_sequence = current_sequence[:unmask_pos] + new_aa + current_sequence[unmask_pos+1:]

            # Calculate final confidence
            final_sample = dataset.__getitem__(0, sequence=current_sequence, masklist=[])
            final_nll, _, final_probs = evaluate_model(model, final_sample, conditioning_info)

            # Calculate average confidence (max probability per position)
            confidence = torch.max(final_probs, dim=-1).values.mean().item()

            results.append({
                'sequence': current_sequence,
                'confidence': confidence,
                'nll': final_nll,
                'length': len(current_sequence),
                'attempt': attempt
            })

            # Clean up temp file
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            continue

    return results

def generate_for_targets(targets_csv, condition_number, n_samples_per_target=20, temperature=0.8, output_file=None):
    """
    Main function to generate CDR3b sequences for all targets
    """
    print(f"Starting CDR3b generation for wet-lab targets...")
    print(f"Using condition {condition_number}: {condition_sets[condition_number]}")

    # Load model
    model = load_model(condition_number)
    conditioning_info = condition_sets[condition_number]

    # Load targets
    targets_df = pd.read_csv(targets_csv)
    print(f"Loaded {len(targets_df)} target peptide-MHC pairs")

    all_results = []

    for idx, row in targets_df.iterrows():
        peptide = row['pep']
        mhc = row['mhc']
        print(f"\\nGenerating for target {idx+1}/{len(targets_df)}: {peptide} + {mhc}")

        # Generate multiple candidates for this target
        target_results = []

        for sample_num in range(n_samples_per_target):
            print(f"  Sample {sample_num+1}/{n_samples_per_target}", end='\\r')

            # Try different CDR3b lengths (common range: 8-20)
            cdr3b_lengths = [10, 11, 12, 13, 14, 15]  # Most common lengths
            length = random.choice(cdr3b_lengths)

            # Generate sequences
            sequences = iterative_generate_cdr3b(
                target_row=row,
                model=model,
                conditioning_info=conditioning_info,
                cdr3b_length=length,
                temperature=temperature,
                max_attempts=3
            )

            # Add best sequence from this generation
            if sequences:
                best_seq = max(sequences, key=lambda x: x['confidence'])
                best_seq.update({
                    'target_idx': idx,
                    'peptide': peptide,
                    'mhc': mhc,
                    'sample_num': sample_num,
                    'condition': condition_number
                })
                target_results.append(best_seq)

        all_results.extend(target_results)
        print(f"  Generated {len(target_results)} candidates")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Sort by confidence within each target
    results_df = results_df.sort_values(['target_idx', 'confidence'], ascending=[True, False])

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"\\nResults saved to: {output_file}")

    # Print summary
    print(f"\\nGeneration Summary:")
    print(f"Total candidates generated: {len(results_df)}")
    print(f"Average confidence: {results_df['confidence'].mean():.3f}")
    print(f"Average CDR3b length: {results_df['length'].mean():.1f}")

    # Show top candidates per target
    print(f"\\nTop candidates per target:")
    for target_idx in results_df['target_idx'].unique():
        target_data = results_df[results_df['target_idx'] == target_idx].head(3)
        peptide = target_data.iloc[0]['peptide']
        mhc = target_data.iloc[0]['mhc']
        print(f"\\nTarget: {peptide} + {mhc}")
        for _, row in target_data.iterrows():
            print(f"  {row['sequence']} (conf: {row['confidence']:.3f}, len: {row['length']})")

    return results_df

def main():
    parser = argparse.ArgumentParser(description='Generate CDR3b sequences for wet-lab targets')
    parser.add_argument('--targets', type=str, default='wetlab_targets.csv',
                        help='CSV file with target peptide-MHC pairs')
    parser.add_argument('--condition', type=int, choices=range(1, 8), default=1,
                        help='Conditioning scheme (1-7, default: 1 - all features)')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of CDR3b candidates per target')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower = more confident)')
    parser.add_argument('--output', type=str, default='results/wetlab_cdr3b_candidates.csv',
                        help='Output CSV file for results')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Specific model epoch to load (default: auto-select best)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Generate sequences
    results_df = generate_for_targets(
        targets_csv=args.targets,
        condition_number=args.condition,
        n_samples_per_target=args.n_samples,
        temperature=args.temperature,
        output_file=args.output
    )

    print(f"\\nCDR3b generation completed successfully!")
    print(f"Results available in: {args.output}")

if __name__ == "__main__":
    main()