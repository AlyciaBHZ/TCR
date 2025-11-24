"""
use masklist to evaluate different conditioned model
run by: python evaluate.py -c 1
"""

import argparse
import torch
import pandas as pd
import esm
import data
import ast
import os
import re
import numpy as np
from pathlib import Path

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e','--condition_epochs', type=str, default='',
                    help='Comma-separated list of condition:epoch pairs, e.g., "1:580,2:600"')
parser.add_argument('-c', '--condition_number', type=int, choices=range(1, 8), default=1,
                    help='Condition number to evaluate (1-6)')
args = parser.parse_args()

# Resolve repository paths robustly (repo root is parent of this file's directory)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'
RESULT_DIR = REPO_ROOT / 'result' / 'unique_condition'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Parse condition_numbers, check if its multiple
condition_numbers = [args.condition_number]
if len(condition_numbers) > 1:
    condition_numbers = [int(x) for x in args.condition_number.split(',')]

# Backward compat: several places use a single `condition_number`
condition_number = condition_numbers[0]

# Validate condition_numbers
for condition_number in condition_numbers:
    if condition_number < 1 or condition_number > 8:
        raise ValueError(f"Invalid condition number: {condition_number}. Must be between 1 and 8.")

# Parse condition_epochs into a dictionary
condition_epochs = {}
if args.condition_epochs:
    pairs = args.condition_epochs.split(',')
    for pair in pairs:
        condition_str, epoch_str = pair.split(':')
        condition_num = int(condition_str)
        epoch_num = int(epoch_str)
        condition_epochs[condition_num] = epoch_num
else:
    condition_epochs = {}  # Empty dict, we will use latest epochs

# Define the conditioning information for each condition set
condition_sets = {
    1: ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'],                 # All
    2: ['pep', 'lv', 'lj', 'hv', 'hj'],                        # No mhc
    3: ['mhc', 'lv', 'lj', 'hv', 'hj'],                        # No pep
    4: ['lv', 'lj', 'hv', 'hj'],                               # No pep and mhc
    5: ['mhc', 'pep'],                                         # No lv lj hv hj
    6: [],                                                     # All gone
    7: ['pep'],                                                # Only pep
}

cfg = {}
cfg['s_in_dim'] = 22
cfg['z_in_dim'] = 2
cfg['s_dim'] = 512
cfg['z_dim'] = 128
cfg['N_elayers'] = 18

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nll_loss_withmask(pred, native, mask):
    device = get_device()
    pred = pred.to(device)
    native = native.to(device)
    mask = mask.to(device)

    # Reshape the mask to match the shape of pred and native
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)  # Add an extra dimension at the end
    
    return (-(pred * native * mask).sum()) / (mask.sum())

device = get_device()

# Load models for each condition
from model import Embedding2nd

models = {}
for condition_number in condition_numbers:
    conditioning_info = condition_sets[condition_number]
    print(f'Loading model for condition {condition_number} with conditioning info: {conditioning_info}')
    model_cfg = cfg.copy()
    model_instance = Embedding2nd(model_cfg)
    model_instance.to(device)

    # Set up the model path based on the condition number
    model_path = os.path.join('./saved_model', f'condition_{condition_number}')

    # Determine the epoch to load
    if condition_number in condition_epochs:
        epoch_to_load = condition_epochs[condition_number]
        model_checkpoint = os.path.join(model_path, f'model_epoch_{epoch_to_load}')
    else:
        # Load the latest model
        if os.path.isdir(model_path):
            model_files = [f for f in os.listdir(model_path) if f.startswith('model_epoch_') and not f.endswith('.opt')]
            if model_files:
                # Extract epoch numbers from filenames
                epochs = [int(re.findall(r'\d+', f)[0]) for f in model_files]
                latest_epoch = max(epochs)
                model_checkpoint = os.path.join(model_path, f'model_epoch_{latest_epoch}')
            else:
                raise FileNotFoundError(f"No model files found in {model_path}")
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist")
    # Load the model
    print(model_checkpoint)
    model_instance.load_state_dict(torch.load(model_checkpoint, map_location=device))
    print(f"Loaded model for condition {condition_number} from {model_checkpoint}")

    # Store the model and conditioning_info
    models[condition_number] = {'model': model_instance, 'conditioning_info': conditioning_info}

# # Load ESM model
# esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# batch_converter = alphabet.get_batch_converter()
# esm_model.eval()  # Set the model to evaluation mode

def calculate_accuracy(pred_probs, true_labels, mask):
    # Get predicted amino acids by taking argmax over probabilities
    pred_labels = torch.argmax(pred_probs, dim=-1).to(device)
    true_labels = torch.argmax(true_labels, dim=-1).to(device)

    masked_positions = mask.bool().to(device)
    # print(masked_positions)
    # print(f'pred: {pred_labels[masked_positions]}')
    # print(f'true: {true_labels[masked_positions]}')

    total_masked = masked_positions.sum().item()

    if len(pred_labels.shape) != len(masked_positions.shape) or len(true_labels.shape) != len(masked_positions.shape):
        raise ValueError("Mismatch in tensor dimensions: Ensure that all tensors have compatible shapes.") 

    # Compare predictions with true labels at masked positions
    correct_predictions = (pred_labels[masked_positions] == true_labels[masked_positions]).sum().item() 
    
    accuracy = correct_predictions / total_masked if total_masked > 0 else 0.0
    return accuracy

def evaluate_model(model, indict, conditioning_info):
    with torch.no_grad():
        predicted_aa_prob = model(indict, computeloss=False, conditioning_info=conditioning_info)
        predicted_aa = torch.argmax(predicted_aa_prob, dim=-1)
        acc = calculate_accuracy(predicted_aa_prob, indict['hd'], indict['mask'])
        nll = model(indict, computeloss=True, conditioning_info=conditioning_info)
    return nll, predicted_aa, acc

def evaluate_esm(esm_model, hd_sequence, hd_masked, mask_positions):
    data = [('seq1', hd_masked)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        logits = esm_model.lm_head(token_representations)  # Apply the language model head to get logits
    
    masked_logits = logits[0, mask_positions, :]
    predicted_aa = torch.argmax(masked_logits, dim=-1) 
    ground = [('seq1', ''.join([hd_sequence[pos] for pos in mask_positions]))]
    _, _, ground_truth_tokens = batch_converter(ground)
    ground_truth_aa = ground_truth_tokens[0, 1:-1]  # Remove special tokens (BOS/EOS)

    correct = (predicted_aa == ground_truth_aa).sum().item() 
    
    accuracy = correct / len(mask_positions) if len(mask_positions) > 0 else 0.0

    mask_tensor = torch.ones_like(ground_truth_aa, dtype=torch.float32)
    
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    native_one_hot = torch.nn.functional.one_hot(ground_truth_aa, num_classes=log_probs.size(-1)).float()

    loss = nll_loss_withmask(log_probs, native_one_hot, mask_tensor)
    
    return predicted_aa, loss.item(), accuracy

def apply_mask(hd_sequence, mask_positions):
    masked_sequence = list(hd_sequence)
    
    for pos in mask_positions:
        if 0 <= pos < len(masked_sequence):  # Ensure the position is within bounds
            masked_sequence[pos] = '<mask>'  # Replace with '<mask>' token
    
    return ''.join(masked_sequence)

mask_path = DATA_DIR / 'masklist.csv'
mask = pd.read_csv(mask_path)
print(f"masking data path: {mask_path}")

val_path = DATA_DIR / 'val.csv'
cond_path = DATA_DIR / f'{condition_number}.csv'
df = pd.read_csv(val_path)

mask_list = []
indices_to_drop = []
for idx, row in df.iterrows():
    values = [row[cond] for cond in condition_sets[condition_number]]

    # Check if any of the values are NaN or empty after stripping whitespace
    if any(pd.isna(value) or str(value).strip() == '' for value in values):
        # print(f'Insufficient information for sample {idx}')
        indices_to_drop.append(idx)
        continue
    else:
        mask_row = mask.loc[mask['hd_original'] == row['cdr3_b']]
        value_str = mask_row.iloc[0]['hd_mask_pos']
        parsed_list = ast.literal_eval(value_str)
        mask_list.append(parsed_list)

df.drop(indices_to_drop, inplace=True)
df.to_csv(cond_path, index=False)
val_set = data.Load_Dataset(cond_path)
print(f'Loaded validation set from {cond_path} with {len(val_set)} samples')

# print(f'Loaded {len(mask_list)} mask positions from {mask_path}')
result = []

# Evaluate each condition
model_info = models[condition_number]
model_instance = model_info['model']
conditioning_info = model_info['conditioning_info']
model_instance.to(device)
model_instance.eval()
with torch .no_grad():
    for idx in range(len(val_set)):
        sample = val_set.__getitem__(idx)
        # print(sample)
        # exit()
        mask_positions = mask_list[idx]
        sample_masked = val_set.__getitem__(idx, mask_positions)
        predicted_aa_prob = model_instance(sample_masked, computeloss=False, conditioning_info=conditioning_info)
        predicted_aa = torch.argmax(predicted_aa_prob, dim=-1)
        acc = calculate_accuracy(predicted_aa_prob, sample_masked['hd'], sample_masked['mask'])
        predicted_score = model_instance(sample_masked, computeloss=True, conditioning_info=conditioning_info)
        nll = predicted_score
        
        result.append({
            # 'original_sequence': sample['cdr3_b'],
            # 'mask_pos': mask_positions,
            f'condition{condition_number}_nll': nll.item(),
            f'condition{condition_number}_acc': acc
        })


# # Evaluate ESM model
# esm_aa, esm_score, esm_acc = evaluate_esm(esm_model, hd_sequence, hd_masked, maskids)
# result['esm_nll'] = esm_score
# result['esm_acc'] = esm_acc
# print(result)

results_df = pd.DataFrame(result)

results_filename = RESULT_DIR / f'{condition_number}_unre.csv'
results_df.to_csv(results_filename, index=False)
print(f"Results saved to {results_filename}")
