""" single point of mutation evaluation
python evaluate_single.py -c 1

 """

import argparse
import torch
import pandas as pd
import data
import os
import re
import numpy as np

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e','--condition_epochs', type=str, default='',
                    help='Comma-separated list of condition:epoch pairs, e.g., "1:580,2:600"')
parser.add_argument('-c','--condition_number', type=int, default='1,2,3,4,5,6',
                    help='condition number to evaluate, e.g., "1/2/3"')
args = parser.parse_args()

# Parse condition_numbers, check if its multiple
condition_number = int(args.condition_number)

# Validate condition_numbers
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
print(f"Using device: {device}")

# Load models for each condition
from model import Embedding2nd


conditioning_info = condition_sets[condition_number]
print(f'Loading model for condition {condition_number} with conditioning info: {conditioning_info}')
model_cfg = cfg.copy()
model_instance = Embedding2nd(model_cfg)
model_instance.to(device)
# Set up the model path based on the condition number
model_path = os.path.join('saved_model', f'{condition_number}')

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
model_instance.load_state_dict(torch.load(model_checkpoint, map_location=device))
print(f"Loaded model for condition {condition_number} from {model_checkpoint}")

# Store the model and conditioning_info
models = {'model': model_instance, 'conditioning_info': conditioning_info}

# # Load ESM model
# esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# batch_converter = alphabet.get_batch_converter()
# esm_model.eval()  # Set the model to evaluation mode

def calculate_accuracy(pred_probs, true_labels, mask):
    # Get predicted amino acids by taking argmax over probabilities
    pred_labels = torch.argmax(pred_probs, dim=-1).to(device)
    true_labels = torch.argmax(true_labels, dim=-1).to(device)

    masked_positions = mask.bool().to(device)

    total_masked = masked_positions.sum().item()

    if len(pred_labels.shape) != len(masked_positions.shape) or len(true_labels.shape) != len(masked_positions.shape):
        raise ValueError("Mismatch in tensor dimensions: Ensure that all tensors have compatible shapes.") 

    # Compare predictions with true labels at masked positions
    correct_predictions = (pred_labels[masked_positions] == true_labels[masked_positions]).sum().item() 
    
    accuracy = correct_predictions / total_masked if total_masked > 0 else 0.0
    return accuracy

# def evaluate_model(model, indict, conditioned, conditioning_info):
#     with torch.no_grad():
#         predicted_aa_prob = model(indict, computeloss=False, conditioned=conditioned, conditioning_info=conditioning_info)
#         predicted_aa = torch.argmax(predicted_aa_prob, dim=-1)
#         acc = calculate_accuracy(predicted_aa_prob, indict['hd'], indict['mask'])
#         predicted_score = model(indict, computeloss=True, conditioned=conditioned, conditioning_info=conditioning_info)
#     return predicted_score, predicted_aa, acc

def evaluate_model(model, indict,  conditioning_info):
    with torch.no_grad():
        predicted_aa_prob = model(indict, computeloss=False,  conditioning_info=conditioning_info)
        predicted_aa = torch.argmax(predicted_aa_prob, dim=-1)

        # # Debugging: Print shapes and masked positions
        # print(f"predicted_aa_prob shape: {predicted_aa_prob.shape}")
        # print(f"indict['hd'] shape: {indict['hd'].shape}")
        # print(f"indict['mask'] shape: {indict['mask'].shape}")
        # print(f"indict['mask'] sum: {indict['mask'].sum().item()}")  # Should be 1.0 for single masking
        # print(f"Mask positions (non-zero indices): {torch.nonzero(indict['mask']).squeeze().tolist()}")

        acc = calculate_accuracy(predicted_aa_prob, indict['hd'], indict['mask'])
        predicted_score = model(indict, computeloss=True, conditioning_info=conditioning_info)
    return predicted_score, predicted_aa, acc


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

results = []
 
val_path = f'../data/tst_cond/{condition_number}.csv' 
df = pd.read_csv(val_path)
print(f"Loaded testing data from {val_path}")
val_set = data.Load_Dataset(val_path, None) 

for idx, row in df.iterrows():
    hd_sequence = row['cdr3_b']
    seq_len = len(hd_sequence)
    print(f"Processing sequence {idx}/{len(df)}: {hd_sequence}")
    
    for i in range(seq_len):
        mask_positions = [i]
        hd_masked = apply_mask(hd_sequence, mask_positions)
        sample = val_set.__getitem__(idx, mask_positions)

        # Initialize a result dictionary for this sample
        result = {
            'hd_original': hd_sequence,
            'hd_mask_count': len(mask_positions),
            'hd_mask_pos': mask_positions,
        }

        model_info = models
        model_instance = model_info['model']
        conditioning_info = model_info['conditioning_info']
        model_instance.to(device)
        model_instance.eval()
        score, pred_aa, acc = evaluate_model(model_instance, sample, conditioning_info)
        result[f'condition{condition_number}_nll'] = score.item()
        result[f'condition{condition_number}_acc'] = acc

        # # Evaluate ESM model
        # esm_aa, esm_score, esm_acc = evaluate_esm(esm_model, hd_sequence, hd_masked, mask_positions)
        # result['esm_nll'] = esm_score
        # result['esm_acc'] = esm_acc

        results.append(result)
        print(results[-1]) # Print the last result
    print(f"Results for sequence {idx} saved to results list")

results_df = pd.DataFrame(results)
# name = name of the condition and epoch
results_filename = f'../result/single/{condition_number}_re.csv'
results_df.to_csv(results_filename, index=False)
print(f"Results saved to {results_filename}")
