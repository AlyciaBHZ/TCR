""" evaluate different training epoches
run with: python inself_evaluate.py -c 1 --epochs "0,500,1000,1250,1500,1750,2000,2500,3000,3450"
"""

import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

import data
import model


# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--condition_number', type=int, choices=range(1, 5), default=1,
                    help='Condition number to evaluate (1-6)')
parser.add_argument('-e','--epochs', type=str, default='100,200,300',
                    help='Comma-separated list of epochs to evaluate, e.g., "100,200,300"')
args = parser.parse_args()

# Parse condition number and epochs
condition_number = args.condition_number
epochs = [int(epoch.strip()) for epoch in args.epochs.split(',')]

# Define the conditioning information for each condition set
condition_sets = {
    1: ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'],                 # All
    2: ['pep', 'lv', 'lj', 'hv', 'hj'],                        # No mhc
    3: ['mhc', 'lv', 'lj', 'hv', 'hj'],                        # No pep
    4: ['mhc', 'pep', 'lj', 'hj'],                             # No hv + lv
    5: ['mhc', 'pep', 'lv', 'hv'],                             # No hj + lj
    6: [],                                                     # All gone
}

conditioning_info = condition_sets[condition_number]
print(f'Evaluating condition {condition_number} with conditioning info: {conditioning_info}')

cfg = {}
cfg['s_in_dim'] = 22
cfg['z_in_dim'] = 2
cfg['s_dim'] = 512
cfg['z_dim'] = 128
cfg['N_elayers'] = 18

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f'Using device: {device}')

def nll_loss_withmask(pred, native, mask):
    pred = pred.to(device)
    native = native.to(device)
    mask = mask.to(device)

    # Reshape the mask to match the shape of pred and native
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)  # Add an extra dimension at the end
    
    return (-(pred * native * mask).sum()) / (mask.sum())

def calculate_accuracy(pred_probs, true_labels, mask):
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

def load_model(condition_number, epoch):
    from model import Embedding2nd
    model_cfg = cfg.copy()
    model_instance = Embedding2nd(model_cfg)
    model_instance.to(device)
    
    model_path = os.path.join('saved_model', f'condition_{condition_number}')
    model_checkpoint = os.path.join(model_path, f'model_epoch_{epoch}')
    if not os.path.isfile(model_checkpoint):
        raise FileNotFoundError(f"Model file {model_checkpoint} does not exist")
    model_instance.load_state_dict(torch.load(model_checkpoint, map_location=device))
    print(f"Loaded model for condition {condition_number} at epoch {epoch} from {model_checkpoint}")
    return model_instance

def evaluate_model_on_dataset(model, conditioning_info, masking_positions_list):
    total_nll = 0.0
    total_correct = 0
    total_masked_positions = 0
    model.eval()
    with torch.no_grad():
        for idx in range(len(val_set)):
            sample = val_set.__getitem__(idx)
            mask_positions = masking_positions_list[idx]
            sample_masked = val_set.__getitem__(idx, mask_positions)
            predicted_aa_prob = model(sample_masked, computeloss=False,  conditioning_info=conditioning_info)
            predicted_aa = torch.argmax(predicted_aa_prob, dim=-1)
            acc = calculate_accuracy(predicted_aa_prob, sample_masked['hd'], sample_masked['mask'])
            predicted_score = model(sample_masked, computeloss=True,  conditioning_info=conditioning_info)
            nll = predicted_score.item()
            
            total_nll += nll
            # For accuracy, we need to sum correct predictions and total positions
            masked_positions = sample_masked['mask'].bool().to(device)
            total_masked = masked_positions.sum().item()
            true_labels = torch.argmax(sample_masked['hd'], dim=-1).to(device)
            pred_labels = torch.argmax(predicted_aa_prob, dim=-1).to(device)
            correct_predictions = (pred_labels[masked_positions] == true_labels[masked_positions]).sum().item()
            total_correct += correct_predictions
            total_masked_positions += total_masked
    avg_nll = total_nll / len(val_set)
    avg_acc = total_correct / total_masked_positions if total_masked_positions > 0 else 0.0
    return avg_nll, avg_acc

# Load validation data
val_path = '../data/val.csv' 
cond_path = f'../data/{condition_number}.csv'
df = pd.read_csv(val_path)
mask_path = '../data/masklist.csv' 
mask = pd.read_csv(mask_path)

mask_list =[]

indices_to_drop = []
for idx, row in df.iterrows():
    values = [row.get(cond) for cond in condition_sets[condition_number]]

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
val_set = data.Load_Dataset(cond_path, None)
print(f'Loaded validation set from {cond_path} with {len(val_set)} samples')

print(f'Loaded {len(mask_list)} mask positions from {mask_path}')

results = []

for epoch in epochs:
    model_instance = load_model(condition_number, epoch)
    avg_nll, avg_acc = evaluate_model_on_dataset(model_instance, conditioning_info, mask_list)
    results.append({'epoch': epoch, 'avg_nll': avg_nll, 'avg_acc': avg_acc})
    print(f'Epoch {epoch}: Avg NLL = {avg_nll:.4f}, Avg Accuracy = {avg_acc:.4f}')

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot Average NLL over Epochs
plt.figure(figsize=(10, 6))
plt.plot(results_df['epoch'], results_df['avg_nll'], marker='o')
plt.title(f'Average NLL over Epochs for Condition {condition_number}')
plt.xlabel('Epoch')
plt.ylabel('Average NLL')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'1111{condition_number}_nll_over_epochs.png')
plt.show()

# Plot Average Accuracy over Epochs
plt.figure(figsize=(10, 6))
plt.plot(results_df['epoch'], results_df['avg_acc'], marker='o')
plt.title(f'Average Accuracy over Epochs for Condition {condition_number}')
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'1111{condition_number}_accuracy_over_epochs.png')
plt.show()
