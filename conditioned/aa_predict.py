"""
use this script to generate predicted cdr3b
the confidence is calculated by the average of the max probability of the masked
run by: python aa_predict.py -c 1 -m "[4,5,6,7,8,9]" -t 1 -p '../data/1G4-TCR.csv'
"""
import argparse
import torch
import pandas as pd
import os
import re
import numpy as np
import itertools
import random
from math import log, exp
from torch.nn.functional import softmax
import ast

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--condition_epochs', type=str, default='',
                    help='Comma-separated list of condition:epoch pairs, e.g., "1:580"')
parser.add_argument('-c', '--condition_number', type=int, choices=range(1, 7), default=1,
                    help='Condition number to evaluate (1-6)')
parser.add_argument('-n', '--num_sequences', type=int, default=5,
                    help='Number of sequences to select for testing')
parser.add_argument('-p', '--pred_file', type=str, default='../data/aa_target.csv',
                    help='Path to the prediction CSV file containing one line of data')
parser.add_argument('-i', '--idx_number', type=int, default=0,
                    help='Index number of the sequence in the pred_file to test')
parser.add_argument('-m', '--masklist', type=ast.literal_eval, default=[3,4,5,6,7,8,9,10],
                    help='List of mask positions (e.g., -m "[2,3,4]")')
parser.add_argument('-t', '--temp', type=float, default=1.0,
                    help='lower (< 1) make the distribution sharper and higher (> 1) make the distribution uniform')
args = parser.parse_args()

# Parse condition_numbers
condition_numbers = [args.condition_number]

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
    1: ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj'],  # All sequences
    2: ['pep', 'lv', 'lj', 'hv', 'hj'],         # No mhc
    3: ['mhc', 'lv', 'lj', 'hv', 'hj'],         # No pep
    4: ['lv', 'lj', 'hv', 'hj'],                # No pep and mhc
    5: ['mhc', 'pep'],                          # Only mhc and pep
    6: [],                                      # No conditioning
}

cfg = {}
cfg['s_in_dim'] = 22
cfg['z_in_dim'] = 2
cfg['s_dim'] = 512
cfg['z_dim'] = 128
cfg['N_elayers'] = 18

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"Using device: {device}")    

from model import Embedding2nd
import data

condition_number = args.condition_number
conditioning_info = condition_sets[condition_number]
# print(f'Loading model for condition {condition_number} with conditioning info: {conditioning_info}')

model_cfg = cfg.copy()
model_instance = Embedding2nd(model_cfg)
model_instance.to(device)# Set up the model path based on the condition number
model_path = os.path.join('saved_model', f'condition_{condition_number}')# Determine the epoch to load
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
model_instance.load_state_dict(torch.load(model_checkpoint, map_location=device,weights_only=True))

print(f"Loaded model for condition {condition_number} from {model_checkpoint}")
models= {'model': model_instance, 'conditioning_info': conditioning_info}


def evaluate_model(model, indict, conditioning_info):
    with torch.no_grad():
        predicted_aa_prob = model(indict, computeloss=False, conditioning_info=conditioning_info)
        predicted_aa = torch.argmax(predicted_aa_prob, dim=-1)
        nll = model(indict, computeloss=True, conditioning_info=conditioning_info)
    return nll, predicted_aa, predicted_aa_prob

def apply_random_mask(sequence):
    # Randomly decide the percentage of the sequence to mask (1% to 90%)
    sequence_length = len(sequence)
    max_mask_len = max(1, int(sequence_length * 0.9))
    min_mask_len = 1
    sel_num = random.randint(min_mask_len, max_mask_len)
    # Randomly select positions to mask
    mask_positions = random.sample(range(sequence_length), sel_num)
    mask_positions.sort()
    return mask_positions

aa_list = [
    'A',  # 0
    'R',  # 1
    'N',  # 2
    'D',  # 3
    'C',  # 4
    'Q',  # 5
    'E',  # 6
    'G',  # 7
    'H',  # 8
    'I',  # 9
    'L',  # 10
    'K',  # 11
    'M',  # 12
    'F',  # 13
    'P',  # 14
    'S',  # 15
    'T',  # 16
    'W',  # 17
    'Y',  # 18
    'V',  # 19
    'X',  # 20 (ambiguous amino acids)
    '-',  # 21 (gap or stop codon)
]

idx_to_aa = {idx: aa for idx, aa in enumerate(aa_list)}
aa_to_idx = {aa: idx for idx, aa in idx_to_aa.items()}

def indices_to_sequence(indices, idx_to_aa):
    # indices: torch tensor of shape (sequence_length,)
    indices = indices.view(-1)
    sequence = ''.join([idx_to_aa.get(idx.item(), 'X') for idx in indices])
    return sequence

def sequence_to_indices(sequence, aa_to_idx):
    # Converts a sequence string to indices using the provided mapping
    indices = [aa_to_idx.get(aa, aa_to_idx['X']) for aa in sequence]
    return indices

# for confidence score and new sequence
def unmasking(sample, sequence, masklist, unmask_pos, idx_to_aa, temperature=1.0):
    """
    Parameters:
    - sequence (str): The current amino acid sequence with masks ('X') applied.
    - masklist (list of int): List of positions of current mask, like [2,3,4,5] - > [3,4,5]
    - unmask_pos (int): The specific position to unmask in this iteration.
    - aa_to_idx (dict): Mapping from amino acid letters to indices.
    - idx_to_aa (dict): Mapping from indices to amino acid letters.
    """
    current_mask = masklist.copy()
    
    # Get model evaluation information
    nll, predicted_aa, predicted_aa_prob = evaluate_model(model_instance, sample, conditioning_info)
    predicted_sequence = indices_to_sequence(predicted_aa, idx_to_aa)  # Remove batch dimension

    masked_positions = sample['mask'].bool().to(device)

    # Extract position probabilities for masked positions
    masked_probs = predicted_aa_prob[masked_positions]  # Filter rows based on mask

    masked_positions_indices = masked_positions.nonzero(as_tuple=True)[0]
    relative_index = (masked_positions_indices == unmask_pos).nonzero(as_tuple=True)[0].item()

    # for choosing the highest probability:
    # max_probs = torch.max(softmax(masked_probs, dim=-1), dim=-1).values
    # # Map unmask_pos to its relative position in masked_positions
    # mask_confidence = max_probs[relative_index].item()
    # predicted_aa = predicted_sequence[unmask_pos]
    # print("Max probability for unmask_pos:", mask_confidence)

    # for diverse sampling (multimonial sampling):
    position_probs = masked_probs[relative_index]
    top_k_probs, top_k_indices = torch.topk(position_probs, 3) # get top n probabilities
    scaled_probs = softmax(top_k_probs / temperature, dim=-1)
    sampled_idx_top = torch.multinomial(scaled_probs, 1).item() # sample the aa from the scaled probability
    sampled_idx = top_k_indices[sampled_idx_top].item()

    predicted_aa = idx_to_aa[sampled_idx]
    mask_confidence = scaled_probs[sampled_idx_top].item()

    # Update the sequence by replacing 'X' with the predicted amino acid at the specified position
    updated_sequence = list(sequence)
    updated_sequence[unmask_pos] = predicted_aa
    updated_sequence = ''.join(updated_sequence)
    # print(updated_sequence)

    # Update the mask list by removing the unmasked position
    new_mask_positions = [pos for pos in current_mask if pos != unmask_pos]
    
    return updated_sequence, new_mask_positions, predicted_aa, mask_confidence, nll.item()

# for iteratively calling the unmasking function
def iterative_unmasking(sequence, csv_file, init_mask, idx_to_aa,output_file):
    # Initialize dataset
    dataset = data.Load_Dataset(csv_file)
    summary = []

    # Generate all permutations of the initial mask positions
    all_permutations = list(itertools.permutations(init_mask))

    # Iterate through each permutation of the mask list
    # for i in len(all_permutations): # use when list is small
    # Iterate randomly by the len(all_permutations)
    while True:
        j = np.random.choice(len(all_permutations))
        permuted_mask = all_permutations[j]
        # print(permuted_mask)
        current_sequence = sequence
        current_mask_positions = list(permuted_mask)
        results = []
        confidence_scores = []

        # Iteratively unmask each position in the permuted order
        for unmask_pos in current_mask_positions:
            sample = dataset.__getitem__(args.idx_number, sequence=current_sequence, masklist=current_mask_positions)

            # Unmask the selected position
            updated_sequence, new_mask_positions, predicted_aa, confidence_score, nll = unmasking(
                sample,
                current_sequence,
                current_mask_positions,
                unmask_pos,
                idx_to_aa,
                temperature=args.temp
            )

            # Store the results of this iteration
            result = {
                'iteration': len(results) + 1,
                'position_unmasked': unmask_pos,
                'predicted_aa': predicted_aa,
                'confidence_score': confidence_score,
                'nll': nll,
                'updated_sequence': updated_sequence,
                'remaining_masks': new_mask_positions.copy()
            }

            results.append(result)
            confidence_scores.append(log(confidence_score)) # log it so by summing it, it will be multiplied

            # Update the current sequence and mask positions for the next iteration
            current_sequence = updated_sequence
            current_mask_positions = new_mask_positions

        average_confidence = sum(confidence_scores) / len(confidence_scores)
        average_confidence = exp(average_confidence) # convert it back to normal scale

        # Store final sequence, unmask order, and average confidence score
        # print(permuted_mask)
        summary = {
            'original_seq': sequence,
            'mutate_list_choice': list(permuted_mask),
            'final_sequence': current_sequence,
            'average_confidence': average_confidence
        }
        # print(summary)

        # append the final_result of each iter to the csv file:
        df = pd.DataFrame([summary])

        df.to_csv(output_file, mode='a', header=False, index=False)

    # return summary

pred_df = pd.read_csv(args.pred_file)
pred_set = data.Load_Dataset(args.pred_file)

pep = pred_df['peptide'].iloc[args.idx_number] 
print(pep)

sample_df = pred_df.iloc[args.idx_number]

# Extract the sequence and conditioning information
sequence = sample_df['cdr3_b']
conditioning_info_keys = condition_sets[args.condition_number]
conditioning_data = {key: sample_df[key] for key in conditioning_info_keys if key in sample_df}

# Ensure all conditioning data are sequences or skip them if missing
for key, value in conditioning_data.items():
    if pd.isna(value) or str(value).strip() == '':
        print(f"Conditioning data '{key}' is missing or empty. It will be ignored.")
        conditioning_data[key] = None
    elif not isinstance(value, str):
        raise ValueError(f"The conditioning data '{key}' should be a string sequence.")

model_instance = models['model']
conditioning_info = models['conditioning_info']
model_instance.to(device)
model_instance.eval()

iter_path = f'../result/aa_predict/{pep}_model{args.condition_number}_temp{args.temp}.csv'

if os.path.isfile(iter_path):
    print("file exist")

else:
    # initilize the csv file with a title first
    df = pd.DataFrame(columns=['original_seq', 'mutate_list_choice', 'final_sequence', 'average_confidence'])   
    df.to_csv(iter_path, index=False)
    
iterative_unmasking(sequence, args.pred_file, args.masklist, idx_to_aa,iter_path)

exit()


result_random = []
result_margin = []
def random_test():
    for i in range(30):

        mask_positions = apply_random_mask(sequence)

        # for now we assume only one sample to test
        sample = pred_set.__getitem__(0, mask_positions)

        nll, predicted_aa, predicted_aa_prob = evaluate_model(model_instance, sample, conditioning_info)

        # print(f"predicted_aa shape: {predicted_aa.shape}")
        # print(f"predicted_aa[0] shape: {predicted_aa[0].shape}")
        # print(f"predicted_aa[0]: {predicted_aa[0]}")

        # Convert predicted amino acids to sequence string
        predicted_sequence = indices_to_sequence(predicted_aa, idx_to_aa)  # Remove batch dimension

        # Compute confidence score (average max probability at masked positions)
        masked_positions = sample['mask'].bool().to(device)[0]  # Remove batch dimension
        masked_probs = predicted_aa_prob[0][masked_positions]
        max_probs = torch.max(softmax(masked_probs, dim=-1), dim=-1).values
        confidence_score = max_probs.mean().item()

        # Store the results
        result_random.append({
            'mutation_idx': i,
            'original_sequence': sequence,
            'predicted_sequence': predicted_sequence,
            'mask_pos': mask_positions,
            'num_masks': len(mask_positions),
            f'condition{args.condition_number}_nll': nll.item(),
            'confidence_score': confidence_score,
        })

def margin_test():
    sequence_length = len(sequence)
    for N in [1, 2]:
        if N * 2 >= sequence_length:
            # Not enough positions to mask; skip this N
            print(f"Skipping N={N} because sequence is too short.")
            continue
       
        positions_to_keep = list(range(N)) + list(range(sequence_length - N, sequence_length))
       
        mask_positions = [pos for pos in range(sequence_length) if pos not in positions_to_keep]
       
        sample = pred_set.__getitem__(0, mask_positions)
        nll, predicted_aa, predicted_aa_prob = evaluate_model(model_instance, sample, conditioning_info)

        predicted_sequence = indices_to_sequence(predicted_aa, idx_to_aa)
  
        masked_positions = sample['mask'].bool().to(device)[0]
        masked_probs = predicted_aa_prob[0][masked_positions]
        max_probs = torch.max(softmax(masked_probs, dim=-1), dim=-1).values
        confidence_score = max_probs.mean().item()

        result_margin.append({
            'margin_N': N,
            'original_sequence': sequence,
            'predicted_sequence': predicted_sequence,
            'mask_pos': mask_positions,
            'num_masks': len(mask_positions),
            f'condition{args.condition_number}_nll': nll.item(),
            'confidence_score': confidence_score,
        })

# random_test()
margin_test()

random = pd.DataFrame(result_random)

# Select sequences with highest confidence scores
N = args.num_sequences  # Number of sequences to select
top_sequences_df = random.sort_values(by='confidence_score', ascending=False).head(N)

top_sequences_filename = f'../result/aa_predict/top_{N}_{pep}.csv'
top_sequences_df.to_csv(top_sequences_filename, index=False)
print(f"Top {N} sequences saved to {top_sequences_filename}")

random_filename = f'../result/aa_predict/{pep}_all_random.csv'
random.to_csv(random_filename, index=False)
print(f"All mutation results saved to {random_filename}")

margin = pd.DataFrame(result_margin)
margin_filename = f'../result/aa_predict/{pep}_all_margin.csv'
margin.to_csv(margin_filename, index=False)