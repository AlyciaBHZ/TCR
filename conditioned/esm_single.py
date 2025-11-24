"""
use this script to calculate single mutation for esm model
run by: python esm_single.py -i ../data/tst_cond/1.csv -o ../result/single/1.csv
"""

import argparse
import torch
import pandas as pd
import esm

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_csv', type=str, required=True,
                    help='Path to the input CSV file containing sequences.')
parser.add_argument('-o', '--output_csv', type=str, default='esm_results.csv',
                    help='Output CSV file to save the results.')
args = parser.parse_args()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# Load ESM model
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
esm_model.to(device)
esm_model.eval()  # Set the model to evaluation mode

def evaluate_esm(esm_model, alphabet, hd_sequence, mask_positions):
    data = [('seq1', hd_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    mask_idx = alphabet.mask_idx

    # Create labels
    labels = batch_tokens.clone()
    # Mask the positions in the input tokens
    batch_tokens[0, mask_positions] = mask_idx
    # For labels, set positions not masked to -100 (ignore index in PyTorch)
    labels[0, batch_tokens[0] != mask_idx] = -100

    with torch.no_grad():
        # Get the token representations
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        # Compute logits
        logits = esm_model.lm_head(token_representations)

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Get predictions at masked positions
        masked_logits = logits[0, mask_positions, :]
        predicted_aa = torch.argmax(masked_logits, dim=-1)

        # Ground truth amino acids at masked positions
        ground_truth_aa = labels[0, mask_positions]

        correct = (predicted_aa == ground_truth_aa).sum().item()
        accuracy = correct / len(mask_positions) if len(mask_positions) > 0 else 0.0

    return predicted_aa, loss.item(), accuracy

results = []

# Load the dataset
df = pd.read_csv(args.input_csv)
print(f"Loaded testing data from {args.input_csv}")

for idx, row in df.iterrows():
    hd_sequence = row['cdr3_b']
    seq_len = len(hd_sequence)
    print(f"Processing sequence {idx+1}/{len(df)}: {hd_sequence}")
    
    for i in range(seq_len):
        # Adjust mask positions for BOS token at position 0
        mask_positions = [i + 1]  # +1 because of the BOS token

        # Initialize a result dictionary for this sample
        result = {
            'hd_original': hd_sequence,
            'hd_mask_count': 1,
            'hd_mask_pos': i,  # Original position without adjustment
        }

        # Evaluate ESM model
        esm_aa, esm_score, esm_acc = evaluate_esm(esm_model, alphabet, hd_sequence, mask_positions)
        # Convert predicted amino acid index to amino acid letter
        predicted_aa = alphabet.get_tok(esm_aa.item())

        result['esm_predicted_aa'] = predicted_aa
        result['esm_nll'] = esm_score
        result['esm_acc'] = esm_acc

        results.append(result)
        print(results[-1])  # Print the last result
    print(f"Results for sequence {idx+1} saved to results list")

results_df = pd.DataFrame(results)
# Save results
results_df.to_csv(args.output_csv, index=False)
print(f"Results saved to {args.output_csv}")
