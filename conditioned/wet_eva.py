#!/usr/bin/env python3

import os
import re
import torch
import pandas as pd
import numpy as np
from math import log, exp
from torch.nn.functional import softmax
from scipy.stats import pearsonr, spearmanr

# -----------------------------
# 1. Import your modules
# -----------------------------
import data
from model import Embedding2nd

# -----------------------------
# 2. Global configuration
# -----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()

# Example: If your model needs these specs. Adjust as needed.
cfg = {
    's_in_dim': 22,
    'z_in_dim': 2,
    's_dim': 512,
    'z_dim': 128,
    'N_elayers': 18
}

# If your model uses specific conditioning info (like 'mhc', 'pep', etc.)
conditioning_info = ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']

# Example amino acid indices
aa_list = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V',
    'X','*'
]
idx_to_aa = {i: aa for i, aa in enumerate(aa_list)}
aa_to_idx = {aa: i for i, aa in idx_to_aa.items()}

original_seq = "CASSYVGNTGELFF"
mask_positions = [4, 5, 6, 7, 8, 9]  # positions to mask (example)

# -----------------------------
# 3. Model-loading function
# -----------------------------
def load_latest_model(condition_number=1, model_dir='saved_model'):
    """
    Loads the latest model checkpoint from something like:
      saved_model/condition_{condition_number}/model_epoch_X
    """
    model_instance = Embedding2nd(cfg)
    model_instance.to(device)

    model_path = os.path.join(model_dir, f'condition_{condition_number}')
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Not found: {model_path}")

    # Find all 'model_epoch_' files (not .opt)
    model_files = [
        f for f in os.listdir(model_path)
        if f.startswith('model_epoch_') and not f.endswith('.opt')
    ]
    if not model_files:
        raise FileNotFoundError(f"No model_epoch_ files in {model_path}")

    # Parse out epochs
    epochs = [int(re.findall(r'\d+', f)[0]) for f in model_files]
    latest_epoch = max(epochs)
    model_checkpoint = os.path.join(model_path, f'model_epoch_{latest_epoch}')
    print(model_checkpoint)
    state_dict = torch.load(model_checkpoint, map_location=device)
    model_instance.load_state_dict(state_dict)
    model_instance.eval()

    print(f"[INFO] Loaded model from {model_checkpoint}")
    return model_instance


# -----------------------------
# 4. Evaluate model
# -----------------------------
@torch.no_grad()
def evaluate_model(model, sample, conditioning_info=None):

    logits = model(sample, computeloss=False, conditioning_info=conditioning_info)
    # predicted_aa = torch.argmax(logits, dim=-1)

    # nll = model(sample, computeloss=True, conditioning_info=conditioning_info)
    return logits

# -----------------------------
# 5. Iterative wet-lab probability check
# -----------------------------
import math
import torch
from torch.nn.functional import softmax

def check_wet_lab_prob(model,dataset,orig_seq,wet_lab_seq,mask_positions,seq_idx=0,temperature=1.0):

    if len(wet_lab_seq) != len(orig_seq):
        return wet_lab_seq, 0.0

    log_probs = []

    for pos in mask_positions:
        # Convert the mutated sequence to a list so we can modify one position
        seq_list = list(wet_lab_seq)

        # Mask the mutated position with 'X'
        seq_list[pos] = 'X'
        masked_seq = ''.join(seq_list)

        # Build a single sample for the dataset:
        #   We mask exactly this position so the model predicts only that site.
        sample = dataset.__getitem__(
            idx=seq_idx,
            sequence=masked_seq,
            masklist=[pos]
        )

        # Evaluate the model; you need to supply the relevant context or conditioning.
        pred_aa, logits = evaluate_model(model, sample, conditioning_info)

        # If there's only one masked position, logits[0] corresponds to that position.
        position_probs = softmax(logits[0] / temperature, dim=-1)

        # The correct residue is whatever is in wet_lab_seq at this position.
        wet_res = wet_lab_seq[pos]
        target_idx = aa_to_idx.get(wet_res, aa_to_idx['X'])
        prob = position_probs[target_idx].item()

        # Accumulate log probability
        log_probs.append(math.log(prob))

    # Geometric mean = exp(average of log probabilities)
    if len(log_probs) > 0:
        avg_conf = math.exp(sum(log_probs) / len(log_probs))
    else:
        avg_conf = 0.0

    return wet_lab_seq, avg_conf

def main():
    model_instance = load_latest_model(
        condition_number=1,
        model_dir='saved_model' 
    )

    dataset = data.Load_Dataset("../data/yq.csv") # load data

    df = pd.read_csv("../exp/nnk.csv")  # exp count 

    results = []

    # 4) For each row in nnk.csv, evaluate
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"Processing row {i}...")

        wet_mut = row['mut']       # e.g. "ASQDDF"
        exp_count = row['count']   # experimental read count

        final_seq, avg_conf = check_wet_lab_prob(
            model=model_instance,
            dataset=dataset,
            orig_seq=original_seq,
            wet_lab_mut=wet_mut,
            mask_positions=mask_positions,
            seq_idx=0,          # or whichever index is relevant in the dataset
            temperature=1.0
        )

        results.append({
            'mut': wet_mut,
            'final_sequence': final_seq,
            'confidence': avg_conf,
            'exp_count': exp_count
        })

    # 5) Save results to CSV
    out_csv = "march12eva.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[INFO] Saved evaluation to: {out_csv}")

    # 6) (Optional) Check correlation
    #    We'll do a quick Pearson and Spearman correlation
    #    between 'confidence' and 'exp_count'.
    conf_list = [r['confidence'] for r in results]
    count_list = [r['exp_count'] for r in results]

    # Filter out any rows where exp_count is missing or zero if needed
    # (If your data can handle zero or negative, adapt as you see fit)
    valid_data = [(c, e) for c, e in zip(conf_list, count_list) if not pd.isna(e)]
    if len(valid_data) > 1:
        conf_vals, count_vals = zip(*valid_data)
        pear_corr, _ = pearsonr(conf_vals, count_vals)
        spear_corr, _ = spearmanr(conf_vals, count_vals)
        print(f"[CORRELATION] Pearson:  {pear_corr:.4f}")
        print(f"[CORRELATION] Spearman: {spear_corr:.4f}")
    else:
        print("[CORRELATION] Not enough valid data points to compute correlation.")

if __name__ == "__main__":
    main()
