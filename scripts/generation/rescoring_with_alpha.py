"""
Rescore beta candidates including mined alpha V/J as conditioning.

Input:
- results/synthesis_ready_paired_constructs.csv

Output:
- results/synthesis_ready_paired_constructs_rescored.csv (adds rescored_nll, rescored_confidence)

Note: Uses conditioned.model Embedding2nd and only scores the beta CDR3 positions
with conditioning fields ['mhc','pep','lv','lj','hv','hj'] as per model.
"""

import os
import pandas as pd
import torch
from pathlib import Path

# Model imports
import sys
sys.path.append('conditioned')
sys.path.append('conditioned/src')
from conditioned.model import Embedding2nd
import conditioned.data as data

REPO_ROOT = Path(__file__).resolve().parent
IN_PATH = REPO_ROOT / 'results' / 'synthesis_ready_paired_constructs.csv'
OUT_PATH = REPO_ROOT / 'results' / 'synthesis_ready_paired_constructs_rescored.csv'

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_temp_row(row):
    # Build a single-line CSV expected by conditioned.data.Load_Dataset
    # Columns: pep,mhc,lv,lj,hv,hd,hj
    return {
        'pep': row['peptide'],
        'mhc': row['mhc'],
        'lv': row['v_alpha'] or '',
        'lj': row['j_alpha'] or '',
        'hv': row['v_beta'] or '',
        'hd': row['cdr3b_sequence'] or '',
        'hj': row['j_beta'] or ''
    }

def main():
    df = pd.read_csv(IN_PATH)
    device = get_device()
    cfg = {'s_in_dim':22,'z_in_dim':2,'s_dim':512,'z_dim':128,'N_elayers':18}
    model = Embedding2nd(cfg).to(device)

    # Load best Condition 1 by default if exists
    ckpt = REPO_ROOT / 'conditioned' / 'saved_model' / 'condition_1' / 'model_epoch_3450'
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    conditioning_info = ['mhc','pep','lv','lj','hv','hj']

    rescored_nll = []
    rescored_conf = []

    for _, row in df.iterrows():
        temp_dict = build_temp_row(row)
        temp_csv = REPO_ROOT / f"temp_eval_{os.getpid()}_{_}.csv"
        pd.DataFrame([temp_dict]).to_csv(temp_csv, index=False)
        try:
            ds = data.Load_Dataset(str(temp_csv))
            sample = ds.__getitem__(0)
            # Move tensors to device
            for k in list(sample.keys()):
                if torch.is_tensor(sample[k]):
                    sample[k] = sample[k].to(device)
            with torch.no_grad():
                nll = model(sample, computeloss=True, conditioning_info=conditioning_info)
                probs = model(sample, computeloss=False, conditioning_info=conditioning_info)
                conf = torch.max(probs, dim=-1).values.mean().item()
            rescored_nll.append(float(nll.item()))
            rescored_conf.append(float(conf))
        except Exception:
            rescored_nll.append(float('nan'))
            rescored_conf.append(float('nan'))
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    df['rescored_nll'] = rescored_nll
    df['rescored_confidence'] = rescored_conf
    df.to_csv(OUT_PATH, index=False)
    print(f'Saved rescored CSV to {OUT_PATH}')

if __name__ == '__main__':
    main()

