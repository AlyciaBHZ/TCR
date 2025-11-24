""" 
Train the model with the specified conditioning information
1. Early stop if no improvement(need to be improved because 
    right now save the last model, not the best model)
"""

import random
import numpy as np
import torch
import os, sys, re
import argparse

import torch.optim as opt
from torch.nn import functional as F
import data
import model as Model
from src import basic
import math
np.random.seed(42)

torch.cuda.empty_cache()

Batch_size = 512
ACCUMULATION_STEP = 128
TEST_STEP = 50
VISION_STEP = 10

def get_device():
    print("cuda:", torch.cuda.is_available())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments for conditioning
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--condition_set', type=int, choices=range(1, 9), default=1,
                    help='Condition set to use (1-8)')
args = parser.parse_args()

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

conditioning_info = condition_sets[args.condition_set]
print('Using conditioning info:', conditioning_info)

# Load datasets without conditioning information
train_set = data.Load_Dataset('../data/trn.csv')
test_set  = data.Load_Dataset('../data/tst.csv')

expdir = os.path.dirname(os.path.abspath(__file__))

# Create a model path specific to the condition set
model_path = os.path.join(expdir, 'saved_model', f'condition_{args.condition_set}')

def test(model):
    test_num = len(test_set)
    model.eval()
    
    losses = []
    
    with torch.no_grad():
        for iidx in range(test_num):
            sample_1 = test_set.__getitem__(iidx)
            loss = model(sample_1, True, conditioning_info=conditioning_info)
            losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    pll = math.exp(avg_loss)

    return avg_loss, pll

def train(model, optimizer, start_bat_time):

    # early stop if no improvement
    best_pll = None
    better_count = 0

    numsamples = len(train_set)
    current_batch = start_bat_time
    
    while True:

        batch_indexes = np.random.choice(numsamples, Batch_size, replace=False)

        optimizer.zero_grad()

        total_loss = 0  

        for i, anindex in enumerate(batch_indexes):
            asample = train_set.__getitem__(anindex)
            loss = model(asample, True, conditioning_info=conditioning_info)
            loss.backward()
            total_loss += loss.item()

            if (i + 1) % ACCUMULATION_STEP == 0 or (i + 1) == len(batch_indexes):
                optimizer.step()
                optimizer.zero_grad()
        
        avg_loss = total_loss / len(batch_indexes)

        if current_batch % VISION_STEP == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        
        if current_batch % TEST_STEP == 0:
            # Test the model
            print("\nTesting model:")
            tms, pll = test(model)
            print(f'Batch {current_batch}, trn_loss: {avg_loss}, tst_loss: {tms}, tst_pll: {pll}')

            if best_pll is None or pll < best_pll:
                best_pll = pll
                better_count = 0
            else:
                better_count += TEST_STEP

            if better_count >= 128:
                print("Early stopping: no improvement in perplexity for {} batches.".format(better_count))
                break

            # Save the model
            savemodel = os.path.join(model_path, f'model_epoch_{current_batch}')
            saveopt = os.path.join(model_path, f'model_epoch_{current_batch}.opt')
            torch.save(model.state_dict(), savemodel)
            torch.save(optimizer.state_dict(), saveopt)
        
        current_batch += 1

def classifier():
    device = get_device()
    cfg = {
        's_in_dim': 22,
        'z_in_dim': 2,
        's_dim': 512,
        'z_dim': 128,
        'N_elayers': 18
    }

    model = Model.Embedding2nd(cfg)
    model.to(device)

    optimizer = opt.Adam(model.parameters())

    # Check if the model path exists, create it if not
    if not os.path.isdir(model_path):
        start_epoch = 0
        os.makedirs(model_path)
        
    else:
        # Load the latest checkpoint if available
        model_files = [f for f in os.listdir(model_path) if f.startswith('model_epoch_') and not f.endswith('.opt')]
        if model_files:
            # Extract epoch numbers from filenames
            epochs = [int(re.findall(r'\d+', f)[0]) for f in model_files]
            start_epoch = max(epochs)
            saved_model = os.path.join(model_path, f'model_epoch_{start_epoch}')
            print(f'Loading latest model from epoch {start_epoch}:', saved_model)
            model.load_state_dict(torch.load(saved_model))
            optimizer.load_state_dict(torch.load(saved_model + '.opt'))
        else:
            start_epoch = 0

    print('Starting from batch:', start_epoch)

    train(model, optimizer, start_epoch)    

if __name__ == '__main__':
    classifier()
