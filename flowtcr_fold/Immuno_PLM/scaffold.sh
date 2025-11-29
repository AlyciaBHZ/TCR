#!/bin/bash
#SBATCH --job-name=TCR_scaffold
#SBATCH --time=24:00:00
#SBATCH --output=~/logs/TCR_scaffold_%j.out
#SBATCH --error=~/logs/TCR_scaffold_%j.err
#SBATCH --partition=GPUA40             # A40 queue (change to GPUA100 for A100)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1               # For A100: gpu:a100:1

# Environment
mkdir -p ~/logs
source ~/.bashrc
conda activate torch

cd /mnt/rna01/zwlexa/project/TCR

echo "Job: $SLURM_JOB_ID on $(hostname) at $(date)"

# Training
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --val_data flowtcr_fold/data/val.jsonl \
    --use_esm \
    --use_lora \
    --esm_model esm2_t12_35M_UR50D \
    --batch_size 64 \
    --out_dir checkpoints/scaffold_v1

echo "Done at $(date)"
