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

# Force unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

cd /mnt/rna01/zwlexa/project/TCR

echo "Job: $SLURM_JOB_ID on $(hostname) at $(date)"

# Training
# - Uses -u for unbuffered output (real-time logging)
# - Auto-resumes from latest checkpoint if exists
# - Checkpoints saved to flowtcr_fold/Immuno_PLM/checkpoints/
# - Add --no_resume to force fresh start
python -u -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --val_data flowtcr_fold/data/val.jsonl \
    --use_esm \
    --use_lora \
    --esm_model esm2_t12_35M_UR50D \
    --batch_size 64 \
    --ckpt_interval 10

echo "Done at $(date)"
