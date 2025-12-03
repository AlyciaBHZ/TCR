#!/bin/bash
#SBATCH --job-name=flowtcrgen_abl
#SBATCH --output=slurm-ablation-%j.out
#SBATCH --error=slurm-ablation-%j.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# FlowTCR-Gen Ablation Study (Stage 2)
# Runs: no_collapse ablation

cd /mnt/rna01/zwlexa/project/TCR

# Activate environment
conda activate torch

echo "============================================"
echo "FlowTCR-Gen Ablation: no_collapse"
echo "============================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================"

python flowtcr_fold/FlowTCR_Gen/train.py --ablation no_collapse

echo "============================================"
echo "Ablation completed at $(date)"
echo "============================================"

