#!/bin/bash
#SBATCH --job-name=flowtcrgen
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# FlowTCR-Gen Training (Stage 2)
# Usage: sbatch run_train.sh [ablation_mode]
# Ablation modes: no_collapse, no_hier, no_cfg

cd /mnt/rna01/zwlexa/project/TCR

# Activate environment
conda activate torch

echo "============================================"
echo "FlowTCR-Gen Training (Stage 2)"
echo "============================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Working dir: $(pwd)"
echo "============================================"

# Check for ablation mode
ABLATION_ARG=""
if [ -n "$1" ]; then
    ABLATION_ARG="--ablation $1"
    echo "Ablation mode: $1"
fi

# Run training
python flowtcr_fold/FlowTCR_Gen/train.py $ABLATION_ARG

echo "============================================"
echo "Training completed at $(date)"
echo "============================================"

