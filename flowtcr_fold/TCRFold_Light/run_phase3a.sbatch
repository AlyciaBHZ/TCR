#!/bin/bash
#SBATCH --job-name=phase3a
#SBATCH --output=logs/phase3a_%j.out
#SBATCH --error=logs/phase3a_%j.err
#SBATCH --partition=GPUA40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

###############################################################################
# Phase 3A: PPI Structure Pretraining for TCRFold-Prophet
#
# Training objective:
#   - Distance prediction (64 bins, 2-22Å)
#   - Contact prediction (binary, interface-weighted)
#
# Data: 76,407 merged PPI samples
# Target: Contact Precision > 50%, Distance MAE < 2.0Å
#
# Estimated time: 24-72 hours (8 layers, batch=4)
###############################################################################

echo "=========================================="
echo "Phase 3A: PPI Structure Pretraining"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Environment setup
cd /mnt/rna01/zwlexa/project/TCR
source ~/.bashrc
conda activate torch
module load cuda 2>/dev/null || true

# Create log directory
mkdir -p logs
mkdir -p checkpoints/stage3_phase_a

# GPU info
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Config
DATA_DIR="flowtcr_fold/data/ppi_merged"
OUT_DIR="checkpoints/stage3_phase_a"
EPOCHS=100
BATCH_SIZE=4
LR=1e-4
S_DIM=384
Z_DIM=128
N_LAYERS=8
MAX_LENGTH=512

echo "Configuration:"
echo "  DATA_DIR: $DATA_DIR"
echo "  OUT_DIR: $OUT_DIR"
echo "  EPOCHS: $EPOCHS"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  LR: $LR"
echo "  Model: s_dim=$S_DIM, z_dim=$Z_DIM, n_layers=$N_LAYERS"
echo "  MAX_LENGTH: $MAX_LENGTH"
echo ""

# Run training
echo "Starting training..."
python flowtcr_fold/TCRFold_Light/train_phase3a.py \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --s_dim "$S_DIM" \
    --z_dim "$Z_DIM" \
    --n_layers "$N_LAYERS" \
    --max_length "$MAX_LENGTH" \
    --warmup_epochs 5 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --val_split 0.1 \
    --save_every 10 \
    --num_workers 4 \
    --seed 42

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="

