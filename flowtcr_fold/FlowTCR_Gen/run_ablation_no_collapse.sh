#!/bin/bash
#SBATCH --job-name=FlowGen_noCollapse
#SBATCH --time=48:00:00
#SBATCH --partition=GPUA100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ============================================
# FlowTCR-Gen Training (Stage 2) - ABLATION
# NO COLLAPSE TOKEN
# Tests importance of the ψ global observer
# ============================================

# Environment
source ~/.bashrc
conda activate torch

# Force unbuffered output
export PYTHONUNBUFFERED=1

cd /mnt/rna01/zwlexa/project/TCR

# Add project root to PYTHONPATH
export PYTHONPATH="/mnt/rna01/zwlexa/project/TCR:$PYTHONPATH"

# Output directory
OUT_DIR="flowtcr_fold/FlowTCR_Gen/saved_model/stage2/ablation_no_collapse"
mkdir -p "$OUT_DIR/checkpoints"
mkdir -p "$OUT_DIR/best_model"
mkdir -p "$OUT_DIR/other_results"
LOG_FILE="$OUT_DIR/train_${SLURM_JOB_ID:-local}.log"

echo "========================================" | tee "$LOG_FILE"
echo "FlowTCR-Gen (Stage 2) Training" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Job: ${SLURM_JOB_ID:-local} on $(hostname)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "Mode: ABLATION - No Collapse Token" | tee -a "$LOG_FILE"
echo "Output: $OUT_DIR" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Training with ablation
stdbuf -oL python -u flowtcr_fold/FlowTCR_Gen/train.py --ablation no_collapse 2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "Done: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

