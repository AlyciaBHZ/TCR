#!/bin/bash
#SBATCH --job-name=PLM_ablation
#SBATCH --time=48:00:00
#SBATCH --partition=GPUA100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Environment
source ~/.bashrc
conda activate torch

# Force unbuffered output
export PYTHONUNBUFFERED=1

cd /mnt/rna01/zwlexa/project/TCR

# Output directory (same as model save dir)
OUT_DIR="flowtcr_fold/Immuno_PLM/saved_model/ablation_peptide_off"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/train_${SLURM_JOB_ID:-local}.log"

echo "========================================" | tee "$LOG_FILE"
echo "Job: ${SLURM_JOB_ID:-local} on $(hostname)" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "Mode: ABLATION (peptide-off)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Training with line-buffered output to log file
stdbuf -oL python -u -m flowtcr_fold.Immuno_PLM.train --ablation --resume 2>&1 | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "Done: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
