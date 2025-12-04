#!/bin/bash
#SBATCH --job-name=phase0_ppi           # Job name
#SBATCH --partition=Normal              # CPU partition (no GPU needed)
#SBATCH --nodes=1                       # Single node
#SBATCH --ntasks-per-node=1             # Single task
#SBATCH --cpus-per-task=32              # 32 CPU cores for parallel EvoEF2
#SBATCH --mem=64G                       # 64GB memory
#SBATCH --time=24:00:00                 # 24 hours with parallelism
#SBATCH --output=logs/phase0_%j.out     # Standard output
#SBATCH --error=logs/phase0_%j.err      # Standard error

# =============================================================================
# Phase 0: PPI Data Preparation Pipeline (CPU Only)
# =============================================================================
# Step 1: Tier 2 structure features (preprocess_ppi_pairs.py)
# Step 2: Tier 1+3 energy features (compute_evoef2_batch.py) 
# Step 3: Merge all features (merge_energy_struct.py)
# =============================================================================

echo "=============================================="
echo "Phase 0 Pipeline Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=============================================="

# Setup environment
cd /mnt/rna01/zwlexa/project/TCR
source ~/.bashrc
conda activate torch

# Set Python path
export PYTHONPATH=/mnt/rna01/zwlexa/project/TCR:$PYTHONPATH

# Create log directory
mkdir -p flowtcr_fold/logs

# =============================================================================
# Configuration
# =============================================================================
RAW_DIR="flowtcr_fold/data/pdb_structures/raw"
PROCESSED_DIR="flowtcr_fold/data/pdb_structures/processed"
MERGED_DIR="flowtcr_fold/data/ppi_merged"
ENERGY_CACHE="flowtcr_fold/data/energy_cache_full.jsonl"

# =============================================================================
# Step 1: Tier 2 Structure Features
# =============================================================================
echo ""
echo "=============================================="
echo "Step 1: Extracting Tier 2 structure features"
echo "=============================================="

python flowtcr_fold/TCRFold_Light/process_pdb/preprocess_ppi_pairs.py \
    --pdb_dir $RAW_DIR \
    --out_dir $PROCESSED_DIR \
    --cutoff 8.0 \
    --min_len 30 \
    --min_contacts 10

STEP1_STATUS=$?
if [ $STEP1_STATUS -ne 0 ]; then
    echo "[ERROR] Step 1 failed with exit code $STEP1_STATUS"
    exit 1
fi

echo "[OK] Step 1 completed: $(date)"
echo "Processed .npz count: $(find $PROCESSED_DIR -name '*.npz' | wc -l)"

# =============================================================================
# Step 2: Tier 1+3 Energy Features (EvoEF2)
# =============================================================================
echo ""
echo "=============================================="
echo "Step 2: Computing Tier 1+3 energies (EvoEF2)"
echo "=============================================="

NUM_WORKERS=32  # Match cpus-per-task

python flowtcr_fold/TCRFold_Light/process_pdb/compute_evoef2_batch.py \
    --pdb_dir $RAW_DIR \
    --output $ENERGY_CACHE \
    --pairwise \
    --append \
    --repair \
    --num_workers $NUM_WORKERS

STEP2_STATUS=$?
if [ $STEP2_STATUS -ne 0 ]; then
    echo "[ERROR] Step 2 failed with exit code $STEP2_STATUS"
    exit 1
fi

echo "[OK] Step 2 completed: $(date)"
echo "Energy cache entries: $(wc -l < $ENERGY_CACHE)"

# =============================================================================
# Step 3: Merge Tier 1+2+3 Features
# =============================================================================
echo ""
echo "=============================================="
echo "Step 3: Merging Tier 1+2+3 features"
echo "=============================================="

python flowtcr_fold/TCRFold_Light/process_pdb/merge_structure_energy.py \
    --npz_dir $PROCESSED_DIR \
    --energy_json $ENERGY_CACHE \
    --out_dir $MERGED_DIR \
    --skip_missing_energy

STEP3_STATUS=$?
if [ $STEP3_STATUS -ne 0 ]; then
    echo "[ERROR] Step 3 failed with exit code $STEP3_STATUS"
    exit 1
fi

echo "[OK] Step 3 completed: $(date)"
echo "Merged .npz count: $(find $MERGED_DIR -name '*.npz' | wc -l)"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Phase 0 Pipeline Complete: $(date)"
echo "=============================================="
echo "Summary:"
echo "  - Raw PDB/CIF files: $(find $RAW_DIR \( -name '*.pdb' -o -name '*.cif' \) | wc -l)"
echo "  - Processed .npz (Tier 2): $(find $PROCESSED_DIR -name '*.npz' | wc -l)"
echo "  - Energy cache entries: $(wc -l < $ENERGY_CACHE)"
echo "  - Merged .npz (Tier 1+2+3): $(find $MERGED_DIR -name '*.npz' | wc -l)"
echo "=============================================="

