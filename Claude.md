# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This TCR (T-Cell Receptor) design project has **two complementary goals**:

1. **Goal 1 (COMPLETED)**: Design CDR3β region - the most critical part of TCR β-chain
   - Located in: `conditioned/`, `psi_model/`, `pretrain_TCR/`
   - Status: Models trained, generation pipeline functional, 36 synthesis-ready constructs

2. **Goal 2 (IN PROGRESS)**: Design full TCR chains (α and β) including framework regions (LV, LJ, HV, HJ)
   - Located in: `tcr_sidechain/`
   - Strategy: Generate frameworks first, then integrate Goal 1 CDR3β models

**Key Relationship**: Goal 2 will use Goal 1 models to fill in CDR3β after generating framework regions.

## Architecture Overview

**Core Model**: Evoformer-based transformer with:
- Sequence and pairwise representations
- Hierarchical conditioning on peptide-MHC-TCR features
- Composite loss functions (NLL + entropy + contrastive terms)

**Dataset**: 220K training samples from IEDB, McPAS-TCR, VDJdb
- CDR3β: 100% coverage (Goal 1)
- β V/J: 30-35% coverage (Goal 2)
- α V/J: 13% coverage (Goal 2 - sparse)

---

## Goal 1: CDR3β Design (COMPLETED)

### Training Commands

**Phase 1 - Baseline Models:**
```bash
cd conditioned/
python train.py -c 1  # Condition 1 (all features) - RECOMMENDED
python train.py -c <condition_number>  # Other conditions: 2-7
```

**Best Model**: Condition 1, epoch 3450
- Location: `conditioned/saved_model/condition_1/model_epoch_3450`
- Features: Full conditioning `['mhc','pep','lv','lj','hv','hj']`

**Phase 2 - Enhanced psiCLM:**
```bash
cd psi_model/
python train.py --loss_type standard        # Standard NLL only
python train.py --loss_type composite       # Composite loss
python train.py --staged_training           # Staged approach
python train.py --use_monte_carlo           # With Monte Carlo
```

**Phase 3 - Pretraining:**
```bash
cd pretrain_TCR/
python train.py          # Stage 1: CDR3β pure LM
python finetune_pairs.py # Stage 2: Paired finetuning
```

### Generation Commands

**Generate CDR3β for targets:**
```bash
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets wetlab_targets.csv \
  --condition 1 \
  --n_samples 50 \
  --output results/new_cdr3b_candidates.csv
```

**Create full β-chain constructs:**
```bash
python scripts/generation/create_synthesis_constructs.py
```

**Rescore with full conditioning:**
```bash
python scripts/generation/rescoring_with_alpha.py
```

**Export to FASTA:**
```bash
python scripts/generation/generate_synthesis_fasta.py
```

See `scripts/generation/README.md` for detailed usage.

### Current Status ✅

- **36 synthesis-ready β-chain constructs** for 12 targets
- **Output file**: `results/synthesis_ready_constructs.fasta`
- **Instructions**: `results/wetlab_instructions.md`
- **Average confidence**: 0.755, Average length: 11.6 AA

---

## Goal 2: Full Chain Design (IN PROGRESS)

### Architecture (`tcr_sidechain/`)

**Target Output**:
- α-chain: LV + LJ (CDR3α implicit in V-J junction)
- β-chain: HV + **CDR3β (from Goal 1)** + HJ

**Key Innovation - 1-D Projection**:
- Project variable-length regions (LV, LJ, HV, HJ) to fixed vectors
- Maximize similarity with cognate antigen via contrastive learning
- InfoNCE loss with hard negatives (FAISS, K=3)

**Training Commands** (when implementation complete):
```bash
cd tcr_sidechain/

# Train Model-C (contrastive)
python train.py \
  --task joint \
  --loss_weights 1 1 0.3 0 \
  --epochs 20 \
  --save checkpoints/model_C.pt

# Train Model-M (baseline MLM)
python train.py \
  --task mlm_only \
  --loss_weights 0 1 0.3 0 \
  --epochs 15 \
  --save checkpoints/model_M.pt
```

### Implementation Status

**✅ Completed**:
- Architecture design
- Core modules: `src/models/encoder.py`, `src/models/decoder.py`
- Tasks: `src/tasks/contrastive.py`, `src/tasks/mlm.py`
- Training script skeleton: `train.py`

**⚠️ TODO** (see `tcr_sidechain/TECHNICAL_TODO.md`):
- Data preprocessing pipeline
- Dataset class (`src/utils/dataset.py`)
- JSONL export with negative sampling
- Complete training loops
- Integration with Goal 1 models

---

## Directory Structure & Key Components

### Goal 1: CDR3β Design

**`conditioned/` - Phase 1**:
- `model.py`: 18-layer Evoformer with 7 conditioning schemes
- `data.py`: Dataset loading with attention-based subsampling
- `train.py`: Training script (use `-c` flag for condition selection)
- `saved_model/condition_X/`: Trained checkpoints

**`psi_model/` - Phase 2**:
- `model.py`: Enhanced architecture with CLS token, collapse mechanism
- `train.py`: Composite loss training
- `eva/`: Evaluation framework

**`pretrain_TCR/` - Phase 3**:
- `train.py`: CDR3β pure language model pretraining
- `finetune_pairs.py`: Paired data finetuning
- `pretrained_model/`: Checkpoints

**`scripts/generation/` - Generation Pipeline**:
- `generate_cdr3b_wetlab.py`: Main generation script
- `create_synthesis_constructs.py`: Build full β-chains (V+CDR3β+J)
- `rescoring_with_alpha.py`: Rescore with full α+β conditioning
- `generate_synthesis_fasta.py`: Export to synthesis format
- `README.md`: Detailed usage guide

### Goal 2: Full Chain Design

**`tcr_sidechain/`**:
- `TECHNICAL_TODO.md`: Implementation checklist
- `README.md`: Goal 2 overview
- `src/models/`: Encoder, decoder, projection head
- `src/tasks/`: Contrastive, MLM, length prediction
- `train.py`: Training script (partial)
- `generate.py`: Generation script (partial)

### Data & Results

**`data/collected data/final_data/`**:
- `trn.csv`: 220K training samples
- `val.csv`: Validation set
- `tst.csv`: Test set
- Columns: `peptide,mhc,l_v,l_j,h_v,cdr3_b,h_j`

**`results/`**:
- `synthesis_ready_constructs.csv`: 36 β-chain constructs
- `synthesis_ready_constructs.fasta`: Synthesis format
- `synthesis_ready_paired_constructs_rescored.csv`: α+β pairs
- `wetlab_instructions.md`: Synthesis protocol

---

## Key Technical Details

### Conditioning Schemes (Goal 1)

The Phase 1 model supports 7 conditioning configurations:
1. `['mhc','pep','lv','lj','hv','hj']` - **All features (BEST)**
2. `['pep','lv','lj','hv','hj']` - No MHC
3. `['mhc','lv','lj','hv','hj']` - No peptide
4. `['lv','lj','hv','hj']` - No pep/MHC
5. `['mhc','pep']` - No V/J regions
6. `[]` - No conditioning
7. `['pep']` - Peptide only

**Recommendation**: Use Condition 1 for best results.

### Data Format

**CSV columns**: `peptide,mhc,l_v,l_j,h_v,cdr3_b,h_j`
- `cdr3_b`: CDR3β sequence (Goal 1 target)
- `l_v, l_j`: α-chain V/J genes (Goal 2, 13% coverage)
- `h_v, h_j`: β-chain V/J genes (Goal 2, 30-35% coverage)
- `peptide, mhc`: Conditioning sequences
- Fields may be empty

### Model Configuration

**Phase 1 (Baseline)**:
```python
cfg = {
    's_in_dim': 22,      # Amino acid vocabulary
    'z_in_dim': 2,       # Pairwise input
    's_dim': 512,        # Sequence embedding
    'z_dim': 128,        # Pairwise embedding
    'N_elayers': 18      # Evoformer layers
}
```

**Phase 2 (psiCLM)**:
```python
cfg = {
    's_in_dim': 21,      # Vocabulary
    's_dim': 512,        # Sequence dimension
    'z_dim': 128,        # Pairwise dimension
    'N_elayers': 8       # Layers (lighter)
}
```

### Training Parameters

**Phase 1**:
- Batch: 512, Accum: 128
- Test interval: 50, Early stop: 128
- Optimizer: Adam

**Phase 2**:
- Batch: 640, Accum: 4
- Test interval: 25
- Multi-group LRs:
  - Other: 5e-5 (wd=1e-4)
  - Attention: 1e-4 (wd=1e-5)
  - Collapse: 2e-4 (wd=1e-6)
- Scheduler: ReduceLROnPlateau (factor=0.8, patience=20)

**Phase 3**:
- Stage 1: Batch 4096, LR 1e-4
- Stage 2: Batch 1024, LR 2e-5

### psiCLM Features

- **Staged Training**: Attention-only first, then full model
- **Composite Loss**: NLL + entropy + force terms
- **Monte Carlo Sampling**: Optional simulated annealing
- **Attention Analysis**: Built-in visualization
- **Variants**: Standard NLL vs composite loss

---

## Current Status

### Goal 1: CDR3β Design ✅ COMPLETED

**Achievements**:
- 3 sets of models trained (Phases 1-3)
- Generation pipeline functional
- 36 synthesis-ready β-chain constructs for 12 targets
- Average confidence: 0.755
- Ready for wet-lab validation

**Output Files**:
- `results/synthesis_ready_constructs.fasta` - For gene synthesis
- `results/wetlab_instructions.md` - Cloning protocol
- `results/synthesis_ready_paired_constructs_rescored.csv` - α+β pairs

### Goal 2: Full Chain Design 🚧 IN PROGRESS

**Completed**:
- Architecture design
- Core model implementations
- Training script skeleton

**TODO** (see `tcr_sidechain/TECHNICAL_TODO.md`):
- Data preprocessing pipeline (Week 1-2)
- Complete training implementation (Week 3-4)
- Goal 1 integration (Week 5-6)
- Testing and validation

**Challenge**: α-chain data sparsity (13% coverage) may limit quality

---

## Development Guidelines

### Model Training

- GPU detection: `torch.cuda.is_available()` with auto device assignment
- Automatic checkpoint saving/loading with epoch tracking
- Early stopping based on perplexity improvement
- NaN/infinity loss detection
- Memory efficient with gradient accumulation

### Code Architecture

- **Embedding Layer**: Handles multiple sequence types with attention subsampling
- **Evoformer Backbone**: 18 layers (Phase 1), 8 layers (Phase 2)
- **Conditioning Logic**: Dynamic sequence concatenation based on available features
- **Pairwise Representations**: Hierarchical (Phase 2) or flat 28 types (Phase 1)
- **Device Agnostic**: All tensor ops handle CUDA/CPU automatically

### File Naming Conventions

- Model checkpoints: `model_epoch_X` and `model_epoch_X.opt`
- Condition directories: `condition_X/` (Phase 1)
- Data files: `trn.csv`, `val.csv`, `tst.csv` in `data/collected data/final_data/`
- Results: Organized by target and confidence

### Data Pipeline

- **Raw Data**: IEDB/McPAS-TCR/VDJdb with peptide-MHC-TCR triples
- **Processing**: IMGT indexing, length bucketing, FAISS negative sampling
- **Format**: CSV with `peptide,mhc,l_v,l_j,h_v,cdr3_b,h_j`
- **Masking**: Random 15% for MLM, full masks removed during testing

---

## Quick Reference

### Most Common Tasks

**Generate CDR3β for new targets:**
```bash
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets my_targets.csv \
  --condition 1 \
  --n_samples 50 \
  --output results/output.csv
```

**Train new CDR3β model:**
```bash
cd conditioned/
python train.py -c 1
```

**Train enhanced model:**
```bash
cd psi_model/
python train.py --loss_type composite
```

**Create synthesis constructs:**
```bash
python scripts/generation/create_synthesis_constructs.py
python scripts/generation/generate_synthesis_fasta.py
```

### Key Files

- `README.md`: Full project documentation
- `wetlab_targets.csv`: 12 peptide-MHC targets
- `scripts/generation/README.md`: Generation pipeline guide
- `tcr_sidechain/TECHNICAL_TODO.md`: Goal 2 implementation checklist
- `results/wetlab_instructions.md`: Synthesis protocol

---

## Notes for Claude Code

- **Two separate goals**: Keep Goal 1 (CDR3β) and Goal 2 (full chain) clearly separated
- **Goal 1 is complete**: Focus on Goal 2 implementation or generation for new targets
- **Use best model**: Always use Condition 1, epoch 3450 for Goal 1 generation
- **Data coverage**: Be aware of sparse α-chain data (13%) when working on Goal 2
- **Generation pipeline**: 4 essential scripts in `scripts/generation/` with usage guide
- **Integration point**: Goal 2 will call Goal 1 models to generate CDR3β

---

**Last Updated**: 2025-11-13
**Project Status**: Goal 1 complete (wet-lab ready); Goal 2 in active development
