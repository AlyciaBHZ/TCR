# TCR Design with Conditioned Protein Language Models

**A two-phase approach to T-Cell Receptor (TCR) design using Evoformer-based protein language models with hierarchical conditioning and composite loss functions.**

---

## Project Goals

This project has two complementary goals for antigen-specific TCR design:

### Goal 1: CDR3β Region Design ✅ **COMPLETED**
Design the **most critical** region of the TCR β-chain (CDR3β) based on peptide-MHC input.

- **Status**: Models trained, generation pipeline functional, wet-lab constructs ready
- **Location**: `conditioned/`, `psi_model/`, `pretrain_TCR/`
- **Output**: 36 synthesis-ready β-chain constructs for 12 targets

### Goal 2: Full TCR Chain Design 🚧 **IN PROGRESS**
Design complete α and β chains including framework regions (LV, LJ, HV, HJ).

- **Status**: Architecture designed, code partially implemented
- **Location**: `tcr_sidechain/`
- **Strategy**: Generate framework regions first, then integrate Goal 1 CDR3β models

**Relationship**: Goal 2 will use Goal 1 models to fill in CDR3β after generating framework regions.

---

## Architecture Overview

**Core Model**: Evoformer-based transformer with:
- Sequence and pairwise representations
- Hierarchical conditioning on peptide-MHC-TCR features
- Composite loss functions (NLL + entropy + contrastive terms)

**Dataset**: 220K training samples from IEDB, McPAS-TCR, and VDJdb databases
- Peptide-MHC-TCR binding triples
- Variable coverage: CDR3β (100%), β V/J (30-35%), α V/J (13%)

---

## Goal 1: CDR3β Design (COMPLETED)

### Phase 1: Baseline Models (`conditioned/`)

**Architecture**: 18-layer Evoformer with 7 systematic conditioning schemes

| Condition | Features | Training Status | Best Epoch |
|-----------|----------|----------------|------------|
| **1** | `['mhc','pep','lv','lj','hv','hj']` **(All)** | ✅ Complete | **3450** |
| 2 | `['pep','lv','lj','hv','hj']` (No MHC) | ✅ Complete | 2300 |
| 3 | `['mhc','lv','lj','hv','hj']` (No peptide) | ✅ Complete | 2050 |
| 4 | `['lv','lj','hv','hj']` (V/J only) | ✅ Complete | 1800 |
| 5 | `['mhc','pep']` (pMHC only) | ✅ Complete | 1900 |
| 6 | `[]` (No conditioning) | ✅ Complete | 2750 |
| 7 | `['pep']` (Peptide only) | ✅ Complete | 1250 |

**Best Model**: Condition 1, epoch 3450 - longest stable training with full conditioning

**Training Command**:
```bash
cd conditioned/
python train.py -c 1  # Train Condition 1 (recommended)
```

### Phase 2: Enhanced psiCLM Models (`psi_model/`)

**Improvements over Phase 1**:
- ✅ Fixed data leakage via proper input masking
- ✅ Added CLS token and learnable collapse token (ψ)
- ✅ Hierarchical pairwise embeddings (2-level: 8+4 types vs flat 28)
- ✅ Composite loss: NLL + profile regularization + collapse entropy
- ✅ Multiple attention mechanisms (Profile, Forced, Controlled, Lightweight)

**Model Variants**:
```bash
cd psi_model/
python train.py --loss_type standard        # Standard NLL only
python train.py --loss_type composite       # Full composite loss
python train.py --staged_training           # Train attention first, then full model
python train.py --use_monte_carlo           # With simulated annealing
```

**Model Dimensions**: 8-layer Evoformer, s_dim=512, z_dim=128

### Phase 3: Pretraining Pipeline (`pretrain_TCR/`)

**Two-stage transfer learning**:

**Stage 1 - CDR3β Pure Language Model**:
```bash
cd pretrain_TCR/
python train.py  # Pure LM on CDR3β only
```
- No conditioning, learns intrinsic CDR3β patterns
- 500 epochs: perplexity 5.98 → 3.13
- Batch size: 4096, LR: 1e-4

**Stage 2 - Paired Data Finetuning**:
```bash
cd pretrain_TCR/
python finetune_pairs.py  # Adapt to full conditioning
```
- Full conditioning with pMHC data
- 475 epochs, successful convergence
- Batch size: 1024, LR: 2e-5

### Generation Pipeline

**Generate CDR3β for new targets**:
```bash
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets wetlab_targets.csv \
  --condition 1 \
  --n_samples 50 \
  --output results/new_candidates.csv
```

**Create full β-chain constructs** (V + CDR3β + J):
```bash
python scripts/generation/create_synthesis_constructs.py
```

**Export to FASTA for synthesis**:
```bash
python scripts/generation/generate_synthesis_fasta.py
```

See `scripts/generation/README.md` for complete usage guide.

### Current Results

**Wet-Lab Ready Output**:
- **36 synthesis-ready β-chain constructs** for 12 peptide-MHC targets
- **3 constructs per target** ranked by confidence
- **File**: `results/synthesis_ready_constructs.fasta`
- **Instructions**: `results/wetlab_instructions.md`

**Generation Statistics**:
- Total candidates generated: 240
- After filtering: 120 high-quality (confidence ≥0.6)
- Final selection: 36 top candidates
- Average confidence: 0.755
- Average CDR3β length: 11.6 AA

---

## Goal 2: Full TCR Chain Design (IN PROGRESS)

### Overview (`tcr_sidechain/`)

**Target**: Design complete TCR with both chains
- **α-chain**: LV + LJ regions (CDR3α implicit in V-J junction)
- **β-chain**: HV + **CDR3β (from Goal 1)** + HJ regions

**Key Innovation**: Two-step approach
1. Generate framework regions (LV, LJ, HV, HJ) using contrastive learning
2. Integrate Goal 1 CDR3β models to complete β-chain

### Architecture Design

**1-D Projection Mechanism**:
- Project variable-length regions (LV, LJ, HV, HJ) to fixed vectors
- Maximize similarity with cognate antigen (peptide±MHC)
- Uses learned projection query + attention pooling

**Training Strategy**:
- **InfoNCE Loss**: Contrastive learning with hard negatives (FAISS, K=3)
- **MLM Loss**: 15% masking for token representation quality
- **Length CE Loss**: Bucket prior regularization
- Temperature annealing: τ 0.1→0.05

**Models**:
- **Model-C (Contrastive)**: Joint InfoNCE + MLM + Length CE
- **Model-M (Baseline)**: Pure MLM for comparison

### Implementation Status

**✅ Completed**:
- `src/models/encoder.py` - Evoformer backbone
- `src/models/decoder.py` - Decoder logic
- `src/tasks/contrastive.py` - Contrastive learning task
- `src/tasks/mlm.py` - MLM + length prediction
- `train.py` - Training script skeleton
- `generate.py` - Generation script

**⚠️ TODO** (see `tcr_sidechain/TECHNICAL_TODO.md`):
- Data preprocessing pipeline
- Dataset class implementation (`src/utils/dataset.py`)
- JSONL export with negative sampling
- Complete training loops
- Integration with Goal 1 models
- Validation pipeline

### Data Availability

Training data coverage for Goal 2:
```
h_v (β-chain V):  71,220 samples (32.3%)  ⚠️ Moderate
h_j (β-chain J):  77,351 samples (35.1%)  ⚠️ Moderate
l_v (α-chain V):  29,119 samples (13.2%)  ⚠️ Sparse
l_j (α-chain J):  28,236 samples (12.8%)  ⚠️ Sparse
```

**Challenge**: α-chain data sparsity may limit full design quality

### Training Commands (When Complete)

```bash
cd tcr_sidechain/

# Train Model-C (contrastive)
python train.py \
  --task joint \
  --loss_weights 1 1 0.3 0 \
  --epochs 20 \
  --save checkpoints/model_C.pt

# Train Model-M (baseline)
python train.py \
  --task mlm_only \
  --loss_weights 0 1 0.3 0 \
  --epochs 15 \
  --save checkpoints/model_M.pt
```

---

## Repository Structure

```
TCR/
├── README.md                      # This file
├── Claude.md                      # Instructions for Claude Code
├── wetlab_targets.csv             # Target peptide-MHC pairs
│
├── docs/                          # Documentation
│   └── archive/                   # Historical documents
│
├── scripts/                       # Utility scripts
│   └── generation/                # CDR3β generation & synthesis prep
│       ├── README.md              # Usage guide for generation scripts
│       ├── generate_cdr3b_wetlab.py
│       ├── create_synthesis_constructs.py
│       ├── generate_synthesis_fasta.py
│       └── rescoring_with_alpha.py
│
├── conditioned/                   # Goal 1: Phase 1 - Baseline models
│   ├── saved_model/               # Trained checkpoints (7 conditions)
│   ├── model.py                   # Evoformer architecture
│   ├── data.py                    # Data loading
│   └── train.py                   # Training scripts
│
├── psi_model/                     # Goal 1: Phase 2 - Enhanced models
│   ├── eva/                       # Evaluation framework
│   ├── model.py                   # psiCLM architecture
│   └── train.py                   # Composite loss training
│
├── pretrain_TCR/                  # Goal 1: Phase 3 - Pretraining
│   ├── pretrained_model/          # Checkpoints
│   ├── train.py                   # CDR3β pretraining
│   └── finetune_pairs.py          # Paired data finetuning
│
├── tcr_sidechain/                 # Goal 2: Full chain design (WIP)
│   ├── TECHNICAL_TODO.md          # Implementation checklist
│   ├── README.md                  # Goal 2 overview
│   ├── src/                       # Model implementations
│   ├── train.py                   # Training script
│   └── generate.py                # Generation script
│
├── data/                          # Training datasets
│   └── collected data/
│       └── final_data/
│           ├── trn.csv            # 220K training samples
│           ├── val.csv            # Validation set
│           └── tst.csv            # Test set
│
└── results/                       # Generation outputs
    ├── synthesis_ready_constructs.csv         # 36 β-chain constructs
    ├── synthesis_ready_constructs.fasta       # Synthesis format
    ├── synthesis_ready_paired_constructs_rescored.csv  # α+β pairs
    ├── wetlab_instructions.md                 # Synthesis protocol
    └── [intermediate files...]
```

---

## Quick Start

### Generate CDR3β for New Targets

```bash
# 1. Create target file
cat > my_targets.csv << EOF
peptide,mhc
GILGFVFTL,A*02:01
NLVPMVATV,A*02:01
EOF

# 2. Generate CDR3β sequences
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets my_targets.csv \
  --condition 1 \
  --n_samples 50 \
  --output results/my_candidates.csv

# 3. Process and export (requires full pipeline - see scripts/generation/README.md)
```

### Train New Model

```bash
# Goal 1: Train CDR3β model with specific conditioning
cd conditioned/
python train.py -c 1  # Condition 1 (all features)

# Goal 1: Train enhanced psiCLM model
cd psi_model/
python train.py --loss_type composite

# Goal 2: Train sidechain model (when implementation complete)
cd tcr_sidechain/
python train.py --task joint
```

---

## Key Files

| File | Purpose |
|------|---------|
| `wetlab_targets.csv` | 12 peptide-MHC pairs for wet-lab validation |
| `results/synthesis_ready_constructs.fasta` | Final FASTA for gene synthesis |
| `results/wetlab_instructions.md` | Cloning and synthesis protocol |
| `scripts/generation/README.md` | Usage guide for generation scripts |
| `tcr_sidechain/TECHNICAL_TODO.md` | Goal 2 implementation checklist |

---

## Model Configuration

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

**Training Parameters**:
- Phase 1: Batch 512, Accum 128, Test interval 50
- Phase 2: Batch 640, Accum 4, Test interval 25
- Phase 3: Stage 1 - Batch 4096, LR 1e-4; Stage 2 - Batch 1024, LR 2e-5

---

## Project Status

| Component | Status | Location |
|-----------|--------|----------|
| **Goal 1: CDR3β Design** | ✅ Complete | `conditioned/`, `psi_model/`, `pretrain_TCR/` |
| Phase 1: Baseline models | ✅ Trained | `conditioned/saved_model/` |
| Phase 2: psiCLM models | ✅ Trained | `psi_model/` |
| Phase 3: Pretraining | ✅ Complete | `pretrain_TCR/pretrained_model/` |
| Generation pipeline | ✅ Functional | `scripts/generation/` |
| Wet-lab constructs | ✅ Ready | `results/` |
| **Goal 2: Full Chain Design** | 🚧 In Progress | `tcr_sidechain/` |
| Architecture design | ✅ Complete | `tcr_sidechain/src/` |
| Data pipeline | ⚠️ TODO | See TECHNICAL_TODO.md |
| Training implementation | ⚠️ Partial | `tcr_sidechain/train.py` |
| Goal 1 integration | ⚠️ TODO | Design complete, not implemented |

---

## Next Steps

### Immediate (Goal 1 - Wet-Lab Validation)
1. Order gene synthesis for 36 β-chain constructs
2. Clone into expression vectors
3. Validate binding with peptide-MHC tetramers
4. Assess functional activity

### Short-term (Goal 2 - Implementation)
1. Complete data preprocessing pipeline (Week 1-2)
2. Implement FAISS negative sampling
3. Train Model-C and Model-M (Week 3-4)
4. Integrate Goal 1 CDR3β models (Week 5-6)
5. Generate full α/β TCR pairs

### Long-term (Validation & Deployment)
1. Structural validation (AlphaFold2, Rosetta)
2. Experimental validation (functional T-cell assays)
3. Iterate based on wet-lab results
4. Production deployment for therapeutic TCR design

---

## Technical Requirements

**Environment**:
- Python 3.11+
- PyTorch 2.3+ with CUDA 12.1
- GPU: A100-40GB minimum (8× A100-80GB recommended for Goal 2)
- RAM: 32GB minimum (64GB recommended)

**Key Dependencies**:
```bash
pip install torch pandas numpy biopython faiss-gpu wandb
```

**Data Format**:
CSV with columns: `peptide,mhc,l_v,l_j,h_v,cdr3_b,h_j`
- `cdr3_b` (CDR3β): Always present (Goal 1 target)
- `l_v, l_j` (α-chain V/J): 13% coverage
- `h_v, h_j` (β-chain V/J): 30-35% coverage

---

## References

- **Databases**: IEDB, McPAS-TCR, VDJdb
- **Architecture**: Evoformer-inspired (AlphaFold2)
- **Methods**: Masked language modeling, contrastive learning, transfer learning

---

## Citation

If you use this code or models, please cite:
```
[Citation information - TBD based on publication]
```

---

**Last Updated**: 2025-11-13
**Status**: Goal 1 complete with wet-lab constructs ready; Goal 2 in active development
