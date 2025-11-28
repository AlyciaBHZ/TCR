# FlowTCR-Fold: Physics-Grounded Generative TCR Design

> **Two-Stage TCR Design Framework**: Scaffold Retrieval + CDR3β Generation with Flow Matching, Topology Priors, and Physics Guidance.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Design Philosophy](#2-core-design-philosophy)
3. [Two-Stage Design Strategy](#3-two-stage-design-strategy)
4. [Module Architecture](#4-module-architecture)
5. [Data Infrastructure](#5-data-infrastructure)
6. [Training Workflows](#6-training-workflows)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Code Layout](#8-code-layout)
9. [Quickstart Guide](#9-quickstart-guide)
10. [Legacy Code References](#10-legacy-code-references)
11. [Status & Roadmap](#11-status--roadmap)

---

## 1. Project Overview

### 1.1 Scientific Goal

Design **antigen-specific T Cell Receptors (TCRs)** given a target peptide-MHC (pMHC) complex. This has transformative implications for:
- Cancer immunotherapy (CAR-T, TCR-T)
- Vaccine development
- Autoimmune disease treatment

### 1.2 Technical Challenge

TCR-pMHC recognition is governed by complex sequence-structure interactions:
- **CDR3β loop**: Primary determinant of antigen specificity
- **V/J gene scaffolds**: Provide structural framework and MHC compatibility
- **Multi-chain topology**: TCRα/β chains interact with pMHC in a coordinated manner

### 1.3 Our Approach

A **Retrieve & Generate** framework that decomposes the problem into two tractable sub-tasks:

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Target pMHC (peptide + MHC allele)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: SCAFFOLD RETRIEVAL                                    │
│  ─────────────────────────────                                  │
│  Immuno-PLM retrieves Top-K V/J scaffolds (lv, lj, hv, hj)     │
│  that are compatible with the target MHC                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: CDR3β GENERATION                                      │
│  ─────────────────────────────                                  │
│  FlowTCR-Gen generates CDR3β conditioned on pMHC + scaffold    │
│  using Discrete Flow Matching                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: STRUCTURE CRITIQUE (Optional)                         │
│  ─────────────────────────────────────                          │
│  TCRFold-Light + EvoEF2 filter and rank candidates             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Output: Complete TCR sequence (scaffold + CDR3β)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Design Philosophy

### 2.1 Hybrid Intelligence

We do not rely solely on data fitting. The system combines:

| Component | Role | Source |
|-----------|------|--------|
| **Discrete Flow Matching** | Global sequence search | FlowTCR-Gen |
| **ESM-2 / Topology Embeddings** | Universal protein semantics | Immuno-PLM |
| **EvoEF2 Energy** | First-principles physics | Physics module |
| **Hierarchical Pair Bias** | TCR-pMHC topology awareness | Legacy psi_model |

### 2.2 Retrieve & Generate Paradigm

**Why not generate scaffolds directly?**

| Challenge | Explanation |
|-----------|-------------|
| Discrete space | V/J genes are categorical (e.g., TRBV19*01), not continuous |
| Combinatorial explosion | Vβ × Jβ × Vα × Jα = hundreds of thousands of combinations |
| Data sparsity | Many combinations appear only a few times in training data |

**Solution**: Retrieve scaffolds from a pre-computed bank, then generate CDR3β.

### 2.3 Safe Contrastive Learning

To avoid "false negative" issues in InfoNCE training:

- **Batch Random Negatives**: Use other samples in the same batch as negatives
- **No explicit hard negative mining from database**: Avoids accidentally marking true binders as negatives
- **Synthetic negatives (optional)**: Mutate anchor positions to create guaranteed non-binders

---

## 3. Two-Stage Design Strategy

### 3.1 Stage 1: Scaffold Retrieval

**Objective**: Find V/J gene combinations compatible with the target MHC.

```
┌──────────────────────────────────────────────────────────────┐
│                    SCAFFOLD RETRIEVAL                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:  Target pMHC sequence                                │
│                                                              │
│  Model:  Immuno-PLM (InfoNCE encoder)                        │
│                                                              │
│  Bank:   Pre-computed scaffold embeddings                    │
│          - Key: (h_v, h_j, l_v, l_j) gene combination        │
│          - Value: Germline amino acid sequences              │
│          - Vector: Immuno-PLM embeddings                     │
│                                                              │
│  Method:                                                     │
│    1. Encode pMHC with Immuno-PLM                            │
│    2. Compute cosine similarity with scaffold bank           │
│    3. Retrieve Top-K scaffolds                               │
│                                                              │
│  Output: Top-K scaffold sequences                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Physical Interpretation**: Select V genes whose structural framework is compatible with the target MHC's binding groove.

### 3.2 Stage 2: CDR3β Generation

**Objective**: Generate CDR3β loop that binds the target peptide.

```
┌──────────────────────────────────────────────────────────────┐
│                    CDR3β GENERATION                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:  pMHC + Scaffold (from Stage 1)                      │
│                                                              │
│  Model:  FlowTCR-Gen (Conditional Flow Matching)             │
│                                                              │
│  Conditioning:                                               │
│    - pMHC embedding (from Immuno-PLM)                        │
│    - Scaffold embedding (from Immuno-PLM)                    │
│    - (Optional) TM-align PSSM for structural prior           │
│    - (Optional) Geometry summary from TCRFold-Light          │
│                                                              │
│  Method:                                                     │
│    - Dirichlet Flow Matching on amino acid simplex           │
│    - Vector field prediction: v_θ(x_t, t, cond)              │
│    - Loss: ||v_θ - (y - x_0)||²                              │
│                                                              │
│  Output: CDR3β sequence candidates                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Stage 3: Structure Critique (Optional)

**Objective**: Filter structurally implausible candidates.

```
┌──────────────────────────────────────────────────────────────┐
│                   STRUCTURE CRITIQUE                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:  CDR3β candidates + Scaffold                         │
│                                                              │
│  Model:  TCRFold-Light (MSA-free Evoformer)                  │
│                                                              │
│  Outputs:                                                    │
│    - Contact map prediction                                  │
│    - pLDDT-like confidence score                             │
│    - Energy surrogate (trained on EvoEF2 labels)             │
│                                                              │
│  Filtering:                                                  │
│    - Remove candidates with low interface contact density    │
│    - Remove candidates with high predicted energy            │
│                                                              │
│  (Optional) EvoEF2 Refinement:                               │
│    - Monte Carlo sidechain repacking                         │
│    - Compute precise binding energy (ΔΔG)                    │
│    - Final ranking                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Module Architecture

### 4.1 Immuno-PLM (ESM-2 + Topology Bias) — Status: **partial**

**Role**: Encode TCR and pMHC sequences into embeddings for retrieval and conditioning.

**Core Design**: Topology bias + V/J conditioning. Current code supports BasicTokenizer and optional ESM with in-house LoRA (no  dependency).

```
┌─────────────────────────────────────────────────────────────────┐
│                     Immuno-PLM Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Tokens                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │      ESM-2 (esm2_t33_650M_UR50D) + LoRA Adapters          │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │  Each Self-Attention Layer:                         │  │ │
│  │  │  Q_proj + LoRA | K_proj + LoRA | V_proj + LoRA     │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │  × 33 layers                                               │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│               Sequence Features [B, L, 1280]                    │
│                             │                                   │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│              ▼                             ▼                    │
│      seq_proj [1280→256]        TopologyBias (from psi_model)  │
│              │                   - 7-level hierarchy            │
│              │                   - pair_embed_lvl1/2            │
│              │                             │                    │
│              └──────────► + ◄──────────────┘                    │
│                           │                                     │
│                           ▼                                     │
│                 Fused Features [B, L, 256]                      │
│                           │                                     │
│                           ▼                                     │
│                 Masked Pooling + LayerNorm                      │
│                           │                                     │
│                           ▼                                     │
│                 Pooled [B, 256] ──► contrastive_head            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

_LoRA adapters are part of the design but are not implemented in the current codebase; current backbone is BasicTokenizer or frozen ESM features._

**Training (implemented subset)**:
`ash
python flowtcr_fold/Immuno_PLM/train_plm.py --data data/trn.csv --batch_size 32 --epochs 100
# Optional (frozen ESM features if installed):
python flowtcr_fold/Immuno_PLM/train_plm.py --data data/trn.csv --use_esm --esm_model esm2_t6_8M_UR50D
`

**Loss**: Batch InfoNCE (safe from false negatives)

### 4.2 FlowTCR-Gen (Flow Matching Generator)

**Role**: Generate CDR3β sequences given pMHC and scaffold conditions.

```python
class FlowMatchingModel(nn.Module):
    """
    Discrete Flow Matching for CDR3β generation.
    
    Flow setup:
    - Base x_0: Uniform distribution over amino acids
    - Target y: One-hot ground truth sequence
    - Interpolant: x_t = (1-t) * x_0 + t * y
    - Vector field: v* = y - x_0
    - Loss: ||v_θ(x_t, t, cond) - v*||²
    """
    
    def __init__(self, vocab_size=21, hidden_dim=256, n_layers=6):
        # Conditioning encoder
        # Time embedding
        # Transformer backbone
        # Vector field head
        pass
    
    def forward(self, x_t, t, condition):
        # Returns: predicted vector field [B, L, vocab_size]
        pass
```

**Conditioning Inputs**:
1. pMHC embedding (from Immuno-PLM)
2. Scaffold embedding (from Immuno-PLM)
3. (Optional) TM-align PSSM
4. (Optional) Geometry features from TCRFold-Light

### 4.3 TCRFold-Light (Structure Critic)

**Role**: Predict structural features and energy for candidate filtering.

```python
class TCRFoldLight(nn.Module):
    """
    MSA-free Evoformer-lite for structure prediction.
    
    Outputs:
    - Distance map: [B, L, L, n_bins]
    - Contact map: [B, L, L, 1]
    - Energy: [B, 1] (surrogate for EvoEF2)
    """
    
    def __init__(self, s_dim=512, z_dim=128, n_layers=12):
        # Evoformer blocks (Triangle updates + attention)
        # Distance head
        # Contact head
        # Energy head
        pass
```

**Training**:
- **Phase 1**: Generic PPI pretraining (PDB contacts)
- **Phase 2**: TCR-specific finetuning (STCRDab/TCR3d)
- **Losses**: Distance MSE + Contact BCE (10× weight for interface) + Energy MSE

---

## 5. Data Infrastructure

### 5.1 Data Sources

| Dataset | Size | Fields | Usage |
|---------|------|--------|-------|
| **Paired TCR-pMHC** (trn.csv) | 200K+ | peptide, mhc, cdr3_b, h_v, h_j, l_v, l_j | Scaffold bank, Immuno-PLM, FlowTCR-Gen |
| **TCRdb CDR3β** | Large | cdr3_b only | (Optional) Flow pretraining |
| **STCRDab / TCR3d** | ~500 | PDB structures | TCRFold-Light training |

### 5.2 Data Fields

```
Required:
├── peptide    : Antigenic peptide sequence (8-15 aa)
├── mhc        : MHC allele name or sequence
└── cdr3_b     : CDR3β sequence (target for generation)

Optional:
├── h_v        : Heavy chain V gene
├── h_j        : Heavy chain J gene
├── l_v        : Light chain V gene (alpha)
├── l_j        : Light chain J gene (alpha)
└── cdr3_a     : CDR3α sequence
```

### 5.3 Scaffold Bank Construction

```python
# Extract unique V/J combinations from paired data
import pandas as pd

df = pd.read_csv("data/trn.csv")

# Group by V/J genes
scaffold_bank = df.groupby(['h_v', 'h_j', 'l_v', 'l_j']).agg({
    'peptide': 'first',  # Representative peptide
    'mhc': 'first',      # Representative MHC
    'cdr3_b': 'count'    # Frequency
}).reset_index()

scaffold_bank.columns = ['h_v', 'h_j', 'l_v', 'l_j', 'rep_peptide', 'rep_mhc', 'count']
scaffold_bank.to_csv("data/scaffold_bank.csv", index=False)

print(f"Unique scaffolds: {len(scaffold_bank)}")
```

### 5.4 Hard Negative Strategies

| Type | Strategy | Safety |
|------|----------|--------|
| **Batch Random** | Other samples in batch as negatives | ✅ Safe |
| **Peptide Decoy** | Same MHC, similar peptide (60-90% identity) | ⚠️ Moderate |
| **CDR3 Mutant** | Same pMHC, 2-3 point mutations in CDR3 | ⚠️ Moderate |
| **Synthetic** | Mutate anchor positions to opposite charge | ✅ Safe |

---

## 6. Training Workflows

### 6.1 Immuno-PLM Training

**Objective**: Learn TCR-pMHC compatibility for scaffold retrieval.

```bash
python flowtcr_fold/Immuno_PLM/train_plm.py \
    --data data/trn.csv \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --tau 0.07 \
    --out_dir checkpoints/plm
```

**Loss Function** (Batch InfoNCE):

```python
def compute_batch_infonce(tcr_emb, pmhc_emb, temperature=0.07):
    # tcr_emb: [B, D], pmhc_emb: [B, D]
    logits = tcr_emb @ pmhc_emb.T / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
```

### 6.2 FlowTCR-Gen Training

**Objective**: Learn to generate CDR3β given pMHC + scaffold.

```bash
python flowtcr_fold/FlowTCR_Gen/train_flow.py \
    --data data/trn.csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --out_dir checkpoints/flow
```

**Loss Function** (Flow Matching):

```python
def flow_matching_loss(model, x_0, y, condition):
    # x_0: uniform noise [B, L, vocab]
    # y: one-hot target [B, L, vocab]
    t = torch.rand(x_0.size(0), 1, 1, device=x_0.device)
    x_t = (1 - t) * x_0 + t * y  # Interpolant
    v_target = y - x_0           # Target vector field
    v_pred = model(x_t, t, condition)
    return F.mse_loss(v_pred, v_target)
```

### 6.3 TCRFold-Light Training

**Objective**: Learn structure prediction with energy supervision.

```bash
python flowtcr_fold/TCRFold_Light/train_with_energy.py \
    --pdb_dir data/pdb_structures \
    --cache_dir data/energy_cache \
    --epochs 100 \
    --batch_size 4 \
    --interface_weight 10.0 \
    --out_dir checkpoints/tcrfold
```

**Loss Function** (Physics-guided):

```python
def compute_physics_loss(pred, target, interface_mask):
    # Distance loss
    loss_dist = F.mse_loss(pred['distance'], target['distance'])
    
    # Contact loss (10× weight for interface)
    loss_contact = F.binary_cross_entropy(
        pred['contact'], target['contact'],
        weight=1 + 9 * interface_mask  # 10× for interface
    )
    
    # Energy loss (EvoEF2 supervision)
    loss_energy = F.mse_loss(pred['energy'], target['energy'])
    
    return loss_dist + loss_contact + loss_energy
```

### 6.4 Training Preferences

| Setting | Value | Location |
|---------|-------|----------|
| Checkpoint frequency | Every 50 epochs | `common/utils.py` |
| Early stopping patience | 100 epochs | `common/utils.py` |
| Gradient clipping | max_norm=1.0 | Training scripts |

---

## 7. Inference Pipeline

### 7.1 Complete Workflow

```python
from flowtcr_fold.Immuno_PLM import ImmunoPLM
from flowtcr_fold.FlowTCR_Gen import FlowMatchingModel
from flowtcr_fold.TCRFold_Light import TCRFoldLight
from flowtcr_fold.physics import TCRStructureOptimizer

# 1. Load models
plm = ImmunoPLM.load("checkpoints/plm/immuno_plm.pt")
flow = FlowMatchingModel.load("checkpoints/flow/flow_gen.pt")
critic = TCRFoldLight.load("checkpoints/tcrfold/tcrfold_light.pt")
optimizer = TCRStructureOptimizer()

# 2. Encode target pMHC
target_pmhc = {"peptide": "GILGFVFTL", "mhc": "HLA-A*02:01"}
pmhc_emb = plm.encode_pmhc(target_pmhc)

# 3. Stage 1: Retrieve Top-K scaffolds
scaffold_bank = load_scaffold_bank("data/scaffold_bank.csv")
scaffold_embs = plm.encode_scaffolds(scaffold_bank)
similarities = pmhc_emb @ scaffold_embs.T
top_k_indices = similarities.topk(10).indices
top_scaffolds = [scaffold_bank[i] for i in top_k_indices]

# 4. Stage 2: Generate CDR3β for each scaffold
candidates = []
for scaffold in top_scaffolds:
    scaffold_emb = plm.encode_scaffold(scaffold)
    condition = torch.cat([pmhc_emb, scaffold_emb], dim=-1)
    
    # Sample multiple CDR3β sequences
    for _ in range(100):
        cdr3b = flow.sample(condition)
        candidates.append({
            "scaffold": scaffold,
            "cdr3b": cdr3b,
            "condition": condition
        })

# 5. Stage 3: Critique and rank
scored_candidates = []
for cand in candidates:
    # TCRFold-Light scoring
    score = critic.score(cand["scaffold"], cand["cdr3b"])
    cand["tcrfold_score"] = score
    
    # (Optional) EvoEF2 refinement
    if score > threshold:
        energy = optimizer.compute_binding_energy(cand)
        cand["energy"] = energy
    
    scored_candidates.append(cand)

# 6. Final ranking
scored_candidates.sort(key=lambda x: x.get("energy", x["tcrfold_score"]))
top_designs = scored_candidates[:10]
```

### 7.2 Command-Line Interface

```bash
# Run full pipeline
python flowtcr_fold/FlowTCR_Gen/pipeline_impl.py \
    --peptide "GILGFVFTL" \
    --mhc "HLA-A*02:01" \
    --top_k_scaffolds 10 \
    --samples_per_scaffold 100 \
    --output results/designs.csv
```

---

## 8. Code Layout

```
flowtcr_fold/
├── README.md                           # This file
├── TODO.md                             # Task tracking
├── EVOEF2_INTEGRATION.md               # EvoEF2 setup guide
│
├── docs/
│   ├── USER_MANUAL.md                  # User guide (中文)
│   ├── Plan_v2.0.md                    # Design plan v2.0 (中文)
│   ├── initial_plan.md                 # Original methodology
│   └── initial_plan_update.md          # Updated methodology
│
├── data/
│   ├── dataset.py                      # FlowDataset with triplet sampling
│   ├── tokenizer.py                    # BasicTokenizer / ESM tokenizer
│   └── convert_csv_to_jsonl.py         # Data preprocessing
│
├── common/
│   └── utils.py                        # Checkpointing, early stopping
│
├── Immuno_PLM/
│   ├── immuno_plm.py                   # ImmunoPLM model
│   ├── train_plm.py                    # Training script
│   └── eval_plm.py                     # Evaluation script
│
├── FlowTCR_Gen/
│   ├── flow_gen.py                     # FlowMatchingModel
│   ├── train_flow.py                   # Training script
│   └── pipeline_impl.py                # Full inference pipeline
│
├── TCRFold_Light/
│   ├── tcrfold_light.py                # TCRFoldLight model
│   ├── train_ppi_impl.py               # PPI pretraining
│   ├── train_tcr_impl.py               # TCR finetuning
│   └── train_with_energy.py            # Energy-supervised training
│
├── physics/
│   ├── evoef_runner.py                 # EvoEF2 Python wrapper
│   ├── energy_dataset.py               # Energy-labeled dataset
│   ├── test_evoef.py                   # EvoEF2 tests
│   └── README.md                       # Physics module docs
│
└── tools/
    └── EvoEF2/                         # EvoEF2 binary + params
```

---

## 9. Quickstart Guide

### 9.1 Environment Setup

```bash
# Create environment
conda create -n flowtcr python=3.9
conda activate flowtcr

# Install core dependencies
pip install torch transformers biopython pandas numpy

# Install ESM-2 (required for Immuno-PLM)
pip install fair-esm

# Install PEFT for LoRA (required for ESM-2 fine-tuning)
pip install 

# (Optional) Install wandb for experiment tracking
pip install wandb
```

**Memory Requirements**:
| Mode | VRAM | Notes |
|------|------|-------|
| BasicTokenizer | ~2 GB | For debugging only |
| ESM-2 (frozen) | ~8 GB | Good for testing |
| ESM-2 + LoRA | ~12 GB | Recommended for production |

### 9.2 Data Preparation

```bash
# 1. Prepare training data
head -3 data/trn.csv
# peptide,mhc,cdr3_b,h_v,h_j,l_v,l_j

# 2. Build scaffold bank
python -c "
import pandas as pd
df = pd.read_csv('data/trn.csv')
scaffolds = df.groupby(['h_v','h_j','l_v','l_j']).size().reset_index(name='count')
scaffolds.to_csv('data/scaffold_bank.csv', index=False)
print(f'Unique scaffolds: {len(scaffolds)}')
"

# 3. (Optional) Prepare PDB structures
mkdir -p data/pdb_structures
# Download from STCRDab / TCR3d
```

### 9.3 Training

```bash
# Step 1: Train Immuno-PLM with ESM-2 + LoRA (design target; uses in-house LoRA if no )
python flowtcr_fold/Immuno_PLM/train_plm.py     --data data/trn.csv     --use_esm --use_lora     --esm_model esm2_t33_650M_UR50D     --lora_rank 8     --batch_size 32     --epochs 100     --out_dir checkpoints/plm

# Step 1 (implemented subset): BasicTokenizer or frozen ESM features
python flowtcr_fold/Immuno_PLM/train_plm.py     --data data/trn.csv     --batch_size 32     --epochs 100     --out_dir checkpoints/plm

# Optional: use frozen ESM features if installed
python flowtcr_fold/Immuno_PLM/train_plm.py     --data data/trn.csv     --use_esm     --esm_model esm2_t6_8M_UR50D     --batch_size 32     --epochs 100     --out_dir checkpoints/plm


# Step 2: Train FlowTCR-Gen
python flowtcr_fold/FlowTCR_Gen/train_flow.py \
    --data data/trn.csv \
    --epochs 100

# (Optional) Step 3: Train TCRFold-Light
python flowtcr_fold/TCRFold_Light/train_with_energy.py \
    --pdb_dir data/pdb_structures
```

### 9.4 Inference

```bash
# Run design pipeline
python flowtcr_fold/FlowTCR_Gen/pipeline_impl.py \
    --peptide "GILGFVFTL" \
    --mhc "HLA-A*02:01"
```

---

## 10. Legacy Code References

This project builds upon validated components from previous work:

| Legacy Module | Location | Reused Components |
|---------------|----------|-------------------|
| **Topology Bias** | `conditioned/model.py` | Region/pair embeddings (lines 85-117) |
| **Hierarchical Pairs** | `psi_model/model.py` | `create_hierarchical_pairs`, Collapse token |
| **Evoformer** | `conditioned/src/Evoformer.py` | Triangle updates, Triangle attention |
| **Data Patterns** | `conditioned/data.py` | Masking, amino acid encoding |

---

## 11. Status & Roadmap

### 11.1 Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| **Data Infrastructure** | ✅ 90% | Triplet sampler, tokenizer, scaffold bank |
| **Immuno-PLM** | ✅ 80% | InfoNCE + topology bias working |
| **FlowTCR-Gen** | 🔄 40% | Basic flow matching, needs full conditioning |
| **TCRFold-Light** | ✅ 75% | EvoEF2 integration complete |
| **Physics Module** | ✅ 90% | EvoEF2 wrapper fully functional |
| **Inference Pipeline** | 🔄 50% | Skeleton implemented |

### 11.2 Roadmap

| Phase | Tasks | Priority |
|-------|-------|----------|
| **Phase 1** | Validate Immuno-PLM training (Batch InfoNCE) | 🔴 High |
| **Phase 2** | Implement scaffold retrieval evaluation | 🔴 High |
| **Phase 3** | Complete FlowTCR-Gen with full conditioning | 🟡 Medium |
| **Phase 4** | TCRFold-Light training with PDB data | 🟡 Medium |
| **Phase 5** | End-to-end pipeline integration | 🟡 Medium |
| **Phase 6** | Benchmarking against baselines | 🟢 Low |

---

## References

- **EvoEF2**: Huang X, Pearce R, Zhang Y. Bioinformatics (2020), 36:1135-1142
- **ESM-2**: Lin Z, et al. Science (2023)
- **Flow Matching**: Lipman Y, et al. ICLR (2023)
- **psi_model**: Internal development (hierarchical pair embeddings)

---

**Last Updated**: 2025-11-28  
**Version**: 2.0  
**Maintainers**: FlowTCR-Fold Team
