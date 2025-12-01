# FlowTCR-Fold: Physics-Grounded Generative TCR Design

> **Three-Stage TCR Design Framework**: Scaffold Retrieval → Topology-Aware CDR3β Generation → Physics-Grounded Validation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Design Philosophy](#2-core-design-philosophy)
3. [Two-Stage Design Strategy](#3-two-stage-design-strategy)
4. [Core Methodology Claims](#core-methodology-claims-论文定位) ⬅️ **Paper Positioning**
5. [Master Plan v3.1](#master-plan-v31-flowtcr-fold-execution-frame)
6. [Module Architecture](#4-module-architecture)
7. [Data Infrastructure](#5-data-infrastructure)
8. [Training Workflows](#6-training-workflows)
9. [Inference Pipeline](#7-inference-pipeline)
10. [Code Layout](#8-code-layout)
11. [Quickstart Guide](#9-quickstart-guide)
12. [Legacy Code References](#10-legacy-code-references)
13. [Status & Roadmap](#11-status--roadmap)

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
│  Stage 3: PHYSICS VALIDATION                                    │
│  ───────────────────────────                                    │
│  TCRFold-Prophet (S_ψ) + Energy Surrogate (E_φ) + MC Refinement │
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

### 3.3 Stage 3: Physics Validation (Required)

**Objective**: Validate structural plausibility and energetic feasibility of generated TCRs.

```
┌──────────────────────────────────────────────────────────────┐
│                   PHYSICS VALIDATION                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:  CDR3β candidates + Scaffold + pMHC                  │
│                                                              │
│  Model:  TCRFold-Prophet (Evoformer-Single + IPA)            │
│                                                              │
│  🔴 Must Have:                                               │
│    - S_ψ: Structure predictor (PPI pretrained)               │
│    - E_φ: Energy surrogate (EvoEF2-NN)                       │
│    - Post-hoc screening: Flow → S_ψ → E_φ ranking            │
│                                                              │
│  🟡 Should Have:                                             │
│    - Offline MC refinement with E_φ guidance                 │
│                                                              │
│  🟢 Exploratory:                                             │
│    - Gradient guidance in Flow ODE                           │
│    - MC samples for self-play training                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Core Methodology Claims (论文定位)

### Primary Contribution: Topology-Aware Flow Matching for CDR3β Generation

**FlowTCR-Gen** is the central innovation of this work, featuring:

| Component | Description | Innovation |
|-----------|-------------|------------|
| **Collapse Token (ψ)** | Learnable global observer that aggregates information across regions | Enables cross-region attention without explicit pairwise enumeration |
| **Hierarchical Pair Embeddings** | 7-level topology encoding (ψ↔region, intra-region, CDR3↔peptide, CDR3↔MHC, etc.) | Injects TCR-pMHC structural priors into the embedding space |
| **Dirichlet Flow Matching** | Continuous-time generative model on the amino acid simplex | Supports smooth interpolation and CFG-based conditional control |

**Key Claim**: By combining structural topology priors with discrete flow matching, FlowTCR-Gen generates CDR3β sequences that are both diverse and structurally plausible, outperforming autoregressive and VAE baselines.

### Supporting Contribution: Physics-Grounded Validation

The physics module (TCRFold-Prophet + EvoEF2) serves as **independent validation** rather than the main innovation:

| Purpose | Method | Role in Paper |
|---------|--------|---------------|
| Structural plausibility | TCRFold-Prophet (Evoformer-Single + IPA) | Demonstrates generated sequences fold into valid TCR-pMHC structures |
| Energetic feasibility | E_φ (EvoEF2-NN surrogate) | Shows binding energy distribution matches natural TCRs |
| Controllable refinement | Monte Carlo with E_φ guidance | Optional post-hoc optimization for best candidates |

**Key Claim**: Generated TCRs are not just statistically similar to training data, but are physically realizable (low clash, favorable binding energy).

---

## Master Plan v3.1 (FlowTCR-Fold Execution Frame)

- Goal: given a target pMHC (peptide + MHC allele), Stage 1 outputs biologically reasonable V/J scaffold priors, Stage 2 generates diverse CDR3β on chosen scaffolds, Stage 3 folds TCR–pMHC and scores with geometry + physics. Stay within this frame for iteration.
- Practical vs exploratory: Practical = minimal paper-ready loop; Exploratory = optional guidance/decoys that must not block the mainline.

### Stage 1 — Immuno-PLM (Scaffold Prior)
- Objective: model p(V,J | MHC, peptide) with MHC as strong signal, peptide as weak refinement; CDR3β not fed as input (analysis only).
- Backbone: `esm2_t33_650M_UR50D` + LoRA (rank 16, alpha 32 on Q/K/V/FFN); prepend an allele embedding token; input `<ALLELE_EMB> MHC_seq Peptide_seq [SEP …]`.
- Dual supervision:
  - Multi-positive InfoNCE on sequences: pMHC embedding vs HV/HJ/LV/LJ sequences with two grouping masks (MHC-only main, pMHC auxiliary weight λ_pmhc≈0.3); missing LV/LJ masked out.
  - Multi-label BCE on gene IDs: group by MHC (primary) and optionally pMHC (secondary weight); pos_weight/focal to handle long tails.
- Metrics: Top-K recall per group (MHC + pMHC) and KL(p_emp || p_model) vs (1) frequency baseline and (2) MHC-only model.

### Stage 2 — FlowTCR-Gen (CDR3β Generator)

**Objective**: Topology-aware discrete flow generation conditioned on pMHC + scaffold.

**Legacy Reuse**:
- psi_model Collapse token + hierarchical pair embeddings
- Evoformer backbone over concatenated sequence
- Pair IDs explicitly mark CDR3↔peptide and CDR3↔MHC interactions

**Flow Head**:
- Dirichlet flow matching on CDR3β (x0 = uniform Dirichlet, x1 = one-hot target)
- Loss = MSE(v_pred, v_true) + λ_ent·collapse-entropy + λ_prof·profile reg
- CFG supported (p=0.1 drop cond during training; inference weight w)
- Keep a "model score" hook (flow cost / collapse scalar) for hybrid MC energy

#### Stage 2 Scope Tiers

| Tier | Component | Description | Paper Status |
|------|-----------|-------------|--------------|
| **🔴 Must Have** | Dirichlet Flow Matching | 核心生成模块 + MSE loss | Required |
| **🔴 Must Have** | Collapse + Hierarchical Pairs | 拓扑感知 conditioning encoder | Required (main claim) |
| **🔴 Must Have** | CFG (Classifier-Free Guidance) | p=0.1 drop, w tunable | Required for controllability |
| **🟡 Should Have** | Model Score Hook | Export flow cost for hybrid MC | Recommended |
| **🟢 Exploratory** | Physics Gradient in ODE | `v_θ - w∇E_φ` at sparse steps | Optional, depends on Stage 3 |

### Stage 3 — TCRFold-Prophet (Structure + Energy)

**Architecture**:
- Trunk: Evoformer-Single + IPA structure head
- Energy head: E_φ as EvoEF2-NN surrogate

**Data**:
- A) General PPI (~50k) for trunk/energy pretrain with EvoEF2 labels
- B) TCR3d/STCRDab for TCR-specific finetune

**Phases**:
- 3A: trunk + struct head on PPI (FAPE + dist/contact)
- 3B: energy head (+ last trunk blocks) to fit EvoEF2, with decoy/noisy structures optional
- 3C: TCR-specific finetune for both heads; target ≥0.7 Pearson/Spearman vs EvoEF2 on TCRs

#### Stage 3 Scope Tiers (论文必需 vs 可选)

| Tier | Component | Description | Paper Status |
|------|-----------|-------------|--------------|
| **🔴 Must Have** | S_ψ (Structure Predictor) | General PPI 预训练的折叠网络 | Required for validation |
| **🔴 Must Have** | E_φ (Energy Surrogate) | 基于 PPI + TCR-pMHC 的 EvoEF2-NN | Required for scoring |
| **🔴 Must Have** | Post-hoc Screening | Flow → S_ψ → E_φ 的后验筛选与排序 | Required for pipeline |
| **🟡 Should Have** | Offline MC Refinement | 基于 E_φ 的 Monte Carlo 序列优化 | Strongly recommended |
| **🟢 Exploratory** | Gradient Guidance in Flow ODE | `x_{t+Δt} = x_t + (v_θ - w∇E_φ)Δt` | Optional, high compute |
| **🟢 Exploratory** | MC-to-Training Loop | MC 生成样本用于二次训练 (self-play) | Future work |

**Rationale**: The Must Have tier provides independent evidence that generated sequences are physically valid. The Should Have tier (MC refinement) is straightforward to implement given E_φ and significantly improves best-case results. Exploratory items are computationally expensive and should not block the main paper.

### Execution Timeline
- T1: finalize Stage 1 grouping and loss wiring (dual InfoNCE + BCE); run training with Top-K/KL vs baselines.
- T2: baseline FlowTCR-Gen with collapse/pair reuse + Dirichlet flow + CFG; validate recon/diversity; log model-score hook.
- T3: Stage 3 phases: 3A/3B on PPI (structure then energy), then 3C TCR finetune; export E_φ for fast scoring.
- T4: Integration: Flow samples → TCRFold-Prophet + E_φ screen → MC (E_φ or hybrid) → EvoEF2 final check; later explore guided flow with ∇E_φ.

### Module / Legacy / New-Tech Matrix
| Stage | Task | Model Backbone | Legacy Usage | New Tech |
|-------|------|----------------|--------------|----------|
| 1 | Scaffold sampling (V/J) | ESM-2 650M + LoRA | Avoid heavy Evoformer here | Multi-positive InfoNCE + causal/BCE heads |
| 2 | CDR3β generation | psi_model Evoformer | ✅ Collapse token + hierarchical pairs | Dirichlet flow head + CFG + grad-guidance hook |
| 3 | Validation (structure + energy) | TCRFold-Light / Prophet | ✅ Evoformer trunk | EvoEF2-NN surrogate + PDB/TCR3d + MC |

## Pipeline v3.1 Detail (Practical vs Exploratory)

### Stage 1 — Immuno-PLM (Scaffold Prior)
- Role: scaffold prior; model p(V,J | MHC, peptide). MHC = strong signal, peptide = weak refinement; CDR3β not used as input (analysis only).
- Inputs: prepend allele embedding token; ESM encodes mhc_sequence+peptide. HV/HJ/LV/LJ sequences feed InfoNCE; HV/HJ/LV/LJ ids feed multi-label BCE. CDR3β only for stats.
- Dual channels:
  - Sequence InfoNCE with multi-positive masks: main grouping by MHC, auxiliary by (peptide,MHC) weighted λ_pmhc≈0.3; missing LV/LJ masked; pos_mask precomputed offline.
  - Multi-label BCE on gene ids: grouped by MHC (primary) + optional pMHC weak weight; pos_weight/focal for long tails; allele cold-start fallback to seq-only/NN.
- Loss: L = L_NCE_MHC + λ_pmhc·L_NCE_pMHC + λ_bce·L_BCE (λ_bce≈0.2 start). Metrics: Top-K per group, KL(p_emp||p_model) vs frequency & MHC-only baselines. Target R@10 ≈20–40% (v1 ~1%), KL(model) < KL(baseline).

### Stage 2 — FlowTCR-Gen (CDR3β Generator)
- Input layout: [ψ, CDR3β, peptide, MHC, scaffold]; pair IDs use 7-level hierarchy (psi_model) marking CDR3↔peptide/MHC.
- Backbone reuse: CollapseAwareEmbedding + SequenceProfileEvoformer (MSA-free) + hierarchical pairs. Long-seq caution: truncate/clip MHC or chunked attention.
- x_t injection: use `x_proj(x_t) + pos_emb` for CDR3 region (replace one-hot). Evoformer runs on full concatenated sequence.
- Flow head: Dirichlet flow matching (x0 uniform Dirichlet, x1 one-hot); loss = MSE(v_pred,v_true) + λ_ent·collapse-entropy + λ_prof·profile reg; decide vocab 20/21 and log.
- CFG: train p=0.1 cond drop; infer v_uncond + w(v_cond−v_uncond), w tunable. Keep model-score hook (flow cost / collapse scalar) for hybrid MC energy.
- Practical: flow loss + regs + CFG; physics post-hoc. Exploratory: sparse ∇E_φ guidance inside ODE; grad-informed MC proposals.

### Stage 3 — TCRFold-Prophet (Structure + Energy)
- Trunk/heads: Evoformer-Single + IPA struct head; energy head E_φ (EvoEF2 surrogate, pair-pooling or lightweight GVP).
- Data: A=general PPI (~50k) with EvoEF2; B=TCR3d/STCRDab (~500–1k) for TCR finetune.
- Phases: 3A struct pretrain (FAPE + dist/contact), 3B energy fit (MSE to EvoEF2, decoys/noisy structures encouraged), 3C TCR finetune both heads; target corr ≥0.7 on TCR.
- Integration: MC with E_φ or hybrid α·E_φ+β·model score; guided flow remains exploratory (apply every N ODE steps or only top-N samples).

### End-to-End Loop
1) Stage1 → scaffold bank/top-K priors.  
2) Stage2 → CDR3β samples (CFG) with model-score.  
3) Stage3 → TCRFold-Prophet struct + E_φ screen → MC refine (E_φ or hybrid) → final EvoEF2 check.  
Exploratory: guided flow with ∇E_φ and grad-informed MC proposals.

## Plan Review v3.1 (Feasibility Snapshot)

### Overall
| 维度 | 评分 | 评价 |
|------|------|------|
| 概念完整性 | ⭐⭐⭐⭐⭐ | 三个 Stage 分工明确，逻辑自洽 |
| 技术可行性 | ⭐⭐⭐⭐☆ | 大部分可行，少数需调整 |
| 实现复杂度 | ⭐⭐⭐☆☆ | 中高复杂度，需要排期 |
| 创新性 | ⭐⭐⭐⭐⭐ | 多处创新点，论文价值高 |
| Practical/Exploratory 划分 | ⭐⭐⭐⭐⭐ | 主线清晰，探索不阻塞 |

结论：✅ 高度可行，按此计划执行。

### Stage 1
- 可行：ESM-2+LoRA(rank16)、MHC+allele embedding、双层 InfoNCE（MHC 主 + pMHC 辅 λ≈0.3）、多标签 BCE、Top-K/KL、MHC-only baseline。
- 注意：未见 allele 冷启动（seq-only 或 NN fallback）；pos_mask 预计算；λ_bce 初值 0.2 后续调。
- 预期：R@10 ≈20–40%（现 1.1%）；KL(model) < KL(baseline)。

### Stage 2
- 可行：CollapseAwareEmbedding、SequenceProfileEvoformer、7-level pairs、Dirichlet Flow、CFG(p=0.1)、entropy/profile 正则。
- 调整：长序列需截断 MHC 或 chunked attention；x_t 用 `x_proj(x_t)+pos_emb`; Flow 头输出 20/21 需定。
- 代码改动：在 psi_model 增 flow 分支/头；新增 `FlowTCR_Gen/flow_gen.py`（FlowMatchingModel、flow_matching_loss、ODE sample）。

### Stage 3
- 可行：3A PPI 预训，3B EvoEF2 能量回归，3C TCR 微调；E_φ surrogate + MC（复用 psiMonteCarloSampler）。
- 资源：3A 50k PPI 3–7 天@4×A100(~40GB)；3B 1–2 天(~20GB)；3C 几小时(~16GB)。
- 风险：E_φ 相关性<0.7 → 加 decoy/ranking loss；Guided ODE 计算大 → 留在 Exploratory。

### Execution Timeline (12–16 wks, condensed)
- W1-2: Stage1 Practical（dual InfoNCE/BCE+allele emb；Milestone R@10>20%, KL<baseline）
- W3-5: Stage2 Practical（FlowTCRGen refactor+flow_loss+ODE+CFG；Milestone recovery>30%, ppl<10）
- W6-8: Stage3 3A/3B（PDB+EvoEF2 labels；corr>0.6）
- W9-10: Stage3 3C + MC 集成（corr>0.7 on TCR）
- W11-12: End-to-end eval + paper；W13+: Exploratory（guided ODE, grad-informed MC, self-play）

### Data/Checkpoint Hygiene & Ablations
- Data: `trn_v1.jsonl` (raw), `trn_v2.jsonl` (clean), `scaffold_bank_v1.json`, `energy_labels/` (EvoEF2 cache).
- Checkpoints: `stage1_v1/`, `stage1_v2/`, `stage2_v1/`, `stage3_phase_a/`, `stage3_phase_b/`, `stage3_phase_c/`, `pipeline_v1/`.
- Ablations: Stage1 MHC-only vs pMHC; Stage2 ±collapse, ±hier pairs; Stage3 E_φ vs EvoEF2 ranking.

### Immediate Starts
1) Stage1 dual InfoNCE + multi-label BCE + gene-name cleanup  
2) Stage2 psiCLM→FlowTCRGen refactor（x_t 注入 + flow head）  
3) PDB 下载与 EvoEF2 批处理脚本

---
## 4. Module Architecture

### 4.1 Immuno-PLM (Scaffold Prior) — Status: 🔄 **In Progress**

**Role**: Model p(V, J | MHC, peptide) — MHC as strong signal, peptide as weak refinement.

**Core Design**: ESM-2 + LoRA backbone with dual supervision (multi-positive InfoNCE + multi-label BCE).

```
┌─────────────────────────────────────────────────────────────────┐
│                   Immuno-PLM v3.1 Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: <ALLELE_EMB> + MHC_seq + Peptide_seq + [SEP]           │
│       │                                                         │
│       ▼                                                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │    ESM-2 (esm2_t33_650M_UR50D) + LoRA (rank16, α=32)      │ │
│  │    + Allele Embedding Table (HLA-A*02:01 → vector)        │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│               z_pmhc [B, 256] (CLS pooling + projection)        │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│     ┌─────────────┐  ┌───────────┐  ┌────────────┐             │
│     │ Multi-pos   │  │ Multi-pos │  │ Multi-label│             │
│     │ InfoNCE     │  │ InfoNCE   │  │ BCE        │             │
│     │ (MHC group) │  │ (pMHC grp)│  │ (gene IDs) │             │
│     └─────────────┘  └───────────┘  └────────────┘             │
│            │               │               │                    │
│            └───────────────┴───────────────┘                    │
│                             │                                   │
│                             ▼                                   │
│     L = L_NCE_MHC + λ_pmhc·L_NCE_pMHC + λ_bce·L_BCE            │
│         (λ_pmhc≈0.3, λ_bce≈0.2)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Design Choices**:
- **Dual-group InfoNCE**: MHC-only grouping (main) + pMHC grouping (auxiliary λ≈0.3)
- **Multi-label BCE**: V/J gene IDs as multi-hot targets with pos_weight/focal
- **Metrics**: Top-K recall + KL(p_emp || p_model) vs frequency baseline

**Training**:
```bash
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --use_esm --use_lora --lora_rank 16 \
    --epochs 100 --batch_size 16
```

### 4.2 FlowTCR-Gen (CDR3β Generator) — Status: 🔄 **40% Complete**

**Role**: Topology-aware discrete flow generation conditioned on pMHC + scaffold.

**Core Innovation**: Reuses `psi_model` components for rich conditioning.

```
┌─────────────────────────────────────────────────────────────────┐
│                   FlowTCR-Gen v3.1 Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [ψ, CDR3β(x_t), peptide, MHC, HV, HJ, LV, LJ]          │
│       │                                                         │
│       ▼                                                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │    CollapseAwareEmbedding (from psi_model)                │ │
│  │    + Hierarchical Pair IDs (7 levels)                     │ │
│  │    + Region-specific adaptive weights                     │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │    SequenceProfileEvoformer (from psi_model)              │ │
│  │    + Time embedding injection                             │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │    Flow Head: Linear(s_dim → 20/21)                       │ │
│  │    Output: v_pred for CDR3β region only                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Loss = MSE(v_pred, v_true) + λ_ent·collapse_entropy           │
│       + λ_prof·profile_reg                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- **Collapse Token (ψ)**: Global observer aggregating cross-region information
- **Hierarchical Pairs**: 7-level topology encoding CDR3↔peptide, CDR3↔MHC interactions
- **Dirichlet Flow**: x0 = uniform Dirichlet, x1 = one-hot target
- **CFG Support**: p=0.1 drop conditioning during training; tunable w at inference

**Training**:
```bash
python -m flowtcr_fold.FlowTCR_Gen.train_flow \
    --data flowtcr_fold/data/trn.jsonl \
    --epochs 100 --batch_size 32 --lr 1e-4
```

### 4.3 TCRFold-Prophet (Structure + Energy) — Status: 🔄 **75% Complete**

**Role**: Validate structural plausibility and predict binding energy for candidate filtering.

**Architecture**: Evoformer-Single + IPA structure head + Energy surrogate E_φ

```
┌─────────────────────────────────────────────────────────────────┐
│                  TCRFold-Prophet Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: TCR + pMHC sequences (concatenated)                     │
│       │                                                         │
│       ▼                                                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │    ESM-2 per-residue features + chain type embedding      │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │    Evoformer-Single Trunk (N layers)                      │ │
│  │    - Triangle attention + pair update + single attention  │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│     ┌─────────────┐  ┌───────────┐  ┌────────────┐             │
│     │ S_ψ: IPA    │  │ Distance  │  │ E_φ: Energy│             │
│     │ Struct Head │  │ + Contact │  │ Surrogate  │             │
│     └─────────────┘  └───────────┘  └────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Training Phases**:
| Phase | Data | Objective | Target |
|-------|------|-----------|--------|
| 3A | General PPI (~50k) | FAPE + dist/contact | Trunk pretraining |
| 3B | PPI + EvoEF2 labels | MSE(E_φ, E_EvoEF2) | Energy surrogate |
| 3C | TCR3d + STCRDab | Finetune all heads | ≥0.7 corr with EvoEF2 |

**Scope Tiers**:
- 🔴 **Must**: S_ψ + E_φ + post-hoc screening
- 🟡 **Should**: Offline MC refinement with E_φ
- 🟢 **Exploratory**: Gradient guidance in Flow ODE

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
└── l_j        : Light chain J gene (alpha)
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

### 6.1 Scaffold Prior Training (Immuno-PLM)

**Objective**: Model p(V, J | MHC, peptide) with dual supervision.

```bash
# Production mode (ESM-2 + LoRA + dual supervision)
python -m flowtcr_fold.Immuno_PLM.train_scaffold_retrieval \
    --data flowtcr_fold/data/trn.jsonl \
    --use_esm --use_lora --lora_rank 16 \
    --epochs 100 --batch_size 16 \
    --lambda_pmhc 0.3 --lambda_bce 0.2
```

**Loss Function** (Dual-Group InfoNCE + Multi-label BCE):

```python
# Multi-positive InfoNCE with dual grouping
loss_nce_mhc = multi_pos_infonce(z_pmhc, z_hv, pos_mask_mhc) + ...  # MHC grouping
loss_nce_pmhc = multi_pos_infonce(z_pmhc, z_hv, pos_mask_pmhc) + ... # pMHC grouping

# Multi-label BCE for gene ID prediction
loss_bce = BCEWithLogits(logits_hv, multi_hot_hv, pos_weight=class_weights) + ...

# Total (λ_pmhc≈0.3, λ_bce≈0.2)
loss = loss_nce_mhc + λ_pmhc * loss_nce_pmhc + λ_bce * loss_bce
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

### 6.3 TCRFold-Prophet Training (3-Phase)

**Objective**: Learn structure prediction + energy surrogate for physics validation.

```bash
# Phase 3A: General PPI structure pretraining
python -m flowtcr_fold.TCRFold_Light.train_ppi_impl \
    --pdb_dir data/pdb_structures \
    --epochs 100 --batch_size 4 \
    --out_dir checkpoints/stage3_phase_a

# Phase 3B: Energy surrogate fitting
python -m flowtcr_fold.TCRFold_Light.train_energy_surrogate \
    --pdb_dir data/pdb_structures \
    --evoef2_cache data/energy_labels \
    --epochs 50 --batch_size 8 \
    --out_dir checkpoints/stage3_phase_b

# Phase 3C: TCR-specific finetuning
python -m flowtcr_fold.TCRFold_Light.train_tcr_impl \
    --tcr_pdb_dir data/tcr_structures \
    --pretrain_ckpt checkpoints/stage3_phase_b/best.pt \
    --epochs 50 --batch_size 4 \
    --out_dir checkpoints/stage3_phase_c
```

**Loss Functions by Phase**:

```python
# Phase 3A: Structure losses
L_3A = L_FAPE + 0.3 * L_dist + 0.3 * L_contact

# Phase 3B: Add energy surrogate
L_3B = L_FAPE + 0.3 * L_dist + L_energy  # MSE(E_φ, E_EvoEF2)

# Phase 3C: TCR-specific (all heads)
L_3C = L_FAPE + 0.3 * L_dist + 0.3 * L_contact + L_energy
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

# Note: LoRA uses built-in implementation (no PEFT dependency)

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

## 11. Status & Roadmap (Plan v3.1)

### 11.1 Implementation Status

| Stage | Module | Status | Key Milestones |
|-------|--------|--------|----------------|
| 1 | **Immuno-PLM** | 🔄 70% | Dual InfoNCE + BCE pending; R@10 target 20-40% |
| 2 | **FlowTCR-Gen** | 🔄 40% | Collapse/pairs integrated; CFG + flow head pending |
| 3A | **TCRFold-Prophet (PPI)** | 🔄 30% | PDB download + FAPE training pending |
| 3B | **Energy Surrogate (E_φ)** | 🔄 20% | EvoEF2 batch processing ready; NN fitting pending |
| 3C | **TCR Finetune** | ⏳ 0% | Depends on 3A/3B completion |
| — | **End-to-end Pipeline** | 🔄 50% | Skeleton implemented; integration pending |

### 11.2 Execution Timeline (12-16 weeks)

| Week | Stage | Tasks | Milestone |
|------|-------|-------|-----------|
| W1-2 | Stage 1 | Dual InfoNCE + BCE + allele emb | R@10 > 20%, KL < baseline |
| W3-5 | Stage 2 | FlowTCRGen refactor + ODE + CFG | Recovery > 30%, PPL < 10 |
| W6-8 | Stage 3A/3B | PPI pretrain + energy fit | Corr > 0.6 with EvoEF2 |
| W9-10 | Stage 3C | TCR finetune + MC integration | Corr > 0.7 on TCR |
| W11-12 | Integration | End-to-end eval + paper draft | Full pipeline functional |
| W13+ | Exploratory | Guided ODE, grad-MC, self-play | Optional enhancements |

### 11.3 Immediate Priorities

1. 🔴 **Stage 1**: Dual-group InfoNCE + multi-label BCE + gene-name cleanup
2. 🔴 **Stage 2**: psiCLM → FlowTCRGen refactor (x_t injection + flow head)
3. 🟡 **Stage 3**: PDB download + EvoEF2 batch processing scripts

---

## References

- **EvoEF2**: Huang X, Pearce R, Zhang Y. Bioinformatics (2020), 36:1135-1142
- **ESM-2**: Lin Z, et al. Science (2023)
- **Flow Matching**: Lipman Y, et al. ICLR (2023)
- **psi_model**: Internal development (hierarchical pair embeddings)

---

**Last Updated**: 2025-12-01  
**Version**: 3.1  
**Maintainers**: FlowTCR-Fold Team
