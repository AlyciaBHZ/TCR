# FlowTCR-Fold Agent Charter

## Scope
- Primary job: help build the full pipeline in `/mnt/rna01/zwlexa/project/TCR/flowtcr_fold`. Other folders are legacy reference/data; do not drift focus.
- Track progress and self-reminders here; update as milestones complete.
- Code preferences (recorded for all stages):
  - Always enable early stopping and checkpoint saving.
  - Save outputs per stage under `saved_model/` with subdirs `checkpoints/`, `other_results/`, `best_model/`.
- Keep project layout clean: `src/`, `train.py`, `model.py` per module for clarity.
- Avoid version numbers in filenames/modules; use clear names (e.g., `data_scaffold.py`, not `*_v1.py`).
- Before making code changes, confirm the need and scope with the user; default to no-op until requirements are explicit.
- Keep runtime configs in code (fixed paths/ESM+LoRA defaults); only expose minimal CLI toggles (e.g., ablation) to avoid long argument lists.
- Stage 3 hygiene: keep Phase0 PDB/energy scripts inside `flowtcr_fold/TCRFold_Light/process_pdb/` with stage-specific names; avoid scattering under `scripts/`.

## Environment Setup
- **Conda environment**: `conda activate torch`
- **Working directory**: `/mnt/rna01/zwlexa/project/TCR`
- **Python**: Use `conda activate torch` before running any scripts

## Non-Negotiable Reminders
1) Base new models on legacy `psi_model` pairwise embeddings + Evoformer logic.  
2) Integrate EvoDesign/EvoEF2 energies and Monte Carlo sidechain moves into scoring/ranking.  
3) First milestone: train an easy all-PDB protein-protein interaction model for `flowtcr_fold/TCRFold_Light` before heavier tasks.  
4) Keep outputs/checkpoints under `flowtcr_fold/checkpoints` or `results/`; avoid polluting legacy dirs.

## Current Snapshot (from flowtcr_fold/README, 2025-12-05)
- Pipeline: Stage1 scaffold retrieval (Immuno-PLM) → Stage2 CDR3β generation (FlowTCR-Gen, Dirichlet flow matching) → Stage3 structure critique (TCRFold-Prophet + EvoEF2).
- Module status: 
  - **Immuno-PLM ✅ 90%** (R@10 Avg=93.2%, pMHC vs MHC-only ablation 完成)
  - **FlowTCR-Gen 🔧 95%** (代码完成 + ODE bug 修复, 待重训验证)
  - **TCRFold-Prophet 🔄 60%** (Phase 0 ✅ 76,407 PPI 样本, Phase 3A 待训练)
  - Physics module ~90%; Inference pipeline ~50%.
- Data fields: peptide, mhc, cdr3_b (required); h_v, h_j, l_v, l_j optional; cdr3_a optional. Scaffold bank built from V/J combos.
- Conditioning sources reused: topology bias + hierarchical pairs from `psi_model/model.py`; Evoformer blocks from legacy `conditioned`.

## Master Plan v3.1 (aligned)
- Stage split: Stage1 Immuno-PLM scaffold prior (strong MHC, weak peptide; no CDR3 input to encoder), Stage2 FlowTCR-Gen CDR3β generation (Dirichlet flow + psi_model collapse/hier pairs), Stage3 TCRFold-Prophet structure/energy (Evoformer trunk + EvoEF2 surrogate + MC).
- Practical vs exploratory: Practical = dual InfoNCE+BCE, flow+CFG, post-hoc E_φ/MC; Exploratory = guided flow with sparse ∇E_φ steps, smarter MC proposals.
- Stage1 model: ESM2 650M + LoRA (rank16, alpha32 on Q/K/V/FFN) with allele token; dual-group multi-positive InfoNCE (MHC main, pMHC λ≈0.3) + multi-label BCE on gene IDs (pos_weight/focal). Metrics: Top-K recall & KL vs frequency baseline and MHC-only model.
- Stage2 model: reuse psi_model Collapse token + hierarchical pair IDs + Evoformer; Dirichlet flow head with entropy/profile regs; CFG (p=0.1 drop cond; tunable w); expose model-score hook for hybrid energy.
- Stage3 model: TCRFold-Prophet = Evoformer-Single trunk + IPA struct head + E_φ (EvoEF2-NN). Data A=PPI for 3A struct pretrain + 3B energy fit (decoys optional); Data B=TCR3d/STCRDab for 3C finetune; target ≥0.7 corr to EvoEF2 on TCRs. MC uses E_φ or α·E_φ+β·model score; optional guided flow integration.
- Execution timeline: (T1) Stage1 grouping/loss wiring + Top-K/KL baselines; (T2) FlowTCR-Gen baseline with CFG + recon/diversity + model-score; (T3) Stage3 phases 3A/3B PPI then 3C TCR finetune; (T4) Integration: Flow samples → TCRFold-Prophet+E_φ screen → MC (hybrid energy) → EvoEF2 final check; later explore guided flow.

## Plan Review v3.1 (feasibility)
- Overall scores: 概念完整性 ⭐⭐⭐⭐⭐, 技术可行性 ⭐⭐⭐⭐☆, 实现复杂度 ⭐⭐⭐☆☆, 创新性 ⭐⭐⭐⭐⭐, Practical/Exploratory 划分 ⭐⭐⭐⭐⭐; 结论 ✅ 高度可行。
- Stage1: feasible (ESM2+LoRA, allele emb, dual InfoNCE, multi-label BCE, Top-K/KL, MHC-only baseline). Watch for allele cold-start (seq-only/NN fallback), precompute pos_mask, start λ_pmhc≈0.3, λ_bce≈0.2. Expected R@10 20–40% (from 1.1%), KL(model)<KL(baseline).
- Stage2: feasible (CollapseAwareEmbedding, SequenceProfileEvoformer, 7-level pairs, Dirichlet flow, CFG, entropy/profile regs). Adjust for long sequences (truncate MHC/chunked attn), inject x_t via `x_proj(x_t)+pos_emb`, fix flow head vocab (20/21).
- Stage3: feasible but resource-heavy: 3A PPI pretrain (50k, 3–7d @4×A100~40GB), 3B energy fit (1–2d ~20GB), 3C TCR finetune (hrs ~16GB). Risks: E_φ corr<0.7 → add decoys/ranking loss; guided ODE heavy → keep exploratory. MC with E_φ using existing sampler.
- Timeline (12–16w): W1-2 Stage1 Practical (R@10>20%, KL<baseline); W3-5 Stage2 Practical (recovery>30%, ppl<10); W6-8 Stage3 3A/3B (corr>0.6); W9-10 Stage3 3C+MC (corr>0.7); W11-12 end-to-end/paper; W13+ exploratory (guided ODE, grad-MC, self-play).
- Hygiene: data versions `trn_v1/2`, `scaffold_bank_v1`, `energy_labels/`; checkpoints `stage1_v1/v2`, `stage2_v1`, `stage3_phase_a/b/c`, `pipeline_v1`; ablations (Stage1 MHC-only vs pMHC; Stage2 ±collapse/±hier pairs; Stage3 E_φ vs EvoEF2 ranking).
- Detailed pipeline alignment:
  - Stage1: scaffold prior; MHC strong, peptide weak; CDR3β excluded from encoder. Inputs include allele token + ESM on mhc_seq+peptide; HV/HJ/LV/LJ seq for InfoNCE; ids for BCE. Loss L = L_NCE_MHC + λ_pmhc·L_NCE_pMHC + λ_bce·L_BCE. Metrics Top-K & KL vs frequency/MHC-only baselines.
  - Stage2: layout [ψ, CDR3β, peptide, MHC, scaffold]; 7-level pair IDs; x_t injected via `x_proj(x_t)+pos_emb`; Dirichlet flow head with entropy/profile regs; CFG p=0.1; long-seq mitigation (truncate/chunk MHC). Keep model-score hook.
  - Stage3: Evoformer-Single + IPA + E_φ surrogate; Data A PPI (3A struct, 3B energy), Data B TCR (3C finetune); MC with E_φ or hybrid; guided flow kept exploratory due to cost.
  - End-to-end: Stage1 priors → Stage2 samples (CFG) → Stage3 struct+E_φ screen → MC refine → EvoEF2 final check.

## Active Focus & Next Steps
- Immediate: Stage3 TCRFold-Light/Prophet PPI pretrain (Phase 3A) with distance/contact losses and checkpoint cadence; in parallel Stage1 dual-group InfoNCE+BCE wiring with Top-K/KL eval scripts.
- Short-term: FlowTCR-Gen baseline with collapse/pair reuse + Dirichlet flow + CFG; export model-score hook and recon/diversity metrics.
- Integration pass: plug E_φ surrogate into pipeline_impl for screening + MC (hybrid α·E_φ+β·model score); keep EvoEF2 as final sanity.
- Always log commands/metrics/artifact paths here; outputs stay in `flowtcr_fold/checkpoints` or `results/`.

## Command Cheatsheet (FlowTCR-Fold)
- Immuno-PLM (LoRA+ESM target): `python flowtcr_fold/Immuno_PLM/train_plm.py --data data/trn.csv --use_esm --use_lora --esm_model esm2_t33_650M_UR50D --lora_rank 8 --batch_size 32 --epochs 100 --out_dir checkpoints/plm`
- FlowTCR-Gen: `python flowtcr_fold/FlowTCR_Gen/train_flow.py --data data/trn.csv --epochs 100 --batch_size 32 --lr 1e-4 --out_dir checkpoints/flow`
- TCRFold-Light PPI/TCR training: pending new entrypoint(s) (previous stubs removed); add commands once real scripts land under `flowtcr_fold/TCRFold_Light/`.
- End-to-end design (once ready): `python flowtcr_fold/FlowTCR_Gen/pipeline_impl.py --peptide <pep> --mhc <allele> --top_k_scaffolds 10 --samples_per_scaffold 100 --output results/designs.csv`
- Stage3 Phase0 (stage-local scripts under `flowtcr_fold/TCRFold_Light/process_pdb/`):
  - Download by ID list: `python flowtcr_fold/TCRFold_Light/process_pdb/download_from_id_list.py --id_file flowtcr_fold/data/pdb/batch1.txt --out_dir flowtcr_fold/data/pdb_structures/raw --num_workers 16`
  - Preprocess PPI pairs: `python flowtcr_fold/TCRFold_Light/process_pdb/preprocess_ppi_pairs.py --pdb_dir flowtcr_fold/data/pdb_structures/raw --out_dir flowtcr_fold/data/pdb_structures/processed --cutoff 8.0 --min_len 30 --min_contacts 10`
  - (After EvoEF2 present) Batch energies: `python flowtcr_fold/TCRFold_Light/process_pdb/compute_evoef2_batch.py --pdb_dir flowtcr_fold/data/pdb_structures/raw --output flowtcr_fold/data/energy_cache.jsonl --repair`

## Multi-Agent Coordination Structure

This window serves as the **Master Planning Window**. Each Stage has its own Implementation Plan:

| Stage | Implementation Plan | Status |
|-------|---------------------|--------|
| Stage 1 | `flowtcr_fold/Immuno_PLM/IMPLEMENTATION_PLAN.md` | ✅ 90% (R@10 Avg=93.2%) |
| Stage 2 | `flowtcr_fold/FlowTCR_Gen/IMPLEMENTATION_PLAN.md` | 🔧 95% (Bug Fixed, 待重训) |
| Stage 3 | `flowtcr_fold/TCRFold_Light/IMPLEMENTATION_PLAN.md` | 🔄 60% (Phase 0 ✅ 76,407 样本) |

### Workflow
1. **This Window**: Master planning, progress review, cross-stage coordination
2. **Other Chats**: Execute specific tasks by referencing Stage-specific `IMPLEMENTATION_PLAN.md`
3. **Sync Back**: Report completed checklist items back to this window for progress tracking

### Key Files
- Master README: `flowtcr_fold/README.md` (v3.1)
- Stage 1 Plan: `flowtcr_fold/Immuno_PLM/IMPLEMENTATION_PLAN.md`
- Stage 2 Plan: `flowtcr_fold/FlowTCR_Gen/IMPLEMENTATION_PLAN.md`
- Stage 3 Plan: `flowtcr_fold/TCRFold_Light/IMPLEMENTATION_PLAN.md`

## Progress Log
- [x] Stage1 dual-group InfoNCE+BCE wired; Top-K/KL vs freq & MHC-only baselines recorded. ✅ (R@10 Avg=93.2%)
- [~] Stage2 FlowTCR-Gen baseline: 代码完成 + Bug 修复; 首轮 buggy 训练分析完成; **待重新训练**
- [x] Stage3 Phase 0 数据管线全量完成: 76,407 PPI 样本 (Tier 1+2+3 完整)
- [ ] Stage3 Phase 3A/3B PPI pretrain/energy fit completed; checkpoints + EvoEF2 corr logged.
- [ ] Stage3 Phase 3C TCR-specific finetune done; corr ≥0.7 achieved/assessed.
- [ ] Pipeline integration: Flow samples → TCRFold-Prophet+E_φ screen → MC (hybrid energy) → EvoEF2 final check; commands + outputs recorded.

### 详细进度 (2025-12-05 更新)
| Stage | 状态 | 关键成果 | 下一步 |
|-------|------|----------|--------|
| Stage 1 | ✅ 完成 | R@10 Avg=93.2%, pMHC vs MHC-only Δ=+0.5% | 可选探索 |
| Stage 2 | 🔧 待重训 | 代码完成, ODE bug 修复, 首轮分析完成 | 重新训练验证 |
| Stage 3 | 🔄 60% | **Phase 0 ✅** 76,407 样本, EvoEF2 能量合并 | Phase 3A 模型训练 |

## Stage-Specific Progress (sync from IMPLEMENTATION_PLAN.md)

### Stage 1: Immuno-PLM (W1-2) ✅ 完成
- [x] Phase 1: 数据准备 (gene name cleanup, AlleleVocab, pos_mask)
- [x] Phase 2: Multi-positive InfoNCE 实现
- [x] Phase 3: Multi-label BCE 实现
- [x] Phase 4: Top-K/KL 评估指标
- [x] Phase 5: Baseline 对比
- [x] **Phase 6 (Ablation)**: pMHC vs MHC-only 完成 (Δ ≈ +1.2% HV, +0.9% HJ)
- [x] **Milestone**: R@10 Avg = 93.2% (远超 20% 目标)

**Latest Results (2025-12-04 @ Epoch 11-13)**:
| Mode | R@10 HV | R@10 HJ | R@10 LV | R@10 LJ | R@10 Avg |
|------|---------|---------|---------|---------|----------|
| Normal (pMHC) | **89.5%** | **83.7%** | 99.7% | 99.9% | **93.2%** |
| Ablation (MHC-only) | 88.3% | 82.8% | 99.7% | 99.8% | 92.7% |
| Frequency Baseline | 39.3% | 74.4% | 33.1% | 23.6% | 42.6% |

**vs Baseline**: HV +50.2%, HJ +9.3%, LV +66.6%, LJ +76.3%

**Exploratory (Stage 1)**:
- [ ] E1: Allele Sequence Fallback
- [ ] E2: Hard Negative Mining
- [ ] E3: Contrastive + Generative Joint Training
- [ ] E4: Causal LM Head for Generative Scaffold

### Stage 2: FlowTCR-Gen (W3-5) 🔧 Bug 已修复，待重训
- [x] Phase 1: 复用 psi_model 组件 (CollapseAwareEmbedding, SequenceProfileEvoformer)
- [x] Phase 2: Dirichlet Flow Matching (dirichlet_flow.py)
- [x] Phase 3: CFG 实现 (CFGWrapper, cfg_drop_prob)
- [x] Phase 4: Model Score Hook (get_model_score, get_collapse_scalar)
- [x] Phase 5: 评估指标 (metrics.py: recovery, diversity, ppl)
- [x] **Phase 5.5: Bug 修复** (2025-12-05)
  - ✅ ODE simplex 投影修复 (softmax → normalize)
  - ✅ 评估参数优化 (n_samples 3→8, n_steps 50→100)
  - ✅ Per-sample conditioning 完整实现
  - ✅ Padding mask 进 entropy 正则
- [ ] **Phase 6 (Ablation)**: ±Collapse, ±Hier Pairs, CFG sweep → 待训练运行
- [ ] **Milestone**: Recovery > 30%, PPL < 10 → 待训练验证

**首轮训练分析 (Buggy Version, 2025-12-04~05)**:

| 发现 | 说明 |
|------|------|
| ✅ Loss 收敛正常 | MSE 从 0.1 降到 0.001，模型架构正确 |
| ❌ Recovery = 0 | ODE simplex 投影 bug 导致 |
| ⚠️ Diversity 急剧下降 | 0.99 → 0.01，可能是 bug + mode collapse |
| 📊 No Collapse 收敛更快 | 参数量少，但可能欠拟合 |
| ⏱️ No Hier 训练更快 | 节省 ~32% 时间 |

详细分析见 `flowtcr_fold/FlowTCR_Gen/IMPLEMENTATION_PLAN.md` Section 10-12。

**新增文件 (2025-12-03)**:
- `encoder.py`: FlowTCRGenEncoder + CollapseAwareEmbedding
- `dirichlet_flow.py`: DirichletFlowMatcher + CFGWrapper
- `model_flow.py`: FlowTCRGen 主模型
- `data.py`: Dataset + Tokenizer
- `metrics.py`: 评估指标
- `train.py`: 训练脚本 (支持 --ablation)

**Exploratory (Stage 2)**:
- [ ] E1: Physics Gradient Guidance in ODE
- [ ] E2: Entropy Scheduling
- [ ] E3: Multi-CDR Generation
- [ ] E4: Self-Play with Stage 3 Feedback

### Stage 3: TCRFold-Prophet (W6-10) 🔄 60%
- [x] **Phase 0: 数据准备** ✅ **全量完成** (2025-12-04)
  - ✅ PDB 下载: 37,867 结构 (35,398 PDB + 2,469 CIF)
  - ✅ PPI 预处理: 78,896 .npz (Tier 2 结构特征)
  - ✅ EvoEF2 能量: 209,826 链对 (Tier 1+3 能量)
  - ✅ **合并样本: 76,407** (Tier 1+2+3 完整，26 字段)
  - ✅ PPIDataset 统一数据集类
- [ ] Phase 3A: PPI 结构预训练 → **数据已就绪，待模型训练**
- [ ] Phase 3B: 能量 Surrogate 训练
- [ ] Phase 3C: TCR 微调
- [ ] Phase MC: Monte Carlo 集成
- [ ] **Phase Ablation**: E_φ vs EvoEF2 ranking, ±Decoy, MC weights
- [ ] **Milestone**: corr > 0.7 with EvoEF2 on TCR

**Phase 0 数据质量**:
- E_bind 范围: [-200, +50] kcal/mol
- 接触数: 10-200 contacts/pair
- 序列长度: 30-500 AA/chain
- 数据路径: `flowtcr_fold/data/pdb_structures/merged/`

**Exploratory (Stage 3)**:
- [ ] E1: Gradient Guidance in Flow ODE
- [ ] E2: MC-to-Training Loop (Self-Play)
- [ ] E3: Gradient-Informed MC Proposal
- [ ] E4: Structure Prediction Head (IPA)
- [ ] E5: Binding Affinity Regression
