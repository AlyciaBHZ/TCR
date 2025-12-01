# FlowTCR-Fold Agent Charter

## Scope
- Primary job: help build the full pipeline in `/mnt/rna01/zwlexa/project/TCR/flowtcr_fold`. Other folders are legacy reference/data; do not drift focus.
- Track progress and self-reminders here; update as milestones complete.

## Non-Negotiable Reminders
1) Base new models on legacy `psi_model` pairwise embeddings + Evoformer logic.  
2) Integrate EvoDesign/EvoEF2 energies and Monte Carlo sidechain moves into scoring/ranking.  
3) First milestone: train an easy all-PDB protein-protein interaction model for `flowtcr_fold/TCRFold_Light` before heavier tasks.  
4) Keep outputs/checkpoints under `flowtcr_fold/checkpoints` or `results/`; avoid polluting legacy dirs.

## Current Snapshot (from flowtcr_fold/README)
- Pipeline: Stage1 scaffold retrieval (Immuno-PLM) → Stage2 CDR3β generation (FlowTCR-Gen, discrete flow matching) → Stage3 structure critique (TCRFold-Light + EvoEF2).
- Module status: Immuno-PLM ✅ (InfoNCE + cls); FlowTCR-Gen ~40% (needs full conditioning); TCRFold-Light ~75% (EvoEF2 integration present); Physics module ~90%; Inference pipeline ~50%.
- Data fields: peptide, mhc, cdr3_b (required); h_v, h_j, l_v, l_j optional; cdr3_a optional. Scaffold bank built from V/J combos.
- Conditioning sources to reuse: topology bias + hierarchical pairs from `psi_model/model.py`; Evoformer blocks from legacy `conditioned` where needed.

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
- TCRFold-Light PPI pretrain (first milestone): `python flowtcr_fold/TCRFold_Light/train_ppi_impl.py --pdb_dir data/pdb_structures --cache_dir data/energy_cache --epochs 100 --batch_size 4 --out_dir checkpoints/tcrfold`
- End-to-end design (once ready): `python flowtcr_fold/FlowTCR_Gen/pipeline_impl.py --peptide <pep> --mhc <allele> --top_k_scaffolds 10 --samples_per_scaffold 100 --output results/designs.csv`

## Multi-Agent Coordination Structure

This window serves as the **Master Planning Window**. Each Stage has its own Implementation Plan:

| Stage | Implementation Plan | Status |
|-------|---------------------|--------|
| Stage 1 | `flowtcr_fold/Immuno_PLM/IMPLEMENTATION_PLAN.md` | 🔄 70% |
| Stage 2 | `flowtcr_fold/FlowTCR_Gen/IMPLEMENTATION_PLAN.md` | 🔄 40% |
| Stage 3 | `flowtcr_fold/TCRFold_Light/IMPLEMENTATION_PLAN.md` | 🔄 30% |

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
- [ ] Stage1 dual-group InfoNCE+BCE wired; Top-K/KL vs freq & MHC-only baselines recorded.
- [ ] Stage2 FlowTCR-Gen baseline (Dirichlet flow + CFG) trained; recon/diversity + model-score hook logged.
- [ ] Stage3 Phase 3A/3B PPI pretrain/energy fit completed; checkpoints + EvoEF2 corr logged.
- [ ] Stage3 Phase 3C TCR-specific finetune done; corr ≥0.7 achieved/assessed.
- [ ] Pipeline integration: Flow samples → TCRFold-Prophet+E_φ screen → MC (hybrid energy) → EvoEF2 final check; commands + outputs recorded.

## Stage-Specific Progress (sync from IMPLEMENTATION_PLAN.md)

### Stage 1: Immuno-PLM (W1-2)
- [ ] Phase 1: 数据准备 (gene name cleanup, AlleleVocab, pos_mask)
- [ ] Phase 2: Multi-positive InfoNCE 实现
- [ ] Phase 3: Multi-label BCE 实现
- [ ] Phase 4: Top-K/KL 评估指标
- [ ] Phase 5: Baseline 对比
- [ ] **Phase 6 (Ablation)**: pMHC vs MHC-only, ±BCE, λ_pmhc sweep
- [ ] **Milestone**: R@10 > 20%, KL < baseline

**Exploratory (Stage 1)**:
- [ ] E1: Allele Sequence Fallback
- [ ] E2: Hard Negative Mining
- [ ] E3: Contrastive + Generative Joint Training
- [ ] E4: Causal LM Head for Generative Scaffold

### Stage 2: FlowTCR-Gen (W3-5)
- [ ] Phase 1: 复用 psi_model 组件
- [ ] Phase 2: Dirichlet Flow Matching
- [ ] Phase 3: CFG 实现
- [ ] Phase 4: Model Score Hook
- [ ] Phase 5: 评估指标
- [ ] **Phase 6 (Ablation)**: ±Collapse, ±Hier Pairs, CFG sweep, Conditioning components
- [ ] **Milestone**: Recovery > 30%, PPL < 10

**Exploratory (Stage 2)**:
- [ ] E1: Physics Gradient Guidance in ODE
- [ ] E2: Entropy Scheduling
- [ ] E3: Multi-CDR Generation
- [ ] E4: Self-Play with Stage 3 Feedback

### Stage 3: TCRFold-Prophet (W6-10)
- [ ] Phase 0: 数据准备 (PDB + EvoEF2)
- [ ] Phase 3A: PPI 结构预训练
- [ ] Phase 3B: 能量 Surrogate 训练
- [ ] Phase 3C: TCR 微调
- [ ] Phase MC: Monte Carlo 集成
- [ ] **Phase Ablation**: E_φ vs EvoEF2 ranking, ±Decoy, MC weights
- [ ] **Milestone**: corr > 0.7 with EvoEF2 on TCR

**Exploratory (Stage 3)**:
- [ ] E1: Gradient Guidance in Flow ODE
- [ ] E2: MC-to-Training Loop (Self-Play)
- [ ] E3: Gradient-Informed MC Proposal
- [ ] E4: Structure Prediction Head (IPA)
- [ ] E5: Binding Affinity Regression
