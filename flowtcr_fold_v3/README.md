# FlowTCR-Fold v3.0 — Implementation Plan & Architecture

This folder captures the plan and architecture for the FlowTCR-Fold v3.0 pipeline, independent of the previous sidechain pipeline.

## 1) High-Level Goals
- Build an autonomous, MSA-free TCR–pMHC generator with geometry awareness.
- Use ESM-2 warm-start, hard negatives, and self-correcting flow-based generation.
- Integrate a lightweight structure head to steer generation without relying on external oracles.

## 2) Modules
**Immuno-PLM (Enhanced Baseline)**  
- 16-layer Transformer (d=768) warm-start from ESM-2 (650M).  
- Loss: `L_PLM = L_MLM + λ_clip L_InfoNCE + λ_aux L_Aux`.  
  - MLM: 15% span masking.  
  - InfoNCE: in-batch + hard negatives (decoy epitope/TCR).  
  - Aux: V/J gene + MHC allele classification from `[CLS]`.

**TCRFold-Light (Geometry Predictor)**  
- MSA-free Evoformer-Lite, 12 layers, 1× recycle, fast inference.  
- Phase 1: general physics pretrain on PPI (dist/contact/SS/RSA).  
- Phase 2: TCR domain refinement (STCRDab/TCR3d) with interface emphasis:  
  `L_struct = L_global + 10.0 * L_interface_fape + 5.0 * L_interface_contact`.

**FlowTCR-Gen (Discrete Flow Matching Generator)**  
- State: sequence simplex; path: Dirichlet conditional flow matching.  
- Self-correcting loop:  
  1) Prior-guided generation with {z_pMHC, z_scaffold}.  
  2) Geometry critic via TCRFold-Light → structure/contact.  
  3) Structure-guided refinement with geometry summary g(Structure).
- Additional terms: attention entropy + attention–contact alignment.

## 3) Data & Hard Negatives
- Format: JSONL with `pep|mhc`, regions {l_v,l_j,h_v,h_j}, length buckets, hard/easy negatives, optional structure labels.  
- Hard negatives: length-bucket FAISS neighbors + decoy epitope/TCR (high identity but non-binder); refresh periodically.  
- Masking: 15% span; negatives include batch shuffle + prebuilt hard pool.

## 4) Training Roadmap
| Phase | Focus | Key Outputs |
|-------|-------|-------------|
| P1 (W1-2) | Infra + data loader + hard neg mining | JSONL + splits; ESM-2 load script |
| P2 (W3-4) | Immuno-PLM (MLM + InfoNCE + Aux) | Domain-adapted checkpoint; InfoNCE ↓ |
| P3 (W5-8) | TCRFold-Light | PPI pretrain → TCR interface finetune |
| P4 (W9-10) | FlowTCR-Gen | Flow matching loss + geometry alignment |
| P5 (W11+) | Full loop & eval | Self-correcting inference; benchmarks/plots |

## 5) Inference (Self-Correcting)
1. Generate coarse sequence with flow (no geometry).  
2. Score with TCRFold-Light → structure/contact.  
3. Refine/re-sample with geometry summary; stop on score threshold or max rounds.  
4. Output FASTA/CSV with scores (InfoNCE, contact alignment, confidence).

## 6) Evaluation
- Speed: vs AF2-Multimer, tFold-TCR on 100 pairs (ms vs L); target 10–50× faster.  
- Virtual screening: EF/AUROC on binder vs decoy sets (hard negatives + random).  
- Novelty vs energy: Levenshtein distance vs interface confidence/energy; aim for high novelty + high confidence.

## 7) Default Hyperparameters (initial draft)
- λ_clip ≈ 1.0, λ_aux ≈ 0.5–1.0; mask ratio 0.15.  
- InfoNCE temperature τ ≈ 0.1 (anneal to 0.05); K_hard=3, K_easy in-batch.  
- Structure weights: interface FAPE 10×, interface contact 5×; 1 recycle.  
- Flow: discrete flow matching steps tuned to keep single-sample latency low; attn entropy/align λ to be calibrated during P4.
