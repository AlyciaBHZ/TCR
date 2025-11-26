# FlowTCR Work Plan & Progress

Status legend: [ ] pending, [~] in-progress, [x] done.

## High-Level Tracks
- [~] Immuno-PLM (Hybrid Encoder)
- [~] TCRFold-Light (Geometry critic + energy head)
- [ ] FlowTCR-Gen (Discrete flow)
- [ ] Physics integration (EvoEF2, TM-align profiles)
- [ ] Inference loop + benchmarks

## Immediate Tasks
- [~] Data layer: triplet sampler with decoy/mutant negatives; region slices for topology bias. (identity window 0.6-0.9, controlled CDR3 mutations, neg_type tags; still heuristic)
- [~] Immuno-PLM code: ESM-backed encoder with topology bias hook; pooled embeddings for InfoNCE.
- [~] PLM training loop: stable InfoNCE with batching, masking, logging. (basic per-step log list; no wandb yet)
- [ ] Integrate real decoy generation (sequence identity filter) and mutant synthesis. (needs BLAST/align; current heuristic only)
- [~] Add evaluation harness (retrieval/InfoNCE metrics, sanity checks on embeddings). (eval_plm scaffold added)
- [ ] Physics hooks: EvoEF2 runner, TM-align profiles (stubs added, need real calls)
- [ ] Flow generator: implement real flow matching loss/conditioning (current placeholder)
- [ ] Inference: replace mock refine with EvoEF2, add scoring/ranking, integrate benchmarks (tFold/AF2 stubs)

## Progress Log
- 2025-11-25: Scaffolded data/model/training/inference packages under `flowtcr_fold/`; restored original README content; initial dataset/encoder/flow stubs in place.
- 2025-11-26: Added decoy/mutation heuristics, batch topology bias support, MLM option, eval scaffold; added physics/benchmarks stubs and inference pipeline skeleton.
