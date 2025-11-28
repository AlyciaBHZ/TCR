# FlowTCR-Fold v2.0 Task Tracking

> **Two-Stage Design Strategy**: Scaffold Retrieval → CDR3β Generation

Status legend: `[ ]` pending, `[~]` in-progress, `[x]` done

---

## High-Level Tracks

| Track | Status | Description |
|-------|--------|-------------|
| **Stage 1: Scaffold Retrieval** | 🔄 In Progress | Immuno-PLM with InfoNCE |
| **Stage 2: CDR3β Generation** | ⬜ Pending | FlowTCR-Gen with conditioning |
| **Stage 3: Structure Critique** | ⬜ Pending | TCRFold-Light + EvoEF2 |
| **Pipeline Integration** | ⬜ Pending | End-to-end workflow |

---

## Phase 1: Immuno-PLM Validation (Current Priority)

### Scaffold Bank Construction
- [ ] Extract unique V/J combinations from trn.csv
- [ ] Filter low-frequency combinations (count < 5)
- [ ] Save to `data/scaffold_bank.csv`
- [ ] Pre-compute scaffold embeddings

### Batch InfoNCE Training
- [~] Implement `compute_batch_infonce()` loss
- [~] Update `train_plm.py` to use batch random negatives
- [ ] Run smoke test with small batch
- [ ] Verify loss convergence

### Retrieval Evaluation
- [ ] Implement `evaluate_retrieval()` function
- [ ] Compute Recall@K (K=1, 5, 10)
- [ ] **Milestone**: Recall@10 > 50%

---

## Phase 2: FlowTCR-Gen Development

### Conditioning Mechanism
- [ ] Encode pMHC with Immuno-PLM
- [ ] Encode scaffold with Immuno-PLM
- [ ] Concatenate conditions for Flow input
- [ ] (Optional) Add TM-align PSSM conditioning

### Flow Matching Core
- [~] Implement `flow_matching_loss()`
- [ ] Implement `FlowMatchingModel.sample()`
- [ ] Train with pMHC + scaffold conditions
- [ ] Validate generation quality

### Generation Evaluation
- [ ] Perplexity metric
- [ ] Length distribution analysis
- [ ] Motif recall (known CDR3 patterns)
- [ ] **Milestone**: Perplexity < 10

---

## Phase 3: Structure Critique Integration

### TCRFold-Light Training
- [x] EvoEF2 wrapper (`physics/evoef_runner.py`)
- [x] Energy dataset (`physics/energy_dataset.py`)
- [x] Training script (`train_with_energy.py`)
- [ ] Collect PDB structures (STCRDab/TCR3d)
- [ ] Train with energy supervision

### Scoring Interface
- [ ] Implement `TCRFoldLight.score(scaffold, cdr3b)`
- [ ] Output: confidence, contact_density, energy
- [ ] Set filtering thresholds

---

## Phase 4: Pipeline Integration

### End-to-End Workflow
- [ ] Create `TCRDesignPipeline` class
- [ ] Stage 1: Retrieve Top-K scaffolds
- [ ] Stage 2: Generate CDR3β candidates
- [ ] Stage 3: Score and filter
- [ ] (Optional) EvoEF2 refinement
- [ ] Final ranking

### Command-Line Interface
- [ ] Update `pipeline_impl.py` with new workflow
- [ ] Add `--peptide`, `--mhc` arguments
- [ ] Add `--top_k_scaffolds`, `--samples_per_scaffold`
- [ ] Output to CSV

---

## Infrastructure

### Documentation
- [x] Update README.md (English, comprehensive)
- [x] Create Plan_v2.0.md (Chinese, detailed)
- [x] Update USER_MANUAL.md
- [ ] Add evaluation scripts documentation

### Testing
- [ ] Smoke test for Immuno-PLM
- [ ] Smoke test for FlowTCR-Gen
- [ ] Integration test for full pipeline

### External Tools
- [x] EvoEF2 wrapper complete
- [ ] Compile EvoEF2 binary
- [ ] (Optional) TM-align integration
- [ ] (Optional) BLAST for hard negatives

---

## Progress Log

| Date | Update |
|------|--------|
| 2025-11-28 | v2.0 Plan: Two-stage strategy (Retrieve & Generate) |
| 2025-11-28 | Updated README.md and created Plan_v2.0.md |
| 2025-11-26 | EvoEF2 integration complete |
| 2025-11-25 | Initial scaffolding |

---

## Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| M1: Immuno-PLM Retrieval | Recall@10 > 50% | ⬜ |
| M2: FlowTCR-Gen Quality | Perplexity < 10 | ⬜ |
| M3: Pipeline E2E | Working end-to-end | ⬜ |
| M4: Baseline Comparison | Beat random baseline | ⬜ |
