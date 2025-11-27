# FlowTCR-Fold: Detailed Design & Plan

> Flow Matching + topology/PSI priors + physics guidance for TCR-pMHC design, with a self-correcting generate–critic–refine loop.

## 0. Core Ideas
- **Flow backbone:** Discrete Flow Matching (Dirichlet conditional flow) replaces AR decoding; global, non-causal sampling eases geometry/physics conditioning.
- **Physics priors:** EvoEF2 energy + structural profiles act as strong priors and differentiable surrogates (Energy Head), not just post-filters.
- **Explicit topology/PSI:** Inject TCR-pMHC multi-chain topology (Region/Pair bias) on top of ESM; reuse psi_model hierarchical pair embeddings and Collapse token for richer pairwise structure.

## 1. Modules (kept separate) and Legacy Reuse
### Immuno-PLM (Hybrid Encoder)
- Role: sequence + pair representations; combine ESM-2 physics with explicit topology/PSI bias.
- Legacy: conditioned/model.py (Embedding region/pair bias, ~85–117); psi_model/model.py (create_hierarchical_pairs, collapse token); data patterns from conditioned/data.py.
- Input format: [CLS] + peptide + [SEP] + MHC + [SEP] + CDR3B + [SEP] (optionally alpha-chain fields when available).
- Loss: InfoNCE + optional MLM.
  - InfoNCE: for anchor z_a, positive z_p, negatives {z_n}, temperature tau:
    
    L_InfoNCE = -log( exp(sim(z_a,z_p)/tau) / (exp(sim(z_a,z_p)/tau) + sum_n exp(sim(z_a,z_n)/tau)) )
    
  - MLM: standard token masking on the same sequence.

### TCRFold-Light (Geometry + Energy Critic)
- Role: MSA-free Evoformer-lite predicting distance/contact plus Energy Head (EvoEF2 surrogate).
- Legacy: conditioned/src/Evoformer.py (keep TriangleUpdate/Attention, drop MSA row/col); pair input = Z_final from Immuno-PLM / PSI.
- Training phases: (3.1) PPI pretrain on generic PDB contacts/physics; (3.2) TCR finetune on STCRDab/TCR3d focusing on interface contact/energy.
- Losses (targeted, to be wired with real data): distance (e.g., binned CE), contact (BCE), energy regression vs EvoEF2 delta-delta-G; weighted toward interface residues.

### FlowTCR-Gen (Two-stage Flow Generation)
- Method: Dirichlet Flow Matching on categorical simplex; conditions = pMHC + V/J scaffold + TM-align PSSM + geometry summaries (from TCRFold-Light).
- Stages: (1) generate CDR3B; (2) inpaint/refine full chain, align attention with contact.
- Flow matching setup:
  - Base x0: uniform over vocab; target y: one-hot tokens.
  - Interpolant: x_t = (1 - t) x0 + t y.
  - Vector field target: v* = y - x0.
  - Loss: L_Flow = || v_theta(x_t, t, cond) - (y - x0) ||^2.
- Additional terms: Attention-contact alignment (cross-attn vs TCRFold contact), Energy surrogate (Energy Head penalty).

### Self-correcting inference (target)
1) Flow sampling (Stage1/Stage2, large pool) conditioned on pMHC/scaffold/profile.
2) TCRFold-Light critique (pLDDT-like, contact density, energy surrogate).
3) EvoEF2 MC repacking + ComputeBinding for top set.
4) Rank by energy + structural confidence.

## 2. Data & Hard Negatives
- Fields: peptide,mhc,cdr3_b,(h_v,h_j,l_v,l_j,cdr3_a optional); clean via data/convert_csv_to_jsonl.py.
- Triplet sampling (data/dataset.py):
  - Peptide decoy: same MHC, identity >=0.8, different peptide.
  - CDR3 decoy/mutant: same MHC, different CDR3, optional 2–3 point mutations; 
eg_type emitted.
- Physics cleaning: EvoEF2 RepairStructure; binding energy via ComputeBinding (stubs in physics/).

## 3. Code Layout (kept modular) and Legacy Pointers
flowtcr_fold/
  README.md, TODO.md
  docs/ USER_MANUAL.md, initial_plan*.md
  data/ dataset.py, tokenizer.py, convert_csv_to_jsonl.py
  common/ utils.py (ckpt every 50 epochs; early stop after 100)
  Immuno_PLM/ immuno_plm.py, train_plm.py, eval_plm.py
  TCRFold_Light/ tcrfold_light.py, train_ppi_impl.py, train_tcr_impl.py, train_struct_impl.py, train_with_energy.py
  FlowTCR_Gen/ flow_gen.py, train_flow.py, pipeline_impl.py
  physics/ evoef_runner.py, energy_dataset.py (EvoEF2 wrapper + labels)
  tools/ EvoEF2/ (binary + params, see EVOEF2_INTEGRATION.md)
legacy refs:
  conditioned/model.py          # topology pair bias
  conditioned/src/Evoformer.py  # Evoformer backbone
  psi_model/model.py            # PSI hierarchical pair & collapse token
  conditioned/data.py           # data loading patterns

## 4. Training Preferences
- Save checkpoint every 50 epochs; early stop if no improvement for 100 epochs (common/utils.py wired into scripts).
- Batch semantics: PLM batches multiple sequences (token-level MLM per sequence); Flow/structure batches carry seq/pair/geometry/energy when available.

## 5. Status & TODO
- Implemented: data/tokenizer, hard-negative sampler, PLM training (InfoNCE+MLM), Evoformer-lite wrapper, simplified flow matching, EvoEF2 wrapper + energy dataset, early stop/ckpt, docs.
- Pipeline: Flow -> TCRFold-Light critic -> optional EvoEF2 refine (scaffold PDB) wired in FlowTCR_Gen/pipeline_impl.py.
- Pending: real PSI+topology fusion (psi_model pair logic), stronger decoys, real structure data + EvoEF2 energies/TM-align PSSM, full flow loss with conditioning + attention-contact alignment, AF2/tFold baselines.

## 6. Quickstart (placeholder)
- Clean: python flowtcr_fold/data/convert_csv_to_jsonl.py --input data/trn.csv --output data/trn.jsonl
- PLM train: python flowtcr_fold/Immuno_PLM/train_plm.py --data data/trn.csv --epochs 1 --batch_size 8
- PLM eval: python flowtcr_fold/Immuno_PLM/eval_plm.py --data data/val.csv --checkpoint checkpoints/plm/immuno_plm.pt
- Flow: python flowtcr_fold/FlowTCR_Gen/train_flow.py --data data/trn.csv
- TCRFold-Light (energy-supervised): python flowtcr_fold/TCRFold_Light/train_with_energy.py --pdb_dir data/pdb_structures

## 7. Target Inference Loop
Flow sampling -> TCRFold-Light critique -> EvoEF2 refine -> rank by energy + structure confidence.
