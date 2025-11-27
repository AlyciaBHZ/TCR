# Repository Guidelines

## Project Structure & Module Organization
- `conditioned/`: baseline CDR3beta Evoformer models; trained checkpoints live in `saved_model/`.
- `psi_model/`: psiCLM enhancements and composite-loss training (`train.py`, `model.py`).
- `pretrain_TCR/`: two-stage pretraining and paired finetuning; checkpoints under `pretrained_model/`.
- `scripts/generation/`: production pipeline for generation and synthesis (`generate_cdr3b_wetlab.py`, `create_synthesis_constructs.py`, `rescoring_with_alpha.py`). Full-chain work in `scripts/generation/tcr_sidechain/` (Goal 2 WIP with demos/tests).
- `flowtcr_fold_v3/`: architecture notes for the future flow-based variant; docs only.
- `data/`: curated CSVs (`final_data/trn.csv`, etc.). Keep raw data elsewhere.
- `results/`: generated candidates and synthesis artifacts; prefer writing new outputs here.
- `docs/`: archived analyses and historical notes.

## Build, Test, and Development Commands
- Generate wet-lab candidates: `python scripts/generation/generate_cdr3b_wetlab.py --targets wetlab_targets.csv --condition 1 --n_samples 50 --output results/new_candidates.csv`.
- Train baseline Goal 1 model: `cd conditioned && python train.py -c 1`.
- Train psiCLM variant: `cd psi_model && python train.py --loss_type composite`.
- Full-chain WIP training: `cd scripts/generation/tcr_sidechain && python train.py --task joint --epochs 1 --output_dir checkpoints/` (adjust flags; torch-heavy).
- Smoke tests: `python scripts/generation/tcr_sidechain/test_training_loop.py` then `python scripts/generation/tcr_sidechain/test_evaluate.py` (mocks wandb; writes to `test_output/`).

## Coding Style & Naming Conventions
- Python-first; follow PEP 8 with 4-space indents, snake_case for functions/vars, PascalCase for classes.
- Keep runs deterministic where possible: seed torch/numpy and reuse small CSV/JSONL fixtures in `data/` or `scripts/generation/tcr_sidechain/dataraw`.
- Do not commit large checkpoints or generated CSV/FASTA; keep under `results/`, `saved_model/`, or `test_output/` and extend `.gitignore` if new.
- Use descriptive filenames (`*_wetlab.py`, `*_constructs.csv`); prefer lower_snake_case for new scripts and config flags.

## Testing Guidelines
- No unified test runner; rely on the targeted smoke scripts above for Goal 2 plus small-batch training/eval runs before long jobs.
- For generation changes, run a short sample: `python scripts/generation/generate_cdr3b_wetlab.py --targets wetlab_targets.csv --n_samples 5 --output results/sanity.csv` and verify columns and sequence lengths.
- Validate new data loaders or loss tweaks with tiny JSONL/CSV subsets to avoid GPU-heavy cycles; stash outputs in `results/` or `scripts/generation/tcr_sidechain/test_output/`.
- When altering sampling, scoring, or loss weighting, note the command and key metrics (loss, confidence) alongside the change.

## Commit & Pull Request Guidelines
- Use imperative, concise commit titles (<=72 chars) with scope hints (e.g., `conditioned: fix masking`, `generation: add fasta export`).
- PRs should state the goal (Goal 1 vs Goal 2), commands run, paths to produced artifacts, and data assumptions; attach small sample outputs when possible.
- Avoid committing proprietary datasets or large checkpoints; provide download or regeneration steps instead.
- Keep diffs focused; separate modeling logic from data wrangling and from documentation when feasible.
- Training preferences:
  - Save checkpoints every 50 epochs.
  - Early stop if no improvement for 100 epochs (mirror `conditioned/train.py` early-stop/checkpoint logic).
