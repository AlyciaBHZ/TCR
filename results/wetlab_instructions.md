Wet‑Lab Instructions (Alpha + Beta)

File to use
- results/synthesis_ready_paired_constructs_rescored.csv — master table with rescored metrics.

Columns (what they mean)
- construct_id: Target‑scoped ID (target index + rank).
- peptide, mhc: Antigen context for each design.
- cdr3b_sequence: Designed CDR3β (short loop; ~9–16 aa typical).
- v_beta, j_beta: β‑chain V/J amino‑acid frameworks (from training data).
- v_alpha, j_alpha: α‑chain V/J amino‑acid frameworks (mined from training data). No CDR3α provided in this release.
- confidence, composite_score: Original generation/filtering scores.
- rescored_confidence, rescored_nll: Model rescoring using full conditioning (pep, mhc, lv, lj, hv, hj).

How to use it
- Per target (peptide/mhc), pick the top 2–3 constructs by rescored_confidence.
- Keep diversity: avoid near‑identical CDR3β within the same target.
- If rescored_confidence is missing (NaN), fall back to composite_score and confidence.

Cloning options
- Beta‑only screen: Use results/synthesis_ready_constructs.fasta (Vβ + CDR3β + Jβ amino‑acid) and co‑transfect with a standard α.
- Paired α+β (recommended):
  - Beta: v_beta + cdr3b_sequence + j_beta (from CSV or FASTA).
  - Alpha: v_alpha + j_alpha from CSV (no CDR3α).
  - Clone each variable domain upstream of TRAC/TRBC constant regions per your system.

Synthesis notes
- Beta variable domain length ~90–130 aa (we flag unusual lengths).
- Use standard amino‑acid alphabet; codon‑optimize DNA for your host with your provider.
