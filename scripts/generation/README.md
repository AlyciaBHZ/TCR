# TCR Generation Scripts - Usage Guide

**Location**: `scripts/generation/`
**Purpose**: Core scripts for generating CDR3β sequences and preparing synthesis constructs
**Last Updated**: 2025-11-13

---

## Overview

These 4 scripts form the essential pipeline for generating new TCR candidates and preparing them for wet-lab synthesis. Use these when:
- Designing TCRs for **new peptide-MHC targets**
- **Regenerating** candidates with different parameters
- Creating **alternative constructs** if initial candidates fail
- **Re-exporting** to different synthesis formats

---

## Scripts

### 1. `generate_cdr3b_wetlab.py` - CDR3β Generation

**Purpose**: Generate CDR3β sequences using trained model for given peptide-MHC targets

**Model Used**:
- Condition 1 (all features): `conditioned/saved_model/condition_1/model_epoch_3450`
- Best performing model from Phase 1 training

**Usage**:
```bash
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets wetlab_targets.csv \
  --condition 1 \
  --n_samples 50 \
  --temperature 0.8 \
  --output results/new_cdr3b_candidates.csv
```

**Parameters**:
- `--targets` (required): CSV with columns `peptide,mhc` (and optional `l_v,l_j,h_v,h_j`)
- `--condition` (default: 1): Conditioning scheme (1-7, use 1 for best results)
- `--n_samples` (default: 20): Number of CDR3β sequences to generate per target
- `--temperature` (default: 0.8): Sampling temperature (lower = more conservative)
- `--output` (required): Output CSV file path

**Output Format**:
```
target_idx,peptide,mhc,cdr3_sequence,confidence,length,generation_method
0,VVGAVGVGK,A*11:01,CASSNRNTIYF,0.864,11,iterative
```

**When to Use**:
- Starting with new peptide-MHC targets
- Want to generate more candidates for existing targets
- Testing different generation parameters (temperature, n_samples)

---

### 2. `create_synthesis_constructs.py` - β-Chain Construction

**Purpose**: Combine V region + CDR3β + J region into complete β-chain sequences

**Input Required**:
- `results/candidates_with_vj.csv` (CDR3β with V/J gene assignments)
  - Generated from Step 1 + V/J extraction pipeline

**Usage**:
```bash
python scripts/generation/create_synthesis_constructs.py
```

**No Command-Line Arguments** - Uses hardcoded paths:
- Input: `results/candidates_with_vj.csv`
- Output: `results/synthesis_ready_constructs.csv`

**What It Does**:
1. Loads CDR3β candidates with recommended V/J genes
2. Extracts V region before CDR3 (removes CDR3 start patterns like CASS-)
3. Extracts J region after CDR3 (removes CDR3 end patterns like -YEQYF)
4. Constructs full β-chain: `V_region + CDR3β + J_region`
5. Validates sequence length (90-130 AA typical)
6. Assigns synthesis priority (HIGH/MEDIUM/STANDARD based on confidence)

**Output Format**:
```
construct_id,peptide,mhc,cdr3_sequence,v_gene,j_gene,full_sequence,sequence_length,confidence,...
TCR_B_00_01,VVGAVGVGK,A*11:01,CASSNRNTIYF,DVKVTQSSR...,SNQPQHFGD...,DVKVT...CASSNR...HFGDGTRL,111,0.864,...
```

**When to Use**:
- After generating new CDR3β candidates
- After V/J gene extraction/assignment
- Before exporting to FASTA for synthesis

---

### 3. `rescoring_with_alpha.py` - Full Conditioning Rescoring

**Purpose**: Rescore β-chain constructs using full α+β conditioning for improved confidence

**Model Used**:
- Condition 1 model with all conditioning: `['mhc','pep','lv','lj','hv','hj']`
- Provides more accurate confidence when α-chain V/J available

**Input Required**:
- `results/synthesis_ready_paired_constructs.csv` (β-chains paired with α V/J)

**Usage**:
```bash
python scripts/generation/rescoring_with_alpha.py
```

**No Command-Line Arguments** - Uses hardcoded paths:
- Input: `results/synthesis_ready_paired_constructs.csv`
- Output: `results/synthesis_ready_paired_constructs_rescored.csv`

**What It Does**:
1. Loads paired α+β constructs
2. For each construct, creates temp CSV with full conditioning data
3. Runs model inference with all V/J genes as context
4. Calculates rescored NLL and confidence
5. Adds `rescored_nll` and `rescored_confidence` columns

**Output Enhancement**:
- Original: `confidence` (CDR3β only)
- Added: `rescored_confidence` (with full α+β context)
- Use rescored values for final ranking

**When to Use**:
- After pairing β-chains with α-chain V/J sequences
- Want more accurate confidence estimates
- Before final synthesis selection

**Note**: Requires CUDA/GPU for reasonable speed (CPU fallback available but slow)

---

### 4. `generate_synthesis_fasta.py` - FASTA Export

**Purpose**: Export synthesis-ready constructs to FASTA format for gene synthesis companies

**Input Required**:
- `results/synthesis_ready_constructs.csv` (from create_synthesis_constructs.py)

**Usage**:
```bash
python scripts/generation/generate_synthesis_fasta.py
```

**No Command-Line Arguments** - Uses hardcoded paths:
- Input: `results/synthesis_ready_constructs.csv`
- Output: `results/synthesis_ready_constructs.fasta`
- Metadata: `results/synthesis_metadata.json`

**What It Does**:
1. Loads valid constructs (validation = is_valid)
2. Sorts by target and composite score
3. Formats sequences with 80-char line breaks
4. Creates descriptive FASTA headers with confidence and scores
5. Assigns synthesis priority labels
6. Exports metadata JSON with detailed construct information

**Output FASTA Format**:
```fasta
>TCR_B_00_01_VVGAVGVGK_A1101_Conf0.864_Score0.870
DVKVTQSSRYLVKRTGEKVFLECVQDMDHENMFWYRQDPGLGLRLIYFSYDVKMKEKGDI
PEGYSVSREKKERFSLILESASTNQTSMYLCASSLCASSNRNTIYFSYEQYFGPGTRLTVT
```

**Output Metadata**:
```json
{
  "construct_id": "TCR_B_00_01",
  "target": "VVGAVGVGK_A*11:01",
  "synthesis_priority": "HIGH",
  "confidence": 0.864,
  ...
}
```

**When to Use**:
- Ready to order gene synthesis
- Need standard FASTA format for synthesis companies
- Want to include only top candidates (script filters by validity)

**Note**: Most synthesis companies accept FASTA directly. Provide the JSON metadata for reference.

---

## Complete Pipeline Workflow

### For New Targets:

```bash
# Step 1: Create target input file
cat > new_targets.csv << EOF
peptide,mhc
GILGFVFTL,A*02:01
NLVPMVATV,A*02:01
EOF

# Step 2: Generate CDR3β sequences
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets new_targets.csv \
  --condition 1 \
  --n_samples 50 \
  --output results/new_cdr3b_raw.csv

# Step 3: Run data pipeline (V/J extraction, filtering, ranking)
# NOTE: These scripts were deleted as one-time processing
# You would need to adapt or recreate if processing new targets
# For now, use existing results/candidates_with_vj.csv as template

# Step 4: Build full β-chain constructs
python scripts/generation/create_synthesis_constructs.py

# Step 5: (Optional) Rescore with full α+β conditioning
python scripts/generation/rescoring_with_alpha.py

# Step 6: Export to FASTA for synthesis
python scripts/generation/generate_synthesis_fasta.py
```

### For Existing Targets (Regeneration):

If you just want to tweak parameters or regenerate:

```bash
# Regenerate with higher temperature (more diversity)
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets wetlab_targets.csv \
  --temperature 1.0 \
  --n_samples 100 \
  --output results/high_diversity_candidates.csv

# Or with lower temperature (more conservative)
python scripts/generation/generate_cdr3b_wetlab.py \
  --targets wetlab_targets.csv \
  --temperature 0.5 \
  --n_samples 50 \
  --output results/conservative_candidates.csv
```

---

## Model Dependencies

All scripts depend on trained models in:
- `conditioned/saved_model/condition_1/model_epoch_3450`

Required Python packages:
- `torch`
- `pandas`
- `numpy`

Required project modules:
- `conditioned.model` (Embedding2nd)
- `conditioned.data` (Load_Dataset)

---

## Output Files Reference

| File | Description | Size |
|------|-------------|------|
| `results/new_cdr3b_candidates.csv` | Raw CDR3β sequences | ~240 candidates/target |
| `results/candidates_with_vj.csv` | CDR3β + V/J assignments | Same + V/J columns |
| `results/synthesis_ready_constructs.csv` | Full β-chains | ~36 constructs |
| `results/synthesis_ready_constructs.fasta` | Synthesis format | Same in FASTA |
| `results/synthesis_metadata.json` | Construct details | JSON metadata |
| `results/synthesis_ready_paired_constructs.csv` | α+β paired | With α V/J |
| `results/synthesis_ready_paired_constructs_rescored.csv` | Rescored | +rescored columns |

---

## Troubleshooting

### Error: "Model checkpoint not found"
- Ensure `conditioned/saved_model/condition_1/model_epoch_3450` exists
- Check path in script matches your directory structure

### Error: "CUDA out of memory"
- Reduce batch size in rescoring script
- Use CPU (slower): Set device to 'cpu' in script
- Generate fewer samples per target

### Error: "Input CSV missing columns"
- Check input CSV has required columns (see each script's Input Required section)
- Ensure column names match exactly (case-sensitive)

### Low confidence scores
- Try different temperature (0.5-1.0 range)
- Check if peptide/MHC have training data coverage
- Consider generating more samples and filtering top candidates

---

## Notes

- **Generation time**: ~1-5 min per target (depends on n_samples)
- **GPU recommended**: Especially for rescoring step
- **Data quality**: Better results when targets similar to training data
- **Confidence threshold**: Typical good candidates have confidence >0.7

---

## Related Documentation

- **Main README**: `../../README.md` - Full project overview
- **Claude Instructions**: `../../Claude.md` - Development guidelines
- **Wet-Lab Instructions**: `../../results/wetlab_instructions.md` - Synthesis protocol
- **Training Data**: `../../data/collected data/final_data/` - Source data

---

**Questions?** Check the main README or Claude.md for project context.
