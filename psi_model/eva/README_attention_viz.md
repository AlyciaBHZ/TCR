# Multi-Model Attention Visualization Script

This script (`multi_model_attention_viz.py`) visualizes attention patterns across different model architectures for comparison. It has been updated to be fully runnable in the current project structure.

## Features

✅ **Multi-model support**: Compares attention patterns across different model architectures
✅ **Layer-wise analysis**: Analyzes attention patterns at each layer
✅ **Entropy metrics**: Computes attention entropy for quantitative analysis
✅ **Visual comparisons**: Creates side-by-side attention heatmaps
✅ **Flexible data input**: Handles various data file formats and locations
✅ **Robust error handling**: Gracefully handles missing models or data

## Usage

### Basic Usage
```bash
python multi_model_attention_viz.py
```

### With Custom Parameters
```bash
python multi_model_attention_viz.py \
  --data_path ../../data/tst.csv \
  --sample_ids 0 1 2 5 10 \
  --output_dir ./attention_results \
  --conditioning_info mhc pep lv lj hv hj
```

### Data Preparation Only
```bash
python multi_model_attention_viz.py --skip_model_loading --data_path ../../data/tst.csv
```

## Command Line Arguments

- `--data_path`: Path to experimental data (default: `../../data/tst.csv`)
- `--output_dir`: Output directory for results (default: `./nll/attention_analysis`)
- `--conditioning_info`: List of conditioning variables (default: `['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']`)
- `--sample_ids`: Specific sample IDs to analyze (default: `[0, 2, 15]`)
- `--skip_model_loading`: Skip model loading and focus on data preparation

## Supported Models

The script currently supports these CLP models:
- **Optimized Composite** (8 layers): `../saved_model/optimized_composite_condition_1/model_epoch_100`
- **Staged Composite** (6 layers): `../saved_model/staged_composite_condition_1/model_epoch_100`
- **Background Composite** (4 layers): `../saved_model/bkcomposite_condition_1/model_epoch_100`

## Output Files

The script generates:

1. **Attention comparison plots** (`attention_comparison_sample{X}_layer{Y}.png`):
   - Side-by-side heatmaps for each model
   - Two attention types: "First Token → All" and "CDR3 → All"
   - Color-coded region boundaries
   - Statistics (max, mean attention weights)

2. **Analysis summary** (`attention_analysis_summary.csv`):
   - Entropy metrics for each model/sample/layer combination
   - Columns: model, sample_idx, layer, collapse_entropy, mean_entropy, max_entropy

3. **Formatted data** (`formatted_attention_data.csv`):
   - Data preprocessed for model consumption
   - Standardized column format

## Data Format

The script expects CSV data with these columns (handles various naming conventions):
- **CDR3**: `cdr3_b`, `cdr3`, `hd`, `CDR3`, `CDR3_seq`
- **Peptide**: `peptide`, `pep`
- **MHC**: `mhc`
- **V/J genes**: `l_v`/`lv`, `l_j`/`lj`, `h_v`/`hv`, `h_j`/`hj`

## Requirements

- Python environment with PyTorch
- CLP model modules available (`model.py`, `data_clp.py`)
- Required packages: `torch`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `tqdm`

## Example Output

```
✅ CLP model imports available
🚀 Using device: cuda
📊 Conditioning info: ['mhc', 'pep', 'lv', 'lj', 'hv', 'hj']
✅ Loaded dataset with 3497 samples
✅ Successfully loaded 3/3 models
📊 Analyzing samples: [0, 1, 2]
📊 Saved attention analysis summary with 54 records
✅ Analysis complete! Results saved in ./nll/attention_analysis
```

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure you're in the correct environment and directory
2. **Model file not found**: Check that model files exist in `../saved_model/` directory
3. **No valid CDR3 data**: Verify your data file contains CDR3 sequences with proper column names
4. **CUDA out of memory**: Reduce the number of samples or use CPU with smaller models

### Environment Setup
```bash
conda activate torch  # Or your PyTorch environment
cd plm/eva
```

### Data Preparation
If models fail to load, you can still prepare data:
```bash
python multi_model_attention_viz.py --skip_model_loading
```

This will create `formatted_attention_data.csv` that can be used with other scripts. 