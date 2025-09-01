# Comprehensive Super Weight Analysis Script

This directory contains the main script for automated super weight detection and analysis.

## Main Script

### `comprehensive_super_weight_analysis.py`
A comprehensive analysis script that performs automated super weight detection and evaluation on language models with visualizations.

**Features:**
- Detect super weights with configurable parameters
- Analyze perplexity impact on WikiText dataset
- Measure accuracy impact on MMLU, HellaSwag, ARC, and GSM8K
- Test individual and combined super weight effects
- Generate comprehensive reports and visualizations
- Save results in JSON format with timestamp tracking

**Usage:**
```bash
python scripts/comprehensive_super_weight_analysis.py \
    --models "allenai/OLMo-1B-0724-hf" "mistralai/Mistral-7B-v0.1" \
    --spike-threshold 50.0 \
    --perplexity-samples 500 \
    --accuracy-samples 100 \
    --accuracy-tasks hellaswag arc_easy mmlu gsm8k \
    --verbose
```

**Key Arguments:**
- `--models`: Model names to analyze (required)
- `--spike-threshold`: Threshold for super weight detection (default: 50.0)
- `--max-iterations`: Maximum detection iterations (default: 10)
- `--perplexity-samples`: Samples for perplexity evaluation (default: 500)
- `--accuracy-samples`: Samples for accuracy evaluation (default: 100)
- `--accuracy-tasks`: Tasks to evaluate (default: hellaswag, arc_easy, mmlu, gsm8k)
- `--results-dir`: Output directory for results (default: results/comprehensive_super_weight_analysis)
- `--verbose/-v`: Enable detailed logging
- `--quiet/-q`: Suppress most output

**MoE-Specific Arguments:**
- `--router-analysis-samples`: Samples for router analysis (default: 10)
- `--p-active-floor`: Minimum expert activation probability (default: 0.01)
- `--co-spike-threshold`: Co-spike threshold for MoE detection (default: 0.12)
- `--enable-causal-scoring`: Enable causal scoring for MoE models

## Output Structure

Results are saved in the following directory structure:

```
results/
├── comprehensive_super_weight_analysis/
│   ├── complete_analysis_YYYYMMDD_HHMMSS.json  # Complete results
│   ├── summary_plots/
│   │   └── model_comparison.png               # Cross-model comparison
│   └── {model_name}/
│       ├── data/
│       │   ├── analysis_YYYYMMDD_HHMMSS.json           # Model-specific results
│       │   ├── combined_analysis_YYYYMMDD_HHMMSS.json  # Combined super weight analysis
│       │   └── individual_analyses/
│       │       ├── sw_0_analysis_YYYYMMDD_HHMMSS.json  # Individual super weight analyses
│       │       ├── sw_1_analysis_YYYYMMDD_HHMMSS.json
│       │       └── ...
│       ├── plots/
│       │   ├── detection_iterations.png           # Detection progress with star markers
│       │   ├── super_weight_distribution.png      # Input vs output values with layer labels
│       │   ├── impact_comparison.png              # Baseline vs modified with log scaling
│       │   ├── combined_analysis.png              # Combined effects with log scaling
│       │   └── moe_analysis.png                   # MoE-specific plots (if applicable)
│       └── logs/
│           └── analysis_YYYYMMDD_HHMMSS.log       # Analysis logs for this model
```

## Analysis Types

### Individual Super Weight Analysis
Each detected super weight is analyzed individually:
- **Perplexity Impact**: How zeroing the weight affects language modeling performance
- **Accuracy Impact**: Effect on downstream tasks (MMLU, HellaSwag, ARC, GSM8K)

### Combined Analysis
All super weights are analyzed together to understand cumulative effects.

### Cross-Model Analysis
When multiple models are analyzed, comparative statistics are computed:
- Super weight count distributions
- Impact severity comparisons
- Architecture-specific patterns

## Examples

### Quick Test on Small Model
```bash
python scripts/comprehensive_super_weight_analysis.py \
    --models "allenai/OLMo-1B-0724-hf" \
    --spike-threshold 50.0 \
    --perplexity-samples 200 \
    --accuracy-samples 50
```

### MoE Model Analysis
```bash
python scripts/comprehensive_super_weight_analysis.py \
    --models "Qwen/Qwen1.5-MoE-A2.7B" \
    --spike-threshold 60.0 \
    --router-analysis-samples 15 \
    --co-spike-threshold 0.095 \
    --enable-causal-scoring \
    --perplexity-samples 300 \
    --accuracy-samples 75
```

### Comparative Study
```bash
python scripts/comprehensive_super_weight_analysis.py \
    --models \
        "allenai/OLMo-1B-0724-hf" \
        "mistralai/Mistral-7B-v0.1" \
        "meta-llama/Llama-2-7b-hf" \
    --spike-threshold 50.0 \
    --perplexity-samples 400 \
    --accuracy-samples 80
```

### Skip Certain Analyses
```bash
python scripts/comprehensive_super_weight_analysis.py \
    --models "mistralai/Mistral-7B-v0.1" \
    --skip-perplexity \
    --accuracy-tasks hellaswag arc_easy \
    --accuracy-samples 50
```

## Requirements

Make sure you have the required dependencies installed:
```bash
pip install torch transformers datasets matplotlib seaborn tqdm numpy
```

## Notes

- Results are automatically saved with timestamps to avoid overwrites
- Large models may require significant GPU memory
- Use `--verbose` for detailed logging or `--quiet` to suppress output
- The script automatically handles different model architectures (standard vs MoE)
- Individual analyses are saved separately for detailed inspection
- All plots use log scales and value labels where appropriate

