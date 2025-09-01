# Super Weight Analysis Scripts

This directory contains automated scripts for super weight detection and analysis.

## Scripts Overview

### `comprehensive_super_weight_analysis.py`
A comprehensive analysis script that performs automated super weight detection and evaluation on language models with visualizations.

### `run_vocab_analyses.py`
A specialized vocabulary analysis script focused on detailed vocabulary effects, interventional impacts, and cascade analysis using the VocabularyAnalyzer class.

## Script Details

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

### `run_vocab_analyses.py`
A specialized vocabulary analysis script that provides detailed analysis of vocabulary effects, interventional impacts, and cascade effects using the VocabularyAnalyzer class.

**Features:**
- Analyze neuron vocabulary effects with optional cosine similarity
- Measure interventional impacts on loss, entropy, top-k margin, stopword mass
- **Dual token analysis**: Both neuron vocabulary effects AND interventional token probability changes
- Generate cascade effect analysis across layers
- Token class enrichment analysis (digits, parentheses, function words, etc.)
- Control experiments for validation (random coordinates, full neuron ablation)
- Bootstrap confidence intervals
- **Smart token validation**: Automatic validation and recommendations for token/window parameters
- Machine-readable JSONL output format
- Reproducible results with fixed seeds
- GPU-aware memory management

**Usage:**
```bash
python scripts/run_vocab_analyses.py \
    --models "allenai/OLMo-1B-0724-hf" \
    --dataset wikitext-2 \
    --tokens 30000 --window-len 1024 --stride 512 --drop-first-token \
    --enable-cosine --enable-enrichment --enable-cascade \
    --output-dir ./vocab_runs --bootstrap 500
```

**Key Arguments:**
- `--models`: Model names or local paths (HF style) (required)
- `--dataset`: Held-out text source (default: wikitext-2)
- `--tokens`: Total tokens to evaluate (default: 30000)
- `--window-len`: Window length for sliding windows (default: 1024)
- `--stride`: Stride between windows (default: 512)
- `--drop-first-token`: Drop first token in each window
- `--output-dir`: Output directory for results (default: results/vocab-analysis)
- `--seed`: Random seed for reproducibility (default: 42)

**Feature Flags:**
- `--enable-cosine`: Compute cosine variant for vocab effects
- `--enable-cascade`: Enable cascade effect analysis
- `--enable-enrichment`: Enable token class enrichment analysis
- `--bootstrap`: Number of bootstrap samples for confidence intervals (0 to disable)

**Detection Parameters:**
- `--spike-threshold`: Spike threshold for super weight detection (default: 50.0)
- `--max-iterations`: Maximum iterations for super weight detection (default: 10)
- `--router-analysis-samples`: Number of samples for MoE router analysis (default: 5)
- `--p-active-floor`: Minimum expert activation probability for MoE (default: 0.01)
- `--co-spike-threshold`: Co-spike threshold for MoE detection (default: 0.12)

**Vocabulary Analysis Output Structure:**
```
results/vocab-analysis/
├── run_config.json                    # Global configuration
├── cross_model_summary.json           # Cross-model comparison
├── README.md                          # Auto-generated summary
└── {model_name}/
    ├── metrics/
    │   ├── neurons/
    │   │   └── neurons.jsonl           # Neuron vocabulary effects (one JSON per line)
    │   └── super_weights/
    │       └── super_weights.jsonl     # Super weight intervention effects
    ├── plots/
    │   ├── neurons/
    │   │   └── {layer}-{neuron}-effects.png      # Vocabulary effect visualizations
    │   └── super_weights/
    │       ├── {tensor}-L{layer}-R{row}-C{col}-deltas.png    # Intervention effects
    │       └── {tensor}-L{layer}-R{row}-C{col}-cascade.png   # Cascade effects
    ├── summaries/
    │   ├── neurons.csv                 # Flat table for neuron analyses
    │   └── super_weights.csv           # Flat table for super weight analyses
    ├── run_config.json                # Model-specific configuration
    └── analysis.log                   # Analysis logs
```

**Data Schemas:**

*Neuron Card (neurons.jsonl):*
```json
{
  "model": "allenai/OLMo-1B-0724-hf",
  "layer": 10, "neuron": 523,
  "moments": {"var": 2.34, "skew": 1.12, "kurt": 3.45},
  "moments_cos": {"var": 1.89, "skew": 0.98, "kurt": 2.87},
  "top_tokens_up": [{"id": 123, "tok": "Ġthe", "str": " the", "effect": 1.23}],
  "top_tokens_down": [{"id": 456, "tok": "Ġnot", "str": " not", "effect": -0.89}],
  "enrichment": [{"class": "digits_years", "score": 0.42, "examples": ["1999","2020"]}],
  "label": "prediction"
}
```

*Super Weight Card (super_weights.jsonl):*
```json
{
  "model": "allenai/OLMo-1B-0724-hf",
  "tensor": "mlp_out", "layer": 10, "row": 523, "col": 768,
  "baselines": {"loss": 2.34, "entropy": 8.45, "topk_margin": 0.12, "stopword_mass": 0.089},
  "deltas": {"loss": 0.018, "entropy": 0.012, "topk_margin": -0.007, "stopword_mass": -0.004},
  "controls": {
    "random_coord_delta": {"loss": 0.001, "entropy": 0.000, "topk_margin": 0.000},
    "neuron_zero_delta": {"loss": 0.091, "entropy": 0.052, "topk_margin": -0.041}
  },
  "cascade": {
    "layers": [0,4,8,12,16,20,24],
    "delta_entropy": [0.001, 0.003, 0.012, 0.008, 0.002, 0.001, 0.000],
    "delta_margin": [-0.001, -0.002, -0.007, -0.004, -0.001, 0.000, 0.000]
  },
  "bootstrap": {
    "loss": {"mean": 0.018, "ci_2.5": 0.012, "ci_97.5": 0.024}
  }
}
```

## Output Structure (Comprehensive Analysis)

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

### Comprehensive Analysis Examples

#### Quick Test on Small Model
```bash
python scripts/comprehensive_super_weight_analysis.py \
    --models "allenai/OLMo-1B-0724-hf" \
    --spike-threshold 50.0 \
    --perplexity-samples 200 \
    --accuracy-samples 50
```

#### MoE Model Analysis
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

#### Comparative Study
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

#### Skip Certain Analyses
```bash
python scripts/comprehensive_super_weight_analysis.py \
    --models "mistralai/Mistral-7B-v0.1" \
    --skip-perplexity \
    --accuracy-tasks hellaswag arc_easy \
    --accuracy-samples 50
```

### Vocabulary Analysis Examples

#### Basic Vocabulary Analysis
```bash
python scripts/run_vocab_analyses.py \
    --models "allenai/OLMo-1B-0724-hf" \
    --dataset wikitext-2 \
    --tokens 30000 \
    --window-len 1024 --stride 512
```

#### Full Feature Analysis with Bootstrap
```bash
python scripts/run_vocab_analyses.py \
    --models "allenai/OLMo-1B-0724-hf" \
    --dataset wikitext-2 \
    --tokens 50000 --window-len 1024 --stride 512 --drop-first-token \
    --enable-cosine --enable-enrichment --enable-cascade \
    --bootstrap 500 \
    --output-dir ./detailed_vocab_analysis
```

#### Multi-Model Vocabulary Comparison
```bash
python scripts/run_vocab_analyses.py \
    --models "allenai/OLMo-1B-0724-hf" "mistralai/Mistral-7B-v0.1" \
    --dataset wikitext-2 \
    --tokens 40000 \
    --enable-cosine --enable-cascade \
    --spike-threshold 60.0 \
    --bootstrap 200
```

#### MoE Vocabulary Analysis
```bash
python scripts/run_vocab_analyses.py \
    --models "Qwen/Qwen1.5-MoE-A2.7B" \
    --dataset wikitext-2 \
    --tokens 35000 \
    --enable-cosine --enable-enrichment --enable-cascade \
    --router-analysis-samples 10 \
    --co-spike-threshold 0.10 \
    --bootstrap 300
```

## Which Script to Use?

### Use `comprehensive_super_weight_analysis.py` when:
- You want to measure **downstream task accuracy** impact (MMLU, HellaSwag, ARC, GSM8K)
- You need **perplexity analysis** on standard evaluation datasets
- You want **high-level summaries** and cross-model comparisons
- You prefer **JSON output** with timestamped results
- You're doing **initial exploration** or **performance impact assessment**

### Use `run_vocab_analyses.py` when:
- You want **detailed vocabulary effect analysis** (neuron-level vocabulary impacts)
- You need **dual token analysis** (both vocabulary effects AND interventional probability changes)
- You need **interventional analysis** with sliding window evaluation
- You want **cascade effect analysis** across transformer layers
- You need **token class enrichment** analysis (digits, parentheses, function words)
- You want **control experiments** (random coordinates, full neuron ablation)
- You need **smart token validation** with automatic parameter recommendations
- You prefer **JSONL output** with CSV summaries for easy data analysis
- You're doing **mechanistic interpretability research** on vocabulary effects

**Quick Decision Guide:**
- **Performance Impact Research** → `comprehensive_super_weight_analysis.py`
- **Mechanistic Interpretability** → `run_vocab_analyses.py`
- **Both** -> Run vocabulary analysis first for detailed mechanistic insights, then comprehensive analysis for performance metrics

## Requirements

Make sure you have the required dependencies installed:
```bash
pip install torch transformers datasets matplotlib seaborn tqdm numpy
```

## Notes

- Results are automatically saved with timestamps to avoid overwrites
- Large models may require significant GPU memory
- Use `--verbose` for detailed logging or `--quiet` to suppress output
- Both scripts automatically handle different model architectures (standard vs MoE)
- Individual analyses are saved separately for detailed inspection
- All plots use log scales and value labels where appropriate
- **Vocabulary analysis** generates JSONL files for easy programmatic access
- **Comprehensive analysis** generates JSON files with complete structured results
- Both scripts support reproducible results with fixed seeds
- GPU memory is managed automatically with cleanup after each model

