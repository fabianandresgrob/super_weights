# Super Weight Attack Module

This module provides adversarial attacks on super weights in large language models. The main script `run_large_model_attack_evaluation.py` provides a comprehensive evaluation framework.

## Module Components

- **`attack.py`**: Core attack implementation with 5 hypothesis classes (A, B, C, D, E) and the SuperWeightAttacker class
- **`run_large_model_attack_evaluation.py`**: Comprehensive evaluation script for running attacks across multiple models
- **`attack_eval.py`**: Evaluation utilities for consistency testing and perplexity bake-off

## Overview

The main evaluation script `run_large_model_attack_evaluation.py`:
1. **Detects super weights** in the specified models
2. **Runs attacks** for all super weights (except last layer) with Hypothesis D (primary) and A (secondary)  
3. **Validates attack consistency** using multi-seed evaluation
4. **Runs perplexity bake-off** evaluation with 4 conditions:
   - Baseline (no prefix, normal model)
   - AdvPrefix (learned adversarial prefix)
   - RandPrefix (random prefix matched in token length)
   - ZeroSW (no prefix, but target super weight is zeroed)
5. **Plots and saves** comprehensive results

## Quick Start

### Basic Usage
```bash
# Quick test with a small model
python attack/run_large_model_attack_evaluation.py \
  --models "allenai/OLMo-1B-0724-hf" \
  --output_dir ./results

# Multiple larger models
python attack/run_large_model_attack_evaluation.py \
  --models "mistralai/Mistral-7B-v0.1" "microsoft/Phi-3-mini-4k-instruct" \
  --output_dir ./results_large \
  --attack_batch_size 128 --bakeoff_batch_size 8
```

## GPU Setup Recommendations

### Memory Management
```bash
# For 7B models
python attack/run_large_model_attack_evaluation.py --attack_batch_size 128 --bakeoff_batch_size 8

# For 8B+ models  
python attack/run_large_model_attack_evaluation.py --attack_batch_size 64 --bakeoff_batch_size 4

# If still OOM, add:
--max_super_weights 3 --skip_consistency
```

## Command Line Arguments

### Core Arguments
- `--models MODEL [MODEL ...]`: Model names to evaluate (required)
- `--output_dir DIR`: Output directory for results (default: ./attack_evaluation_results)
- `--seed INT`: Random seed for reproducibility (default: 42)
- `--cache_dir DIR`: Directory to cache models (default: ~/models/)

### Model Loading
- `--cache_dir DIR`: Directory to cache models (default: ~/models/)

### Super Weight Detection
- `--spike_threshold FLOAT`: Threshold for super weight detection (default: 50.0)
- `--detection_max_iterations INT`: Max iterations for detection (default: 10)
- `--detection_prompt STR`: Custom detection prompt (default: random WikiText-2)
- `--skip_last_layer`: Skip super weights in last layer for attacks (default: True)
- `--max_super_weights INT`: Limit number of super weights to process

### Attack Configuration
- `--hypotheses {A,B,C,D,E} [{A,B,C,D,E} ...]`: Which hypotheses to test (default: D A)
- `--head_reduction {single,mean,weighted,topk}`: Head reduction for Hypothesis D (default: mean)
- `--attack_num_steps INT`: Attack optimization steps (default: 200)
- `--attack_batch_size INT`: Attack batch size (default: 256)
- `--attack_search_width INT`: Attack search width (default: 512)
- `--attack_top_k INT`: Top-k for attack search (default: 256)
- `--placement {prefix,suffix}`: Adversarial string placement (default: prefix)
- `--adv_string_init STR`: Initial adversarial string (default: "<bos> ~ <bos> ~ <bos>")
- `--allow_non_ascii`: Allow non-ASCII characters in adversarial strings (default: True)
- `--attack_prompt STR`: Custom prompt for attacks (default: random WikiText-2)

### Evaluation Configuration
- `--skip_consistency`: Skip consistency validation to save time
- `--consistency_n_prompts INT`: Prompts for consistency evaluation (default: 100)
- `--consistency_min_tokens INT`: Minimum tokens for consistency evaluation prompts (default: 6)
- `--consistency_max_tokens INT`: Maximum tokens for consistency evaluation prompts (default: 40)
- `--skip_bakeoff`: Skip perplexity bake-off to save time
- `--bakeoff_n_prompts INT`: Prompts for perplexity evaluation (default: 100)
- `--bakeoff_batch_size INT`: Batch size for perplexity evaluation (default: 16)
- `--bakeoff_min_tokens INT`: Minimum tokens for bakeoff evaluation prompts (default: 50)
- `--bakeoff_max_tokens INT`: Maximum tokens for bakeoff evaluation prompts (default: 150)
- `--verbose`: Enable verbose logging

## Output Files

The script generates several output files:

### JSON Results
- `{model_name}/data/attack_evaluation_results_{timestamp}.json`: Complete results per model
- `combined_attack_evaluation_results_{timestamp}.json`: Combined results for all models

### CSV Summary  
- `{model_name}/data/attack_evaluation_summary_{timestamp}.csv`: Tabular summary with key metrics

### Plots
- `{model_name}/plots/perplexity_bakeoff_{layer}_{row}_{col}_{hypothesis}.png`: Perplexity bake-off plots

### Logs
- `{model_name}/logs/attack_evaluation_{timestamp}.log`: Detailed execution logs

## Key Metrics Saved

The CSV output includes all important metrics:

### Attack Results
- `final_loss`: Final attack loss value
- `adv_string`: Best adversarial string found
- `final_down_proj_in_col_at_sink`: **Critical super weight input activation**
- `final_down_proj_out_row_at_sink`: **Critical super weight output activation**

### Consistency Validation
- `consistency_pass`: Whether attack passes multi-seed consistency tests
- `consistency_down_proj_in_*`: Consistency metrics for input activations
- `consistency_down_proj_out_*`: Consistency metrics for output activations

### Perplexity Bake-off
- `ppl_delta_adv_vs_baseline`: Perplexity change for adversarial vs baseline
- `ppl_delta_zero_vs_baseline`: Perplexity change when super weight is zeroed
- `baseline_mean_activation`: Baseline activation level
- `adv_mean_activation`: Activation level with adversarial prefix

## Example Configurations

### Quick Test (Development)
```bash
python attack/run_large_model_attack_evaluation.py \
  --models "allenai/OLMo-1B-0724-hf" \
  --attack_num_steps 100 \
  --consistency_n_prompts 50 \
  --bakeoff_n_prompts 50 \
  --max_super_weights 2
```

### Production Run (Large Models)
```bash
python attack/run_large_model_attack_evaluation.py \
  --models "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-3.1-8B" \
  --spike_threshold 50.0 \
  --attack_num_steps 200 \
  --attack_batch_size 64 \
  --bakeoff_batch_size 4 \
  --hypotheses D A \
  --output_dir ./production_results
```

### Memory-Optimized (Very Large Models)
```bash
python attack/run_large_model_attack_evaluation.py \
  --models "meta-llama/Llama-3.1-8B" \
  --spike_threshold 70.0 \
  --attack_batch_size 32 \
  --bakeoff_batch_size 2 \
  --max_super_weights 3 \
  --hypotheses D \
  --skip_consistency
```
