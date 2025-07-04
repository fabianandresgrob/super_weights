# Super Weight Attack Evaluation Script

This script provides a comprehensive evaluation framework for running adversarial attacks on super weights in large language models.

## Overview

The script:
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
  --output_dir ./results \
  --use_fp16 --device_map_auto

# Multiple larger models (optimized for 2x GPU with ~42GB VRAM)
python attack/run_large_model_attack_evaluation.py \
  --models "mistralai/Mistral-7B-v0.1" "microsoft/Phi-3-mini-4k-instruct" \
  --output_dir ./results_large \
  --use_fp16 --device_map_auto \
  --attack_batch_size 128 --bakeoff_batch_size 8
```

## GPU Setup Recommendations

For your **2x NVIDIA GPU setup with ~42GB total VRAM**:

### Essential Flags
- `--use_fp16`: Use float16 precision (essential for 7B+ models)
- `--device_map_auto`: Automatically distribute model across GPUs
- `--attack_batch_size 64-128`: Reduce if you get OOM errors
- `--bakeoff_batch_size 4-8`: Very small batches for perplexity evaluation

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
- `--device_map_auto`: Use automatic device mapping for multi-GPU (recommended)
- `--use_fp16`: Use float16 precision to save VRAM (recommended for large models)

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
- `--placement {prefix,suffix}`: Adversarial string placement (default: prefix)
- `--adv_string_init STR`: Initial adversarial string (default: "! ! ! ! ! ! ! ! ! !")

### Evaluation Configuration
- `--skip_consistency`: Skip consistency validation to save time
- `--consistency_n_prompts INT`: Prompts for consistency evaluation (default: 100)
- `--skip_bakeoff`: Skip perplexity bake-off to save time
- `--bakeoff_n_prompts INT`: Prompts for perplexity evaluation (default: 100)
- `--bakeoff_batch_size INT`: Batch size for perplexity evaluation (default: 16)

## Output Files

The script generates several output files:

### JSON Results
- `attack_evaluation_results_{model}_{timestamp}.json`: Complete results per model
- `combined_attack_evaluation_results_{timestamp}.json`: Combined results for all models

### CSV Summary  
- `attack_evaluation_summary_{model}_{timestamp}.csv`: Tabular summary with key metrics

### Plots
- `plots/perplexity_bakeoff_{layer}_{row}_{col}_{hypothesis}.png`: Perplexity bake-off plots

### Logs
- `attack_evaluation_{model}_{timestamp}.log`: Detailed execution logs

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
  --max_super_weights 2 \
  --use_fp16 --device_map_auto
```

### Production Run (2x GPU, 42GB VRAM)
```bash
python attack/run_large_model_attack_evaluation.py \
  --models "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-3.1-8B" \
  --spike_threshold 50.0 \
  --attack_num_steps 200 \
  --attack_batch_size 64 \
  --bakeoff_batch_size 4 \
  --hypotheses D A \
  --use_fp16 --device_map_auto \
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
  --skip_consistency \
  --use_fp16 --device_map_auto
```

## Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce `--attack_batch_size` (try 64, 32, 16)
2. Reduce `--bakeoff_batch_size` (try 4, 2, 1)
3. Add `--max_super_weights N` to limit processing
4. Use `--skip_consistency` or `--skip_bakeoff` to save memory

### Slow Execution
1. Reduce `--attack_num_steps` (try 150, 100)
2. Reduce `--consistency_n_prompts` and `--bakeoff_n_prompts` (try 75, 50)
3. Use `--max_super_weights` to limit scope
4. Add `--skip_consistency` for faster runs

### Model Loading Issues
1. Check model name spelling and availability on HuggingFace
2. Ensure sufficient disk space in `--cache_dir`
3. Try without `--device_map_auto` if encountering device mapping issues

## Performance Notes

- **7B models**: Expect 2-4 hours per model with full evaluation
- **1B models**: Expect 30-60 minutes per model with full evaluation  
- **Memory usage**: Peak ~35-40GB for 7B models with our recommended settings
- **Disk space**: ~20-50GB per model for caching, plus results storage

## Advanced Usage

See `attack/example_attack_evaluation.py` for more detailed examples and configurations.
