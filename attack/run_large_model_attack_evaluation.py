#!/usr/bin/env python3
"""
Comprehensive script for running super weight attacks on larger models.

This script:
1. Detects super weights in the given models
2. Runs attacks for all super weights (except last layer) with Hypothesis D (primary) and A (secondary)
3. Validates attack consistency using multi-seed evaluation
4. Runs perplexity bake-off evaluation with 4 conditions
5. Plots and saves results

Usage:
    python run_large_model_attack_evaluation.py --models "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.1"
    python run_large_model_attack_evaluation.py --models "microsoft/Phi-3-mini-4k-instruct" --output_dir ./results_custom
"""

import argparse
import json
import logging
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from research.researcher import SuperWeightResearchSession
from attack.attack import SuperWeightAttacker, SuperWeightAttackConfig, SuperWeightTarget
from attack.attack_eval import (
    run_heldout_evaluation_single_seed,
    run_multi_seed_consistency_evaluation,
    run_perplexity_bakeoff,
    plot_perplexity_bakeoff
)
from utils.datasets import DatasetLoader


def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def setup_logging(output_dir: Path, model_name: str) -> logging.Logger:
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"attack_evaluation_{model_name.replace('/', '_')}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("AttackEvaluation")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def get_random_wikitext_prompt(session, seed: int = None) -> str:
    """Get a single random prompt from WikiText-2."""
    if seed is not None:
        original_seed_state = random.getstate()
        np_seed_state = np.random.get_state()
        torch_seed_state = torch.get_rng_state()
        set_all_seeds(seed)
    
    try:
        # Use DatasetLoader to get a random sample
        dataset_loader = DatasetLoader(seed=seed or 42)
        texts = dataset_loader.load_perplexity_dataset(
            dataset_name='wikitext',
            config='wikitext-2-raw-v1', 
            split='validation',
            n_samples=1,
            min_length=50
        )
        
        if not texts:
            # Fallback prompt
            return "The quick brown fox jumps over the lazy dog and continues its journey through the forest."
        
        return texts[0]
    
    finally:
        if seed is not None:
            # Restore original random states
            random.setstate(original_seed_state)
            np.random.set_state(np_seed_state)
            torch.set_rng_state(torch_seed_state)


def detect_super_weights(session, logger: logging.Logger, spike_threshold: float = 50.0, 
                        max_iterations: int = 10, detection_prompt: str = None) -> List:
    """Detect super weights in the model."""
    logger.info("Starting super weight detection...")
    
    # Use provided prompt or get a random one
    if detection_prompt is None:
        detection_prompt = get_random_wikitext_prompt(session, seed=42)
    logger.info(f"Detection prompt: {detection_prompt[:100]}...")
    
    # Detect super weights
    super_weights = session.detect_super_weights(
        input_text=detection_prompt,
        spike_threshold=spike_threshold,
        max_iterations=max_iterations
    )
    
    logger.info(f"Detected {len(super_weights)} super weights")
    for i, sw in enumerate(super_weights):
        logger.info(f"  {i+1}. {sw}")
    
    return super_weights


def filter_super_weights_for_attack(super_weights: List, logger: logging.Logger) -> List:
    """Filter out super weights from the last layer for attack."""
    if not super_weights:
        return []
    
    # Find the maximum layer index
    max_layer = max(sw.layer for sw in super_weights)
    
    # Filter out last layer super weights
    filtered_weights = [sw for sw in super_weights if sw.layer != max_layer]
    
    logger.info(f"Filtered {len(super_weights) - len(filtered_weights)} super weights from last layer (layer {max_layer})")
    logger.info(f"Remaining {len(filtered_weights)} super weights for attack")
    
    return filtered_weights


def run_attack_for_super_weight(session, super_weight, hypothesis: str, logger: logging.Logger,
                               head_reduction: str = "mean", attack_config: dict = None) -> Optional[Dict[str, Any]]:
    """Run attack for a single super weight with given hypothesis."""
    logger.info(f"Running Hypothesis {hypothesis} attack on {super_weight}")
    
    try:
        # Get attack prompt - use provided or random
        if attack_config and attack_config.get('attack_prompt'):
            attack_prompt = attack_config['attack_prompt']
        else:
            attack_prompt = get_random_wikitext_prompt(session, seed=42)
        
        # Create attack config
        target = SuperWeightTarget(
            super_weight=super_weight,
            head_idx=None  # Will be auto-selected for hypothesis D
        )
        
        config_params = {
            'target': target,
            'hypothesis': hypothesis,
            'num_steps': attack_config.get('num_steps', 200) if attack_config else 200,
            'adv_string_init': attack_config.get('adv_string_init', "! ! ! ! ! ! ! ! ! !") if attack_config else "! ! ! ! ! ! ! ! ! !",
            'search_width': attack_config.get('search_width', 512) if attack_config else 512,
            'batch_size': attack_config.get('batch_size', 256) if attack_config else 256,
            'top_k_search': attack_config.get('top_k_search', 256) if attack_config else 256,
            'allow_non_ascii': attack_config.get('allow_non_ascii', True) if attack_config else True,
            'prompt_text': attack_prompt,
            'placement': attack_config.get('placement', "prefix") if attack_config else "prefix",
            'head_reduction': head_reduction if hypothesis == 'D' else "single",
            'target_all_content_tokens': True
        }
        
        config = SuperWeightAttackConfig(**config_params)
        
        # Run attack
        attacker = SuperWeightAttacker(session.model, session.tokenizer, config, log_level=logging.WARNING)
        attack_result = attacker.attack()
        
        logger.info(f"Attack completed. Final loss: {attack_result['final_loss']:.6f}")
        logger.info(f"Best adversarial string: '{attack_result['best_adv_string']}'")
        
        return {
            'super_weight': super_weight,
            'hypothesis': hypothesis,
            'attack_result': attack_result,
            'attacker': attacker,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Attack failed for {super_weight} with hypothesis {hypothesis}: {e}")
        logger.error(traceback.format_exc())
        return None


def validate_attack_consistency(session, attack_result: Dict[str, Any], logger: logging.Logger,
                               consistency_config: dict = None) -> Dict[str, Any]:
    """Validate attack consistency using multi-seed evaluation."""
    logger.info(f"Validating attack consistency for {attack_result['super_weight']} (Hypothesis {attack_result['hypothesis']})")
    
    try:
        # Get config parameters
        seeds = consistency_config.get('seeds', [41, 42, 43]) if consistency_config else [41, 42, 43]
        n_prompts = consistency_config.get('n_prompts', 100) if consistency_config else 100
        min_tokens = consistency_config.get('min_tokens', 6) if consistency_config else 6
        max_tokens = consistency_config.get('max_tokens', 40) if consistency_config else 40
        
        consistency_result = run_multi_seed_consistency_evaluation(
            session=session,
            attacker=attack_result['attacker'],
            adv_string=attack_result['attack_result']['best_adv_string'],
            seeds=seeds,
            n_prompts=n_prompts,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            metrics_of_interest=(
                'down_proj_in_col_at_sink',
                'down_proj_out_row_at_sink',
            ),
            dataset_name='wikitext',
            dataset_config='wikitext-2-raw-v1',
            split='validation',
            show_progress=False,
            set_all_seeds_fn=set_all_seeds
        )
        
        # Check if attack passes consistency tests
        overall_pass = consistency_result.get('final_evaluation', {}).get('overall_pass', False)
        logger.info(f"Attack consistency: {'PASS' if overall_pass else 'FAIL'}")
        
        return consistency_result
        
    except Exception as e:
        logger.error(f"Consistency validation failed: {e}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def run_perplexity_evaluation(session, attack_result: Dict[str, Any], logger: logging.Logger,
                             output_dir: Path, bakeoff_config: dict = None) -> Dict[str, Any]:
    """Run perplexity bake-off evaluation."""
    logger.info(f"Running perplexity bake-off for {attack_result['super_weight']} (Hypothesis {attack_result['hypothesis']})")
    
    try:
        # Get config parameters
        batch_size = bakeoff_config.get('batch_size', 16) if bakeoff_config else 16
        min_tokens = bakeoff_config.get('min_tokens', 50) if bakeoff_config else 50
        max_tokens = bakeoff_config.get('max_tokens', 150) if bakeoff_config else 150
        n_prompts = bakeoff_config.get('n_prompts', 100) if bakeoff_config else 100
        
        bakeoff_result = run_perplexity_bakeoff(
            session=session,
            attacker=attack_result['attacker'],
            target_sw=attack_result['super_weight'],
            adv_prefix=attack_result['attack_result']['best_adv_string'],
            prompts=None,  # Will sample automatically
            seed=42,
            activation_metric='down_proj_in_col_at_sink',
            include_additive=False,
            random_prefix_seed=123,
            batch_size=batch_size,
            dataset_name='wikitext',
            dataset_config='wikitext-2-raw-v1',
            split='validation',
            min_tokens=min_tokens,
            max_tokens=max_tokens
        )
        
        # Plot and save results
        plot_filename = f"perplexity_bakeoff_{attack_result['super_weight'].layer}_{attack_result['super_weight'].row}_{attack_result['super_weight'].column}_{attack_result['hypothesis']}.png"
        plot_path = output_dir / "plots" / plot_filename
        plot_path.parent.mkdir(exist_ok=True)
        
        plot_perplexity_bakeoff(bakeoff_result, figsize=(12, 8), save_path=str(plot_path))
        logger.info(f"Perplexity bake-off plot saved: {plot_path}")
        
        return bakeoff_result
        
    except Exception as e:
        logger.error(f"Perplexity evaluation failed: {e}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def save_results(results: Dict[str, Any], output_dir: Path, model_name: str):
    """Save all results to JSON and CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace('/', '_')
    
    # Save comprehensive JSON results
    json_filename = f"attack_evaluation_results_{safe_model_name}_{timestamp}.json"
    json_path = output_dir / json_filename
    
    # Prepare JSON-serializable results
    json_results = {}
    for key, value in results.items():
        if key == 'model_info':
            json_results[key] = value
        elif key == 'super_weights':
            json_results[key] = [str(sw) for sw in value]
        elif key == 'attack_results':
            json_results[key] = {}
            for sw_key, sw_results in value.items():
                json_results[key][sw_key] = {}
                for hypothesis, result in sw_results.items():
                    if result is None:
                        json_results[key][sw_key][hypothesis] = None
                        continue
                    
                    # Extract serializable parts
                    json_results[key][sw_key][hypothesis] = {
                        'super_weight': str(result.get('super_weight')),
                        'hypothesis': result.get('hypothesis'),
                        'attack_result': {
                            'best_adv_string': result.get('attack_result', {}).get('best_adv_string'),
                            'final_loss': result.get('attack_result', {}).get('final_loss'),
                            'total_iterations': result.get('attack_result', {}).get('total_iterations'),
                            'final_metrics': result.get('attack_result', {}).get('final_metrics')
                        },
                        'consistency_result': result.get('consistency_result'),
                        'bakeoff_result': result.get('bakeoff_result')
                    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # Create summary CSV
    csv_data = []
    if 'attack_results' in results:
        for sw_key, sw_results in results['attack_results'].items():
            for hypothesis, result in sw_results.items():
                if result is None:
                    continue
                
                row = {
                    'model': model_name,
                    'super_weight': str(result.get('super_weight')),
                    'layer': result.get('super_weight').layer,
                    'row': result.get('super_weight').row,
                    'column': result.get('super_weight').column,
                    'hypothesis': hypothesis,
                    'adv_string': result.get('attack_result', {}).get('best_adv_string'),
                    'final_loss': result.get('attack_result', {}).get('final_loss'),
                    'total_iterations': result.get('attack_result', {}).get('total_iterations'),
                    'consistency_pass': result.get('consistency_result', {}).get('final_evaluation', {}).get('overall_pass', False)
                }
                
                # Add attack final metrics - these contain the down_proj metrics
                final_metrics = result.get('attack_result', {}).get('final_metrics', {})
                if final_metrics:
                    row.update({
                        'final_down_proj_in_col_at_sink': final_metrics.get('down_proj_in_col_at_sink'),
                        'final_down_proj_out_row_at_sink': final_metrics.get('down_proj_out_row_at_sink'),
                        'final_gate_norm_at_sink': final_metrics.get('gate_norm_at_sink'),
                        'final_up_norm_at_sink': final_metrics.get('up_norm_at_sink'),
                        'final_content_attn_to_sink_head': final_metrics.get('content_attn_to_sink_head'),
                        'final_stopword_mass_next_token': final_metrics.get('stopword_mass_next_token')
                    })
                
                # Add consistency metrics for down_proj
                consistency_result = result.get('consistency_result', {})
                if 'seed_results' in consistency_result:
                    # Extract metrics from multi-seed evaluation
                    per_metric_results = consistency_result.get('final_evaluation', {}).get('per_metric', {})
                    if 'down_proj_in_col_at_sink' in per_metric_results:
                        down_proj_in_stats = per_metric_results['down_proj_in_col_at_sink']
                        row.update({
                            'consistency_down_proj_in_mean_median': down_proj_in_stats.get('mean_median'),
                            'consistency_down_proj_in_min_median': down_proj_in_stats.get('min_median_observed'),
                            'consistency_down_proj_in_metric_pass': down_proj_in_stats.get('metric_pass')
                        })
                    if 'down_proj_out_row_at_sink' in per_metric_results:
                        down_proj_out_stats = per_metric_results['down_proj_out_row_at_sink']
                        row.update({
                            'consistency_down_proj_out_mean_median': down_proj_out_stats.get('mean_median'),
                            'consistency_down_proj_out_min_median': down_proj_out_stats.get('min_median_observed'),
                            'consistency_down_proj_out_metric_pass': down_proj_out_stats.get('metric_pass')
                        })
                
                # Add bakeoff metrics if available
                bakeoff = result.get('bakeoff_result', {})
                if 'comparisons' in bakeoff:
                    comparisons = bakeoff['comparisons']
                    if 'adv_vs_baseline' in comparisons:
                        row.update({
                            'ppl_delta_adv_vs_baseline': comparisons['adv_vs_baseline'].get('delta_mean_ppl'),
                            'ppl_pvalue_adv_vs_baseline': comparisons['adv_vs_baseline'].get('ppl_t_test', {}).get('p'),
                            'activation_corr_delta_ppl_adv_baseline': comparisons['adv_vs_baseline'].get('activation_corr_delta_ppl')
                        })
                    if 'random_vs_baseline' in comparisons:
                        row.update({
                            'ppl_delta_random_vs_baseline': comparisons['random_vs_baseline'].get('delta_mean_ppl'),
                            'ppl_pvalue_random_vs_baseline': comparisons['random_vs_baseline'].get('ppl_t_test', {}).get('p')
                        })
                    if 'zeroSW_vs_baseline' in comparisons:
                        row.update({
                            'ppl_delta_zero_vs_baseline': comparisons['zeroSW_vs_baseline'].get('delta_mean_ppl'),
                            'ppl_pvalue_zero_vs_baseline': comparisons['zeroSW_vs_baseline'].get('ppl_t_test', {}).get('p')
                        })
                    if 'adv_vs_random' in comparisons:
                        row.update({
                            'ppl_delta_adv_vs_random': comparisons['adv_vs_random'].get('delta_mean_ppl'),
                            'ppl_pvalue_adv_vs_random': comparisons['adv_vs_random'].get('ppl_t_test', {}).get('p')
                        })
                
                # Add baseline activation values for comparison
                if 'conditions' in bakeoff:
                    baseline_cond = bakeoff['conditions'].get('baseline', {})
                    if 'summary' in baseline_cond:
                        baseline_summary = baseline_cond['summary']
                        row.update({
                            'baseline_mean_ppl': baseline_summary.get('ppl', {}).get('mean'),
                            'baseline_mean_activation': baseline_summary.get('activation', {}).get('mean'),
                            'baseline_std_activation': baseline_summary.get('activation', {}).get('std')
                        })
                    
                    adv_cond = bakeoff['conditions'].get('adv', {})
                    if 'summary' in adv_cond:
                        adv_summary = adv_cond['summary']
                        row.update({
                            'adv_mean_ppl': adv_summary.get('ppl', {}).get('mean'),
                            'adv_mean_activation': adv_summary.get('activation', {}).get('mean'),
                            'adv_std_activation': adv_summary.get('activation', {}).get('std')
                        })
                
                csv_data.append(row)
    
    if csv_data:
        csv_filename = f"attack_evaluation_summary_{safe_model_name}_{timestamp}.csv"
        csv_path = output_dir / csv_filename
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        print(f"Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")


def process_model(model_name: str, output_dir: Path, logger: logging.Logger, args) -> Dict[str, Any]:
    """Process a single model: detect super weights, run attacks, and evaluate."""
    logger.info(f"Processing model: {model_name}")
    
    results = {
        'model_info': {
            'name': model_name,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'spike_threshold': args.spike_threshold,
                'detection_max_iterations': args.detection_max_iterations,
                'attack_num_steps': args.attack_num_steps,
                'head_reduction': args.head_reduction,
                'consistency_n_prompts': args.consistency_n_prompts,
                'bakeoff_n_prompts': args.bakeoff_n_prompts,
                'skip_last_layer': args.skip_last_layer,
                'hypotheses': args.hypotheses
            }
        },
        'super_weights': [],
        'attack_results': {}
    }
    
    # Initialize session with cache and device options
    try:
        session = SuperWeightResearchSession.from_model_name(
            model_name, 
            cache_dir=args.cache_dir,
            device_map="auto" if args.device_map_auto else None,
            torch_dtype=torch.float16 if args.use_fp16 else None
        )
        session.model.eval()
        logger.info(f"Model loaded successfully: {model_name}")
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        results['error'] = f"Model loading failed: {e}"
        return results
    
    # Detect super weights
    try:
        super_weights = detect_super_weights(
            session, logger, 
            spike_threshold=args.spike_threshold,
            max_iterations=args.detection_max_iterations,
            detection_prompt=args.detection_prompt
        )
        if not super_weights:
            logger.warning("No super weights detected")
            results['error'] = "No super weights detected"
            return results
        
        results['super_weights'] = super_weights
        
        # Filter super weights for attack (exclude last layer if requested)
        if args.skip_last_layer:
            attack_weights = filter_super_weights_for_attack(super_weights, logger)
        else:
            attack_weights = super_weights
            logger.info(f"Processing all {len(attack_weights)} super weights (including last layer)")
        
        if not attack_weights:
            logger.warning("No super weights remain after filtering")
            results['error'] = "No attackable super weights found"
            return results
        
        # Limit number of super weights if specified
        if args.max_super_weights and len(attack_weights) > args.max_super_weights:
            attack_weights = attack_weights[:args.max_super_weights]
            logger.info(f"Limited to first {args.max_super_weights} super weights")
        
    except Exception as e:
        logger.error(f"Super weight detection failed: {e}")
        results['error'] = f"Detection failed: {e}"
        return results
    
    # Prepare attack and evaluation configs
    attack_config = {
        'num_steps': args.attack_num_steps,
        'adv_string_init': args.adv_string_init,
        'search_width': args.attack_search_width,
        'batch_size': args.attack_batch_size,
        'top_k_search': args.attack_top_k,
        'allow_non_ascii': args.allow_non_ascii,
        'placement': args.placement,
        'attack_prompt': args.attack_prompt
    }
    
    consistency_config = {
        'seeds': [41, 42, 43],
        'n_prompts': args.consistency_n_prompts,
        'min_tokens': args.consistency_min_tokens,
        'max_tokens': args.consistency_max_tokens
    }
    
    bakeoff_config = {
        'batch_size': args.bakeoff_batch_size,
        'min_tokens': args.bakeoff_min_tokens,
        'max_tokens': args.bakeoff_max_tokens,
        'n_prompts': args.bakeoff_n_prompts
    }
    
    # Run attacks for each super weight
    for sw_idx, sw in enumerate(attack_weights):
        sw_key = f"layer_{sw.layer}_row_{sw.row}_col_{sw.column}"
        results['attack_results'][sw_key] = {}
        logger.info(f"Processing super weight {sw_idx + 1}/{len(attack_weights)}: {sw}")
        
        # Clean up CUDA memory between super weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for hypothesis in args.hypotheses:
            logger.info(f"Running Hypothesis {hypothesis} attack for {sw}")
            
            hypothesis_result = run_attack_for_super_weight(
                session, sw, hypothesis, logger, 
                head_reduction=args.head_reduction,
                attack_config=attack_config
            )
            
            if hypothesis_result is not None:
                # Validate consistency if requested
                if not args.skip_consistency:
                    hypothesis_result['consistency_result'] = validate_attack_consistency(
                        session, hypothesis_result, logger, consistency_config
                    )
                
                # Run perplexity bake-off if requested
                if not args.skip_bakeoff:
                    hypothesis_result['bakeoff_result'] = run_perplexity_evaluation(
                        session, hypothesis_result, logger, output_dir, bakeoff_config
                    )
                
            results['attack_results'][sw_key][hypothesis] = hypothesis_result
    
    # Save results
    save_results(results, output_dir, model_name)
    
    logger.info(f"Model processing completed: {model_name}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run super weight attack evaluation on larger models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--models', nargs='+', required=True,
                       help='List of model names to evaluate')
    parser.add_argument('--output_dir', type=str, default='./attack_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--cache_dir', type=str, default='~/models/',
                       help='Directory to cache downloaded models')
    
    # Model loading options
    parser.add_argument('--device_map_auto', action='store_true',
                       help='Use automatic device mapping for multi-GPU (recommended for 2x GPUs)')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use float16 precision to save VRAM')
    
    # Super weight detection
    parser.add_argument('--spike_threshold', type=float, default=50.0,
                       help='Threshold for super weight detection')
    parser.add_argument('--detection_max_iterations', type=int, default=10,
                       help='Maximum iterations for super weight detection')
    parser.add_argument('--detection_prompt', type=str, default=None,
                       help='Custom prompt for detection (if not provided, uses random WikiText-2)')
    parser.add_argument('--skip_last_layer', action='store_true', default=True,
                       help='Skip super weights in the last layer for attacks')
    parser.add_argument('--max_super_weights', type=int, default=None,
                       help='Maximum number of super weights to process per model')
    
    # Attack configuration
    parser.add_argument('--hypotheses', nargs='+', default=['D', 'A'],
                       choices=['A', 'B', 'C', 'D', 'E'],
                       help='Which hypotheses to test')
    parser.add_argument('--head_reduction', type=str, default='mean',
                       choices=['single', 'mean', 'weighted', 'topk'],
                       help='Head reduction method for Hypothesis D')
    parser.add_argument('--attack_num_steps', type=int, default=200,
                       help='Number of attack optimization steps')
    parser.add_argument('--adv_string_init', type=str, default="! ! ! ! ! ! ! ! ! !",
                       help='Initial adversarial string')
    parser.add_argument('--attack_search_width', type=int, default=512,
                       help='Attack search width')
    parser.add_argument('--attack_batch_size', type=int, default=256,
                       help='Attack batch size (reduce if out of memory)')
    parser.add_argument('--attack_top_k', type=int, default=256,
                       help='Top-k for attack search')
    parser.add_argument('--allow_non_ascii', action='store_true', default=True,
                       help='Allow non-ASCII characters in adversarial strings')
    parser.add_argument('--placement', type=str, default='prefix',
                       choices=['prefix', 'suffix'],
                       help='Whether to place adversarial string as prefix or suffix')
    parser.add_argument('--attack_prompt', type=str, default=None,
                       help='Custom prompt for attacks (if not provided, uses random WikiText-2)')
    
    # Consistency evaluation
    parser.add_argument('--skip_consistency', action='store_true',
                       help='Skip consistency validation to save time')
    parser.add_argument('--consistency_n_prompts', type=int, default=100,
                       help='Number of prompts for consistency evaluation')
    parser.add_argument('--consistency_min_tokens', type=int, default=6,
                       help='Minimum tokens for consistency evaluation prompts')
    parser.add_argument('--consistency_max_tokens', type=int, default=40,
                       help='Maximum tokens for consistency evaluation prompts')
    
    # Perplexity bake-off
    parser.add_argument('--skip_bakeoff', action='store_true',
                       help='Skip perplexity bake-off to save time')
    parser.add_argument('--bakeoff_n_prompts', type=int, default=100,
                       help='Number of prompts for perplexity evaluation')
    parser.add_argument('--bakeoff_batch_size', type=int, default=16,
                       help='Batch size for perplexity evaluation (reduce if out of memory)')
    parser.add_argument('--bakeoff_min_tokens', type=int, default=50,
                       help='Minimum tokens for bakeoff evaluation prompts')
    parser.add_argument('--bakeoff_max_tokens', type=int, default=150,
                       help='Maximum tokens for bakeoff evaluation prompts')
    
    # Performance and debugging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    
    # Set random seed
    set_all_seeds(args.seed)
    
    # GPU optimization for multi-GPU setup
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Set memory management for better multi-GPU performance
        torch.cuda.empty_cache()
        if args.device_map_auto:
            print("Using automatic device mapping for multi-GPU")
    else:
        print("CUDA not available, using CPU")
    
    print(f"\nAttack Evaluation Script")
    print(f"Models: {args.models}")
    print(f"Hypotheses: {args.hypotheses}")
    print(f"Output directory: {output_dir}")
    print(f"Detection threshold: {args.spike_threshold}")
    print(f"Attack steps: {args.attack_num_steps}")
    print(f"Head reduction: {args.head_reduction}")
    print(f"Skip consistency: {args.skip_consistency}")
    print(f"Skip bakeoff: {args.skip_bakeoff}")
    print(f"Use FP16: {args.use_fp16}")
    print(f"Random seed: {args.seed}")
    print("-" * 50)
    
    all_results = {}
    
    for model_idx, model_name in enumerate(args.models):
        print(f"\n{'='*60}")
        print(f"Processing {model_idx + 1}/{len(args.models)}: {model_name}")
        print(f"{'='*60}")
        
        # Set up logging for this model
        logger = setup_logging(output_dir, model_name)
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        try:
            model_results = process_model(model_name, output_dir, logger, args)
            all_results[model_name] = model_results
            
        except Exception as e:
            logger.error(f"Unexpected error processing {model_name}: {e}")
            logger.error(traceback.format_exc())
            all_results[model_name] = {
                'error': f"Unexpected error: {e}",
                'model_info': {'name': model_name, 'timestamp': datetime.now().isoformat()}
            }
        
        finally:
            # Clean up CUDA memory between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Log memory usage after cleanup
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"After cleanup - GPU {i}: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_results_path = output_dir / f"combined_attack_evaluation_results_{timestamp}.json"
    
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Combined results saved: {combined_results_path}")
    print(f"Individual model results and plots saved in: {output_dir}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print(f"\nSummary:")
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"  {model_name}: ERROR - {results['error']}")
        else:
            n_super_weights = len(results.get('super_weights', []))
            n_attacks = sum(len([r for r in sw_results.values() if r is not None]) 
                          for sw_results in results.get('attack_results', {}).values())
            print(f"  {model_name}: {n_super_weights} super weights, {n_attacks} successful attacks")


if __name__ == "__main__":
    main()
