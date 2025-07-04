import random, math, numpy as np
from typing import List, Dict, Any, Tuple
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import existing dataset utilities
from utils.datasets import DatasetLoader

# --- Enhanced Prompt Sampling ---
def sample_prompts_with_token_filtering(
    tokenizer,
    num_prompts: int = 100,
    seed: int = 42,
    dataset_name: str = 'wikitext',
    dataset_config: str = 'wikitext-2-raw-v1',
    split: str = 'validation',
    min_tokens: int = 6,
    max_tokens: int = 40,
    domain: str = None,  # For custom domain-specific samples
) -> List[str]:
    """
    Sample prompts using DatasetLoader with token length filtering.
    
    Args:
        tokenizer: Tokenizer for length calculation
        num_prompts: Number of prompts to collect
        seed: Random seed
        dataset_name: Dataset to use ('wikitext', 'custom', etc.)
        dataset_config: Dataset configuration
        split: Dataset split
        min_tokens: Minimum token count (without special tokens)
        max_tokens: Maximum token count (without special tokens)
        domain: For custom samples, specify domain type
    """
    loader = DatasetLoader(seed=seed)
    
    if dataset_name == 'custom' and domain:
        # Use custom domain-specific samples - load extra to account for filtering
        raw_texts = loader.load_custom_text_samples(domain=domain, n_samples=num_prompts * 3)
    else:
        # Use standard dataset loading - load extra for filtering
        raw_texts = loader.load_perplexity_dataset(
            dataset_name=dataset_name,
            config=dataset_config,
            split=split,
            n_samples=num_prompts * 3,
            min_length=10
        )
    
    # Filter by token count
    collected = []
    for text in raw_texts:
        if not text.strip():
            continue
            
        # Check token length without special tokens
        token_ids = tokenizer(text, add_special_tokens=False)['input_ids']
        token_count = len(token_ids)
        
        if min_tokens <= token_count <= max_tokens:
            collected.append(text)
        
        if len(collected) >= num_prompts:
            break
    
    if len(collected) < num_prompts:
        print(f"[WARN] Only collected {len(collected)} prompts (wanted {num_prompts}). Consider adjusting token limits or using a different dataset.")
    
    return collected[:num_prompts]


def sample_wikitext_prompts_filtered(
    tokenizer,
    num_prompts: int = 100,
    seed: int = 42,
    dataset_config: str = 'wikitext-2-raw-v1',
    split: str = 'validation',
    min_tokens: int = 6,
    max_tokens: int = 40,
    max_stream: int = 5000,
    deduplicate: bool = True,
) -> List[str]:
    """
    Backward compatibility wrapper - now uses the enhanced sampling method.
    """
    return sample_prompts_with_token_filtering(
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        seed=seed,
        dataset_name='wikitext',
        dataset_config=dataset_config,
        split=split,
        min_tokens=min_tokens,
        max_tokens=max_tokens
    )


# --- Per-Prompt Evaluation ---

def evaluate_prompts_metrics(
    attacker,
    prompts: List[str],
    adv_text: str = "",
    metrics_of_interest: Tuple[str, ...] = (
        'down_proj_in_col_at_sink',
        'down_proj_out_row_at_sink',
    ),
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Evaluate metrics on each prompt using attacker.eval_metrics().
    Returns raw metric arrays with one value per prompt.
    """
    original_prompt = attacker.config.prompt_text
    results = {m: [] for m in metrics_of_interest}
    layouts = []  # Store layout information for debugging
    
    iterator = tqdm(prompts, desc="Evaluating prompts" + (" (adv)" if adv_text else " (baseline)")) if show_progress else prompts
    
    for p in iterator:
        attacker.config.prompt_text = p
        m = attacker.eval_metrics(adv_text=adv_text)
        for k in metrics_of_interest:
            results[k].append(m.get(k, float('nan')))
        layouts.append(m['layout'])
    
    # Restore original prompt
    attacker.config.prompt_text = original_prompt
    
    return {
        'metrics_per_prompt': results,
        'prompts': prompts,
        'adv_text': adv_text,
        'layouts': layouts
    }

# --- Reduction Analysis and Summary Statistics ---

def compute_reductions_and_summary(
    baseline: Dict[str, Any],
    attacked: Dict[str, Any],
    metrics_of_interest: Tuple[str, ...],
    eps: float = 1e-12
) -> Dict[str, Any]:
    """
    Compute percent reductions: (baseline - attacked) / baseline * 100.
    Returns per-metric arrays and summary statistics (median, p10, p90, mean).
    """
    summary = {}
    per_metric = {}
    
    for metric in metrics_of_interest:
        base_vals = np.array(baseline['metrics_per_prompt'][metric], dtype=float)
        att_vals = np.array(attacked['metrics_per_prompt'][metric], dtype=float)
        
        # Compute percent reduction
        with np.errstate(divide='ignore', invalid='ignore'):
            reduction = (base_vals - att_vals) / (base_vals + eps) * 100.0
            reduction[~np.isfinite(reduction)] = np.nan
        
        per_metric[metric] = {
            'baseline': base_vals,
            'attacked': att_vals,
            'reduction_percent': reduction
        }
        
        # Calculate summary statistics for valid reductions
        valid = reduction[~np.isnan(reduction)]
        if valid.size == 0:
            med = p10 = p90 = mn = float('nan')
        else:
            med = float(np.median(valid))
            p10 = float(np.percentile(valid, 10))
            p90 = float(np.percentile(valid, 90))
            mn = float(np.mean(valid))
        
        summary[metric] = {
            'median_reduction_percent': med,
            'p10_reduction_percent': p10,
            'p90_reduction_percent': p90,
            'mean_reduction_percent': mn,
            'n_valid': int(valid.size),
            'n_total': int(reduction.size)
        }
    return {
        'per_metric': per_metric,
        'summary': summary
    }

# --- Pass/Fail Logic ---

def evaluate_pass_fail(
    seed_summaries: Dict[int, Dict[str, Any]],
    metrics_of_interest: Tuple[str, ...],
    min_median_reduction: float = 20.0,
    min_p10_reduction: float = 10.0,
    median_variation_tolerance: float = 5.0
) -> Dict[str, Any]:
    """
    Combine multi-seed summaries into final pass/fail evaluation.
    Checks consistency across seeds and minimum reduction thresholds.
    """
    final = {}
    overall_pass = True
    
    for metric in metrics_of_interest:
        medians = []
        p10s = []
        for seed, seed_data in seed_summaries.items():
            medians.append(seed_data['summary'][metric]['median_reduction_percent'])
            p10s.append(seed_data['summary'][metric]['p10_reduction_percent'])
        
        medians_arr = np.array(medians, dtype=float)
        p10_arr = np.array(p10s, dtype=float)
        
        mean_median = float(np.nanmean(medians_arr))
        max_dev = float(np.nanmax(np.abs(medians_arr - mean_median)))
        min_median_observed = float(np.nanmin(medians_arr))
        min_p10_observed = float(np.nanmin(p10_arr))
        
        pass_median = (min_median_observed >= min_median_reduction)
        pass_p10 = (min_p10_observed >= min_p10_reduction)
        pass_stability = (max_dev <= median_variation_tolerance)
        
        metric_pass = pass_median and pass_p10 and pass_stability
        overall_pass = overall_pass and metric_pass
        
        final[metric] = {
            'seed_medians': medians,
            'seed_p10s': p10s,
            'mean_median': mean_median,
            'max_abs_deviation_median': max_dev,
            'min_median_observed': min_median_observed,
            'min_p10_observed': min_p10_observed,
            'thresholds': {
                'min_median_reduction': min_median_reduction,
                'min_p10_reduction': min_p10_reduction,
                'median_variation_tolerance': median_variation_tolerance
            },
            'passes': {
                'median_threshold': pass_median,
                'p10_threshold': pass_p10,
                'stability': pass_stability
            },
            'metric_pass': metric_pass
        }
    
    return {'per_metric': final, 'overall_pass': overall_pass}

# --- Single Seed Evaluation ---

def run_heldout_evaluation_single_seed(
    session,
    attacker,
    adv_string: str,
    seed: int,
    n_prompts: int = 100,
    min_tokens: int = 6,
    max_tokens: int = 40,
    metrics_of_interest: Tuple[str, ...] = (
        'down_proj_in_col_at_sink',
        'down_proj_out_row_at_sink',
    ),
    dataset_name: str = 'wikitext',
    dataset_config: str = 'wikitext-2-raw-v1',
    split: str = 'validation',
    domain: str = None,
    show_progress: bool = True,
    set_all_seeds_fn = None
) -> Dict[str, Any]:
    """
    Run single-seed evaluation with flexible dataset options.
    """
    if set_all_seeds_fn is not None:
        set_all_seeds_fn(seed)
    
    tokenizer = session.tokenizer
    
    # Sample prompts with token filtering
    prompts = sample_prompts_with_token_filtering(
        tokenizer=tokenizer,
        num_prompts=n_prompts,
        seed=seed,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        domain=domain
    )
    
    baseline_eval = evaluate_prompts_metrics(
        attacker=attacker,
        prompts=prompts,
        adv_text="",
        metrics_of_interest=metrics_of_interest,
        show_progress=show_progress
    )
    attacked_eval = evaluate_prompts_metrics(
        attacker=attacker,
        prompts=prompts,
        adv_text=adv_string,
        metrics_of_interest=metrics_of_interest,
        show_progress=show_progress
    )
    
    reductions = compute_reductions_and_summary(
        baseline=baseline_eval,
        attacked=attacked_eval,
        metrics_of_interest=metrics_of_interest
    )
    
    return {
        'seed': seed,
        'prompts': prompts,
        'baseline': baseline_eval,
        'attacked': attacked_eval,
        'reductions': reductions,
        'dataset_info': {
            'name': dataset_name,
            'config': dataset_config,
            'split': split,
            'domain': domain,
            'n_prompts': len(prompts)
        }
    }

# --- Multi-Seed Consistency Evaluation ---

def run_multi_seed_consistency_evaluation(
    session,
    attacker,
    adv_string: str,
    seeds: List[int] = [41, 42, 43],
    n_prompts: int = 100,
    min_tokens: int = 6,
    max_tokens: int = 40,
    metrics_of_interest: Tuple[str, ...] = (
        'down_proj_in_col_at_sink',
        'down_proj_out_row_at_sink',
    ),
    dataset_name: str = 'wikitext',
    dataset_config: str = 'wikitext-2-raw-v1', 
    split: str = 'validation',
    domain: str = None,
    thresholds: Dict[str, float] = None,
    show_progress: bool = True,
    set_all_seeds_fn = None
) -> Dict[str, Any]:
    """
    Run multi-seed evaluation with flexible dataset options.
    """
    if thresholds is None:
        thresholds = {
            'min_median_reduction': 20.0,
            'min_p10_reduction': 10.0,
            'median_variation_tolerance': 5.0
        }
    
    seed_results = {}
    
    print("=== Multi-seed held-out evaluation ===")
    print(f"Seeds: {seeds}")
    print(f"Dataset: {dataset_name} ({dataset_config}, {split})")
    if domain:
        print(f"Domain: {domain}")
    print(f"Metrics: {metrics_of_interest}")
    print(f"Thresholds: {thresholds}")
    print("-------------------------------------")
    
    for sd in seeds:
        print(f"\n[Seed {sd}] Running held-out evaluation...")
        res = run_heldout_evaluation_single_seed(
            session=session,
            attacker=attacker,
            adv_string=adv_string,
            seed=sd,
            n_prompts=n_prompts,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            metrics_of_interest=metrics_of_interest,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            domain=domain,
            show_progress=show_progress,
            set_all_seeds_fn=set_all_seeds_fn
        )
        seed_results[sd] = res
        
        # Quick per-seed print
        for metric in metrics_of_interest:
            summ = res['reductions']['summary'][metric]
            print(f"  Metric: {metric}")
            print(f"    Median reduction: {summ['median_reduction_percent']:.2f}% "
                  f"(p10={summ['p10_reduction_percent']:.2f}%, p90={summ['p90_reduction_percent']:.2f}%) "
                  f"n_valid={summ['n_valid']}/{summ['n_total']}")
    
    # Build seed summaries for pass/fail
    seed_summaries = {sd: seed_results[sd]['reductions'] for sd in seeds}
    
    pass_fail = evaluate_pass_fail(
        seed_summaries=seed_summaries,
        metrics_of_interest=metrics_of_interest,
        min_median_reduction=thresholds['min_median_reduction'],
        min_p10_reduction=thresholds['min_p10_reduction'],
        median_variation_tolerance=thresholds['median_variation_tolerance']
    )
    
    print("\n=== PASS/FAIL SUMMARY ===")
    for metric, info in pass_fail['per_metric'].items():
        print(f"Metric: {metric}")
        print(f"  Seed medians: {['{:.2f}'.format(x) for x in info['seed_medians']]}")
        print(f"  Mean median: {info['mean_median']:.2f}% | Max abs dev: {info['max_abs_deviation_median']:.2f}%")
        print(f"  Min median observed: {info['min_median_observed']:.2f}% "
              f"(threshold {info['thresholds']['min_median_reduction']:.2f}%) "
              f"-> pass={info['passes']['median_threshold']}")
        print(f"  Min p10 observed: {info['min_p10_observed']:.2f}% "
              f"(threshold {info['thresholds']['min_p10_reduction']:.2f}%) "
              f"-> pass={info['passes']['p10_threshold']}")
        print(f"  Stability pass (≤ {info['thresholds']['median_variation_tolerance']:.2f}%): "
              f"{info['passes']['stability']}")
        print(f"  METRIC PASS: {info['metric_pass']}")
    print(f"\nOVERALL PASS: {pass_fail['overall_pass']}")
    
    return {
        'seed_results': seed_results,
        'pass_fail': pass_fail,
        'config': {
            'seeds': seeds,
            'n_prompts': n_prompts,
            'min_tokens': min_tokens,
            'max_tokens': max_tokens,
            'metrics_of_interest': metrics_of_interest,
            'thresholds': thresholds,
            'adv_string': adv_string,
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'split': split,
            'domain': domain
        }
    }

# --- Statistical Test Helpers ---

def _summary_from_array(arr: np.ndarray) -> Dict[str, float]:
    """Compute basic summary statistics from array."""
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'p25': float('nan'),
            'p75': float('nan'),
            'count': 0
        }
    
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {
            'mean': float('nan'),
            'median': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'p25': float('nan'),
            'p75': float('nan'),
            'count': 0
        }
    
    return {
        'mean': float(np.mean(valid)),
        'median': float(np.median(valid)),
        'std': float(np.std(valid)),
        'min': float(np.min(valid)),
        'max': float(np.max(valid)),
        'p25': float(np.percentile(valid, 25)),
        'p75': float(np.percentile(valid, 75)),
        'count': int(len(valid))
    }


def _paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Perform paired t-test between two arrays."""
    try:
        from scipy import stats
        valid_mask = ~(np.isnan(a) | np.isnan(b))
        if np.sum(valid_mask) < 2:
            return {'t': float('nan'), 'p': float('nan')}
        
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]
        
        t_stat, p_val = stats.ttest_rel(a_valid, b_valid)
        return {'t': float(t_stat), 'p': float(p_val)}
    except ImportError:
        # Fallback implementation without scipy
        valid_mask = ~(np.isnan(a) | np.isnan(b))
        if np.sum(valid_mask) < 2:
            return {'t': float('nan'), 'p': float('nan')}
        
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]
        diff = a_valid - b_valid
        
        if len(diff) < 2 or np.std(diff) == 0:
            return {'t': float('nan'), 'p': float('nan')}
        
        t_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
        # Simple approximation for p-value (not exact without scipy)
        p_val = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / 2)))  # rough approximation
        return {'t': float(t_stat), 'p': float(p_val)}


def _bootstrap_ci(data: np.ndarray, seed: int = 42, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0 or np.all(np.isnan(data)):
        return (float('nan'), float('nan'))
    
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return (float('nan'), float('nan'))
    
    rng = np.random.RandomState(seed)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(valid_data, size=len(valid_data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


# --- Per-Prompt Perplexity Calculation ---

@torch.no_grad()
def ppl_for_prompts(
    model,
    tokenizer,
    prompts: List[str],
    attacker,
    prefix_str: str = None,
    batch_size: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Compute per-prompt negative log-likelihood (mean CE) and perplexity.
    
    If prefix_str is None or empty:
        - Computes perplexity on the entire prompt (including BOS if present)
    If prefix_str is provided:
        - Uses the attacker's layout calculation to identify content tokens
        - Computes perplexity only on content tokens (excluding prefix/suffix and BOS)
    
    Returns:
        {
            'nll': np.ndarray[float]  shape (N,)
            'ppl': np.ndarray[float]  shape (N,)
            'token_counts': np.ndarray[int] (# tokens contributing to loss)
        }
    """
    model.eval()
    device = next(model.parameters()).device
    
    if not prompts:
        return {'nll': np.array([]), 'ppl': np.array([]), 'token_counts': np.array([])}

    # Store original attacker state to restore later
    original_prompt = attacker.config.prompt_text
    original_adv_string = attacker.config.adv_string_init
    
    nll_list = []
    tok_count_list = []

    # Check if we have adversarial text ONCE before the loop
    has_adversarial = prefix_str and prefix_str.strip()

    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_nll = []
            batch_tok_counts = []
            
            # Process each prompt individually
            for prompt in batch:
                has_bos = attacker._has_bos_token()
                
                if not has_adversarial:
                    # BASELINE CASE: No adversarial text - compute perplexity on entire prompt
                    prompt_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'].to(device)
                    inputs = prompt_ids
                    
                    # For regular perplexity, we want the entire sequence
                    target_start = 0
                    target_end = prompt_ids.shape[1]
                    
                else:
                    # ADVERSARIAL CASE: Use attacker's layout to identify content tokens
                    # Only NOW do we set the prompt and recalculate layout
                    attacker.config.prompt_text = prompt
                    # Also need to set the adversarial string for layout calculation
                    attacker.config.adv_string_init = prefix_str
                    # Recalculate layout for this specific prompt and adversarial text
                    attacker._calculate_content_layout()
                    
                    # Get tokenized components
                    if has_bos:
                        prompt_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'].to(device)
                        bos_ids = prompt_ids[:, 0:1]
                        content_ids = prompt_ids[:, 1:]  # Exclude BOS
                    else:
                        prompt_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'].to(device)
                        bos_ids = None
                        content_ids = prompt_ids  # All tokens are content
                    
                    prefix_ids = tokenizer(prefix_str, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)
                    
                    # Build input sequence according to attacker's placement
                    if has_bos:
                        if attacker.config.placement == "prefix":
                            # [BOS][prefix][content_tokens]
                            inputs = torch.cat([bos_ids, prefix_ids, content_ids], dim=1)
                        else:  # suffix
                            # [BOS][content_tokens][prefix]
                            inputs = torch.cat([bos_ids, content_ids, prefix_ids], dim=1)
                    else:
                        if attacker.config.placement == "prefix":
                            # [prefix][content_tokens]
                            inputs = torch.cat([prefix_ids, content_ids], dim=1)
                        else:  # suffix
                            # [content_tokens][prefix]
                            inputs = torch.cat([content_ids, prefix_ids], dim=1)
                    
                    # Use the calculated content token positions
                    target_start = attacker.content_start_idx
                    target_end = attacker.content_end_idx
                
                # Prepare for next-token prediction
                input_ids = inputs[:, :-1]  # Input for prediction
                target_ids = inputs[:, 1:]  # Targets to predict
                
                # Create attention mask
                attention_mask = torch.ones_like(input_ids)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [1, seq_len-1, vocab_size]
                
                # Extract logits and targets for the target range
                # Adjust indices for the shifted prediction (inputs[:, :-1])
                if target_end > target_start:
                    pred_target_start = max(0, target_start - 1)
                    pred_target_end = min(logits.shape[1], target_end - 1)
                    
                    if pred_target_end > pred_target_start:
                        target_logits = logits[:, pred_target_start:pred_target_end, :]  # [1, target_len, vocab]
                        target_targets = target_ids[:, pred_target_start:pred_target_end]  # [1, target_len]
                        
                        # Compute cross-entropy loss
                        ce_loss = F.cross_entropy(
                            target_logits.reshape(-1, target_logits.size(-1)),
                            target_targets.reshape(-1),
                            reduction='none'
                        )
                        
                        # Average over target tokens
                        mean_ce = ce_loss.mean()
                        tok_count = target_targets.numel()
                        
                        batch_nll.append(mean_ce.item())
                        batch_tok_counts.append(tok_count)
                    else:
                        # Edge case: no target tokens
                        batch_nll.append(float('nan'))
                        batch_tok_counts.append(0)
                else:
                    # Edge case: no target tokens
                    batch_nll.append(float('nan'))
                    batch_tok_counts.append(0)
            
            nll_list.extend(batch_nll)
            tok_count_list.extend(batch_tok_counts)
    
    finally:
        # Restore original attacker state
        attacker.config.prompt_text = original_prompt
        attacker.config.adv_string_init = original_adv_string

    nll = np.array(nll_list, dtype=np.float64)
    ppl = np.exp(nll)
    
    return {
        'nll': nll,
        'ppl': ppl,
        'token_counts': np.array(tok_count_list, dtype=np.int32)
    }

# --- Random Prefix Generation ---

def generate_random_prefix_like(tokenizer, adv_prefix: str, seed: int = 0) -> str:
    """
    Generate a random prefix with the same token count as adv_prefix,
    sampling uniformly from the tokenizer's vocabulary.
    """
    rng = random.Random(seed)
    
    # Get target token count
    adv_tok_ids = tokenizer(adv_prefix, add_special_tokens=False)['input_ids']
    n_tokens = len(adv_tok_ids)
    
    if n_tokens == 0:
        return ""
    
    # Get vocabulary size and filter out special tokens
    vocab_size = len(tokenizer.get_vocab())
    
    # Filter out special tokens, control tokens, etc.
    valid_token_ids = []
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            # Skip special tokens, control characters, etc.
            if (not token_str.startswith('<') and 
                not token_str.startswith('[') and 
                not token_str.startswith('▁') and  # SentencePiece prefix
                len(token_str.strip()) > 0):
                valid_token_ids.append(token_id)
        except:
            continue
    
    if not valid_token_ids:
        # Fallback if filtering is too aggressive
        valid_token_ids = list(range(min(1000, vocab_size)))  # Skip token 0 (often padding)
    
    # Sample random tokens
    random_token_ids = [rng.choice(valid_token_ids) for _ in range(n_tokens)]
    
    # Decode to string
    random_string = tokenizer.decode(random_token_ids, skip_special_tokens=True)
    return random_string


# --- Activation Metric Collection ---

def collect_activation_metric_per_prompt(
    attacker,
    prompts: List[str],
    prefix: str,
    activation_metric: str
) -> np.ndarray:
    """
    Collect activation metrics per prompt using the attacker's eval_metrics method.
    """
    ev = evaluate_prompts_metrics(
        attacker=attacker,
        prompts=prompts,
        adv_text=prefix,
        metrics_of_interest=(activation_metric,),
        show_progress=False
    )
    return np.array(ev['metrics_per_prompt'][activation_metric], dtype=float)


# --- Perplexity Bake-off Orchestrator ---

def run_perplexity_bakeoff(
    session,
    attacker,
    target_sw,
    adv_prefix: str,
    prompts: List[str] = None,
    seed: int = 42,
    activation_metric: str = 'down_proj_in_col_at_sink',
    include_additive: bool = False,
    random_prefix_seed: int = 123,
    batch_size: int = 16,
    dataset_name: str = 'wikitext',
    dataset_config: str = 'wikitext-2-raw-v1',
    split: str = 'validation',
    domain: str = None,
    min_tokens: int = 6,
    max_tokens: int = 40
) -> Dict[str, Any]:
    """
    Run comprehensive perplexity bake-off evaluation across multiple conditions.
    
    Compares perplexity and activation metrics across four conditions:
    1. Baseline: No prefix, normal model
    2. Adversarial: Learned adversarial prefix  
    3. Random: Random prefix matched in token length
    4. Zero SW: No prefix, but target super weight is zeroed
    
    Also collects activation metrics and performs statistical comparisons
    to assess attack effectiveness.
    
    Args:
        session: SuperWeightResearchSession with model and tokenizer
        attacker: SuperWeightAttacker instance for layout calculation
        target_sw: Target super weight for zeroing condition
        adv_prefix: Learned adversarial prefix string
        prompts: Evaluation prompts (if None, samples from dataset)
        seed: Random seed for prompt sampling
        activation_metric: Metric to collect ('down_proj_in_col_at_sink', etc.)
        batch_size: Batch size for perplexity calculation
        dataset_*: Dataset sampling parameters
        
    Returns:
        Dictionary containing per-condition results, comparisons, and statistics
    """
    if prompts is None:
        # Use enhanced prompt sampling
        prompts = sample_prompts_with_token_filtering(
            tokenizer=session.tokenizer,
            num_prompts=100,
            seed=seed,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            domain=domain
        )

    model = session.model
    tokenizer = session.tokenizer

    # Generate random prefix matching adversarial prefix characteristics
    rand_prefix = generate_random_prefix_like(tokenizer, adv_prefix, seed=random_prefix_seed)
    
    print(f"Dataset: {dataset_name} ({dataset_config}, {split})" + (f", domain: {domain}" if domain else ""))
    print(f"Adversarial prefix: '{adv_prefix}'")
    print(f"Random prefix: '{rand_prefix}'")
    print(f"Evaluating {len(prompts)} prompts across conditions...")

    # Compute PPLs for all conditions using attacker-based layout
    print("Computing baseline perplexities...")
    baseline = ppl_for_prompts(model, tokenizer, prompts, attacker, prefix_str=None, batch_size=batch_size)
    
    print("Computing adversarial perplexities...")
    adv_res = ppl_for_prompts(model, tokenizer, prompts, attacker, prefix_str=adv_prefix, batch_size=batch_size)
    
    print("Computing random prefix perplexities...")
    rand_res = ppl_for_prompts(model, tokenizer, prompts, attacker, prefix_str=rand_prefix, batch_size=batch_size)
    
    print("Computing zero super-weight perplexities...")
    with session.manager.temporary_zero([target_sw]):
        zero_res = ppl_for_prompts(model, tokenizer, prompts, attacker, prefix_str=None, batch_size=batch_size)
    
    if include_additive:
        print("Computing additive condition (adv_prefix + zero super-weight)...")
        with session.manager.temporary_zero([target_sw]):
            add_res = ppl_for_prompts(model, tokenizer, prompts, attacker, prefix_str=adv_prefix, batch_size=batch_size)
    else:
        add_res = None

    # Collect activation metrics for correlation analysis
    print("Collecting activation metrics...")
    baseline_act = collect_activation_metric_per_prompt(attacker, prompts, prefix="", activation_metric=activation_metric)
    adv_act = collect_activation_metric_per_prompt(attacker, prompts, prefix=adv_prefix, activation_metric=activation_metric)
    rand_act = collect_activation_metric_per_prompt(attacker, prompts, prefix=rand_prefix, activation_metric=activation_metric)
    
    with session.manager.temporary_zero([target_sw]):
        zero_act = collect_activation_metric_per_prompt(attacker, prompts, prefix="", activation_metric=activation_metric)
    
    if include_additive:
        with session.manager.temporary_zero([target_sw]):
            add_act = collect_activation_metric_per_prompt(attacker, prompts, prefix=adv_prefix, activation_metric=activation_metric)
    else:
        add_act = None

    # Organize results by condition
    conds = {
        'baseline': {
            'ppl': baseline['ppl'], 
            'nll': baseline['nll'], 
            'activation': baseline_act,
            'token_counts': baseline['token_counts']
        },
        'adv': {
            'ppl': adv_res['ppl'], 
            'nll': adv_res['nll'], 
            'activation': adv_act, 
            'prefix': adv_prefix,
            'token_counts': adv_res['token_counts']
        },
        'random': {
            'ppl': rand_res['ppl'], 
            'nll': rand_res['nll'], 
            'activation': rand_act, 
            'prefix': rand_prefix,
            'token_counts': rand_res['token_counts']
        },
        'zeroSW': {
            'ppl': zero_res['ppl'], 
            'nll': zero_res['nll'], 
            'activation': zero_act,
            'token_counts': zero_res['token_counts']
        },
    }
    
    if include_additive:
        conds['adv_zeroSW'] = {
            'ppl': add_res['ppl'], 
            'nll': add_res['nll'], 
            'activation': add_act, 
            'prefix': adv_prefix,
            'token_counts': add_res['token_counts']
        }

    # Compute summary statistics
    for k, v in conds.items():
        # Filter out NaN values for summary statistics
        valid_ppl = v['ppl'][~np.isnan(v['ppl'])]
        valid_nll = v['nll'][~np.isnan(v['nll'])]
        valid_act = v['activation'][~np.isnan(v['activation'])]
        
        v['summary'] = {
            'ppl': _summary_from_array(valid_ppl) if len(valid_ppl) > 0 else _summary_from_array(np.array([np.nan])),
            'nll': _summary_from_array(valid_nll) if len(valid_nll) > 0 else _summary_from_array(np.array([np.nan])),
            'activation': _summary_from_array(valid_act) if len(valid_act) > 0 else _summary_from_array(np.array([np.nan]))
        }

    # Pairwise comparisons with statistical tests
    def make_comp(a, b, name):
        A = conds[a]
        B = conds[b]
        
        # Filter paired data (remove NaN pairs)
        valid_mask = ~(np.isnan(A['ppl']) | np.isnan(B['ppl']))
        if not np.any(valid_mask):
            return {
                'A': a, 'B': b,
                'delta_mean_ppl': float('nan'),
                'delta_mean_ppl_pct': float('nan'),
                'delta_median_ppl': float('nan'),
                'ppl_t_test': {'t': float('nan'), 'p': float('nan')},
                'nll_t_test': {'t': float('nan'), 'p': float('nan')},
                'bootstrap_ci_delta_ppl': (float('nan'), float('nan')),
                'delta_mean_activation': float('nan'),
                'activation_corr_delta_ppl': float('nan')
            }
        
        A_ppl_valid = A['ppl'][valid_mask]
        B_ppl_valid = B['ppl'][valid_mask]
        A_nll_valid = A['nll'][valid_mask]
        B_nll_valid = B['nll'][valid_mask]
        A_act_valid = A['activation'][valid_mask]
        B_act_valid = B['activation'][valid_mask]
        
        diff_ppl = A_ppl_valid - B_ppl_valid
        diff_nll = A_nll_valid - B_nll_valid
        
        # Statistical tests
        t_ppl = _paired_t_test(A_ppl_valid, B_ppl_valid)
        t_nll = _paired_t_test(A_nll_valid, B_nll_valid)
        ci_low, ci_high = _bootstrap_ci(diff_ppl, seed=seed)
        
        # Correlation between activation change and perplexity change
        act_diff = A_act_valid - B_act_valid
        if len(act_diff) > 1 and act_diff.std() > 0 and diff_ppl.std() > 0:
            corr = float(np.corrcoef(act_diff, diff_ppl)[0, 1])
        else:
            corr = float('nan')

        return {
            'A': a,
            'B': b,
            'delta_mean_ppl': float(np.mean(diff_ppl)),
            'delta_mean_ppl_pct': float((np.mean(A_ppl_valid) - np.mean(B_ppl_valid)) / np.mean(B_ppl_valid) * 100),
            'delta_median_ppl': float(np.median(diff_ppl)),
            'ppl_t_test': t_ppl,
            'nll_t_test': t_nll,
            'bootstrap_ci_delta_ppl': (ci_low, ci_high),
            'delta_mean_activation': float(np.mean(act_diff)),
            'activation_corr_delta_ppl': corr,
            'n_valid_pairs': int(len(diff_ppl))
        }

    # Define comparisons
    comparisons = {
        'adv_vs_baseline': make_comp('adv', 'baseline', 'adv_vs_baseline'),
        'random_vs_baseline': make_comp('random', 'baseline', 'random_vs_baseline'),
        'adv_vs_random': make_comp('adv', 'random', 'adv_vs_random'),
        'zeroSW_vs_baseline': make_comp('zeroSW', 'baseline', 'zeroSW_vs_baseline'),
    }
    
    if include_additive:
        comparisons['adv_zeroSW_vs_zeroSW'] = make_comp('adv_zeroSW', 'zeroSW', 'adv_zeroSW_vs_zeroSW')
        comparisons['adv_zeroSW_vs_adv'] = make_comp('adv_zeroSW', 'adv', 'adv_zeroSW_vs_adv')

    return {
        'prompts': prompts,
        'conditions': conds,
        'comparisons': comparisons,
        'activation_metric': activation_metric,
        'config': {
            'seed': seed,
            'include_additive': include_additive,
            'batch_size': batch_size,
            'adv_prefix': adv_prefix,
            'rand_prefix': rand_prefix,
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'split': split,
            'domain': domain,
            'min_tokens': min_tokens,
            'max_tokens': max_tokens
        }
    }

def plot_perplexity_bakeoff(bake_results: Dict[str, Any], figsize=(12, 8), save_path: str = None):
    """
    Minimalist visualization of perplexity bake-off results with log scale for large differences.
    
    Args:
        bake_results: Output from run_perplexity_bakeoff()
        figsize: Figure size tuple
        save_path: If provided, save the plot to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    conditions = bake_results['conditions']
    comparisons = bake_results['comparisons']
    
    # Color scheme
    colors = {
        'baseline': '#2E8B57',    # Sea green
        'adv': '#DC143C',         # Crimson  
        'random': '#4169E1',      # Royal blue
        'zeroSW': '#FF8C00'       # Dark orange
    }
    
    # 1. Mean Perplexity by Condition (LOG SCALE)
    ax1 = axes[0, 0]
    condition_names = list(conditions.keys())
    mean_ppls = [conditions[name]['summary']['ppl']['mean'] for name in condition_names]
    
    bars = ax1.bar(range(len(condition_names)), mean_ppls, 
                   color=[colors[name] for name in condition_names], 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_yscale('log')  # LOG SCALE
    ax1.set_title('Mean Perplexity by Condition (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perplexity (log scale)')
    ax1.set_xticks(range(len(condition_names)))
    ax1.set_xticklabels(condition_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, which='both')
    
    # Add value labels on bars
    for bar, val in zip(bars, mean_ppls):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Perplexity Change vs Baseline (LOG SCALE for absolute values)
    ax2 = axes[0, 1]
    comparisons_to_plot = ['adv_vs_baseline', 'random_vs_baseline', 'zeroSW_vs_baseline']
    comp_labels = ['Adversarial', 'Random', 'Zero SW']
    comp_colors = ['#DC143C', '#4169E1', '#FF8C00']
    
    delta_ppls = [comparisons[comp]['delta_mean_ppl'] for comp in comparisons_to_plot]
    p_values = [comparisons[comp]['ppl_t_test']['p'] for comp in comparisons_to_plot]
    
    # Use absolute values for bar heights, track signs separately
    abs_delta_ppls = [abs(d) for d in delta_ppls]
    bar_colors = [comp_colors[i] if delta_ppls[i] >= 0 else 'lightgray' for i in range(len(delta_ppls))]
    
    bars = ax2.bar(range(len(comp_labels)), abs_delta_ppls, 
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_yscale('log')  # LOG SCALE for absolute values
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1)  # Reference line at |1|
    ax2.set_title('|Perplexity Change| vs Baseline (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('|ΔPerplexity| (log scale)')
    ax2.set_xticks(range(len(comp_labels)))
    ax2.set_xticklabels(comp_labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, which='both')
    
    # Add significance indicators and values
    for i, (bar, val, abs_val, p) in enumerate(zip(bars, delta_ppls, abs_delta_ppls, p_values)):
        if not np.isnan(val):
            # Show actual signed value
            sign = '+' if val >= 0 else ''
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                    f'{sign}{val:.1f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
            # Add significance stars
            if not np.isnan(p):
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if stars:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 2.0,
                            stars, ha='center', va='bottom', 
                            fontsize=8, color='red')
    
    # 3. Activation Change vs Perplexity Change Correlation
    ax3 = axes[1, 0]
    
    # Use adversarial vs baseline comparison for correlation plot
    adv_baseline_comp = comparisons['adv_vs_baseline']
    
    if 'activation_corr_delta_ppl' in adv_baseline_comp and not np.isnan(adv_baseline_comp['activation_corr_delta_ppl']):
        # Get the raw data for scatter plot
        baseline_cond = conditions['baseline']
        adv_cond = conditions['adv']
        
        # Filter valid data
        valid_mask = ~(np.isnan(baseline_cond['ppl']) | np.isnan(adv_cond['ppl']) | 
                      np.isnan(baseline_cond['activation']) | np.isnan(adv_cond['activation']))
        
        if np.any(valid_mask):
            ppl_change = adv_cond['ppl'][valid_mask] - baseline_cond['ppl'][valid_mask]
            act_change = adv_cond['activation'][valid_mask] - baseline_cond['activation'][valid_mask]
            
            ax3.scatter(act_change, ppl_change, alpha=0.6, color='#DC143C', s=30)
            
            # Add correlation line if significant correlation
            corr = adv_baseline_comp['activation_corr_delta_ppl']
            if abs(corr) > 0.1:  # Only show line if correlation is meaningful
                z = np.polyfit(act_change, ppl_change, 1)
                p = np.poly1d(z)
                x_line = np.linspace(act_change.min(), act_change.max(), 100)
                ax3.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1)
            
            # Add correlation coefficient text
            ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10, fontweight='bold')
    
    ax3.set_title('Activation vs Perplexity Change', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Δ Activation')
    ax3.set_ylabel('Δ Perplexity')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # 4. Distribution Comparison (Box plot with LOG SCALE)
    ax4 = axes[1, 1]
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    box_colors = []
    
    for name in ['baseline', 'adv', 'random', 'zeroSW']:
        if name in conditions:
            valid_ppl = conditions[name]['ppl'][~np.isnan(conditions[name]['ppl'])]
            if len(valid_ppl) > 0:
                box_data.append(valid_ppl)
                box_labels.append(name)
                box_colors.append(colors[name])
    
    if box_data:
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, 
                        showfliers=False, widths=0.6)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Style other elements
        for element in ['whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color('black')
                item.set_linewidth(1)
    
    ax4.set_yscale('log')  # LOG SCALE
    ax4.set_title('Perplexity Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Perplexity (log scale)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print key statistics with better formatting for large ranges
    print("\nKey Statistics:")
    print("-" * 60)
    for comp_name, comp_data in comparisons.items():
        if comp_name in ['adv_vs_baseline', 'random_vs_baseline', 'adv_vs_random']:
            delta_ppl = comp_data['delta_mean_ppl']
            p_val = comp_data['ppl_t_test']['p']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            # Format large numbers more readably
            if abs(delta_ppl) >= 100:
                delta_str = f"{delta_ppl:+,.0f}"
            else:
                delta_str = f"{delta_ppl:+.2f}"
                
            print(f"{comp_name:20s}: ΔPL={delta_str:>8s} (p={p_val:.3g}) {significance}")
