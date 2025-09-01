import torch
import torch.nn as nn
import numpy as np
import scipy.stats
from typing import List, Dict, Any, Optional

from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalMLPHandler
from utils.datasets import DatasetLoader
from utils.device_utils import safe_matmul


class VocabularyAnalyzer:
    """
    Analyzes how Super Weights affect model vocabulary and token probabilities.
    
    Provides methods for analyzing neuron vocabulary effects, measuring interventional 
    impacts on loss/entropy, cascade effects through layers, token class enrichment
    analysis, and control baselines for validation.
    """

    def __init__(self, model, tokenizer, manager, mlp_handler: UniversalMLPHandler):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        self.mlp_handler = mlp_handler
        self.dataset_loader = DatasetLoader(seed=42)
        
        # Cache for preprocessed unembedding matrix and stopword IDs
        self._cached_unembedding = None
        self._english_stop_ids = None
        self._init_stopword_ids()

    def _init_stopword_ids(self):
        """Initialize cached stopword token IDs for efficient evaluation"""
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        self._english_stop_ids = []
        
        for word in stopwords:
            try:
                # Try both with and without leading space
                for variant in [word, ' ' + word]:
                    token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    if len(token_ids) == 1:  # Single token
                        self._english_stop_ids.append(token_ids[0])
            except:
                pass
        
        self._english_stop_ids = list(set(self._english_stop_ids))  # Remove duplicates

    def _eval_windows(self, texts: List[str], window_len: int = 1024, stride: int = 512) -> Dict[str, float]:
        """
        Evaluate metrics using sliding windows for held-out evaluation.
        
        Tokenizes texts to fixed windows, computes per-token loss, entropy, 
        top-k margin and stopword mass, then averages across all windows.
        
        Args:
            texts: List of text strings to evaluate
            window_len: Length of each sliding window
            stride: Stride between windows
            
        Returns:
            Dictionary with averaged metrics across all windows
        """
        self.model.eval()
        with torch.inference_mode():
            losses, entropies, margins, stop_masses = [], [], [], []
            
            for text in texts:
                # Tokenize without truncation to get full sequence
                toks = self.tokenizer(text, return_tensors='pt', truncation=False)
                ids = toks.input_ids[0]
                
                # Process sliding windows
                for s in range(0, max(1, ids.size(0) - 1), stride):
                    win = ids[s:s + window_len]
                    if win.size(0) < 2:
                        break
                    
                    # Prepare inputs and targets
                    inputs = {
                        'input_ids': win[:-1].unsqueeze(0).to(self.model.device),
                        'attention_mask': torch.ones_like(win[:-1]).unsqueeze(0).to(self.model.device)
                    }
                    targets = win[1:].unsqueeze(0).to(self.model.device)
                    
                    # Forward pass
                    logits = self.model(**inputs).logits[0]  # [T, V]
                    logprobs = logits.log_softmax(-1)
                    
                    # Per-token loss (skip first token in window)
                    lp_true = logprobs.gather(-1, targets[0].unsqueeze(-1)).squeeze(-1)  # [T]
                    window_loss = (-lp_true[1:]).mean().item() if lp_true.size(0) > 1 else (-lp_true).mean().item()
                    losses.append(window_loss)
                    
                    # Per-token entropy (skip first token in window)  
                    p = logprobs.exp()
                    window_entropy = (-(p * logprobs).sum(-1)[1:]).mean().item() if logprobs.size(0) > 1 else (-(p * logprobs).sum(-1)).mean().item()
                    entropies.append(window_entropy)
                    
                    # Top-k margin (skip first token in window)
                    eval_probs = p[1:] if p.size(0) > 1 else p
                    top2 = eval_probs.topk(2, dim=-1).values  # [T-1, 2]
                    margin = (top2[:, 0] - top2[:, 1]).mean().item()
                    margins.append(margin)
                    
                    # Stopword mass (skip first token in window)
                    if self._english_stop_ids:
                        eval_probs = p[1:] if p.size(0) > 1 else p
                        stopword_mass = eval_probs[:, self._english_stop_ids].sum(-1).mean().item()
                        stop_masses.append(stopword_mass)
            
            return {
                'loss': float(np.mean(losses)) if losses else float('inf'),
                'entropy': float(np.mean(entropies)) if entropies else 0.0,
                'topk_margin': float(np.mean(margins)) if margins else 0.0,
                'stopword_mass': float(np.mean(stop_masses)) if stop_masses else None,
            }

    def _compute_robust_correlation(self, effects1: torch.Tensor, effects2: torch.Tensor) -> float:
        """
        Compute correlation with robustness against constant vectors.
        
        torch.corrcoef returns nan when a vector is (near) constant. This method
        adds a small guard to handle such cases gracefully.
        
        Args:
            effects1: First effect vector
            effects2: Second effect vector
            
        Returns:
            Correlation coefficient, or 0.0 if either vector is near-constant
        """
        # Check for near-constant vectors
        if effects1.std() < 1e-8 or effects2.std() < 1e-8:
            return 0.0
        
        try:
            corr_matrix = torch.corrcoef(torch.stack([effects1, effects2]))
            return float(corr_matrix[0, 1])
        except:
            return 0.0

    def _load_token_filtered_texts(self, dataset_name: str = 'wikitext', 
                                 config: str = 'wikitext-2-raw-v1',
                                 n_samples: int = 100,
                                 min_tokens: int = 10,
                                 max_tokens: int = 2048) -> List[str]:
        """
        Load texts with token-level filtering instead of character-level.
        
        Uses tokenizer for precise length control and provides fallback for offline usage.
        """
        try:
            # Try to load from dataset loader
            candidate_texts = self.dataset_loader.load_perplexity_dataset(
                dataset_name=dataset_name,
                config=config,
                n_samples=n_samples * 2,  # Load extra to account for filtering
                min_length=1  # We'll do token-level filtering
            )
        except Exception:
            # Fallback to simple texts if dataset loading fails
            candidate_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models can be trained on large datasets.",
                "Python is a popular programming language for data science.",
                "Natural language processing involves understanding human language.",
                "Deep learning uses neural networks with multiple layers."
            ] * (n_samples // 5 + 1)
        
        # Filter by token length
        filtered_texts = []
        for text in candidate_texts:
            try:
                # Tokenize and check length
                tokens = self.tokenizer(text, truncation=False, return_tensors='pt')
                token_count = tokens['input_ids'].shape[1]
                
                if min_tokens <= token_count <= max_tokens:
                    filtered_texts.append(text)
                elif token_count > max_tokens:
                    # Truncate and decode back to text
                    truncated_tokens = self.tokenizer(text, max_length=max_tokens, 
                                                    truncation=True, return_tensors='pt')
                    truncated_text = self.tokenizer.decode(truncated_tokens['input_ids'][0], 
                                                         skip_special_tokens=True)
                    filtered_texts.append(truncated_text)
                    
                if len(filtered_texts) >= n_samples:
                    break
            except:
                continue
        
        return filtered_texts[:n_samples]

    def analyze_neuron_vocabulary_effects(self, super_weight: SuperWeight, 
                                         apply_universal_neurons_processing: bool = True) -> Dict[str, Any]:
        """
        Analyze vocabulary effects of the neuron containing the super weight.
        
        Computes e = W_U @ w_out for the intermediate neuron and reports statistical
        moments, top affected tokens, and function classification.
        
        Args:
            super_weight: SuperWeight to analyze (analyzes the neuron containing it)
            apply_universal_neurons_processing: Apply layer norm folding and mean-centering
        
        Returns:
            Dictionary containing vocabulary effects, statistics, classification, and top tokens
        """
        try:
            with torch.no_grad():
                # Get the down projection matrix containing the super weight
                components = self.mlp_handler.get_mlp_components(super_weight.layer)
                if 'down' not in components:
                    raise ValueError(f"No down projection found in layer {super_weight.layer}")
                
                down_proj = components['down']
                
                # Get sw_neuron: the neuron that contains our super weight
                # This is the column vector showing how this neuron affects all output dimensions
                sw_neuron = down_proj.weight[:, super_weight.column]  # [2048]
                
                # Get unembedding matrix with Universal Neurons processing
                unembedding_matrix = self._get_unembedding_matrix(apply_universal_neurons_processing)
                
                # Apply mean-centering to neuron if processing is enabled (paper-faithful)
                if apply_universal_neurons_processing:
                    sw_neuron = sw_neuron - sw_neuron.mean()
                
                # Compute vocabulary effects: e = W_U @ w_out
                vocab_effects_raw = safe_matmul(unembedding_matrix, sw_neuron, result_device='cpu')
                
                # Cosine similarity variant (norm-robust)
                vocab_effects_cosine = self._compute_cosine_vocab_effects(unembedding_matrix, sw_neuron)
            
            # Analyze both variants
            raw_analysis = self._analyze_vocab_effects(vocab_effects_raw, "raw_dot_product")
            cosine_analysis = self._analyze_vocab_effects(vocab_effects_cosine, "cosine_similarity")
            
            return {
                'super_weight': super_weight,
                'sw_neuron_coordinates': (super_weight.layer, super_weight.column),
                'processing_applied': apply_universal_neurons_processing,
                'raw_analysis': raw_analysis,
                'cosine_analysis': cosine_analysis,
                # Main results
                'vocab_effects': vocab_effects_raw.cpu().numpy(),
                'statistics': raw_analysis['statistics'],
                'classification': raw_analysis['classification'],
                'top_tokens': raw_analysis['top_tokens'],
                'enrichment': raw_analysis['enrichment']
            }
            
        except Exception as e:
            return {
                'super_weight': super_weight,
                'error': f"Failed to analyze sw_neuron: {str(e)}"
            }

    def analyze_super_weight_intervention(self, super_weight: SuperWeight, 
                                        test_texts: Optional[List[str]] = None,
                                        n_samples: int = 100) -> Dict[str, Any]:
        """
        Measure the interventional effect of zeroing a super weight.
        
        Compares model behavior with and without the super weight to quantify
        its contribution to loss, entropy, top-k margin, and stopword probability mass
        using sliding window evaluation for robust metrics.
        
        Args:
            super_weight: SuperWeight to analyze
            test_texts: Optional custom texts (otherwise uses wikitext)
            n_samples: Number of samples to use
            
        Returns:
            Dictionary with baseline and modified metrics plus deltas
        """
        # Get test texts with improved token-level filtering
        if test_texts is None:
            test_texts = self._load_token_filtered_texts(
                dataset_name='wikitext',
                config='wikitext-2-raw-v1',
                n_samples=n_samples,
                min_tokens=10,
                max_tokens=2048
            )
        
        # Measure baseline metrics using windowed evaluation
        baseline_metrics = self._eval_windows(test_texts)
        
        # Measure with intervention
        with self.manager.temporary_zero([super_weight]):
            modified_metrics = self._eval_windows(test_texts)
        
        return {
            'super_weight': super_weight,
            'baseline_loss': baseline_metrics['loss'],
            'modified_loss': modified_metrics['loss'],
            'delta_loss': modified_metrics['loss'] - baseline_metrics['loss'],
            'baseline_entropy': baseline_metrics['entropy'],
            'modified_entropy': modified_metrics['entropy'],
            'delta_entropy': modified_metrics['entropy'] - baseline_metrics['entropy'],
            'baseline_topk_margin': baseline_metrics['topk_margin'],
            'modified_topk_margin': modified_metrics['topk_margin'],
            'delta_topk_margin': modified_metrics['topk_margin'] - baseline_metrics['topk_margin'],
            'baseline_stopword_mass': baseline_metrics['stopword_mass'],
            'modified_stopword_mass': modified_metrics['stopword_mass'],
            'delta_stopword_mass': (modified_metrics['stopword_mass'] - baseline_metrics['stopword_mass']) if baseline_metrics['stopword_mass'] is not None else None,
            'n_samples': len(test_texts)
        }

    def analyze_cascade_effects(self, super_weight: SuperWeight, 
                               input_text: str = "Apple Inc. is a tech company.",
                               num_layers: int = 5,
                               top_k_margin: int = 10) -> Dict[str, Any]:
        """
        Analyze how super weight effects propagate through model layers.
        
        Captures residual stream activations of the last token at multiple layers both 
        with and without the super weight intervention. Projects each layer's residual 
        through the unembedding matrix to measure vocabulary effects at different depths.
        
        Args:
            super_weight: SuperWeight to analyze
            input_text: Text to analyze cascade effects on
            num_layers: Number of layers to sample for cascade analysis
            top_k_margin: K value for computing top-k margin changes
            
        Returns:
            Dictionary with per-layer entropy changes, effect patterns, and actual layer indices
        """
        try:
            # Tokenize input
            tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
            
            # Get unembedding matrix for projections
            unembedding_matrix = self._get_unembedding_matrix(apply_universal_neurons_processing=False)
            
            # Capture residual streams at key layers (now returns layer indices too)
            baseline_residuals, actual_layer_indices = self._capture_residual_layers(tokens, num_layers=num_layers)
            
            with self.manager.temporary_zero([super_weight]):
                modified_residuals, _ = self._capture_residual_layers(tokens, num_layers=num_layers)
            
            # Analyze differences at each layer
            layer_effects = {}
            for i, (baseline_res, modified_res) in enumerate(zip(baseline_residuals, modified_residuals)):
                actual_layer_idx = actual_layer_indices[i]
                
                # Project to vocabulary space
                baseline_logits = safe_matmul(baseline_res, unembedding_matrix.T, result_device='cpu')
                modified_logits = safe_matmul(modified_res, unembedding_matrix.T, result_device='cpu')
                
                # Compute differences
                logit_diff = baseline_logits - modified_logits
                
                # Compute entropy change
                baseline_entropy = -torch.sum(torch.softmax(baseline_logits, dim=-1) * torch.log_softmax(baseline_logits, dim=-1))
                modified_entropy = -torch.sum(torch.softmax(modified_logits, dim=-1) * torch.log_softmax(modified_logits, dim=-1))
                entropy_change = float(baseline_entropy - modified_entropy)
                
                # Compute top-k margin change (now configurable)
                baseline_top_k = torch.topk(baseline_logits, top_k_margin).values
                modified_top_k = torch.topk(modified_logits, top_k_margin).values
                margin_change = float((baseline_top_k[0] - baseline_top_k[1]) - (modified_top_k[0] - modified_top_k[1]))
                
                layer_effects[actual_layer_idx] = {
                    'entropy_change': entropy_change,
                    'margin_change': margin_change,
                    'effect_magnitude': float(torch.norm(logit_diff)),
                    'layer_index': actual_layer_idx  # Include actual layer index
                }
            
            return {
                'super_weight': super_weight,
                'input_text': input_text,
                'layer_effects': layer_effects,
                'actual_layer_indices': actual_layer_indices,  # Return actual layer indices
                'amplification_pattern': self._classify_cascade_pattern(layer_effects),
                'num_layers_analyzed': len(actual_layer_indices),
                'top_k_margin': top_k_margin
            }
            
        except Exception as e:
            return {
                'super_weight': super_weight,
                'error': f"Cascade analysis failed: {str(e)}"
            }

    def analyze_token_class_enrichment(self, vocab_effects: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze enrichment of vocabulary effects within semantic token classes.
        
        Tests whether vocabulary effects are concentrated within specific token
        categories (digits, years, punctuation, etc.) using variance reduction.
        
        Args:
            vocab_effects: Vocabulary effects vector [vocab_size]
            
        Returns:
            Dictionary with enrichment scores and best enriched token class
        """
        # Define token classes for enrichment analysis
        token_classes = {
            'digits': lambda t: t.strip().isdigit(),
            'years': lambda t: t.strip().isdigit() and 1700 <= int(t.strip()) <= 2050 if t.strip().isdigit() else False,
            'parens': lambda t: any(char in t for char in '()[]{}'),
            'whitespace_prefixed': lambda t: t.startswith(' '),
            'caps': lambda t: t.strip() and t.strip()[0].isupper(),
            'pronouns': lambda t: t.strip().lower() in {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'},
            'punctuation': lambda t: any(char in t for char in '.,!?;:'),
            'stopwords': lambda t: t.strip().lower() in {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        }
        
        enrichment_scores = {}
        class_effects = {}
        
        # Compute enrichment for each token class
        for class_name, classifier_func in token_classes.items():
            class_effects_list = []
            non_class_effects_list = []
            special_token_ids = set(getattr(self.tokenizer, 'all_special_ids', []))
            
            for token_id in range(vocab_effects.shape[0]):
                if token_id in special_token_ids:
                    continue
                    
                try:
                    # Use improved token handling
                    raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                    token_str = self.tokenizer.convert_tokens_to_string([raw_token])
                    effect = vocab_effects[token_id].item()
                    
                    if classifier_func(token_str):
                        class_effects_list.append(effect)
                    else:
                        non_class_effects_list.append(effect)
                except:
                    # Skip problematic tokens
                    non_class_effects_list.append(vocab_effects[token_id].item())
            
            if len(class_effects_list) < 5:  # Too few examples
                enrichment_scores[class_name] = 0.0
                class_effects[class_name] = {'mean': 0.0, 'count': len(class_effects_list)}
                continue
            
            # Compute variance reduction (enrichment score)
            class_var = np.var(class_effects_list)
            non_class_var = np.var(non_class_effects_list) if non_class_effects_list else 1.0
            overall_var = np.var(vocab_effects.numpy())
            
            # Enrichment score: how much variance is explained by this class
            enrichment_score = (overall_var - (len(class_effects_list) * class_var + len(non_class_effects_list) * non_class_var) / len(vocab_effects)) / overall_var
            enrichment_scores[class_name] = max(0.0, enrichment_score)
            
            class_effects[class_name] = {
                'mean': np.mean(class_effects_list),
                'var': class_var,
                'count': len(class_effects_list),
                'top_effects': sorted(class_effects_list, key=abs, reverse=True)[:5]
            }
        
        # Find best theme
        best_theme = max(enrichment_scores.items(), key=lambda x: x[1])
        
        return {
            'enrichment_scores': enrichment_scores,
            'class_effects': class_effects,
            'best_theme': {
                'class': best_theme[0],
                'score': best_theme[1],
                'description': f"Effects concentrated in {best_theme[0]} tokens"
            }
        }

    @torch.no_grad()
    def analyze_controls_and_baselines(self, super_weight: SuperWeight,
                                     test_texts: Optional[List[str]] = None,
                                     n_samples: int = 50) -> Dict[str, Any]:
        """
        Run control experiments to validate super weight effects.
        
        Performs three control experiments: magnitude-matched random ablation,
        full neuron vs single scalar ablation comparison, and deterministic
        no-op verification.
        
        Args:
            super_weight: SuperWeight to analyze
            test_texts: Optional test texts for evaluation
            n_samples: Number of samples for analysis
            
        Returns:
            Dictionary with control experiment results
        """
        if test_texts is None:
            test_texts = self._load_token_filtered_texts(
                dataset_name='wikitext',
                config='wikitext-2-raw-v1',
                n_samples=n_samples,
                min_tokens=10,
                max_tokens=2048
            )
        
        # 1. Magnitude-matched random scalar ablation
        random_coord = self._get_random_coordinate_same_magnitude(super_weight)
        sw_effect = self.analyze_super_weight_intervention(super_weight, test_texts)
        random_effect = self.analyze_super_weight_intervention(random_coord, test_texts)
        
        # 2. Neuron vs scalar comparison
        neuron_vs_scalar = self._compare_neuron_vs_scalar_ablation(super_weight, test_texts)
        
        # 3. No-op check (deterministic verification)
        noop_check = self._verify_deterministic_noop(super_weight, test_texts)
        
        return {
            'super_weight': super_weight,
            'random_baseline': {
                'random_coord': random_coord,
                'sw_delta_loss': sw_effect['delta_loss'],
                'random_delta_loss': random_effect['delta_loss'],
                'specificity_ratio': abs(sw_effect['delta_loss']) / max(abs(random_effect['delta_loss']), 1e-6)
            },
            'neuron_vs_scalar': neuron_vs_scalar,
            'noop_check': noop_check
        }

    # Helper methods
    def _compute_cosine_vocab_effects(self, unembedding_matrix: torch.Tensor, 
                                    sw_neuron: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity variant (norm-robust).
        
        Uses F.cosine_similarity for cleaner implementation and better numerical stability.
        """
        # Use functional cosine similarity for cleaner, more robust computation
        cosine_similarities = torch.nn.functional.cosine_similarity(
            unembedding_matrix, sw_neuron.unsqueeze(0), dim=1
        )
        return cosine_similarities.cpu()
    
    def _analyze_vocab_effects(self, vocab_effects: torch.Tensor, analysis_type: str) -> Dict[str, Any]:
        """Analyze vocabulary effects for either raw or cosine variants"""
        statistics = self._compute_effect_statistics(vocab_effects)
        classification = self._classify_super_weight_function(vocab_effects)
        top_tokens = self._get_top_affected_tokens(vocab_effects)
        enrichment = self.analyze_token_class_enrichment(vocab_effects)
        
        return {
            'analysis_type': analysis_type,
            'statistics': statistics,
            'classification': classification,
            'top_tokens': top_tokens,
            'enrichment': enrichment
        }
    
    def _compute_effect_statistics(self, vocab_effects: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive statistics: moments, percentiles, and significance counts.
        
        Tracks both moments and percentiles for robust analysis across different
        effect distributions (raw dot products vs cosine similarities).
        """
        vocab_effects = vocab_effects.detach().cpu()
        effects_np = vocab_effects.numpy()
        
        # Compute percentiles for robust statistics
        percentiles = np.percentile(effects_np, [1, 5, 25, 50, 75, 95, 99])
        
        return {
            'mean': float(torch.mean(vocab_effects)),
            'std': float(torch.std(vocab_effects)),
            'variance': float(torch.var(vocab_effects)),
            'kurtosis': float(scipy.stats.kurtosis(effects_np)),
            'skew': float(scipy.stats.skew(effects_np)),
            'max_effect': float(torch.max(vocab_effects)),
            'min_effect': float(torch.min(vocab_effects)),
            'num_significant': int(torch.sum(torch.abs(vocab_effects) > 1.0)),
            'percentiles': {
                'p1': float(percentiles[0]),
                'p5': float(percentiles[1]),
                'p25': float(percentiles[2]),
                'p50': float(percentiles[3]),
                'p75': float(percentiles[4]),
                'p95': float(percentiles[5]),
                'p99': float(percentiles[6])
            }
        }
    
    def _classify_super_weight_function(self, vocab_effects: torch.Tensor) -> Dict[str, Any]:
        """
        Classify neuron function as prediction/suppression/partition based on effect distribution.
        
        Uses kurtosis/skew for concentrated effects and variance + both-sign mass for partitions.
        """
        effects_np = vocab_effects.numpy()
        kurtosis = scipy.stats.kurtosis(effects_np)
        skew = scipy.stats.skew(effects_np)
        variance = float(torch.var(vocab_effects))
        
        # Check for partition pattern: high variance with both positive and negative effects
        pos_frac = (vocab_effects > 0).float().mean().item()
        neg_frac = (vocab_effects < 0).float().mean().item()
        
        if kurtosis > 10:  # Very concentrated effects
            if skew > 0:
                function_type = "prediction"
                description = "Boosts probability of specific token sets"
            else:
                function_type = "suppression" 
                description = "Reduces probability of specific token sets"
        elif variance > 1.0 and pos_frac >= 0.1 and neg_frac >= 0.1:  # Partition check with both-sign mass
            function_type = "partition"
            description = "Affects broad token classes (boost some, suppress others)"
        else:
            function_type = "unclear"
            description = "Effects too small or distributed to classify clearly"
        
        confidence = min(1.0, max(0.0, (abs(kurtosis) + variance) / 20.0))
        
        return {
            'type': function_type,
            'description': description,
            'confidence': confidence,
            'kurtosis': kurtosis,
            'skew': skew,
            'variance': variance,
            'pos_fraction': pos_frac,
            'neg_fraction': neg_frac
        }
    
    def _get_top_affected_tokens(self, vocab_effects: torch.Tensor, top_k: int = 20) -> Dict[str, List[Dict]]:
        """
        Get top boosted and suppressed tokens with improved token string handling.
        
        Uses convert_ids_to_tokens for raw tokens and convert_tokens_to_string for display,
        filters out special tokens.
        """
        effects_cpu = vocab_effects.cpu()
        special_token_ids = set(getattr(self.tokenizer, 'all_special_ids', []))
        
        # Get top boosted (most positive effects)
        top_boosted_indices = torch.topk(effects_cpu, min(top_k * 3, len(effects_cpu))).indices  # Get extra to filter
        top_boosted = []
        for idx in top_boosted_indices:
            token_id = idx.item()
            if token_id in special_token_ids:
                continue
                
            try:
                # Get raw token representation
                raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                # Get pretty string representation
                token_str = self.tokenizer.convert_tokens_to_string([raw_token])
                
                top_boosted.append({
                    'token_id': token_id,
                    'token_str': token_str,
                    'raw_token': raw_token,
                    'effect_magnitude': effects_cpu[idx].item()
                })
                
                if len(top_boosted) >= top_k:
                    break
            except:
                pass
        
        # Get top suppressed (most negative effects)
        top_suppressed_indices = torch.topk(-effects_cpu, min(top_k * 3, len(effects_cpu))).indices
        top_suppressed = []
        for idx in top_suppressed_indices:
            token_id = idx.item()
            if token_id in special_token_ids:
                continue
                
            try:
                # Get raw token representation
                raw_token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                # Get pretty string representation
                token_str = self.tokenizer.convert_tokens_to_string([raw_token])
                
                top_suppressed.append({
                    'token_id': token_id,
                    'token_str': token_str,
                    'raw_token': raw_token,
                    'effect_magnitude': effects_cpu[idx].item()
                })
                
                if len(top_suppressed) >= top_k:
                    break
            except:
                pass
        
        return {
            'top_boosted': top_boosted,
            'top_suppressed': top_suppressed
        }
    
    def _get_unembedding_matrix(self, apply_universal_neurons_processing: bool = True) -> torch.Tensor:
        """
        Get unembedding matrix with optional preprocessing for cleaner analysis.
        
        Applies paper-faithful preprocessing: mean-center rows to remove softmax translation bias.
        Caches the processed matrix for efficiency.
        """
        # Return cached version if available and processing matches
        cache_key = apply_universal_neurons_processing
        if hasattr(self, '_cached_unembedding') and hasattr(self, '_cache_processed') and self._cache_processed == cache_key:
            return self._cached_unembedding
            
        # Find unembedding matrix
        if hasattr(self.model, 'lm_head'):
            unembedding_weight = self.model.lm_head.weight.detach()
        elif hasattr(self.model, 'output'):
            unembedding_weight = self.model.output.weight.detach()
        else:
            raise ValueError("Could not find unembedding matrix")
        
        if not apply_universal_neurons_processing:
            self._cached_unembedding = unembedding_weight
            self._cache_processed = cache_key
            return unembedding_weight
        
        # Apply paper-faithful preprocessing: mean-center rows to remove softmax translation bias
        processed_matrix = unembedding_weight.clone()
        vocab_mean = processed_matrix.mean(dim=0, keepdim=True)
        processed_matrix = processed_matrix - vocab_mean
        
        # Cache the result
        self._cached_unembedding = processed_matrix
        self._cache_processed = cache_key
        
        return processed_matrix
    
    def _capture_residual_layers(self, tokens: torch.Tensor, num_layers: int = 5) -> tuple[List[torch.Tensor], List[int]]:
        """
        Capture residual stream activations at evenly spaced layers.
        
        Returns:
            Tuple of (residuals, layer_indices) so caller knows which layers were actually captured
        """
        total_layers = len(list(self.model.model.layers))
        layer_indices = [int(i * total_layers / num_layers) for i in range(num_layers)]
        
        residuals = []
        
        def capture_residual(layer_idx):
            def hook_fn(module, input_tensor, output_tensor):
                if len(residuals) == layer_idx:  # Only capture once per layer
                    if isinstance(output_tensor, tuple):
                        residual = output_tensor[0][0, -1, :].detach().cpu()  # Last token
                    else:
                        residual = output_tensor[0, -1, :].detach().cpu()
                    residuals.append(residual)
            return hook_fn
        
        # Register hooks
        hooks = []
        for i, layer_idx in enumerate(layer_indices):
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(capture_residual(i))
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = self.model(**tokens)
        finally:
            for hook in hooks:
                hook.remove()
        
        return residuals, layer_indices
    
    def _classify_cascade_pattern(self, layer_effects: Dict) -> str:
        """Classify the amplification pattern of effects across layers"""
        entropies = [effects['entropy_change'] for effects in layer_effects.values()]
        
        if all(e > 0 for e in entropies):
            return "amplifying"
        elif all(e < 0 for e in entropies):
            return "dampening"
        elif abs(entropies[-1]) > abs(entropies[0]) * 2:
            return "accelerating"
        else:
            return "stable"
    
    def _get_random_coordinate_same_magnitude(self, super_weight: SuperWeight) -> SuperWeight:
        """Generate a random coordinate with similar magnitude for baseline comparison"""
        components = self.mlp_handler.get_mlp_components(super_weight.layer)
        down_proj = components['down']
        
        # Find coordinate with similar magnitude using original_value
        if super_weight.original_value is not None:
            target_magnitude = abs(float(super_weight.original_value))
        else:
            # Fallback: use magnitude_product 
            target_magnitude = super_weight.magnitude_product
        weight_matrix = down_proj.weight.abs()
        
        # Find all weights within 10% of target magnitude
        close_weights = torch.where(
            torch.abs(weight_matrix - target_magnitude) < target_magnitude * 0.1
        )
        
        if len(close_weights[0]) == 0:
            # Fallback: any random coordinate
            random_row = torch.randint(0, weight_matrix.shape[0], (1,)).item()
            random_col = torch.randint(0, weight_matrix.shape[1], (1,)).item()
        else:
            # Pick random from magnitude-matched coordinates
            idx = torch.randint(0, len(close_weights[0]), (1,)).item()
            random_row = close_weights[0][idx].item()
            random_col = close_weights[1][idx].item()
        
        return SuperWeight(
            layer=super_weight.layer,
            row=random_row,
            column=random_col,
            original_value=down_proj.weight[random_row, random_col].item(),
            component="down_proj",
            iteration_found=-1,  # We don't care about the iteration or input/output values
            input_value=-1,
            output_value=-1
        )
    
    def _compare_neuron_vs_scalar_ablation(self, super_weight: SuperWeight, 
                                         test_texts: List[str]) -> Dict[str, Any]:
        """Compare full neuron ablation vs single scalar ablation"""
        # Single scalar ablation
        scalar_effect = self.analyze_super_weight_intervention(super_weight, test_texts)
        
        # Full neuron ablation (zero entire column)
        neuron_effect = self._analyze_full_neuron_ablation(super_weight, test_texts)
        
        return {
            'scalar_delta_loss': scalar_effect['delta_loss'],
            'neuron_delta_loss': neuron_effect['delta_loss'],
            'scalar_contribution_ratio': abs(scalar_effect['delta_loss']) / max(abs(neuron_effect['delta_loss']), 1e-6)
        }
    
    def _analyze_full_neuron_ablation(self, super_weight: SuperWeight, test_texts: List[str]) -> Dict[str, Any]:
        """Ablate the entire neuron (full column) and measure effects using windowed evaluation"""
        # Create temporary weights for full neuron ablation
        components = self.mlp_handler.get_mlp_components(super_weight.layer)
        down_proj = components['down']
        
        # Store original column
        original_column = down_proj.weight[:, super_weight.column].clone()
        
        try:
            # Zero entire column
            down_proj.weight[:, super_weight.column] = 0
            
            # Measure effects using windowed evaluation
            metrics = self._eval_windows(test_texts)
            
            return {
                'delta_loss': metrics['loss'], 
                'delta_entropy': metrics['entropy'],
                'delta_topk_margin': metrics['topk_margin'],
                'delta_stopword_mass': metrics['stopword_mass']
            }
            
        finally:
            # Restore original weights
            down_proj.weight[:, super_weight.column] = original_column
    
    def _verify_deterministic_noop(self, super_weight: SuperWeight, test_texts: List[str]) -> Dict[str, Any]:
        """Verify deterministic model behavior by running identical computations twice using windowed evaluation"""
        # Run twice with identical conditions
        metrics1 = self._eval_windows(test_texts)
        metrics2 = self._eval_windows(test_texts)
        
        return {
            'loss_difference': abs(metrics1['loss'] - metrics2['loss']),
            'entropy_difference': abs(metrics1['entropy'] - metrics2['entropy']),
            'topk_margin_difference': abs(metrics1['topk_margin'] - metrics2['topk_margin']),
            'is_deterministic': abs(metrics1['loss'] - metrics2['loss']) < 1e-6
        }

    def display_vocabulary_card(self, analysis_results: Dict[str, Any], top_k_display: int = 5) -> None:
        """
        Display comprehensive vocabulary analysis results in a readable format
        
        Args:
            analysis_results: Results from analyze_neuron_vocabulary_effects
            top_k_display: Number of top boosted/suppressed tokens to display
        """
        sw = analysis_results['super_weight']
        
        print(f"=== Vocabulary Effect Card: {sw} ===")
        print(f"Processing applied: {analysis_results.get('processing_applied', 'Unknown')}")
        
        # Display raw analysis
        if 'raw_analysis' in analysis_results:
            raw = analysis_results['raw_analysis']
            stats = raw['statistics']
            classification = raw['classification']
            enrichment = raw['enrichment']
            top_tokens = raw['top_tokens']
            
            print(f"\n--- Raw Dot Product Analysis ---")
            print(f"Classification: {classification['type'].upper()} ({classification['confidence']:.2f})")
            print(f"  {classification['description']}")
            print(f"Moments: var={stats['variance']:.3f}, skew={stats['skew']:.2f}, kurt={stats['kurtosis']:.2f}")
            print(f"Percentiles: p5={stats['percentiles']['p5']:.3f}, p50={stats['percentiles']['p50']:.3f}, p95={stats['percentiles']['p95']:.3f}")
            print(f"Best theme: {enrichment['best_theme']['class']} (score: {enrichment['best_theme']['score']:.3f})")
            
            print(f"Top {top_k_display} boosted:")
            for i, token in enumerate(top_tokens['top_boosted'][:top_k_display]):
                print(f"  {i+1}. {repr(token['token_str'])} ({token['effect_magnitude']:+.3f})")
            
            print(f"Top {top_k_display} suppressed:")
            for i, token in enumerate(top_tokens['top_suppressed'][:top_k_display]):
                print(f"  {i+1}. {repr(token['token_str'])} ({token['effect_magnitude']:+.3f})")
        
        # Display cosine analysis
        if 'cosine_analysis' in analysis_results:
            cosine = analysis_results['cosine_analysis']
            cos_stats = cosine['statistics']
            cos_classification = cosine['classification']
            
            print(f"\n--- Cosine Similarity Analysis ---")
            print(f"Classification: {cos_classification['type'].upper()} ({cos_classification['confidence']:.2f})")
            print(f"Moments: var={cos_stats['variance']:.3f}, skew={cos_stats['skew']:.2f}, kurt={cos_stats['kurtosis']:.2f}")
            print(f"Percentiles: p5={cos_stats['percentiles']['p5']:.3f}, p50={cos_stats['percentiles']['p50']:.3f}, p95={cos_stats['percentiles']['p95']:.3f}")
        
        # Fallback to legacy format if new structure not available
        if 'statistics' in analysis_results:
            stats = analysis_results['statistics']
            classification = analysis_results['classification']
            enrichment = analysis_results['enrichment']
            top_tokens = analysis_results['top_tokens']
            
            print(f"Classification: {classification['type'].upper()} ({classification['confidence']:.2f})")
            print(f"  {classification['description']}")
            print(f"Moments: var={stats['variance']:.2f}, skew={stats['skew']:.2f}, kurt={stats['kurtosis']:.2f}")
            print(f"Best theme: {enrichment['best_theme']['class']} (score: {enrichment['best_theme']['score']:.3f})")
            
            print(f"Top {top_k_display} boosted:")
            for i, token in enumerate(top_tokens['top_boosted'][:top_k_display]):
                print(f"  {i+1}. {repr(token['token_str'])} ({token['effect_magnitude']:+.3f})")
            
            print(f"Top {top_k_display} suppressed:")
            for i, token in enumerate(top_tokens['top_suppressed'][:top_k_display]):
                print(f"  {i+1}. {repr(token['token_str'])} ({token['effect_magnitude']:+.3f})")

    def display_intervention_results(self, intervention_results: Dict[str, Any]) -> None:
        """Display intervention analysis results in a readable format"""
        sw = intervention_results['super_weight']
        
        print(f"=== Intervention Effect Card: {sw} ===")
        print(f"Loss: {intervention_results['baseline_loss']:.4f} → {intervention_results['modified_loss']:.4f} (Δ{intervention_results['delta_loss']:+.4f})")
        print(f"Entropy: {intervention_results['baseline_entropy']:.4f} → {intervention_results['modified_entropy']:.4f} (Δ{intervention_results['delta_entropy']:+.4f})")
        print(f"Top-K Margin: {intervention_results['baseline_topk_margin']:.4f} → {intervention_results['modified_topk_margin']:.4f} (Δ{intervention_results['delta_topk_margin']:+.4f})")
        
        if intervention_results['delta_stopword_mass'] is not None:
            print(f"Stopword Mass: {intervention_results['baseline_stopword_mass']:.4f} → {intervention_results['modified_stopword_mass']:.4f} (Δ{intervention_results['delta_stopword_mass']:+.4f})")
        else:
            print("Stopword Mass: Not available")
        
        print(f"Evaluated on {intervention_results['n_samples']} samples")

    def display_cascade_results(self, cascade_results: Dict[str, Any]) -> None:
        """Display cascade analysis results in a readable format with actual layer indices"""
        if 'error' in cascade_results:
            print(f"=== Cascade Analysis Error ===")
            print(f"Error: {cascade_results['error']}")
            return
            
        sw = cascade_results['super_weight']
        
        print(f"=== Cascade Effect Card: {sw} ===")
        print(f"Input: {cascade_results['input_text']}")
        print(f"Pattern: {cascade_results['amplification_pattern']}")
        print(f"Layers analyzed: {cascade_results['num_layers_analyzed']}")
        print(f"Actual layer indices: {cascade_results['actual_layer_indices']}")
        print(f"Top-K margin parameter: {cascade_results['top_k_margin']}")
        
        print("\nLayer-by-layer effects:")
        for layer_idx, effects in cascade_results['layer_effects'].items():
            print(f"  Layer {layer_idx}: entropy_Δ={effects['entropy_change']:+.4f}, "
                  f"margin_Δ={effects['margin_change']:+.4f}, "
                  f"magnitude={effects['effect_magnitude']:.4f}")
        
        # Show progression pattern
        entropies = [effects['entropy_change'] for effects in cascade_results['layer_effects'].values()]
        if len(entropies) > 1:
            print(f"\nEntropy change progression: {entropies[0]:+.4f} → {entropies[-1]:+.4f} "
                  f"(amplification: {abs(entropies[-1]/entropies[0]):.2f}x)" if entropies[0] != 0 else "")
