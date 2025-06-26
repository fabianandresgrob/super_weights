import torch
import numpy as np
import scipy.stats
from typing import List, Dict, Any, Optional
from datasets import load_dataset

from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalMLPHandler
from analysis.results_manager import VocabularyResultsManager


class VocabularyAnalyzer:
    """
    Analyzes vocabulary effects of super weights using causal intervention.
    Adapted from Universal Neurons methodology.
    """

    def __init__(self, model, tokenizer, manager, results_dir: str = "results"):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        self.mlp_handler = UniversalMLPHandler(model)
        self.results_manager = VocabularyResultsManager(results_dir)
    
    def analyze_vocabulary_effects(self, super_weight: SuperWeight, 
                                test_texts: Optional[List[str]] = None,
                                dataset_name: Optional[str] = None,
                                dataset_config: Optional[str] = None,
                                n_samples: int = 500,
                                max_length: int = 512) -> Dict[str, Any]:
        """
        Analyze vocabulary effects using either custom texts or dataset samples.
        
        Args:
            super_weight: SuperWeight to analyze
            test_texts: Optional list of custom test texts (takes priority if provided)
            dataset_name: Dataset name (e.g., 'wikitext')
            dataset_config: Dataset config (e.g., 'wikitext-2-raw-v1') 
            n_samples: Number of dataset samples to use
            max_length: Maximum sequence length for dataset samples
            
        Returns:
            Dictionary with vocabulary analysis results
        """
        
        # Determine which text source to use
        if test_texts is not None:
            texts = test_texts
            text_source = f"custom_texts (n={len(test_texts)})"
        else:
            if dataset_name is None:
                dataset_name = 'wikitext'
                dataset_config = 'wikitext-2-raw-v1'
            
            # Load dataset texts (reuse logic from metrics.py)
            texts = self._load_dataset_texts(dataset_name, dataset_config, n_samples, max_length)
            text_source = f"{dataset_name}/{dataset_config} (n={len(texts)})"
        
        # Get baseline vocabulary distribution
        baseline_logits = self._compute_average_logits(texts)
        
        # Get modified distribution with super weight zeroed
        with self.manager.temporary_zero([super_weight]):
            modified_logits = self._compute_average_logits(texts)
        
        # Calculate the effect (what the super weight contributes)
        vocab_effects = baseline_logits - modified_logits
        
        # Analyze the effects
        statistics = self._compute_effect_statistics(vocab_effects)
        classification = self._classify_super_weight_function(vocab_effects)
        top_tokens = self._get_top_affected_tokens(vocab_effects)
        patterns = self._analyze_token_patterns(vocab_effects)
        
        return {
            'super_weight': super_weight,
            'vocab_effects': vocab_effects.numpy(),
            'statistics': statistics,
            'classification': classification,
            'top_tokens': top_tokens,
            'patterns': patterns,
            'baseline_logits': baseline_logits.numpy(),
            'modified_logits': modified_logits.numpy(),
            'text_source': text_source
        }
    
    def analyze_neuron_vocabulary_effects(self, super_weight: SuperWeight) -> Dict[str, Any]:
        """
        Analyze vocabulary effects of the full neuron containing the super weight.
        Uses direct computation (W_U @ w_out) like Universal Neurons paper.
        
        Args:
            super_weight: SuperWeight to analyze (identifies the neuron)
            
        Returns:
            Dictionary with neuron vocabulary analysis results
        """
        
        try:
            with torch.no_grad():
                # Get the full neuron output vector
                layer = self._get_layer(super_weight.layer)
                _, _, module = self._get_mlp_component_info(super_weight.layer)
                neuron_output_vector = module.weight[:, super_weight.column].clone().detach()
                
                # Get unembedding matrix (same as lm_head weight)
                if hasattr(self.model, 'lm_head'):
                    unembedding_matrix = self.model.lm_head.weight
                elif hasattr(self.model, 'embed_out'):
                    unembedding_matrix = self.model.embed_out.weight
                else:
                    # Try to find unembedding matrix
                    unembedding_matrix = self.model.model.embed_tokens.weight
                
                # Compute vocabulary effects: W_U @ w_out
                vocab_effects = torch.matmul(neuron_output_vector, unembedding_matrix.T).cpu()
                
                # Analyze the effects using same methods as intervention analysis
                statistics = self._compute_effect_statistics(vocab_effects)
                classification = self._classify_super_weight_function(vocab_effects)
                top_tokens = self._get_top_affected_tokens(vocab_effects)
                patterns = self._analyze_token_patterns(vocab_effects)
                
                return {
                    'super_weight': super_weight,
                    'analysis_type': 'neuron_direct',
                    'neuron_coordinates': (super_weight.layer, super_weight.row),
                    'vocab_effects': vocab_effects.numpy(),
                    'statistics': statistics,
                    'classification': classification,
                    'top_tokens': top_tokens,
                    'patterns': patterns,
                    'neuron_output_norm': float(torch.norm(neuron_output_vector))
                }
            
        except Exception as e:
            return {
                'super_weight': super_weight,
                'analysis_type': 'neuron_direct',
                'error': f"Failed to analyze neuron: {str(e)}"
            }
        
    def _compute_average_logits(self, test_texts: List[str]) -> torch.Tensor:
        """Compute average logits across test texts for the last token"""
        all_logits = []
        
        for text in test_texts:
            tokens = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**tokens)
                # Take logits for the last token
                last_token_logits = outputs.logits[0, -1, :].cpu()
                all_logits.append(last_token_logits)
        
        # Average across all test texts
        return torch.stack(all_logits).mean(dim=0)
    
    def _load_dataset_texts(self, dataset_name: str, dataset_config: str, 
                       n_samples: int, max_length: int) -> List[str]:
        """Load and preprocess texts from dataset (adapted from metrics.py)"""
        
        dataset = load_dataset(dataset_name, dataset_config, split='test', streaming=True)
        # shuffle the dataset to get diverse samples
        dataset = dataset.shuffle(seed=42)
        # Extract texts
        texts = []
        for i, item in enumerate(dataset):
            if i >= n_samples:
                break
            text = item['text'].strip() if 'text' in item else item['sentence'].strip()
            if not text:
                continue
            # Truncate to max_length
            if len(text) > max_length:
                text = text[:max_length]
            texts.append(text)

        if not texts:
            raise ValueError(f"No valid texts found in dataset {dataset_name}/{dataset_config}. "
                             f"Ensure the dataset has a 'text' or 'sentence' field.")
        return texts
    
    def _compute_effect_statistics(self, vocab_effects: torch.Tensor) -> Dict[str, float]:
        """Compute statistical properties of vocabulary effects"""
        
        # Ensure tensor is detached and on CPU
        vocab_effects = vocab_effects.detach().cpu()
        effects_np = vocab_effects.numpy()
        
        return {
            'mean': float(torch.mean(vocab_effects)),
            'std': float(torch.std(vocab_effects)),
            'variance': float(torch.var(vocab_effects)),
            'kurtosis': float(scipy.stats.kurtosis(effects_np)),
            'skew': float(scipy.stats.skew(effects_np)),
            'max_effect': float(torch.max(vocab_effects)),
            'min_effect': float(torch.min(vocab_effects)),
            'num_positive': int(torch.sum(vocab_effects > 0)),
            'num_negative': int(torch.sum(vocab_effects < 0)),
            'num_significant': int(torch.sum(torch.abs(vocab_effects) > 1.0))
        }
    
    def _classify_super_weight_function(self, vocab_effects: torch.Tensor) -> Dict[str, Any]:
        """
        Classify super weight function based on vocabulary effects.
        Adapted from Universal Neurons paper classification system.
        """
        effects_np = vocab_effects.numpy()
        kurtosis = scipy.stats.kurtosis(effects_np)
        skew = scipy.stats.skew(effects_np)
        variance = float(torch.var(vocab_effects))
        
        # Classification logic from Universal Neurons paper
        if kurtosis > 10:  # Very concentrated effects
            if skew > 0:
                function_type = "prediction"
                description = "Boosts probability of specific token sets"
            else:
                function_type = "suppression"
                description = "Reduces probability of specific token sets"
        elif variance > 1.0:  # Broadly distributed effects
            function_type = "partition"
            description = "Affects broad token classes (boost some, suppress others)"
        else:
            function_type = "unclear"
            description = "Effects too small or distributed to classify clearly"
        
        return {
            'type': function_type,
            'description': description,
            'confidence': self._compute_classification_confidence(kurtosis, skew, variance),
            'metrics': {
                'kurtosis': kurtosis,
                'skew': skew,
                'variance': variance
            }
        }
    
    def _compute_classification_confidence(self, kurtosis: float, skew: float, variance: float) -> float:
        """Compute confidence score for classification (0-1)"""
        if kurtosis > 20:
            return 0.9
        elif kurtosis > 10:
            return 0.7
        elif variance > 2.0:
            return 0.6
        else:
            return 0.3
    
    def _get_top_affected_tokens(self, vocab_effects: torch.Tensor, top_k: int = 20) -> Dict[str, List[Dict]]:
        """Get the most positively and negatively affected tokens"""
        
        # Top boosted tokens (positive effects)
        top_boosted_indices = torch.argsort(vocab_effects, descending=True)[:top_k]
        top_boosted = []
        for idx in top_boosted_indices:
            token_id = idx.item()
            try:
                token_str = self.tokenizer.decode([token_id])
                top_boosted.append({
                    'token_id': token_id,
                    'token_str': token_str,
                    'effect_magnitude': vocab_effects[idx].item()
                })
            except:
                # Skip tokens that can't be decoded
                continue
        
        # Top suppressed tokens (negative effects)
        top_suppressed_indices = torch.argsort(vocab_effects, descending=False)[:top_k]
        top_suppressed = []
        for idx in top_suppressed_indices:
            token_id = idx.item()
            try:
                token_str = self.tokenizer.decode([token_id])
                top_suppressed.append({
                    'token_id': token_id,
                    'token_str': token_str,
                    'effect_magnitude': vocab_effects[idx].item()
                })
            except:
                continue
        
        return {
            'top_boosted': top_boosted,
            'top_suppressed': top_suppressed
        }
    
    def _analyze_token_patterns(self, vocab_effects: torch.Tensor, threshold: float = 1.0) -> Dict[str, List[Dict]]:
        """
        Analyze linguistic patterns in affected tokens.
        Based on Universal Neurons paper pattern detection.
        """
        significant_effects = torch.abs(vocab_effects) > threshold
        affected_token_ids = torch.where(significant_effects)[0]
        
        patterns = {
            'years': [],
            'punctuation': [],
            'function_words': [],
            'numbers': [],
            'capitalized': [],
            'space_prefixed': [],
            'sophisticated_words': [],
            'fragments': []
        }
        
        # Common function words
        function_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Sophisticated words
        sophisticated_words = {'improving', 'shaping', 'leading', 'changing', 'developing', 'creating'}
        
        for token_id in affected_token_ids:
            try:
                token_str = self.tokenizer.decode([token_id.item()])
                effect = vocab_effects[token_id.item()].item()
                
                token_info = {
                    'token_id': token_id.item(),
                    'token_str': token_str,
                    'effect': effect
                }
                
                # Pattern classification
                token_stripped = token_str.strip()
                
                # Years (1700-2050 range from Universal Neurons paper)
                if token_stripped.isdigit() and 1700 <= int(token_stripped) <= 2050:
                    patterns['years'].append(token_info)
                
                # Punctuation
                elif any(char in token_str for char in '.,!?;:()[]{}'):
                    patterns['punctuation'].append(token_info)
                
                # Function words
                elif token_stripped.lower() in function_words:
                    patterns['function_words'].append(token_info)
                
                # Numbers (general)
                elif token_stripped.isdigit():
                    patterns['numbers'].append(token_info)
                
                # Capitalized words
                elif token_stripped and token_stripped[0].isupper():
                    patterns['capitalized'].append(token_info)
                
                # Space-prefixed tokens
                elif token_str.startswith(' '):
                    patterns['space_prefixed'].append(token_info)
                
                # Sophisticated words
                elif any(word in token_str.lower() for word in sophisticated_words):
                    patterns['sophisticated_words'].append(token_info)
                
                # Fragments (short tokens)
                elif len(token_stripped) <= 3 and token_stripped.isalpha():
                    patterns['fragments'].append(token_info)
                
            except:
                # Skip tokens that can't be decoded or processed
                continue
        
        # Sort each pattern by effect magnitude
        for pattern_name in patterns:
            patterns[pattern_name].sort(key=lambda x: abs(x['effect']), reverse=True)
        
        return patterns
    
    
    def display_analysis_results(self, analysis_results: Dict[str, Any], top_k: int = 10) -> None:
        """Display vocabulary analysis results in a readable format"""
        
        sw = analysis_results['super_weight']
        stats = analysis_results['statistics']
        classification = analysis_results['classification']
        top_tokens = analysis_results['top_tokens']
        patterns = analysis_results['patterns']
        
        print(f"\n=== Vocabulary Analysis: {sw} ===")
        
        # Basic statistics
        print(f"\nEffect Statistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Kurtosis: {stats['kurtosis']:.4f}")
        print(f"  Skew: {stats['skew']:.4f}")
        print(f"  Significant effects: {stats['num_significant']}")
        
        # Classification
        print(f"\nClassification:")
        print(f"  Type: {classification['type'].upper()}")
        print(f"  Description: {classification['description']}")
        print(f"  Confidence: {classification['confidence']:.2f}")
        
        # Top affected tokens
        print(f"\nTop {top_k} Boosted Tokens:")
        for i, token_info in enumerate(top_tokens['top_boosted'][:top_k]):
            token_repr = repr(token_info['token_str'])  # Shows spaces, newlines clearly
            print(f"  {i+1:2d}. {token_repr:20s} (effect: {token_info['effect_magnitude']:+.3f})")
        
        print(f"\nTop {top_k} Suppressed Tokens:")
        for i, token_info in enumerate(top_tokens['top_suppressed'][:top_k]):
            token_repr = repr(token_info['token_str'])
            print(f"  {i+1:2d}. {token_repr:20s} (effect: {token_info['effect_magnitude']:+.3f})")
        
        # Pattern analysis
        print(f"\nDetected Patterns:")
        for pattern_name, tokens in patterns.items():
            if tokens:
                print(f"  {pattern_name.upper()}: {len(tokens)} tokens")
                # Show a few examples
                examples = [repr(t['token_str']) for t in tokens[:3]]
                print(f"    Examples: {', '.join(examples)}")
    
    def compare_super_weights(self, super_weights: List[SuperWeight], 
                            test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare vocabulary effects across multiple super weights"""
        
        results = {}
        effect_similarities = {}
        
        # Analyze each super weight
        for sw in super_weights:
            results[str(sw)] = self.analyze_vocabulary_effects(sw, test_texts)
        
        # Compute pairwise similarities
        sw_keys = list(results.keys())
        for i, sw1_key in enumerate(sw_keys):
            for j, sw2_key in enumerate(sw_keys[i+1:], i+1):
                effects1 = torch.tensor(results[sw1_key]['vocab_effects'])
                effects2 = torch.tensor(results[sw2_key]['vocab_effects'])
                
                # Compute cosine similarity
                cosine_sim = torch.cosine_similarity(effects1, effects2, dim=0).item()
                
                # Compute correlation
                correlation = torch.corrcoef(torch.stack([effects1, effects2]))[0, 1].item()
                
                effect_similarities[f"{sw1_key} vs {sw2_key}"] = {
                    'cosine_similarity': cosine_sim,
                    'correlation': correlation
                }
        
        return {
            'individual_analyses': results,
            'similarities': effect_similarities,
            'summary': self._create_comparison_summary(results, effect_similarities)
        }
    
    def _create_comparison_summary(self, results: Dict, similarities: Dict) -> Dict[str, Any]:
        """Create a summary of the comparison results"""
        
        # Count function types
        function_types = {}
        for analysis in results.values():
            func_type = analysis['classification']['type']
            function_types[func_type] = function_types.get(func_type, 0) + 1
        
        # Find most similar pair
        most_similar = max(similarities.items(), key=lambda x: x[1]['cosine_similarity'])
        
        # Find most different pair
        most_different = min(similarities.items(), key=lambda x: x[1]['cosine_similarity'])
        
        return {
            'total_super_weights': len(results),
            'function_type_distribution': function_types,
            'most_similar_pair': {
                'pair': most_similar[0],
                'cosine_similarity': most_similar[1]['cosine_similarity']
            },
            'most_different_pair': {
                'pair': most_different[0],
                'cosine_similarity': most_different[1]['cosine_similarity']
            },
            'average_similarity': np.mean([sim['cosine_similarity'] for sim in similarities.values()])
        }
    

    def compare_neuron_vs_super_weight(self, super_weight: SuperWeight,
                                    test_texts: Optional[List[str]] = None,
                                    dataset_name: Optional[str] = None,
                                    dataset_config: Optional[str] = None,
                                    n_samples: int = 500) -> Dict[str, Any]:
        """
        Compare vocabulary effects between the full neuron and the super weight contribution.
        
        Args:
            super_weight: SuperWeight to analyze
            test_texts: Optional custom test texts for super weight analysis
            dataset_name: Dataset for super weight analysis (if test_texts is None)
            dataset_config: Dataset config
            n_samples: Number of samples for dataset analysis
            
        Returns:
            Dictionary with comparison results
        """
        
        # Run both analyses
        print(f"Analyzing full neuron containing {super_weight}...")
        neuron_analysis = self.analyze_neuron_vocabulary_effects(super_weight)
        
        print(f"Analyzing super weight contribution via intervention...")
        super_weight_analysis = self.analyze_vocabulary_effects(
            super_weight, test_texts, dataset_name, dataset_config, n_samples
        )
        
        # Check for errors
        if 'error' in neuron_analysis:
            return {
                'comparison_type': 'neuron_vs_super_weight',
                'super_weight': super_weight,
                'neuron_analysis': neuron_analysis,
                'super_weight_analysis': super_weight_analysis,
                'error': f"Neuron analysis failed: {neuron_analysis['error']}"
            }
        
        if 'error' in super_weight_analysis:
            return {
                'comparison_type': 'neuron_vs_super_weight',
                'super_weight': super_weight,
                'neuron_analysis': neuron_analysis,
                'super_weight_analysis': super_weight_analysis,
                'error': f"Super weight analysis failed: {super_weight_analysis['error']}"
            }
        
        # Compare the vocabulary effects
        neuron_effects = torch.tensor(neuron_analysis['vocab_effects'])
        sw_effects = torch.tensor(super_weight_analysis['vocab_effects'])
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([neuron_effects, sw_effects]))[0, 1].item()
        
        # Compute cosine similarity
        cosine_sim = torch.cosine_similarity(neuron_effects, sw_effects, dim=0).item()
        
        # Compute effect magnitude ratios
        neuron_max_effect = float(torch.max(torch.abs(neuron_effects)))
        sw_max_effect = float(torch.max(torch.abs(sw_effects)))
        magnitude_ratio = sw_max_effect / neuron_max_effect if neuron_max_effect > 0 else 0.0
        
        # Compare statistics
        neuron_stats = neuron_analysis['statistics']
        sw_stats = super_weight_analysis['statistics']
        
        stats_comparison = {
            'kurtosis_ratio': sw_stats['kurtosis'] / neuron_stats['kurtosis'] if neuron_stats['kurtosis'] != 0 else 0.0,
            'skew_difference': sw_stats['skew'] - neuron_stats['skew'],
            'std_ratio': sw_stats['std'] / neuron_stats['std'] if neuron_stats['std'] != 0 else 0.0,
            'mean_difference': sw_stats['mean'] - neuron_stats['mean']
        }
        
        # Compare classifications
        neuron_class = neuron_analysis['classification']['type']
        sw_class = super_weight_analysis['classification']['type']
        classification_match = neuron_class == sw_class
        
        # Find top overlapping tokens
        neuron_top_tokens = set(token['token_str'] for token in neuron_analysis['top_tokens']['top_boosted'][:20])
        neuron_top_tokens.update(token['token_str'] for token in neuron_analysis['top_tokens']['top_suppressed'][:20])
        
        sw_top_tokens = set(token['token_str'] for token in super_weight_analysis['top_tokens']['top_boosted'][:20])
        sw_top_tokens.update(token['token_str'] for token in super_weight_analysis['top_tokens']['top_suppressed'][:20])
        
        overlapping_tokens = neuron_top_tokens.intersection(sw_top_tokens)
        token_overlap_ratio = len(overlapping_tokens) / len(neuron_top_tokens.union(sw_top_tokens)) if neuron_top_tokens.union(sw_top_tokens) else 0.0
        
        # Determine relationship strength
        if correlation > 0.8 and classification_match:
            relationship = "strong_alignment"
            description = "Super weight strongly represents the neuron's function"
        elif correlation > 0.5:
            relationship = "moderate_alignment" 
            description = "Super weight partially represents the neuron's function"
        elif correlation > 0.0:
            relationship = "weak_alignment"
            description = "Super weight is a minor component of the neuron's function"
        else:
            relationship = "different_function"
            description = "Super weight and neuron have different or opposing functions"
        
        return {
            'comparison_type': 'neuron_vs_super_weight',
            'super_weight': super_weight,
            'neuron_analysis': neuron_analysis,
            'super_weight_analysis': super_weight_analysis,
            'comparison_metrics': {
                'correlation': correlation,
                'cosine_similarity': cosine_sim,
                'magnitude_ratio': magnitude_ratio,
                'stats_comparison': stats_comparison,
                'classification_match': classification_match,
                'token_overlap_ratio': token_overlap_ratio,
                'overlapping_tokens': list(overlapping_tokens)
            },
            'relationship': {
                'type': relationship,
                'description': description,
                'confidence': abs(correlation)  # Use correlation as confidence measure
            },
            'summary': {
                'neuron_classification': neuron_class,
                'super_weight_classification': sw_class,
                'alignment_strength': correlation,
                'super_weight_contribution': f"{magnitude_ratio:.1%}"
            }
        }

    def _get_layer(self, layer_idx: int):
        """Get layer by index using universal handler"""
        layers = self.mlp_handler.registry.find_layers(self.model)
        return layers[layer_idx]

    def _get_mlp_component_info(self, layer_idx: int):
        """Get MLP component info using universal handler"""
        # Get architecture info and components
        arch_info = self.mlp_handler.get_mlp_architecture(layer_idx)
        components = self.mlp_handler.get_mlp_components(layer_idx)
        
        # Find the down/output projection component
        if 'down' in components:
            # Gated architecture
            down_component = components['down']
            down_info = arch_info.components['down']
            return "mlp", down_info.component_name, down_component
        elif 'output' in components:
            # Standard architecture  
            output_component = components['output']
            output_info = arch_info.components['output']
            return "mlp", output_info.component_name, output_component
        else:
            raise ValueError(f"No down/output projection found in layer {layer_idx}")

    def display_neuron_vs_super_weight_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """
        Display comparison results in a readable format.
        
        Args:
            comparison_results: Results from compare_neuron_vs_super_weight()
        """
        
        if 'error' in comparison_results:
            print(f"❌ Comparison failed: {comparison_results['error']}")
            return
        
        sw = comparison_results['super_weight']
        metrics = comparison_results['comparison_metrics']
        relationship = comparison_results['relationship']
        summary = comparison_results['summary']
        
        print(f"\n{'='*60}")
        print(f"NEURON vs SUPER WEIGHT COMPARISON: {sw}")
        print(f"{'='*60}")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"  Neuron function: {summary['neuron_classification']}")
        print(f"  Super weight function: {summary['super_weight_classification']}")
        print(f"  Alignment strength: {summary['alignment_strength']:.3f}")
        print(f"  Super weight contribution: {summary['super_weight_contribution']}")
        
        # Relationship
        print(f"\n🔗 RELATIONSHIP:")
        print(f"  Type: {relationship['type']}")
        print(f"  Description: {relationship['description']}")
        print(f"  Confidence: {relationship['confidence']:.3f}")
        
        # Detailed metrics
        print(f"\n📈 DETAILED METRICS:")
        print(f"  Correlation: {metrics['correlation']:.3f}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.3f}")
        print(f"  Magnitude ratio: {metrics['magnitude_ratio']:.3f}")
        print(f"  Classification match: {metrics['classification_match']}")
        print(f"  Token overlap: {metrics['token_overlap_ratio']:.1%}")
        
        # Statistical differences
        stats_comp = metrics['stats_comparison']
        print(f"\n📊 STATISTICAL COMPARISON:")
        print(f"  Kurtosis ratio (SW/Neuron): {stats_comp['kurtosis_ratio']:.2f}")
        print(f"  Skew difference (SW - Neuron): {stats_comp['skew_difference']:.2f}")
        print(f"  Std ratio (SW/Neuron): {stats_comp['std_ratio']:.2f}")
        print(f"  Mean difference (SW - Neuron): {stats_comp['mean_difference']:.3f}")
        
        # Overlapping tokens
        if metrics['overlapping_tokens']:
            print(f"\n🎯 OVERLAPPING TOP TOKENS:")
            overlapping = metrics['overlapping_tokens'][:10]  # Show first 10
            print(f"  {', '.join(repr(token) for token in overlapping)}")
            if len(metrics['overlapping_tokens']) > 10:
                print(f"  ... and {len(metrics['overlapping_tokens']) - 10} more")
        else:
            print(f"\n🎯 NO OVERLAPPING TOP TOKENS")
        
        print(f"\n{'-'*60}")
    
    def analyze_vocabulary_cascade(self, super_weight: SuperWeight, 
                                 input_text: str = "Apple Inc. is a tech company.",
                                 method: str = "residual_stream") -> Dict[str, Any]:
        """
        Analyze vocabulary effects throughout the cascade.
        
        Args:
            super_weight: SuperWeight to analyze
            input_text: Text to analyze
            method: 'full_projection', 'residual_stream'
            
        Returns:
            Dictionary with cascade analysis results
        """
        
        print(f"Running cascade analysis using {method} method...")
        
        if method == "full_projection":
            return self.analyze_activation_cascade(super_weight, input_text)
        elif method == "residual_stream":
            return self.analyze_residual_stream_cascade(super_weight, input_text)
        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze_activation_cascade(self, super_weight: SuperWeight, 
                                 input_text: str = "Apple Inc. is a tech company.") -> Dict[str, Any]:
        """
        Analyze how super weight affects activations at each layer using full projection.
        Projects each intermediate activation through remaining layers to get final logits.
        """
        
        try:
            # Get baseline activations through all layers
            print("Capturing baseline activations...")
            baseline_activations = self._capture_all_layer_activations(input_text)
            
            # Get modified activations with super weight zeroed
            print("Capturing modified activations...")
            with self.manager.temporary_zero([super_weight]):
                modified_activations = self._capture_all_layer_activations(input_text)
            
            cascade_effects = {}
            num_layers = len(baseline_activations)
            
            print(f"Projecting activations through {num_layers} layers...")
            
            for layer_idx in range(num_layers):
                print(f"  Processing layer {layer_idx}/{num_layers-1}", end='\r')
                
                # Compute activation difference at this layer
                activation_diff = baseline_activations[layer_idx] - modified_activations[layer_idx]
                
                # Project through remaining layers to get final logits
                projected_baseline = self._project_to_final_logits(baseline_activations[layer_idx], layer_idx)
                projected_modified = self._project_to_final_logits(modified_activations[layer_idx], layer_idx)
                
                logit_effect = projected_baseline - projected_modified
                
                # Analyze the effects
                statistics = self._compute_effect_statistics(logit_effect)
                top_tokens = self._get_top_affected_tokens(logit_effect, top_k=10)
                
                cascade_effects[layer_idx] = {
                    'activation_magnitude': float(torch.norm(activation_diff)),
                    'vocab_effect': logit_effect.numpy(),
                    'statistics': statistics,
                    'top_tokens': top_tokens,
                    'effect_magnitude': float(torch.norm(logit_effect)),
                    'layers_remaining': num_layers - layer_idx - 1
                }
            
            print()  # New line after progress
            
            # Analyze propagation patterns
            propagation_analysis = self._analyze_effect_propagation(cascade_effects)
            convergence_analysis = self._analyze_vocabulary_convergence(cascade_effects)
            
            return {
                'analysis_type': 'full_projection',
                'super_weight': super_weight,
                'input_text': input_text,
                'cascade_effects': cascade_effects,
                'propagation_analysis': propagation_analysis,
                'convergence_analysis': convergence_analysis,
                'summary': self._create_cascade_summary(cascade_effects, 'full_projection')
            }
            
        except Exception as e:
            return {
                'analysis_type': 'full_projection',
                'super_weight': super_weight,
                'error': f"Full projection analysis failed: {str(e)}"
            }

    def analyze_residual_stream_cascade(self, super_weight: SuperWeight, 
                                      input_text: str = "Apple Inc. is a tech company.") -> Dict[str, Any]:
        """
        Analyze cascading effects through residual stream.
        Shows how super weight effects accumulate through residual connections.
        """
        
        try:
            print("Capturing residual stream...")
            
            # Capture residual stream at each layer
            baseline_residuals = self._capture_residual_stream(input_text)
            
            with self.manager.temporary_zero([super_weight]):
                modified_residuals = self._capture_residual_stream(input_text)
            
            residual_effects = {}
            cumulative_effect = torch.zeros(self.model.config.vocab_size)
            
            print(f"Analyzing residual differences across {len(baseline_residuals)} layers...")
            
            for layer_idx, (baseline_res, modified_res) in enumerate(zip(baseline_residuals, modified_residuals)):
                print(f"  Processing layer {layer_idx}/{len(baseline_residuals)-1}", end='\r')
                
                # Residual difference at this layer
                residual_diff = baseline_res - modified_res
                
                # Project this specific difference to vocabulary space
                # Get unembedding matrix
                if hasattr(self.model, 'lm_head'):
                    unembedding_matrix = self.model.lm_head.weight
                elif hasattr(self.model, 'embed_out'):
                    unembedding_matrix = self.model.embed_out.weight
                else:
                    unembedding_matrix = self.model.model.embed_tokens.weight
                
                # Ensure both tensors are on the same device
                residual_diff = residual_diff.to(unembedding_matrix.device)
                layer_vocab_effect = torch.matmul(residual_diff, unembedding_matrix.T).cpu()
                
                # Accumulate effects
                cumulative_effect += layer_vocab_effect
                
                # Analyze this layer's effects
                statistics = self._compute_effect_statistics(layer_vocab_effect)
                top_tokens = self._get_top_affected_tokens(layer_vocab_effect, top_k=10)
                
                residual_effects[layer_idx] = {
                    'residual_magnitude': float(torch.norm(residual_diff)),
                    'direct_vocab_effect': layer_vocab_effect.detach().numpy(),
                    'cumulative_vocab_effect': cumulative_effect.clone().detach().numpy(),
                    'statistics': statistics,
                    'top_tokens': top_tokens,
                    'effect_magnitude': float(torch.norm(layer_vocab_effect)),
                    'cumulative_magnitude': float(torch.norm(cumulative_effect)),
                    'amplification_factor': float(torch.norm(layer_vocab_effect) / torch.norm(residual_diff)) if torch.norm(residual_diff) > 0 else 0.0
                }
            
            print()  # New line after progress
            
            # Analyze accumulation patterns
            accumulation_analysis = self._analyze_effect_accumulation(residual_effects)
            amplification_analysis = self._analyze_amplification_patterns(residual_effects)
            
            return {
                'analysis_type': 'residual_stream',
                'super_weight': super_weight,
                'input_text': input_text,
                'residual_effects': residual_effects,
                'accumulation_analysis': accumulation_analysis,
                'amplification_analysis': amplification_analysis,
                'summary': self._create_cascade_summary(residual_effects, 'residual_stream')
            }
            
        except Exception as e:
            return {
                'analysis_type': 'residual_stream',
                'super_weight': super_weight,
                'error': f"Residual stream analysis failed: {str(e)}"
            }

    def _capture_all_layer_activations(self, input_text: str) -> List[torch.Tensor]:
        """Capture activations at each layer using universal architecture utilities"""
        
        activations = []
        
        # Tokenize input
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        # Hook function to capture activations
        def capture_activation(module, input_tensor, output_tensor):
            try:
                # Handle different output formats safely
                if output_tensor is None:
                    return
                    
                if isinstance(output_tensor, tuple):
                    activation = output_tensor[0]  # First element is usually the main output
                else:
                    activation = output_tensor
                
                if activation is None:
                    return
                
                # Take the last token's activation and ensure it's detached
                last_token_activation = activation[0, -1, :].detach().cpu()
                activations.append(last_token_activation)
                
            except Exception as e:
                print(f"Warning: Could not capture activation: {e}")
                # Add a zero tensor as placeholder
                if activations:
                    # Use same size as previous activation
                    activations.append(torch.zeros_like(activations[-1]))
                else:
                    # Fallback zero tensor
                    activations.append(torch.zeros(self.model.config.hidden_size))
        
        # Register hooks using universal architecture utilities
        hooks = []
        layers = self.mlp_handler.registry.find_layers(self.model)
        
        for layer in layers:
            hook = layer.register_forward_hook(capture_activation)
            hooks.append(hook)
        
        try:
            # Run forward pass
            with torch.no_grad():
                outputs = self.model(**tokens)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return activations

    def _capture_residual_stream(self, input_text: str) -> List[torch.Tensor]:
        """Capture residual stream at each layer using universal architecture utilities"""
        
        residuals = []
        
        # Tokenize input
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        # Use a simpler approach: capture layer outputs and treat them as residual approximations
        # This works universally across architectures without needing model-specific handling
        print("Capturing residual stream using universal layer capture...")
        
        try:
            # For now, use layer activations as residual stream approximation
            # This is a reasonable approximation since transformer layers typically
            # have residual connections where output ≈ input + layer_transform(input)
            residuals = self._capture_all_layer_activations(input_text)
            
            # TODO: In future, could add more sophisticated residual capture
            # by hooking into specific residual connection points using the
            # universal architecture registry
            
            return residuals
                
        except Exception as e:
            print(f"Warning: Could not capture residual stream ({e}), using fallback...")
            # Fallback: return layer activations
            return self._capture_all_layer_activations(input_text)

    def _project_to_final_logits(self, activation: torch.Tensor, from_layer: int) -> torch.Tensor:
        """Project activation to final logits using direct matrix multiplication"""
        
        try:
            with torch.no_grad():
                # For now, use direct projection through the output head
                # This is a reasonable approximation for understanding vocabulary effects
                
                activation = activation.to(self.model.device)
                
                # Apply final layer norm if it exists (this is important for OLMo/LLaMA models)
                if hasattr(self.model.model, 'norm'):
                    # Add batch and sequence dimensions for layer norm
                    x = activation.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
                    x = self.model.model.norm(x)
                    activation = x[0, 0, :]  # Remove dimensions
                elif hasattr(self.model, 'ln_f'):
                    x = activation.unsqueeze(0).unsqueeze(0)
                    x = self.model.ln_f(x)
                    activation = x[0, 0, :]
                
                # Project to vocabulary space
                if hasattr(self.model, 'lm_head'):
                    logits = torch.matmul(activation, self.model.lm_head.weight.T)
                elif hasattr(self.model, 'embed_out'):
                    logits = torch.matmul(activation, self.model.embed_out.weight.T)
                elif hasattr(self.model.model, 'embed_tokens'):
                    # Use embedding matrix transpose (tied weights)
                    logits = torch.matmul(activation, self.model.model.embed_tokens.weight.T)
                else:
                    # Last resort: zero logits
                    logits = torch.zeros(self.model.config.vocab_size, device=activation.device)
                
                return logits.cpu()
                
        except Exception as e:
            print(f"Error in direct projection: {e}")
            # Return zero logits as ultimate fallback
            return torch.zeros(self.model.config.vocab_size)

    def _analyze_effect_propagation(self, cascade_effects: Dict) -> Dict[str, Any]:
        """Analyze how effects propagate and amplify through layers"""
        
        layer_indices = sorted(cascade_effects.keys())
        magnitudes = [cascade_effects[i]['effect_magnitude'] for i in layer_indices]
        activation_magnitudes = [cascade_effects[i]['activation_magnitude'] for i in layer_indices]
        
        # Classify propagation pattern
        if magnitudes[-1] > magnitudes[0] * 2:
            pattern = "amplifying"
        elif magnitudes[-1] < magnitudes[0] * 0.5:
            pattern = "dampening"
        else:
            pattern = "stable"
        
        # Find critical layers (where effects change significantly)
        critical_layers = []
        for i in range(1, len(magnitudes)):
            change_ratio = magnitudes[i] / magnitudes[i-1] if magnitudes[i-1] > 0 else float('inf')
            if change_ratio > 2.0 or change_ratio < 0.5:
                critical_layers.append(layer_indices[i])
        
        return {
            'propagation_pattern': pattern,
            'magnitude_trajectory': magnitudes,
            'activation_trajectory': activation_magnitudes,
            'critical_layers': critical_layers,
            'amplification_ratio': magnitudes[-1] / magnitudes[0] if magnitudes[0] > 0 else 0.0,
            'peak_layer': layer_indices[magnitudes.index(max(magnitudes))]
        }

    def _analyze_vocabulary_convergence(self, cascade_effects: Dict) -> Dict[str, Any]:
        """Check if vocabulary effects converge to the final effect"""
        
        layer_indices = sorted(cascade_effects.keys())
        final_effect = torch.tensor(cascade_effects[layer_indices[-1]]['vocab_effect'])
        
        convergence_scores = {}
        stable_tokens = set()
        
        for layer_idx in layer_indices:
            layer_effect = torch.tensor(cascade_effects[layer_idx]['vocab_effect'])
            
            # Cosine similarity with final effect
            similarity = torch.cosine_similarity(layer_effect, final_effect, dim=0).item()
            convergence_scores[layer_idx] = similarity
            
            # Check for stable top tokens
            if similarity > 0.8:
                layer_top_tokens = set(token['token_str'] for token in cascade_effects[layer_idx]['top_tokens']['top_boosted'][:10])
                if not stable_tokens:
                    stable_tokens = layer_top_tokens
                else:
                    stable_tokens = stable_tokens.intersection(layer_top_tokens)
        
        # Find convergence point (where similarity > 0.9)
        convergence_layer = None
        for layer_idx in layer_indices:
            if convergence_scores[layer_idx] > 0.9:
                convergence_layer = layer_idx
                break
        
        return {
            'convergence_scores': convergence_scores,
            'convergence_layer': convergence_layer,
            'stable_tokens': list(stable_tokens),
            'final_similarity': convergence_scores[layer_indices[-1]] if layer_indices else 0.0
        }

    def _analyze_effect_accumulation(self, residual_effects: Dict) -> Dict[str, Any]:
        """Analyze how effects accumulate in residual stream"""
        
        layer_indices = sorted(residual_effects.keys())
        cumulative_magnitudes = [residual_effects[i]['cumulative_magnitude'] for i in layer_indices]
        direct_magnitudes = [residual_effects[i]['effect_magnitude'] for i in layer_indices]
        
        # Find layers with significant contributions
        significant_layers = []
        for i, layer_idx in enumerate(layer_indices):
            if direct_magnitudes[i] > max(direct_magnitudes) * 0.1:  # > 10% of max
                significant_layers.append(layer_idx)
        
        # Analyze accumulation pattern
        growth_rates = []
        for i in range(1, len(cumulative_magnitudes)):
            if cumulative_magnitudes[i-1] > 0:
                rate = cumulative_magnitudes[i] / cumulative_magnitudes[i-1]
                growth_rates.append(rate)
        
        avg_growth_rate = np.mean(growth_rates) if growth_rates else 1.0
        
        return {
            'accumulation_pattern': 'linear' if 0.9 <= avg_growth_rate <= 1.1 else 'non_linear',
            'cumulative_trajectory': cumulative_magnitudes,
            'direct_contributions': direct_magnitudes,
            'significant_layers': significant_layers,
            'total_accumulation': cumulative_magnitudes[-1] if cumulative_magnitudes else 0.0,
            'average_growth_rate': avg_growth_rate
        }

    def _analyze_amplification_patterns(self, residual_effects: Dict) -> Dict[str, Any]:
        """Analyze amplification factors across layers"""
        
        layer_indices = sorted(residual_effects.keys())
        amplification_factors = [residual_effects[i]['amplification_factor'] for i in layer_indices]
        
        # Find highly amplifying layers
        high_amplification = [layer_indices[i] for i, factor in enumerate(amplification_factors) if factor > 2.0]
        
        # Find dampening layers
        dampening = [layer_indices[i] for i, factor in enumerate(amplification_factors) if factor < 0.5]
        
        return {
            'amplification_factors': amplification_factors,
            'average_amplification': np.mean(amplification_factors),
            'max_amplification': max(amplification_factors) if amplification_factors else 0.0,
            'high_amplification_layers': high_amplification,
            'dampening_layers': dampening,
            'amplification_pattern': 'amplifying' if np.mean(amplification_factors) > 1.0 else 'dampening'
        }

    def _create_cascade_summary(self, effects: Dict, analysis_type: str) -> Dict[str, Any]:
        """Create summary of cascade analysis"""
        
        layer_indices = sorted(effects.keys())
        
        if analysis_type == 'full_projection':
            magnitudes = [effects[i]['effect_magnitude'] for i in layer_indices]
            key_metric = 'effect_magnitude'
        else:  # residual_stream
            magnitudes = [effects[i]['cumulative_magnitude'] for i in layer_indices]
            key_metric = 'cumulative_magnitude'
        
        peak_layer = layer_indices[magnitudes.index(max(magnitudes))]
        final_magnitude = magnitudes[-1] if magnitudes else 0.0
        
        return {
            'analysis_type': analysis_type,
            'total_layers_analyzed': len(layer_indices),
            'peak_effect_layer': peak_layer,
            'final_effect_magnitude': final_magnitude,
            'magnitude_range': [min(magnitudes), max(magnitudes)] if magnitudes else [0.0, 0.0],
            'effect_evolution': 'increasing' if magnitudes[-1] > magnitudes[0] else 'decreasing'
        }
    
    def display_cascade_analysis(self, cascade_results: Dict[str, Any], top_k: int = 5) -> None:
        """
        Display cascade analysis results in a readable format.
        
        Args:
            cascade_results: Results from analyze_vocabulary_cascade()
            top_k: Number of top tokens to show per layer
        """
        
        if 'error' in cascade_results:
            print(f"❌ Cascade analysis failed: {cascade_results['error']}")
            return
        
        analysis_type = cascade_results['analysis_type']
        sw = cascade_results['super_weight']
        summary = cascade_results['summary']
        
        print(f"\n{'='*70}")
        print(f"CASCADE ANALYSIS ({analysis_type.upper()}): {sw}")
        print(f"{'='*70}")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"  Analysis type: {analysis_type}")
        print(f"  Layers analyzed: {summary['total_layers_analyzed']}")
        print(f"  Peak effect at layer: {summary['peak_effect_layer']}")
        print(f"  Final magnitude: {summary['final_effect_magnitude']:.4f}")
        print(f"  Effect evolution: {summary['effect_evolution']}")
        
        # Analysis-specific results
        if analysis_type == 'full_projection':
            self._display_full_projection_results(cascade_results, top_k)
        elif analysis_type == 'residual_stream':
            self._display_residual_stream_results(cascade_results, top_k)
        
        print(f"\n{'-'*70}")

    def _display_full_projection_results(self, results: Dict[str, Any], top_k: int) -> None:
        """Display full projection specific results"""
        
        cascade_effects = results['cascade_effects']
        propagation = results['propagation_analysis']
        convergence = results['convergence_analysis']
        
        print(f"\n🔄 PROPAGATION ANALYSIS:")
        print(f"  Pattern: {propagation['propagation_pattern']}")
        print(f"  Amplification ratio: {propagation['amplification_ratio']:.3f}")
        print(f"  Peak layer: {propagation['peak_layer']}")
        print(f"  Critical layers: {propagation['critical_layers']}")
        
        print(f"\n🎯 CONVERGENCE ANALYSIS:")
        print(f"  Convergence layer: {convergence['convergence_layer']}")
        print(f"  Final similarity: {convergence['final_similarity']:.3f}")
        if convergence['stable_tokens']:
            print(f"  Stable tokens: {', '.join(repr(token) for token in convergence['stable_tokens'][:10])}")
        
        # Show key layers
        key_layers = [0, propagation['peak_layer'], max(cascade_effects.keys())]
        key_layers = sorted(list(set(key_layers)))  # Remove duplicates and sort
        
        print(f"\n📈 KEY LAYERS ANALYSIS:")
        for layer_idx in key_layers:
            if layer_idx in cascade_effects:
                effects = cascade_effects[layer_idx]
                print(f"\n  Layer {layer_idx}:")
                print(f"    Effect magnitude: {effects['effect_magnitude']:.4f}")
                print(f"    Activation magnitude: {effects['activation_magnitude']:.4f}")
                print(f"    Layers remaining: {effects['layers_remaining']}")
                
                # Top boosted tokens
                top_boosted = effects['top_tokens']['top_boosted'][:top_k]
                if top_boosted:
                    tokens_str = ', '.join(f"{repr(t['token_str'])}({t['effect_magnitude']:+.2f})" for t in top_boosted)
                    print(f"    Top boosted: {tokens_str}")

    def _display_residual_stream_results(self, results: Dict[str, Any], top_k: int) -> None:
        """Display residual stream specific results"""
        
        residual_effects = results['residual_effects']
        accumulation = results['accumulation_analysis']
        amplification = results['amplification_analysis']
        
        print(f"\n📈 ACCUMULATION ANALYSIS:")
        print(f"  Pattern: {accumulation['accumulation_pattern']}")
        print(f"  Total accumulation: {accumulation['total_accumulation']:.4f}")
        print(f"  Average growth rate: {accumulation['average_growth_rate']:.3f}")
        print(f"  Significant layers: {accumulation['significant_layers']}")
        
        print(f"\n🔊 AMPLIFICATION ANALYSIS:")
        print(f"  Pattern: {amplification['amplification_pattern']}")
        print(f"  Average amplification: {amplification['average_amplification']:.3f}")
        print(f"  Max amplification: {amplification['max_amplification']:.3f}")
        print(f"  High amplification layers: {amplification['high_amplification_layers']}")
        print(f"  Dampening layers: {amplification['dampening_layers']}")
        
        # Show significant contributing layers
        significant_layers = accumulation['significant_layers'][:5]  # Top 5
        
        print(f"\n📊 SIGNIFICANT LAYERS:")
        for layer_idx in significant_layers:
            if layer_idx in residual_effects:
                effects = residual_effects[layer_idx]
                print(f"\n  Layer {layer_idx}:")
                print(f"    Direct effect: {effects['effect_magnitude']:.4f}")
                print(f"    Cumulative effect: {effects['cumulative_magnitude']:.4f}")
                print(f"    Amplification factor: {effects['amplification_factor']:.3f}")
                print(f"    Residual magnitude: {effects['residual_magnitude']:.4f}")
                
                # Top affected tokens for this layer
                top_boosted = effects['top_tokens']['top_boosted'][:top_k]
                if top_boosted:
                    tokens_str = ', '.join(f"{repr(t['token_str'])}({t['effect_magnitude']:+.2f})" for t in top_boosted)
                    print(f"    Top boosted: {tokens_str}")

    def compare_cascade_methods(self, super_weight: SuperWeight, 
                              input_text: str = "Apple Inc. is a tech company.") -> Dict[str, Any]:
        """
        Compare results from both cascade analysis methods.
        
        Args:
            super_weight: SuperWeight to analyze
            input_text: Text to analyze with both methods
            
        Returns:
            Dictionary with comparison results
        """
        
        print(f"Running cascade method comparison for {super_weight}...")
        
        # Run both analyses
        full_projection = self.analyze_vocabulary_cascade(super_weight, input_text, "full_projection")
        residual_stream = self.analyze_vocabulary_cascade(super_weight, input_text, "residual_stream")
        
        # Check for errors
        if 'error' in full_projection or 'error' in residual_stream:
            return {
                'super_weight': super_weight,
                'full_projection': full_projection,
                'residual_stream': residual_stream,
                'comparison_error': "One or both analyses failed"
            }
        
        # Compare final effects
        fp_final = torch.tensor(full_projection['cascade_effects'][max(full_projection['cascade_effects'].keys())]['vocab_effect'])
        rs_final = torch.tensor(residual_stream['residual_effects'][max(residual_stream['residual_effects'].keys())]['cumulative_vocab_effect'])
        
        # Compute similarity between final effects
        final_correlation = torch.corrcoef(torch.stack([fp_final, rs_final]))[0, 1].item()
        final_cosine_sim = torch.cosine_similarity(fp_final, rs_final, dim=0).item()
        
        # Compare magnitude trajectories
        fp_magnitudes = [full_projection['cascade_effects'][i]['effect_magnitude'] 
                        for i in sorted(full_projection['cascade_effects'].keys())]
        rs_magnitudes = [residual_stream['residual_effects'][i]['cumulative_magnitude'] 
                        for i in sorted(residual_stream['residual_effects'].keys())]
        
        # Align trajectories (in case different number of layers)
        min_len = min(len(fp_magnitudes), len(rs_magnitudes))
        fp_aligned = fp_magnitudes[:min_len]
        rs_aligned = rs_magnitudes[:min_len]
        
        magnitude_correlation = np.corrcoef(fp_aligned, rs_aligned)[0, 1] if min_len > 1 else 0.0
        
        return {
            'super_weight': super_weight,
            'input_text': input_text,
            'full_projection': full_projection,
            'residual_stream': residual_stream,
            'comparison_metrics': {
                'final_effects_correlation': final_correlation,
                'final_effects_cosine_similarity': final_cosine_sim,
                'magnitude_trajectory_correlation': magnitude_correlation,
                'fp_final_magnitude': float(torch.norm(fp_final)),
                'rs_final_magnitude': float(torch.norm(rs_final))
            },
            'summary': {
                'methods_agree': final_correlation > 0.7,
                'preferred_method': 'full_projection' if torch.norm(fp_final) > torch.norm(rs_final) else 'residual_stream',
                'consistency_score': (final_correlation + magnitude_correlation) / 2
            }
        }

    def display_cascade_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """Display cascade method comparison results"""
        
        if 'comparison_error' in comparison_results:
            print(f"❌ Comparison failed: {comparison_results['comparison_error']}")
            return
        
        sw = comparison_results['super_weight']
        metrics = comparison_results['comparison_metrics']
        summary = comparison_results['summary']
        
        print(f"\n{'='*60}")
        print(f"CASCADE METHODS COMPARISON: {sw}")
        print(f"{'='*60}")
        
        print(f"\n📊 COMPARISON METRICS:")
        print(f"  Final effects correlation: {metrics['final_effects_correlation']:.3f}")
        print(f"  Final effects cosine similarity: {metrics['final_effects_cosine_similarity']:.3f}")
        print(f"  Magnitude trajectory correlation: {metrics['magnitude_trajectory_correlation']:.3f}")
        print(f"  Full projection final magnitude: {metrics['fp_final_magnitude']:.4f}")
        print(f"  Residual stream final magnitude: {metrics['rs_final_magnitude']:.4f}")
        
        print(f"\n🎯 SUMMARY:")
        print(f"  Methods agree: {summary['methods_agree']}")
        print(f"  Preferred method: {summary['preferred_method']}")
        print(f"  Consistency score: {summary['consistency_score']:.3f}")
        
        print(f"\n💡 INTERPRETATION:")
        if summary['methods_agree']:
            print("  ✅ Both methods show consistent results")
            print("  → Super weight effects are stable across analysis approaches")
        else:
            print("  ⚠️  Methods show different results")
            print("  → Super weight effects may be context-dependent or method-sensitive")
        
        print(f"\n{'-'*60}")
    
    def analyze_and_save_vocabulary_effects(self, super_weight: SuperWeight, 
                                          model_name: str,
                                          test_texts: Optional[List[str]] = None,
                                          dataset_name: Optional[str] = None,
                                          dataset_config: Optional[str] = None,
                                          n_samples: int = 500,
                                          max_length: int = 512,
                                          save_plots: bool = True,
                                          display_results: bool = True) -> str:
        """
        Analyze vocabulary effects and save results with plots.
        
        Args:
            super_weight: SuperWeight to analyze
            model_name: Name of the model for saving results
            test_texts: Optional list of custom test texts (takes priority if provided)
            dataset_name: Dataset name (e.g., 'wikitext')
            dataset_config: Dataset config (e.g., 'wikitext-2-raw-v1') 
            n_samples: Number of dataset samples to use
            max_length: Maximum sequence length for dataset samples
            save_plots: Whether to generate and save plots
            display_results: Whether to display results to console
            
        Returns:
            Path to saved results file
        """
        
        print(f"Analyzing vocabulary effects for {super_weight} in {model_name}...")
        
        # Run the analysis
        results = self.analyze_vocabulary_effects(
            super_weight, test_texts, dataset_name, dataset_config, n_samples, max_length
        )
        
        # Display results if requested
        if display_results:
            self.display_analysis_results(results)
        
        # Save results
        saved_path = self.results_manager.save_vocabulary_effects_analysis(
            results, model_name, save_plots
        )
        
        print(f"✅ Results saved to: {saved_path}")
        if save_plots:
            print(f"📊 Plots saved to: {self.results_manager.plots_dir}")
        
        return saved_path
    
    def analyze_and_save_neuron_vocabulary(self, super_weight: SuperWeight, 
                                         model_name: str,
                                         save_plots: bool = True,
                                         display_results: bool = True) -> str:
        """
        Analyze neuron vocabulary effects and save results with plots.
        
        Args:
            super_weight: SuperWeight to analyze (identifies the neuron)
            model_name: Name of the model for saving results
            save_plots: Whether to generate and save plots
            display_results: Whether to display results to console
            
        Returns:
            Path to saved results file
        """
        
        print(f"Analyzing neuron vocabulary effects for {super_weight} in {model_name}...")
        
        # Run the analysis
        results = self.analyze_neuron_vocabulary_effects(super_weight)
        
        # Display results if requested and no error
        if display_results and 'error' not in results:
            self.display_analysis_results(results)
        elif 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
        
        # Save results
        saved_path = self.results_manager.save_neuron_vocabulary_analysis(
            results, model_name, save_plots
        )
        
        print(f"✅ Results saved to: {saved_path}")
        if save_plots and 'error' not in results:
            print(f"📊 Plots saved to: {self.results_manager.plots_dir}")
        
        return saved_path
    
    def analyze_and_save_cascade(self, super_weight: SuperWeight, 
                               model_name: str,
                               input_text: str = "Apple Inc. is a tech company.",
                               method: str = "residual_stream",
                               save_plots: bool = True,
                               display_results: bool = True) -> str:
        """
        Analyze vocabulary cascade and save results with plots.
        
        Args:
            super_weight: SuperWeight to analyze
            model_name: Name of the model for saving results
            input_text: Text to analyze
            method: 'full_projection' or 'residual_stream'
            save_plots: Whether to generate and save plots
            display_results: Whether to display results to console
            
        Returns:
            Path to saved results file
        """
        
        print(f"Analyzing vocabulary cascade ({method}) for {super_weight} in {model_name}...")
        
        # Run the analysis
        results = self.analyze_vocabulary_cascade(super_weight, input_text, method)
        
        # Display results if requested and no error
        if display_results and 'error' not in results:
            self.display_cascade_analysis(results)
        elif 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
        
        # Save results
        saved_path = self.results_manager.save_cascade_analysis(
            results, model_name, save_plots
        )
        
        print(f"✅ Results saved to: {saved_path}")
        if save_plots and 'error' not in results:
            print(f"📊 Plots saved to: {self.results_manager.plots_dir}")
        
        return saved_path
    
    def analyze_and_save_neuron_vs_super_weight(self, super_weight: SuperWeight,
                                              model_name: str,
                                              test_texts: Optional[List[str]] = None,
                                              dataset_name: Optional[str] = None,
                                              dataset_config: Optional[str] = None,
                                              n_samples: int = 500,
                                              save_plots: bool = True,
                                              display_results: bool = True) -> str:
        """
        Compare neuron vs super weight effects and save results with plots.
        
        Args:
            super_weight: SuperWeight to analyze
            model_name: Name of the model for saving results
            test_texts: Optional custom test texts for super weight analysis
            dataset_name: Dataset for super weight analysis (if test_texts is None)
            dataset_config: Dataset config
            n_samples: Number of samples for dataset analysis
            save_plots: Whether to generate and save plots
            display_results: Whether to display results to console
            
        Returns:
            Path to saved results file
        """
        
        print(f"Comparing neuron vs super weight for {super_weight} in {model_name}...")
        
        # Run the comparison
        results = self.compare_neuron_vs_super_weight(
            super_weight, test_texts, dataset_name, dataset_config, n_samples
        )
        
        # Display results if requested and no error
        if display_results and 'error' not in results:
            self.display_neuron_vs_super_weight_comparison(results)
        elif 'error' in results:
            print(f"❌ Comparison failed: {results['error']}")
        
        # Save results
        saved_path = self.results_manager.save_comparison_analysis(
            results, model_name, 'neuron_vs_super_weight', save_plots
        )
        
        print(f"✅ Results saved to: {saved_path}")
        if save_plots and 'error' not in results:
            print(f"📊 Plots saved to: {self.results_manager.plots_dir}")
        
        return saved_path
    
    def run_complete_vocabulary_analysis(self, super_weight: SuperWeight,
                                       model_name: str,
                                       test_texts: Optional[List[str]] = None,
                                       dataset_name: Optional[str] = None,
                                       dataset_config: Optional[str] = None,
                                       n_samples: int = 500,
                                       cascade_input: str = "Apple Inc. is a tech company.",
                                       save_plots: bool = True,
                                       display_results: bool = False) -> Dict[str, str]:
        """
        Run all vocabulary analyses for a super weight and save results.
        
        Args:
            super_weight: SuperWeight to analyze
            model_name: Name of the model for saving results
            test_texts: Optional custom test texts
            dataset_name: Dataset name for vocabulary effects analysis
            dataset_config: Dataset config
            n_samples: Number of samples for dataset analysis
            cascade_input: Input text for cascade analysis
            save_plots: Whether to generate and save plots
            display_results: Whether to display results to console (usually False for batch runs)
            
        Returns:
            Dictionary mapping analysis type to saved file path
        """
        
        print(f"\n🔬 Running complete vocabulary analysis for {super_weight} in {model_name}")
        print("=" * 80)
        
        saved_files = {}
        
        # 1. Vocabulary effects analysis
        try:
            saved_files['vocabulary_effects'] = self.analyze_and_save_vocabulary_effects(
                super_weight, model_name, test_texts, dataset_name, dataset_config, 
                n_samples, display_results=display_results, save_plots=save_plots
            )
        except Exception as e:
            print(f"❌ Vocabulary effects analysis failed: {e}")
            saved_files['vocabulary_effects'] = None
        
        # 2. Neuron vocabulary analysis
        try:
            saved_files['neuron_vocabulary'] = self.analyze_and_save_neuron_vocabulary(
                super_weight, model_name, save_plots=save_plots, display_results=display_results
            )
        except Exception as e:
            print(f"❌ Neuron vocabulary analysis failed: {e}")
            saved_files['neuron_vocabulary'] = None
        
        # 3. Neuron vs super weight comparison
        try:
            saved_files['neuron_vs_super_weight'] = self.analyze_and_save_neuron_vs_super_weight(
                super_weight, model_name, test_texts, dataset_name, dataset_config, 
                n_samples, save_plots=save_plots, display_results=display_results
            )
        except Exception as e:
            print(f"❌ Neuron vs super weight comparison failed: {e}")
            saved_files['neuron_vs_super_weight'] = None
        
        # 4. Cascade analysis (residual stream)
        try:
            saved_files['cascade_residual'] = self.analyze_and_save_cascade(
                super_weight, model_name, cascade_input, method="residual_stream",
                save_plots=save_plots, display_results=display_results
            )
        except Exception as e:
            print(f"❌ Cascade analysis (residual stream) failed: {e}")
            saved_files['cascade_residual'] = None
        
        # 5. Cascade analysis (full projection) - optional, can be slow
        try:
            saved_files['cascade_full_projection'] = self.analyze_and_save_cascade(
                super_weight, model_name, cascade_input, method="full_projection",
                save_plots=save_plots, display_results=display_results
            )
        except Exception as e:
            print(f"❌ Cascade analysis (full projection) failed: {e}")
            saved_files['cascade_full_projection'] = None
        
        print(f"\n✅ Complete vocabulary analysis finished for {super_weight}")
        success_count = sum(1 for path in saved_files.values() if path is not None)
        print(f"📊 Successfully saved {success_count}/{len(saved_files)} analyses")
        print("=" * 80)
        
        return saved_files
    
    def create_model_summary_report(self, model_name: str) -> str:
        """
        Create a summary report of all vocabulary analyses for a model.
        
        Args:
            model_name: Name of the model to summarize
            
        Returns:
            Path to the saved summary report
        """
        return self.results_manager.create_summary_report(model_name)