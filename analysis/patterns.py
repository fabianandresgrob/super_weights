import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalMLPHandler


class PatternsAnalyzer:
    """
    Analyzes patterns in super weight locations, activations, and effects.
    """
    
    def __init__(self, model, tokenizer, manager, mlp_handler: UniversalMLPHandler):
        self.model = model
        self.tokenizer = tokenizer
        self.manager = manager
        self.mlp_handler = mlp_handler  # Use passed handler instead of creating new one
    
    def analyze_spatial_patterns(self, super_weights: List[SuperWeight]) -> Dict[str, Any]:
        """
        Analyze spatial patterns in super weight locations.
        
        Args:
            super_weights: List of detected super weights
            
        Returns:
            Dictionary with spatial pattern analysis
        """
        
        # Group by layer
        layer_groups = defaultdict(list)
        for sw in super_weights:
            layer_groups[sw.layer].append(sw)
        
        # Group by input channel across layers
        input_channel_patterns = defaultdict(list)
        for sw in super_weights:
            input_channel_patterns[sw.column].append(sw)
        
        # Group by output channel across layers
        output_channel_patterns = defaultdict(list)
        for sw in super_weights:
            output_channel_patterns[sw.row].append(sw)
        
        # Analyze layer distribution
        layer_stats = self._analyze_layer_distribution(layer_groups)
        
        # Find channel reuse patterns
        channel_reuse = self._analyze_channel_reuse(input_channel_patterns, output_channel_patterns)
        
        # Analyze coordinate clustering
        coordinate_clusters = self._analyze_coordinate_clustering(super_weights)
        
        return {
            'layer_distribution': layer_stats,
            'channel_reuse_patterns': channel_reuse,
            'coordinate_clusters': coordinate_clusters,
            'total_super_weights': len(super_weights),
            'layers_with_super_weights': len(layer_groups),
            'unique_input_channels': len(input_channel_patterns),
            'unique_output_channels': len(output_channel_patterns)
        }
    
    def _analyze_layer_distribution(self, layer_groups: Dict[int, List[SuperWeight]]) -> Dict[str, Any]:
        """Analyze how super weights are distributed across layers"""
        
        layer_counts = {layer: len(weights) for layer, weights in layer_groups.items()}
        
        if not layer_counts:
            return {'error': 'No super weights to analyze'}
        
        layers = list(layer_counts.keys())
        counts = list(layer_counts.values())
        
        return {
            'layer_counts': layer_counts,
            'most_populated_layer': max(layer_counts, key=layer_counts.get),
            'least_populated_layer': min(layer_counts, key=layer_counts.get),
            'early_layers_count': sum(count for layer, count in layer_counts.items() if layer < 5),
            'middle_layers_count': sum(count for layer, count in layer_counts.items() if 5 <= layer < 20),
            'late_layers_count': sum(count for layer, count in layer_counts.items() if layer >= 20),
            'layer_concentration': self._compute_layer_concentration(layers, counts)
        }
    
    def _compute_layer_concentration(self, layers: List[int], counts: List[int]) -> float:
        """Compute concentration metric for layer distribution (0 = uniform, 1 = concentrated)"""
        if not counts or sum(counts) == 0:
            return 0.0
        
        # Normalized counts
        probs = np.array(counts) / sum(counts)
        
        # Compute entropy
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(probs))
        
        # Concentration is 1 - normalized_entropy
        concentration = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        return concentration
    
    def _analyze_channel_reuse(self, input_patterns: Dict, output_patterns: Dict) -> Dict[str, Any]:
        """Analyze channel reuse patterns across layers"""
        
        # Find input channels used in multiple layers
        multi_layer_input_channels = {
            channel: layers for channel, layers in input_patterns.items()
            if len(set(sw.layer for sw in layers)) > 1
        }
        
        # Find output channels used in multiple layers
        multi_layer_output_channels = {
            channel: layers for channel, layers in output_patterns.items()
            if len(set(sw.layer for sw in layers)) > 1
        }
        
        # Find channels that appear as both input and output
        input_channels = set(input_patterns.keys())
        output_channels = set(output_patterns.keys())
        shared_channels = input_channels.intersection(output_channels)
        
        return {
            'multi_layer_input_channels': len(multi_layer_input_channels),
            'multi_layer_output_channels': len(multi_layer_output_channels),
            'shared_input_output_channels': len(shared_channels),
            'input_channel_reuse_examples': dict(list(multi_layer_input_channels.items())[:3]),
            'output_channel_reuse_examples': dict(list(multi_layer_output_channels.items())[:3]),
            'shared_channel_examples': list(shared_channels)[:5]
        }
    
    def _analyze_coordinate_clustering(self, super_weights: List[SuperWeight]) -> Dict[str, Any]:
        """Analyze clustering of super weight coordinates"""
        
        coordinates = [(sw.row, sw.column) for sw in super_weights]
        
        if len(coordinates) < 2:
            return {'error': 'Not enough coordinates for clustering analysis'}
        
        # Simple clustering analysis
        coordinate_distances = []
        for i, (r1, c1) in enumerate(coordinates):
            for j, (r2, c2) in enumerate(coordinates[i+1:], i+1):
                distance = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
                coordinate_distances.append(distance)
        
        avg_distance = np.mean(coordinate_distances)
        min_distance = np.min(coordinate_distances)
        max_distance = np.max(coordinate_distances)
        
        # Find close pairs (distance < mean - std)
        threshold = avg_distance - np.std(coordinate_distances)
        close_pairs = []
        
        for i, (r1, c1) in enumerate(coordinates):
            for j, (r2, c2) in enumerate(coordinates[i+1:], i+1):
                distance = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
                if distance < threshold:
                    close_pairs.append({
                        'sw1': super_weights[i],
                        'sw2': super_weights[j],
                        'distance': distance
                    })
        
        return {
            'average_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'close_pairs_count': len(close_pairs),
            'close_pairs_examples': close_pairs[:5],
            'clustering_score': 1.0 - (avg_distance / max_distance) if max_distance > 0 else 0.0
        }
    
    def analyze_activation_patterns(self, super_weights: List[SuperWeight], 
                                  test_texts: List[str] = None) -> Dict[str, Any]:
        """
        Analyze activation patterns of super weights across different inputs.
        
        Args:
            super_weights: List of super weights to analyze
            test_texts: List of test texts to use
            
        Returns:
            Dictionary with activation pattern analysis
        """
        
        if test_texts is None:
            test_texts = self._get_default_test_texts()
        
        activation_data = {}
        
        for sw in super_weights:
            sw_activations = []
            
            for text in test_texts:
                activation = self._get_super_weight_activation(sw, text)
                sw_activations.append({
                    'text': text,
                    'activation': activation
                })
            
            activation_data[str(sw)] = {
                'super_weight': sw,
                'activations': sw_activations,
                'statistics': self._compute_activation_statistics(sw_activations)
            }
        
        # Cross-activation analysis
        cross_analysis = self._analyze_cross_activations(activation_data, test_texts)
        
        return {
            'individual_patterns': activation_data,
            'cross_activation_analysis': cross_analysis,
            'pattern_summary': self._summarize_activation_patterns(activation_data)
        }
    
    def _get_super_weight_activation(self, super_weight: SuperWeight, text: str) -> float:
        """Get activation value for a super weight on specific text"""
        
        activation_value = None
        
        def create_hook(target_sw):
            def hook(module, inputs, output):
                nonlocal activation_value
                # Get activation at the super weight's column position
                act_input = inputs[0].detach()
                # Average across batch and sequence dimensions, take specific channel
                activation_value = act_input[:, :, target_sw.column].mean().item()
            return hook
        
        # Register hook on the appropriate layer
        layer = self._get_layer(super_weight.layer)
        _, _, module = self._get_mlp_component_info(layer)
        hook = module.register_forward_hook(create_hook(super_weight))
        
        try:
            # Run forward pass
            tokens = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                self.model(**tokens)
        finally:
            hook.remove()
        
        return activation_value if activation_value is not None else 0.0
    
    def _get_layer(self, layer_idx: int):
        """Get layer by index"""
        if hasattr(self.model, "model"):
            return self.model.model.layers[layer_idx]
        else:
            return self.model.layers[layer_idx]
    
    def _get_mlp_component_info(self, layer):
        """Get MLP component info (copied from detector for consistency)"""
        if hasattr(layer, "mlp"):
            base = "mlp"
            if hasattr(layer.mlp, "down_proj"):
                down_name = "down_proj"
                module = layer.mlp.down_proj
            elif hasattr(layer.mlp, "c_proj"):
                down_name = "c_proj"
                module = layer.mlp.c_proj
            else:
                down_name = "fc2"
                module = layer.mlp.fc2
        elif hasattr(layer, "feed_forward"):
            base = "feed_forward"
            down_name = "output_dense"
            module = layer.feed_forward.output_dense
        else:
            raise ValueError(f"Unsupported MLP structure")
        
        return base, down_name, module
    
    def _compute_activation_statistics(self, activations: List[Dict]) -> Dict[str, float]:
        """Compute statistics for activation patterns"""
        
        values = [act['activation'] for act in activations]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0.0
        }
    
    def _analyze_cross_activations(self, activation_data: Dict, test_texts: List[str]) -> Dict[str, Any]:
        """Analyze correlations between super weight activations"""
        
        sw_names = list(activation_data.keys())
        n_weights = len(sw_names)
        
        if n_weights < 2:
            return {'error': 'Need at least 2 super weights for cross-activation analysis'}
        
        # Create activation matrix (super_weights x texts)
        activation_matrix = np.zeros((n_weights, len(test_texts)))
        
        for i, sw_name in enumerate(sw_names):
            for j, act_data in enumerate(activation_data[sw_name]['activations']):
                activation_matrix[i, j] = act_data['activation']
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(activation_matrix)
        
        # Find highly correlated pairs
        high_correlation_pairs = []
        for i in range(n_weights):
            for j in range(i+1, n_weights):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.7:  # High correlation threshold
                    high_correlation_pairs.append({
                        'sw1': sw_names[i],
                        'sw2': sw_names[j],
                        'correlation': corr
                    })
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'average_correlation': np.mean(np.abs(correlation_matrix[np.triu_indices(n_weights, k=1)])),
            'high_correlation_pairs': high_correlation_pairs,
            'max_correlation': np.max(np.abs(correlation_matrix[np.triu_indices(n_weights, k=1)])),
            'super_weight_names': sw_names
        }
    
    def _summarize_activation_patterns(self, activation_data: Dict) -> Dict[str, Any]:
        """Summarize activation patterns across all super weights"""
        
        all_activations = []
        high_variance_weights = []
        low_variance_weights = []
        
        for sw_name, data in activation_data.items():
            stats = data['statistics']
            all_activations.extend([act['activation'] for act in data['activations']])
            
            if stats.get('coefficient_of_variation', 0) > 0.5:
                high_variance_weights.append(sw_name)
            elif stats.get('coefficient_of_variation', 0) < 0.1:
                low_variance_weights.append(sw_name)
        
        return {
            'total_super_weights': len(activation_data),
            'overall_activation_stats': {
                'mean': np.mean(all_activations),
                'std': np.std(all_activations),
                'range': np.max(all_activations) - np.min(all_activations)
            },
            'high_variance_weights': high_variance_weights,
            'low_variance_weights': low_variance_weights,
            'pattern_diversity': len(high_variance_weights) / len(activation_data) if activation_data else 0.0
        }
    
    def analyze_functional_patterns(self, super_weights: List[SuperWeight]) -> Dict[str, Any]:
        """
        Analyze functional patterns in super weights based on their properties.
        
        Args:
            super_weights: List of super weights to analyze
            
        Returns:
            Dictionary with functional pattern analysis
        """
        
        # Group by iteration found
        iteration_groups = defaultdict(list)
        for sw in super_weights:
            iteration_groups[sw.iteration_found].append(sw)
        
        # Analyze magnitude patterns
        magnitude_patterns = self._analyze_magnitude_patterns(super_weights)
        
        # Analyze component patterns
        component_patterns = self._analyze_component_patterns(super_weights)
        
        # Analyze discovery patterns
        discovery_patterns = self._analyze_discovery_patterns(iteration_groups)
        
        return {
            'magnitude_patterns': magnitude_patterns,
            'component_patterns': component_patterns,
            'discovery_patterns': discovery_patterns,
            'iteration_distribution': {iter_num: len(weights) for iter_num, weights in iteration_groups.items()}
        }
    
    def _analyze_magnitude_patterns(self, super_weights: List[SuperWeight]) -> Dict[str, Any]:
        """Analyze patterns in super weight magnitudes"""
        
        input_values = [sw.input_value for sw in super_weights]
        output_values = [sw.output_value for sw in super_weights]
        magnitude_products = [sw.magnitude_product for sw in super_weights]
        
        return {
            'input_value_stats': {
                'mean': np.mean(input_values),
                'std': np.std(input_values),
                'range': np.max(input_values) - np.min(input_values)
            },
            'output_value_stats': {
                'mean': np.mean(output_values),
                'std': np.std(output_values),
                'range': np.max(output_values) - np.min(output_values)
            },
            'magnitude_product_stats': {
                'mean': np.mean(magnitude_products),
                'std': np.std(magnitude_products),
                'range': np.max(magnitude_products) - np.min(magnitude_products)
            },
            'input_output_correlation': np.corrcoef(input_values, output_values)[0, 1],
            'largest_magnitude_weight': max(super_weights, key=lambda sw: sw.magnitude_product),
            'smallest_magnitude_weight': min(super_weights, key=lambda sw: sw.magnitude_product)
        }
    
    def _analyze_component_patterns(self, super_weights: List[SuperWeight]) -> Dict[str, Any]:
        """Analyze patterns in component types"""
        
        component_counts = defaultdict(int)
        for sw in super_weights:
            component_counts[sw.component] += 1
        
        return {
            'component_distribution': dict(component_counts),
            'most_common_component': max(component_counts, key=component_counts.get),
            'component_diversity': len(component_counts),
            'uniform_components': len(set(component_counts.values())) == 1
        }
    
    def _analyze_discovery_patterns(self, iteration_groups: Dict[int, List[SuperWeight]]) -> Dict[str, Any]:
        """Analyze patterns in discovery across iterations"""
        
        iteration_counts = {iter_num: len(weights) for iter_num, weights in iteration_groups.items()}
        
        if not iteration_counts:
            return {'error': 'No iteration data available'}
        
        return {
            'total_iterations': len(iteration_groups),
            'weights_per_iteration': iteration_counts,
            'peak_discovery_iteration': max(iteration_counts, key=iteration_counts.get),
            'early_discovery_ratio': sum(count for iter_num, count in iteration_counts.items() if iter_num <= 2) / sum(iteration_counts.values()),
            'discovery_trend': self._compute_discovery_trend(iteration_counts)
        }
    
    def _compute_discovery_trend(self, iteration_counts: Dict[int, int]) -> str:
        """Compute trend in discovery across iterations"""
        
        iterations = sorted(iteration_counts.keys())
        counts = [iteration_counts[iter_num] for iter_num in iterations]
        
        if len(counts) < 2:
            return "insufficient_data"
        
        # Simple trend analysis
        if counts[0] > counts[-1]:
            if all(counts[i] >= counts[i+1] for i in range(len(counts)-1)):
                return "monotonic_decrease"
            else:
                return "general_decrease"
        elif counts[0] < counts[-1]:
            if all(counts[i] <= counts[i+1] for i in range(len(counts)-1)):
                return "monotonic_increase"
            else:
                return "general_increase"
        else:
            return "stable"
    
    def _get_default_test_texts(self) -> List[str]:
        """Get default test texts for activation analysis"""
        return [
            "The researchers are improving their methodology.",
            "In 2001, the technology changed dramatically.",
            "Hello, world! How are you doing today?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning requires computational resources."
        ]