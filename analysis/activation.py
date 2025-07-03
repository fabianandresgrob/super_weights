"""
Model-agnostic super activation analysis.
Provides mathematical analysis of super activations that works across different model architectures.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalMLPHandler, MLPArchitectureType


class SuperActivationAnalyzer:
    """
    Analyzer for mathematical super activation analysis.
    Works across different transformer architectures by detecting and adapting to their MLP structures.
    """
    
    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler, log_level=logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        self.mlp_handler = mlp_handler  # Use passed handler instead of creating new one
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the analyzer"""
        logger = logging.getLogger(f"SuperActivationAnalyzer_{id(self)}")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def mathematical_super_activation_analysis(self, 
                                             super_weight: SuperWeight, 
                                             input_text: str = "Apple Inc. is a worldwide tech company.") -> Dict[str, Any]:
        """
        Perform mathematical analysis of super activation creation.
        
        This method automatically detects the model architecture and performs the appropriate
        mathematical analysis for the super activation.
        
        Args:
            super_weight: SuperWeight to analyze
            input_text: Input text to use for analysis
            
        Returns:
            Dictionary with mathematical analysis results
        """
        
        self.logger.info(f"Super activation analysis for {super_weight}")
        
        target_layer = super_weight.layer
        target_channel = super_weight.column
        
        # Detect architecture
        arch_info = self.mlp_handler.get_mlp_architecture(target_layer)
        self.logger.info(f"Detected architecture: {arch_info.architecture_type}")
        
        # Get input vector
        input_vector = self._extract_mlp_input_vector(input_text, target_layer)
        
        # Extract weights and biases using the handler
        weights = self.mlp_handler.extract_weights(target_layer)
        biases = self.mlp_handler.extract_biases(target_layer)
        
        # Perform architecture-specific analysis
        if arch_info.architecture_type in [MLPArchitectureType.GATED_MLP, MLPArchitectureType.FUSED_GATED_MLP]:
            return self._analyze_gated_mlp(
                input_vector, weights, biases, target_channel, arch_info, super_weight
            )
        elif arch_info.architecture_type == MLPArchitectureType.STANDARD_MLP:
            return self._analyze_standard_mlp(
                input_vector, weights, biases, target_channel, arch_info, super_weight
            )
        else:
            raise ValueError(f"Unsupported architecture type: {arch_info.architecture_type}")
    
    def _analyze_gated_mlp(self, 
                          input_vector: torch.Tensor,
                          weights: Dict[str, torch.Tensor],
                          biases: Dict[str, Optional[torch.Tensor]],
                          target_channel: int,
                          arch_info,
                          super_weight: SuperWeight) -> Dict[str, Any]:
        """Analyze gated MLP architecture - now supports all gated types including fused"""
        
        x = input_vector.cpu().numpy()
        
        # Handle different gated architectures using the handler's extracted weights
        if arch_info.architecture_type == MLPArchitectureType.FUSED_GATED_MLP:
            # For fused gate+up, weights are already split by the handler
            W_gate = weights['gate'].cpu().numpy()
            W_up = weights['up'].cpu().numpy()
            W_down = weights['down'].cpu().numpy()
            
            b_gate = biases['gate'].cpu().numpy() if biases['gate'] is not None else None
            b_up = biases['up'].cpu().numpy() if biases['up'] is not None else None
            b_down = biases['down'].cpu().numpy() if biases['down'] is not None else None
            
        else:  # Standard gated MLP
            W_gate = weights['gate'].cpu().numpy()
            W_up = weights['up'].cpu().numpy()
            W_down = weights['down'].cpu().numpy()
            
            b_gate = biases['gate'].cpu().numpy() if biases['gate'] is not None else None
            b_up = biases['up'].cpu().numpy() if biases['up'] is not None else None
            b_down = biases['down'].cpu().numpy() if biases['down'] is not None else None
    
        analysis_results = {
            'super_weight': super_weight,
            'architecture_info': arch_info,
            'model_agnostic_analysis': True,
            'mathematical_breakdown': {},
            'weight_analysis': {},
            'attack_vectors': {},
            'zero_activation_strategies': {}
        }
        
        # Step-by-step mathematical analysis
        analysis_results['mathematical_breakdown'] = self._step_by_step_gated_computation(
            x, W_gate, W_up, W_down, b_gate, b_up, b_down, target_channel, arch_info.activation_function, super_weight
        )
        
        # Analyze the weight vectors for target channel
        analysis_results['weight_analysis'] = self._analyze_target_channel_weights_gated(
            W_gate, W_up, W_down, b_gate, b_up, b_down, target_channel
        )
        
        # Find attack vectors
        analysis_results['attack_vectors'] = self._find_gated_attack_vectors(
            x, W_gate, W_up, b_gate, b_up, target_channel, arch_info.activation_function
        )
        
        # Zero activation strategies
        analysis_results['zero_activation_strategies'] = self._find_gated_zero_activation_strategies(
            x, W_gate, W_up, b_gate, b_up, target_channel, arch_info.activation_function
        )
        
        return analysis_results
    
    def _analyze_standard_mlp(self,
                            input_vector: torch.Tensor,
                            weights: Dict[str, torch.Tensor],
                            biases: Dict[str, Optional[torch.Tensor]],
                            target_channel: int,
                            arch_info,
                            super_weight: SuperWeight) -> Dict[str, Any]:
        """Analyze standard MLP architecture (GPT-2, etc.)"""
        
        x = input_vector.cpu().numpy()
        W_hidden = weights['hidden'].cpu().numpy()
        W_output = weights['output'].cpu().numpy()
        
        b_hidden = biases['hidden'].cpu().numpy() if biases['hidden'] is not None else None
        b_output = biases['output'].cpu().numpy() if biases['output'] is not None else None
        
        analysis_results = {
            'super_weight': super_weight,
            'architecture_info': arch_info,
            'model_agnostic_analysis': True,
            'mathematical_breakdown': {},
            'weight_analysis': {},
            'attack_vectors': {},
            'zero_activation_strategies': {}
        }
        
        # Step-by-step mathematical analysis
        analysis_results['mathematical_breakdown'] = self._step_by_step_standard_computation(
            x, W_hidden, W_output, b_hidden, b_output, target_channel, arch_info.activation_function, super_weight
        )
        
        # Analyze the weight vectors for target channel
        analysis_results['weight_analysis'] = self._analyze_target_channel_weights_standard(
            W_hidden, W_output, b_hidden, b_output, target_channel
        )
        
        # Find attack vectors
        analysis_results['attack_vectors'] = self._find_standard_attack_vectors(
            x, W_hidden, b_hidden, target_channel, arch_info.activation_function
        )
        
        # Zero activation strategies
        analysis_results['zero_activation_strategies'] = self._find_standard_zero_activation_strategies(
            x, W_hidden, b_hidden, target_channel, arch_info.activation_function
        )
        
        return analysis_results
    
    def _step_by_step_gated_computation(self, x, W_gate, W_up, W_down, b_gate, b_up, b_down, 
                                      target_channel, activation_fn, super_weight: SuperWeight):
        """Perform exact step-by-step computation for gated MLP"""
        
        # Step 1: Gate projection
        gate_output = W_gate @ x
        if b_gate is not None:
            gate_output += b_gate
        gate_target_output = gate_output[target_channel]
        
        # Step 2: Apply activation function
        activated_gate = self._apply_activation(gate_output, activation_fn)
        activated_gate_target = activated_gate[target_channel]
        
        # Step 3: Up projection
        up_output = W_up @ x
        if b_up is not None:
            up_output += b_up
        up_target_output = up_output[target_channel]
        
        # Step 4: Hadamard product (super activation)
        hadamard_result = activated_gate * up_output
        super_activation = hadamard_result[target_channel]
        
        # Step 5: Down projection
        final_output = W_down @ hadamard_result
        if b_down is not None:
            final_output += b_down
        super_weight_contribution = final_output[super_weight.row]
        
        return {
            'input_analysis': {
                'input_vector_shape': list(x.shape),
                'input_magnitude': float(np.linalg.norm(x)),
                'input_mean': float(np.mean(x)),
                'input_std': float(np.std(x))
            },
            
            'step1_gate_projection': {
                'operation': f"gate_output = W_gate @ x{' + b_gate' if b_gate is not None else ''}",
                # 'weight_vector': W_gate[target_channel].tolist(),
                'bias_term': float(b_gate[target_channel]) if b_gate is not None else None,
                'result_target_channel': float(gate_target_output),
                'has_bias': b_gate is not None
            },
            
            'step2_activation': {
                'operation': f"activated_gate = {activation_fn.upper()}(gate_output)",
                'input_to_activation': float(gate_target_output),
                'activation_output_target_channel': float(activated_gate_target),
                'activation_function': activation_fn,
                'activation_derivative_at_point': float(self._activation_derivative(gate_target_output, activation_fn))
            },
            
            'step3_up_projection': {
                'operation': f"up_output = W_up @ x{' + b_up' if b_up is not None else ''}",
                # 'weight_vector': W_up[target_channel].tolist(),
                'bias_term': float(b_up[target_channel]) if b_up is not None else None,
                'result_target_channel': float(up_target_output),
                'has_bias': b_up is not None
            },
            
            'step4_hadamard_product': {
                'operation': "hadamard_result = activated_gate ⊙ up_output (ELEMENT-WISE MULTIPLICATION)",
                'activated_gate_component': float(activated_gate_target),
                'up_output_component': float(up_target_output),
                'super_activation_result': float(super_activation),
                'formula': f"y[:, {target_channel}] = {activation_fn.upper()}(W_gate[{target_channel}] @ x) * (W_up[{target_channel}] @ x)",
                'this_is_where_super_activation_is_created': True
            },
            
            'step5_down_projection': {
                'operation': f"final_output = W_down @ hadamard_result{' + b_down' if b_down is not None else ''}",
                'super_weight_row': super_weight.row,
                'super_weight_processes_super_activation': float(super_activation),
                'super_weight_contribution': float(super_weight_contribution),
                'has_bias': b_down is not None
            },
            
            'verification': {
                'computed_super_activation': float(super_activation),
                'super_activation_magnitude': float(abs(super_activation)),
                'super_activation_sign': 'positive' if super_activation > 0 else 'negative',
                'expected_vs_actual': f"Expected ~{super_weight.input_value}, Computed: {super_activation:.2f}"
            }
        }
    
    def _step_by_step_standard_computation(self, x, W_hidden, W_output, b_hidden, b_output,
                                         target_channel, activation_fn, super_weight: SuperWeight):
        """Perform exact step-by-step computation for standard MLP"""
        
        # Step 1: Hidden projection
        hidden_output = W_hidden @ x
        if b_hidden is not None:
            hidden_output += b_hidden
        hidden_target_output = hidden_output[target_channel]
        
        # Step 2: Apply activation function
        activated_hidden = self._apply_activation(hidden_output, activation_fn)
        super_activation = activated_hidden[target_channel]  # This is the super activation
        
        # Step 3: Output projection
        final_output = W_output @ activated_hidden
        if b_output is not None:
            final_output += b_output
        super_weight_contribution = final_output[super_weight.row]
        
        return {
            'input_analysis': {
                'input_vector_shape': list(x.shape),
                'input_magnitude': float(np.linalg.norm(x)),
                'input_mean': float(np.mean(x)),
                'input_std': float(np.std(x))
            },
            
            'step1_hidden_projection': {
                'operation': f"hidden_output = W_hidden @ x{' + b_hidden' if b_hidden is not None else ''}",
                # 'weight_vector': W_hidden[target_channel].tolist(),
                'bias_term': float(b_hidden[target_channel]) if b_hidden is not None else None,
                'result_target_channel': float(hidden_target_output),
                'has_bias': b_hidden is not None
            },
            
            'step2_activation': {
                'operation': f"activated_hidden = {activation_fn.upper()}(hidden_output)",
                'input_to_activation': float(hidden_target_output),
                'super_activation_result': float(super_activation),
                'activation_function': activation_fn,
                'this_is_where_super_activation_is_created': True
            },
            
            'step3_output_projection': {
                'operation': f"final_output = W_output @ activated_hidden{' + b_output' if b_output is not None else ''}",
                'super_weight_row': super_weight.row,
                'super_weight_processes_super_activation': float(super_activation),
                'super_weight_contribution': float(super_weight_contribution),
                'has_bias': b_output is not None
            },
            
            'verification': {
                'computed_super_activation': float(super_activation),
                'super_activation_magnitude': float(abs(super_activation)),
                'super_activation_sign': 'positive' if super_activation > 0 else 'negative',
                'expected_vs_actual': f"Expected ~{super_weight.input_value}, Computed: {super_activation:.2f}"
            }
        }
    
    def _apply_activation(self, x, activation_name: str):
        """Apply activation function to input"""
        if activation_name.lower() == 'silu':
            return x * (1.0 / (1.0 + np.exp(-x)))
        elif activation_name.lower() == 'gelu':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif activation_name.lower() == 'relu':
            return np.maximum(0, x)
        else:
            return x  # Linear fallback
    
    def _activation_derivative(self, x, activation_name: str):
        """Compute derivative of activation function at point x"""
        if activation_name.lower() == 'silu':
            sigmoid_x = 1.0 / (1.0 + np.exp(-x))
            return sigmoid_x * (1 + x * (1 - sigmoid_x))
        elif activation_name.lower() == 'gelu':
            # Approximation of GELU derivative
            tanh_term = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))
            return 0.5 * (1 + tanh_term) + 0.5 * x * (1 - tanh_term**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        elif activation_name.lower() == 'relu':
            return 1.0 if x > 0 else 0.0
        else:
            return 1.0  # Linear derivative
    
    def _analyze_target_channel_weights_gated(self, W_gate, W_up, W_down, b_gate, b_up, b_down, target_channel):
        """Analyze weight vectors for target channel in gated MLP"""
        
        gate_weights = W_gate[target_channel]
        up_weights = W_up[target_channel]
        
        return {
            'target_channel_info': {
                'channel_index': target_channel,
                'description': f"Channel {target_channel} in intermediate space where super activation occurs"
            },
            
            'gate_weights_analysis': {
                'operation': f"gate_output[{target_channel}] = W_gate[{target_channel}] @ x{' + b_gate[{target_channel}]' if b_gate is not None else ''}",
                'weight_vector_norm': float(np.linalg.norm(gate_weights)),
                'max_weight': float(np.max(gate_weights)),
                'min_weight': float(np.min(gate_weights)),
                'mean_weight': float(np.mean(gate_weights)),
                'std_weight': float(np.std(gate_weights)),
                'bias_term': float(b_gate[target_channel]) if b_gate is not None else None,
                'top_5_positive_weights': self._get_top_weights(gate_weights, positive=True),
                'top_5_negative_weights': self._get_top_weights(gate_weights, positive=False)
            },
            
            'up_weights_analysis': {
                'operation': f"up_output[{target_channel}] = W_up[{target_channel}] @ x{' + b_up[{target_channel}]' if b_up is not None else ''}",
                'weight_vector_norm': float(np.linalg.norm(up_weights)),
                'max_weight': float(np.max(up_weights)),
                'min_weight': float(np.min(up_weights)),
                'mean_weight': float(np.mean(up_weights)),
                'std_weight': float(np.std(up_weights)),
                'bias_term': float(b_up[target_channel]) if b_up is not None else None,
                'top_5_positive_weights': self._get_top_weights(up_weights, positive=True),
                'top_5_negative_weights': self._get_top_weights(up_weights, positive=False)
            },
            
            'weight_interaction_analysis': {
                'gate_up_correlation': float(np.corrcoef(gate_weights, up_weights)[0, 1]),
                'same_sign_percentage': float(np.mean((gate_weights > 0) == (up_weights > 0)) * 100),
                'weight_magnitude_ratio': float(np.linalg.norm(gate_weights) / np.linalg.norm(up_weights)),
                'coordinated_amplification': 'YES' if np.corrcoef(gate_weights, up_weights)[0, 1] > 0.5 else 'NO'
            }
        }
    
    def _analyze_target_channel_weights_standard(self, W_hidden, W_output, b_hidden, b_output, target_channel):
        """Analyze weight vectors for target channel in standard MLP"""
        
        hidden_weights = W_hidden[target_channel]
        
        return {
            'target_channel_info': {
                'channel_index': target_channel,
                'description': f"Channel {target_channel} in hidden space where super activation occurs"
            },
            
            'hidden_weights_analysis': {
                'operation': f"hidden_output[{target_channel}] = W_hidden[{target_channel}] @ x{' + b_hidden[{target_channel}]' if b_hidden is not None else ''}",
                'weight_vector_norm': float(np.linalg.norm(hidden_weights)),
                'max_weight': float(np.max(hidden_weights)),
                'min_weight': float(np.min(hidden_weights)),
                'mean_weight': float(np.mean(hidden_weights)),
                'std_weight': float(np.std(hidden_weights)),
                'bias_term': float(b_hidden[target_channel]) if b_hidden is not None else None,
                'top_5_positive_weights': self._get_top_weights(hidden_weights, positive=True),
                'top_5_negative_weights': self._get_top_weights(hidden_weights, positive=False)
            }
        }
    
    def _find_gated_attack_vectors(self, x, W_gate, W_up, b_gate, b_up, target_channel, activation_fn):
        """Find attack vectors for gated MLP"""
        
        w_gate_target = W_gate[target_channel]
        w_up_target = W_up[target_channel]
        b_gate_target = b_gate[target_channel] if b_gate is not None else 0.0
        b_up_target = b_up[target_channel] if b_up is not None else 0.0
        
        attacks = {}
        
        # Attack 1: Zero the gate projection
        attacks['zero_gate_attack'] = self._find_zero_projection_attack(
            x, w_gate_target, b_gate_target, "gate"
        )
        
        # Attack 2: Zero the up projection
        attacks['zero_up_attack'] = self._find_zero_projection_attack(
            x, w_up_target, b_up_target, "up"
        )
        
        # Attack 3: Activation saturation (if applicable)
        if activation_fn.lower() == "silu":
            attacks['activation_saturation_attack'] = self._find_silu_saturation_attack(
                x, w_gate_target, b_gate_target
            )
        elif activation_fn.lower() == "relu":
            attacks['activation_saturation_attack'] = self._find_relu_saturation_attack(
                x, w_gate_target, b_gate_target
            )
        
        # Attack 4: Minimal perturbation
        attacks['minimal_perturbation_attack'] = self._find_minimal_perturbation_attack_gated(
            x, w_gate_target, w_up_target, b_gate_target, b_up_target
        )
        
        return attacks
    
    def _find_standard_attack_vectors(self, x, W_hidden, b_hidden, target_channel, activation_fn):
        """Find attack vectors for standard MLP"""
        
        w_hidden_target = W_hidden[target_channel]
        b_hidden_target = b_hidden[target_channel] if b_hidden is not None else 0.0
        
        attacks = {}
        
        # Attack 1: Zero the hidden projection
        attacks['zero_hidden_attack'] = self._find_zero_projection_attack(
            x, w_hidden_target, b_hidden_target, "hidden"
        )
        
        # Attack 2: Activation saturation
        if activation_fn.lower() == "relu":
            attacks['activation_saturation_attack'] = self._find_relu_saturation_attack(
                x, w_hidden_target, b_hidden_target
            )
        
        return attacks
    
    def _find_zero_projection_attack(self, x, w_target, b_target, projection_name):
        """Generic method to find attack that zeros a projection"""
        
        current_output = np.dot(w_target, x) + b_target
        target_change = -current_output
        
        max_weight_idx = np.argmax(np.abs(w_target))
        required_input_change = target_change / w_target[max_weight_idx] if w_target[max_weight_idx] != 0 else float('inf')
        
        return {
            'attack_name': f'Zero {projection_name.title()} Projection Attack',
            'current_output': float(current_output),
            'required_change': float(target_change),
            'modification_strategy': {
                'modify_dimension': int(max_weight_idx),
                'current_value': float(x[max_weight_idx]),
                'required_new_value': float(x[max_weight_idx] + required_input_change),
                'input_change': float(required_input_change)
            },
            'feasibility': 'high' if abs(required_input_change) < 10.0 else 'medium' if abs(required_input_change) < 100.0 else 'low'
        }
    
    def _find_silu_saturation_attack(self, x, w_target, b_target):
        """Find attack using SiLU saturation"""
        
        current_output = np.dot(w_target, x) + b_target
        saturation_target = -10.0  # SiLU ≈ 0 for x << -5
        required_change = saturation_target - current_output
        
        max_weight_idx = np.argmax(np.abs(w_target))
        required_input_change = required_change / w_target[max_weight_idx] if w_target[max_weight_idx] != 0 else float('inf')
        
        return {
            'attack_name': 'SiLU Saturation Attack',
            'current_output': float(current_output),
            'saturation_target': saturation_target,
            'required_change': float(required_change),
            'expected_activation_output': float(self._apply_activation(saturation_target, 'silu')),
            'modification_strategy': {
                'modify_dimension': int(max_weight_idx),
                'current_value': float(x[max_weight_idx]),
                'required_new_value': float(x[max_weight_idx] + required_input_change),
                'input_change': float(required_input_change)
            }
        }
    
    def _find_relu_saturation_attack(self, x, w_target, b_target):
        """Find attack using ReLU saturation"""
        
        current_output = np.dot(w_target, x) + b_target
        saturation_target = -1.0  # Any negative value zeros ReLU
        required_change = saturation_target - current_output
        
        max_weight_idx = np.argmax(np.abs(w_target))
        required_input_change = required_change / w_target[max_weight_idx] if w_target[max_weight_idx] != 0 else float('inf')
        
        return {
            'attack_name': 'ReLU Saturation Attack',
            'current_output': float(current_output),
            'saturation_target': saturation_target,
            'required_change': float(required_change),
            'expected_activation_output': 0.0,
            'modification_strategy': {
                'modify_dimension': int(max_weight_idx),
                'current_value': float(x[max_weight_idx]),
                'required_new_value': float(x[max_weight_idx] + required_input_change),
                'input_change': float(required_input_change)
            }
        }
    
    def _find_minimal_perturbation_attack_gated(self, x, w_gate, w_up, b_gate, b_up):
        """Find minimal perturbation attack for gated MLP"""
        
        current_gate_output = np.dot(w_gate, x) + b_gate
        current_up_output = np.dot(w_up, x) + b_up
        
        # Compare which is easier to zero
        if abs(current_gate_output) < abs(current_up_output):
            target_strategy = 'zero_gate'
            target_change = -current_gate_output
            w_target = w_gate
        else:
            target_strategy = 'zero_up'
            target_change = -current_up_output
            w_target = w_up
        
        # Minimal norm solution
        w_norm_sq = np.dot(w_target, w_target)
        if w_norm_sq > 0:
            delta_x = (target_change / w_norm_sq) * w_target
            perturbation_magnitude = np.linalg.norm(delta_x)
        else:
            delta_x = np.zeros_like(x)
            perturbation_magnitude = float('inf')
        
        return {
            'attack_name': 'Minimal Perturbation Attack',
            'chosen_strategy': target_strategy,
            'perturbation_magnitude': float(perturbation_magnitude),
            'relative_perturbation': float(perturbation_magnitude / np.linalg.norm(x)) if np.linalg.norm(x) > 0 else float('inf'),
            'feasibility': 'high' if perturbation_magnitude < 5.0 else 'medium' if perturbation_magnitude < 50.0 else 'low'
        }
    
    def _find_gated_zero_activation_strategies(self, x, W_gate, W_up, b_gate, b_up, target_channel, activation_fn):
        """High-level strategies for creating zero super activations in gated MLP"""
        
        w_gate_target = W_gate[target_channel]
        w_up_target = W_up[target_channel]
        b_gate_target = b_gate[target_channel] if b_gate is not None else 0.0
        b_up_target = b_up[target_channel] if b_up is not None else 0.0
        
        current_gate_output = np.dot(w_gate_target, x) + b_gate_target
        current_up_output = np.dot(w_up_target, x) + b_up_target
        current_super_activation = self._apply_activation(current_gate_output, activation_fn) * current_up_output
        
        return {
            'current_state': {
                'gate_output': float(current_gate_output),
                'up_output': float(current_up_output),
                'activation_output': float(self._apply_activation(current_gate_output, activation_fn)),
                'super_activation': float(current_super_activation)
            },
            
            'mathematical_conditions_for_zero': [
                {
                    'condition': 'Gate Zeroing',
                    'equation': f'W_gate[{target_channel}] @ x_modified + b_gate = 0',
                    'result': f'{activation_fn.upper()}(0) → super_activation = 0 × up_output = 0',
                    'difficulty': 'medium'
                },
                {
                    'condition': 'Up Zeroing', 
                    'equation': f'W_up[{target_channel}] @ x_modified + b_up = 0',
                    'result': 'super_activation = activation_output × 0 = 0',
                    'difficulty': 'medium'
                }
            ],
            
            'attack_feasibility_ranking': self._rank_attack_feasibility(
                current_gate_output, current_up_output, activation_fn
            )
        }
    
    def _find_standard_zero_activation_strategies(self, x, W_hidden, b_hidden, target_channel, activation_fn):
        """High-level strategies for creating zero super activations in standard MLP"""
        
        w_hidden_target = W_hidden[target_channel]
        b_hidden_target = b_hidden[target_channel] if b_hidden is not None else 0.0
        
        current_hidden_output = np.dot(w_hidden_target, x) + b_hidden_target
        current_super_activation = self._apply_activation(current_hidden_output, activation_fn)
        
        return {
            'current_state': {
                'hidden_output': float(current_hidden_output),
                'super_activation': float(current_super_activation)
            },
            
            'mathematical_conditions_for_zero': [
                {
                    'condition': 'Hidden Zeroing',
                    'equation': f'W_hidden[{target_channel}] @ x_modified + b_hidden = 0',
                    'result': f'{activation_fn.upper()}(0) = 0',
                    'difficulty': 'medium'
                }
            ]
        }
    
    def _rank_attack_feasibility(self, gate_output, up_output, activation_fn):
        """Rank attack strategies by feasibility"""
        
        rankings = []
        
        if abs(up_output) < abs(gate_output):
            rankings.append({
                'rank': 1,
                'strategy': 'Zero Up Projection',
                'reason': f'Current up_output = {up_output:.2f}, smaller magnitude than gate',
                'estimated_difficulty': 'medium'
            })
            rankings.append({
                'rank': 2,
                'strategy': 'Zero Gate Projection',
                'reason': f'Current gate_output = {gate_output:.2f}, larger magnitude',
                'estimated_difficulty': 'medium'
            })
        else:
            rankings.append({
                'rank': 1,
                'strategy': 'Zero Gate Projection',
                'reason': f'Current gate_output = {gate_output:.2f}, smaller magnitude than up',
                'estimated_difficulty': 'medium'
            })
            rankings.append({
                'rank': 2,
                'strategy': 'Zero Up Projection',
                'reason': f'Current up_output = {up_output:.2f}, larger magnitude',
                'estimated_difficulty': 'medium'
            })
        
        if activation_fn.lower() in ['silu', 'relu']:
            rankings.append({
                'rank': 3,
                'strategy': f'{activation_fn.upper()} Saturation',
                'reason': f'Drive gate to negative values where {activation_fn.upper()} ≈ 0',
                'estimated_difficulty': 'high'
            })
        
        return rankings
    
    def _get_top_weights(self, weights, positive=True, top_k=5):
        """Get top k weights (positive or negative)"""
        if positive:
            indices = np.argsort(weights)[-top_k:][::-1]
        else:
            indices = np.argsort(weights)[:top_k]
        
        return [
            {'dimension': int(idx), 'weight': float(weights[idx])}
            for idx in indices
        ]
    
    def _extract_mlp_input_vector(self, input_text: str, target_layer: int) -> torch.Tensor:
        """Extract the exact input vector that feeds into the MLP"""
        
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        input_vector = None
        
        def capture_mlp_input(module, input, output):
            nonlocal input_vector
            mlp_input = input[0]  # [batch, seq, hidden_dim]
            input_vector = mlp_input[0, 0, :].detach().clone()  # First token, all dimensions
        
        # Get the layer and register hook
        layers = self.mlp_handler.registry.find_layers(self.model)
        layer = layers[target_layer]
        mlp_module = self.mlp_handler.registry.find_mlp(layer)
        
        hook = mlp_module.register_forward_hook(capture_mlp_input)
        
        try:
            with torch.no_grad():
                self.model(**tokens)
        finally:
            hook.remove()
        
        return input_vector