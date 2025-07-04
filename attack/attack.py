"""
Gradient-based attack to zero out gate projections using backpropagation to input
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

class GradientGateZeroingAttack:
    """
    Use gradient descent to find input modifications that zero gate outputs
    """
    
    def __init__(self, model, tokenizer, mlp_handler, log_level=logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        self.mlp_handler = mlp_handler
        self.model.eval()  # Freeze model in eval mode
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.logger.info("GradientGateZeroingAttack initialized with shared MLP handler")
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the attack system"""
        logger = logging.getLogger(f"GradientGateZeroingAttack_{id(self)}")
        logger.setLevel(log_level)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def attack_gate_output(self, 
                          input_text: str,
                          target_layer: int,
                          target_channel: int,
                          learning_rate: float = 0.01,
                          max_iterations: int = 1000,
                          target_value: float = 0.0,
                          loss_type: str = 'mse',
                          show_progress: bool = True) -> Dict:
        """
        Attack a specific gate output using gradient descent on input
        
        Args:
            input_text: Original text input
            target_layer: Layer containing the super weight
            target_channel: Channel in gate projection to zero
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum optimization iterations
            target_value: Target value for gate output (usually 0.0)
            loss_type: 'mse', 'mae', or 'huber'
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with attack results and analysis
        """
        
        self.logger.info(f"Starting gradient attack on layer {target_layer}, channel {target_channel}")
        self.logger.info(f"Attack parameters: lr={learning_rate}, max_iter={max_iterations}, target={target_value}")
        
        # Tokenize input
        tokens = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens['input_ids']
        
        # Get initial embeddings
        with torch.no_grad():
            original_embeddings = self.model.get_input_embeddings()(input_ids)
        
        # Create optimizable embeddings
        perturbed_embeddings = original_embeddings.clone().detach().requires_grad_(True)
        
        # Set up optimizer
        optimizer = torch.optim.Adam([perturbed_embeddings], lr=learning_rate)
        
        # Track optimization progress
        history = {
            'iterations': [],
            'losses': [],
            'gate_outputs': [],
            'perturbation_norms': [],
            'gradient_norms': []
        }
        
        original_gate_output = self._get_gate_output(original_embeddings, target_layer, target_channel)
        self.logger.info(f"Original gate output at layer {target_layer}, channel {target_channel}: {original_gate_output:.4f}")
        self.logger.info(f"Target: {target_value:.4f}")
        self.logger.info("Starting optimization...")
        
        best_result = {
            'loss': float('inf'),
            'embeddings': None,
            'gate_output': original_gate_output,
            'iteration': 0
        }
        
        # Create progress bar
        if show_progress:
            pbar = tqdm(range(max_iterations), desc="Gradient Attack", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        else:
            pbar = range(max_iterations)
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # Forward pass to get gate output
            current_gate_output = self._get_gate_output(perturbed_embeddings, target_layer, target_channel)
            
            # Calculate loss
            if loss_type == 'mse':
                loss = F.mse_loss(current_gate_output, torch.tensor(target_value))
            elif loss_type == 'mae':
                loss = F.l1_loss(current_gate_output, torch.tensor(target_value))
            elif loss_type == 'huber':
                loss = F.smooth_l1_loss(current_gate_output, torch.tensor(target_value))
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            # Backward pass
            loss.backward()
            
            # Calculate metrics
            perturbation = perturbed_embeddings - original_embeddings
            perturbation_norm = torch.norm(perturbation).item()
            gradient_norm = torch.norm(perturbed_embeddings.grad).item()
            
            # Track progress
            history['iterations'].append(iteration)
            history['losses'].append(loss.item())
            history['gate_outputs'].append(current_gate_output.item())
            history['perturbation_norms'].append(perturbation_norm)
            history['gradient_norms'].append(gradient_norm)
            
            # Update best result
            if loss.item() < best_result['loss']:
                best_result.update({
                    'loss': loss.item(),
                    'embeddings': perturbed_embeddings.clone().detach(),
                    'gate_output': current_gate_output.item(),
                    'iteration': iteration
                })
            
            # Update progress bar
            if show_progress:
                reduction_pct = abs(original_gate_output - current_gate_output.item()) / abs(original_gate_output) * 100
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Gate': f'{current_gate_output.item():.4f}',
                    'Reduction': f'{reduction_pct:.1f}%',
                    'PertNorm': f'{perturbation_norm:.4f}'
                })
            
            # Progress reporting (less frequent now since we have progress bar)
            if not show_progress and (iteration % 100 == 0 or iteration < 10):
                self.logger.info(f"Iter {iteration:4d}: Loss={loss.item():.6f}, "
                               f"Gate={current_gate_output.item():.4f}, "
                               f"PertNorm={perturbation_norm:.4f}")
            
            # Gradient step
            optimizer.step()
            
            # Early stopping if target achieved
            if abs(current_gate_output.item() - target_value) < 0.01:
                if show_progress:
                    pbar.set_description("🎉 Target achieved!")
                    pbar.close()
                self.logger.info(f"Target achieved at iteration {iteration}!")
                break
            
            # Early stopping if gradient becomes too small
            if gradient_norm < 1e-8:
                if show_progress:
                    pbar.set_description("⚠️ Gradient too small")
                    pbar.close()
                self.logger.warning(f"Gradient too small at iteration {iteration}, stopping.")
                break
        
        # Close progress bar if it wasn't closed by early stopping
        if show_progress and hasattr(pbar, 'close'):
            pbar.close()
        
        # Final analysis
        final_perturbation = best_result['embeddings'] - original_embeddings
        
        result = {
            'success': abs(best_result['gate_output'] - target_value) < 0.1,
            'original_gate_output': original_gate_output,
            'final_gate_output': best_result['gate_output'],
            'target_value': target_value,
            'reduction_achieved': abs(original_gate_output - best_result['gate_output']),
            'reduction_percentage': abs(original_gate_output - best_result['gate_output']) / abs(original_gate_output) * 100,
            'final_loss': best_result['loss'],
            'iterations_used': best_result['iteration'],
            'original_embeddings': original_embeddings,
            'optimized_embeddings': best_result['embeddings'],
            'perturbation': final_perturbation,
            'perturbation_l2_norm': torch.norm(final_perturbation).item(),
            'perturbation_linf_norm': torch.norm(final_perturbation, p=float('inf')).item(),
            'optimization_history': history,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'max_iterations': max_iterations,
                'loss_type': loss_type
            }
        }
        
        # Log final summary
        self.logger.info("=== ATTACK COMPLETE ===")
        self.logger.info(f"Success: {'✅' if result['success'] else '❌'}")
        self.logger.info(f"Final gate output: {result['final_gate_output']:.4f}")
        self.logger.info(f"Reduction achieved: {result['reduction_percentage']:.1f}%")
        self.logger.info(f"Perturbation L2 norm: {result['perturbation_l2_norm']:.4f}")
        self.logger.info(f"Iterations used: {result['iterations_used']}")
        
        return result
    
    def _get_gate_output(self, embeddings: torch.Tensor, target_layer: int, target_channel: int) -> torch.Tensor:
        """
        Forward pass to extract gate output at specific layer and channel
        """
        # Forward pass through layers up to target layer
        hidden_states = embeddings
        
        # Pass through transformer layers up to target layer
        for layer_idx in range(target_layer + 1):
            layer = self.model.model.layers[layer_idx]
            
            # Attention block
            if hasattr(layer, 'self_attn'):
                # Apply input layernorm before attention
                if hasattr(layer, 'input_layernorm'):
                    normed_states = layer.input_layernorm(hidden_states)
                else:
                    normed_states = hidden_states
                
                # Attention needs normed states and attention mask
                attn_output = layer.self_attn(normed_states)[0]
                hidden_states = hidden_states + attn_output
            
            # MLP block - stop at target layer to extract gate output
            if layer_idx == target_layer:
                # Apply pre-MLP normalization
                if hasattr(layer, 'post_attention_layernorm'):
                    mlp_input = layer.post_attention_layernorm(hidden_states)
                elif hasattr(layer, 'input_layernorm'):
                    mlp_input = layer.input_layernorm(hidden_states)
                else:
                    mlp_input = hidden_states
                
                # Get gate output for standard MLP
                return self._extract_gate_output(mlp_input, target_layer, target_channel)
            
            # Complete MLP forward pass for non-target layers
            if hasattr(layer, 'post_attention_layernorm'):
                mlp_input = layer.post_attention_layernorm(hidden_states)
            elif hasattr(layer, 'input_layernorm'):
                mlp_input = layer.input_layernorm(hidden_states)
            else:
                mlp_input = hidden_states
            
            mlp_output = layer.mlp(mlp_input)
            hidden_states = hidden_states + mlp_output
        
        raise RuntimeError("Should not reach here - gate output should be extracted at target layer")
    
    def _extract_gate_output(self, mlp_input: torch.Tensor, target_layer: int, target_channel: int) -> torch.Tensor:
        """
        Extract gate output from standard MLP
        """
        arch_info = self.mlp_handler.get_mlp_architecture(target_layer)
        
        # Skip MoE layers for now
        if arch_info.is_moe:
            raise ValueError(f"MoE layers not supported yet. Layer {target_layer} is MoE.")
        
        if not arch_info.has_gate:
            raise ValueError(f"Layer {target_layer} does not have a gate component")
        
        # Get the gate component
        components = self.mlp_handler.get_mlp_components(target_layer)
        gate_module = components['gate']
        
        # Apply gate projection
        gate_output = gate_module(mlp_input)
        
        # Extract specific channel and average across sequence length
        target_gate_output = gate_output[:, :, target_channel].mean()
        
        return target_gate_output
    
    def validate_attack_target(self, target_layer: int, target_channel: int) -> Dict[str, Any]:
        """
        Validate that the attack target is valid for the model architecture
        """
        self.logger.debug(f"Validating attack target: layer {target_layer}, channel {target_channel}")
        
        try:
            arch_info = self.mlp_handler.get_mlp_architecture(target_layer)
        except ValueError as e:
            error_msg = f"Invalid layer index: {e}"
            self.logger.error(error_msg)
            return {
                'valid': False,
                'error': error_msg
            }
        
        validation_info = {
            'valid': True,
            'layer_type': arch_info.architecture_type.value,
            'has_gate': arch_info.has_gate,
            'is_moe': arch_info.is_moe,
            'activation_function': arch_info.activation_function
        }
        
        # Skip MoE layers for now
        if arch_info.is_moe:
            error_msg = f"MoE layers not supported yet. Layer {target_layer} is MoE."
            self.logger.error(error_msg)
            validation_info['valid'] = False
            validation_info['error'] = error_msg
            return validation_info
        
        if not arch_info.has_gate:
            error_msg = f"Layer {target_layer} does not have a gate component"
            self.logger.error(error_msg)
            validation_info['valid'] = False
            validation_info['error'] = error_msg
            return validation_info
        
        # Check if target channel is valid
        components = self.mlp_handler.get_mlp_components(target_layer)
        gate_module = components['gate']
        
        gate_size = gate_module.weight.shape[0]
        if target_channel >= gate_size:
            error_msg = f"Target channel {target_channel} >= gate size {gate_size}"
            self.logger.error(error_msg)
            validation_info['valid'] = False
            validation_info['error'] = error_msg
        
        validation_info['gate_size'] = gate_size
        
        if validation_info['valid']:
            self.logger.debug(f"Attack target validation successful: {validation_info}")
        
        return validation_info
    
    def validate_super_weight_target(self, super_weight) -> Dict[str, Any]:
        """
        Validate that a super weight can be targeted by this attack
        """
        self.logger.debug(f"Validating super weight target: {super_weight}")
        return self.validate_attack_target(super_weight.layer, super_weight.col)
    
    def attack_super_weight(self, super_weight, input_text: str, **kwargs) -> Dict:
        """
        Convenience method to attack a super weight directly
        """
        self.logger.info(f"Attacking super weight: {super_weight}")
        
        # Validate target
        validation = self.validate_super_weight_target(super_weight)
        if not validation['valid']:
            error_msg = f"Invalid super weight target: {validation['error']}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Super weight target validated successfully")
        
        # Run attack
        return self.attack_gate_output(
            input_text=input_text,
            target_layer=super_weight.layer,
            target_channel=super_weight.col,
            **kwargs
        )
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture for attack planning
        """
        self.logger.info("Generating architecture summary for attack planning")
        
        summary = {
            'num_layers': len(self.mlp_handler.layers),
            'layers': {}
        }
        
        attackable_layers = 0
        for layer_idx in range(len(self.mlp_handler.layers)):
            arch_info = self.mlp_handler.get_mlp_architecture(layer_idx)
            
            layer_summary = {
                'type': arch_info.architecture_type.value,
                'has_gate': arch_info.has_gate,
                'is_moe': arch_info.is_moe,
                'activation': arch_info.activation_function
            }
            
            if arch_info.has_gate and not arch_info.is_moe:
                components = self.mlp_handler.get_mlp_components(layer_idx)
                gate_module = components['gate']
                layer_summary['gate_size'] = gate_module.weight.shape[0]
                layer_summary['attackable'] = True
                attackable_layers += 1
            else:
                layer_summary['attackable'] = False
            
            summary['layers'][layer_idx] = layer_summary
        
        summary['attackable_layers'] = attackable_layers
        
        self.logger.info(f"Architecture summary: {attackable_layers}/{len(self.mlp_handler.layers)} layers attackable")
        
        return summary
