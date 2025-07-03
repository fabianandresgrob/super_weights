import torch
import logging
from typing import Dict, List, Set, Union, Tuple, Optional
from contextlib import contextmanager

from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalMLPHandler


class SuperWeightManager:
    """
    Manages super weight modifications with automatic backup/restore.
    Provides safe manipulation of super weights with context managers.
    """
    
    def __init__(self, model, mlp_handler: UniversalMLPHandler, log_level=logging.INFO):
        self.model = model
        self.mlp_handler = mlp_handler
        self.original_values: Dict[SuperWeight, torch.Tensor] = {}
        self.current_scales: Dict[SuperWeight, float] = {}
        self.currently_modified: Set[SuperWeight] = set()
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        self.logger.info("SuperWeightManager initialized with shared MLP handler")
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the manager"""
        logger = logging.getLogger(f"SuperWeightManager_{id(self)}")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_weight_module(self, super_weight: SuperWeight) -> torch.nn.Module:
        """Get the weight module for a super weight using the MLP handler"""
        layer_idx = super_weight.layer
        
        # Check if this is an MoE layer
        if self.mlp_handler.is_moe_layer(layer_idx):
            # For MoE, we need to parse the component path
            if hasattr(super_weight, 'component') and super_weight.component:
                # Parse component path like "experts.0.down_proj"
                parts = super_weight.component.split('.')
                if len(parts) >= 3 and parts[0] == 'experts':
                    expert_idx = int(parts[1])
                    component_name = parts[2]
                    
                    # Get expert components
                    expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_idx)
                    
                    # Find the right component
                    for comp_type, module in expert_components.items():
                        arch_info = self.mlp_handler.get_mlp_architecture(layer_idx)
                        expert_info = arch_info.moe_info.experts[expert_idx]
                        comp_info = expert_info.components[comp_type]
                        
                        if comp_info.component_name == component_name:
                            return module
                    
                    raise ValueError(f"Component {component_name} not found in expert {expert_idx}")
                else:
                    raise ValueError(f"Invalid MoE component path: {super_weight.component}")
            else:
                raise ValueError(f"MoE super weight missing component information: {super_weight}")
        else:
            # Regular MLP layer
            components = self.mlp_handler.get_mlp_components(layer_idx)
            
            # Find the output/down projection component
            if 'down' in components:
                return components['down']
            elif 'output' in components:
                return components['output']
            else:
                raise ValueError(f"No output projection found in layer {layer_idx}")
    
    def scale_super_weights(self, super_weights: List[SuperWeight], scale_factor: float) -> bool:
        """
        Scale super weights by a given factor.
        
        Args:
            super_weights: List of SuperWeight objects to scale
            scale_factor: Factor to scale by (0.0 = zero, 1.0 = no change, 2.0 = double)
            
        Returns:
            True if all weights were successfully scaled
        """
        success_count = 0
        
        for sw in super_weights:
            try:
                # Get the weight module using MLP handler
                weight_module = self._get_weight_module(sw)
                weight_matrix = weight_module.weight
                
                if weight_matrix.device.type == 'meta':
                    self.logger.error(f"Weight matrix for {sw} is on meta device. Skipping.")
                    continue
                
                # Backup original value if not already done
                if sw not in self.original_values:
                    original_value = weight_matrix[sw.row, sw.column].clone().detach()
                    self.original_values[sw] = original_value
                    sw.original_value = original_value
                
                # Apply scaling
                new_value = self.original_values[sw] * scale_factor
                weight_matrix.data[sw.row, sw.column] = new_value
                
                # Track current state
                self.current_scales[sw] = scale_factor
                self.currently_modified.add(sw)
                
                success_count += 1
                self.logger.debug(f"Scaled {sw} by {scale_factor:.3f}: "
                                f"{self.original_values[sw].item():.4f} -> {new_value.item():.4f}")
                
            except Exception as e:
                self.logger.error(f"Error scaling weight {sw}: {e}")
        
        self.logger.info(f"Successfully scaled {success_count}/{len(super_weights)} super weights by {scale_factor:.3f}")
        return success_count == len(super_weights)
    
    def zero_super_weights(self, super_weights: List[SuperWeight]) -> bool:
        """Zero out super weights (convenience method for scale_factor=0.0)"""
        return self.scale_super_weights(super_weights, 0.0)
    
    def restore_super_weights(self, super_weights: List[SuperWeight]) -> bool:
        """
        Restore specific super weights to their original values.
        
        Args:
            super_weights: List of SuperWeight objects to restore
            
        Returns:
            True if all weights were successfully restored
        """
        success_count = 0
        
        for sw in super_weights:
            try:
                if sw in self.original_values:
                    weight_module = self._get_weight_module(sw)
                    weight_module.weight.data[sw.row, sw.column] = self.original_values[sw]
                    
                    # Update tracking
                    self.current_scales.pop(sw, None)
                    self.currently_modified.discard(sw)
                    
                    success_count += 1
                    self.logger.debug(f"Restored {sw} to {self.original_values[sw].item():.4f}")
                else:
                    self.logger.warning(f"No original value stored for {sw}")
                    
            except Exception as e:
                self.logger.error(f"Failed to restore {sw}: {e}")
        
        self.logger.info(f"Restored {success_count}/{len(super_weights)} weights")
        return success_count == len(super_weights)
    
    def restore_all(self) -> bool:
        """Restore all currently modified super weights"""
        if not self.currently_modified:
            self.logger.info("No weights to restore")
            return True
        
        weights_to_restore = list(self.currently_modified)
        return self.restore_super_weights(weights_to_restore)
    
    def is_modified(self, super_weight: SuperWeight) -> bool:
        """Check if a super weight is currently modified"""
        return super_weight in self.currently_modified
    
    def is_zeroed(self, super_weight: SuperWeight) -> bool:
        """Check if a super weight is currently zeroed"""
        return (super_weight in self.current_scales and 
                abs(self.current_scales[super_weight]) < 1e-6)
    
    def get_current_scale(self, super_weight: SuperWeight) -> float:
        """Get the current scale factor for a super weight"""
        return self.current_scales.get(super_weight, 1.0)
    
    def get_current_value(self, super_weight: SuperWeight) -> torch.Tensor:
        """Get the current value of a super weight"""
        try:
            weight_module = self._get_weight_module(super_weight)
            return weight_module.weight[super_weight.row, super_weight.column].clone()
        except Exception as e:
            self.logger.error(f"Error getting current value for {super_weight}: {e}")
            return None
    
    @contextmanager
    def temporary_scale(self, super_weights: List[SuperWeight], scale_factor: float):
        """
        Context manager for temporary scaling of super weights.
        Automatically restores original values on exit.
        """
        # Store which weights were already modified before we started
        previously_modified = {sw for sw in super_weights if sw in self.currently_modified}
        newly_modified = [sw for sw in super_weights if sw not in previously_modified]
        
        try:
            self.scale_super_weights(super_weights, scale_factor)
            yield
        finally:
            # Only restore weights that we newly modified
            if newly_modified:
                self.restore_super_weights(newly_modified)
    
    @contextmanager
    def temporary_zero(self, super_weights: List[SuperWeight]):
        """Context manager for temporary zeroing of super weights"""
        with self.temporary_scale(super_weights, 0.0):
            yield
    
    def get_modification_summary(self) -> Dict:
        """Get a summary of current modifications"""
        return {
            'total_modified': len(self.currently_modified),
            'total_zeroed': sum(1 for sw in self.currently_modified if self.is_zeroed(sw)),
            'scale_distribution': {
                scale: len([sw for sw, s in self.current_scales.items() if abs(s - scale) < 1e-6])
                for scale in sorted(set(self.current_scales.values()))
            },
            'modified_weights': [
                {
                    'super_weight': str(sw),
                    'original_value': self.original_values[sw].item(),
                    'current_scale': self.current_scales[sw],
                    'current_value': self.get_current_value(sw).item() if self.get_current_value(sw) is not None else None
                }
                for sw in self.currently_modified
            ]
        }
    
    def validate_super_weights(self, super_weights: List[SuperWeight]) -> List[SuperWeight]:
        """
        Validate that super weights are compatible with the current model architecture.
        
        Args:
            super_weights: List of SuperWeight objects to validate
            
        Returns:
            List of valid SuperWeight objects
        """
        valid_weights = []
        
        for sw in super_weights:
            try:
                # Try to get the weight module - this will validate the architecture
                weight_module = self._get_weight_module(sw)
                weight_shape = weight_module.weight.shape
                
                # Check if coordinates are within bounds
                if 0 <= sw.row < weight_shape[0] and 0 <= sw.column < weight_shape[1]:
                    valid_weights.append(sw)
                else:
                    self.logger.warning(f"Super weight {sw} coordinates out of bounds for shape {weight_shape}")
                    
            except Exception as e:
                self.logger.warning(f"Super weight {sw} is invalid: {e}")
        
        self.logger.info(f"Validated {len(valid_weights)}/{len(super_weights)} super weights")
        return valid_weights
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore all modifications"""
        self.restore_all()