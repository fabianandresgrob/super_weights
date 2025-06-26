import torch
import logging
from typing import Dict, List, Set, Union, Tuple
from contextlib import contextmanager

from detection.super_weight import SuperWeight


class SuperWeightManager:
    """
    Manages super weight modifications with automatic backup/restore.
    Provides safe manipulation of super weights with context managers.
    """
    
    def __init__(self, model, log_level=logging.INFO):
        self.model = model
        self.original_values: Dict[SuperWeight, torch.Tensor] = {}
        self.current_scales: Dict[SuperWeight, float] = {}
        self.currently_modified: Set[SuperWeight] = set()
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Model architecture info
        self._analyze_model_architecture()
    
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
    
    def _analyze_model_architecture(self):
        """Analyze the model architecture to understand layer structure"""
        if hasattr(self.model, "model"):
            self.model_body = self.model.model
        else:
            self.model_body = self.model
        
        if hasattr(self.model_body, "layers"):
            self.layers_attr = "layers"
        elif hasattr(self.model_body, "encoder"):
            self.layers_attr = "encoder.layer"
        else:
            raise ValueError("Unsupported model architecture")
        
        self.layers = eval(f"self.model_body.{self.layers_attr}")
    
    def _get_mlp_component_info(self, layer) -> Tuple[str, str, torch.nn.Module]:
        """Get MLP component information for a layer"""
        if hasattr(layer, "mlp"):
            base = "mlp"
            if hasattr(layer.mlp, "down_proj"):
                down_name = "down_proj"
                module = layer.mlp.down_proj
            elif hasattr(layer.mlp, "c_proj"):
                down_name = "c_proj"
                module = layer.mlp.c_proj
            elif hasattr(layer.mlp, "dense_h_to_4h") and hasattr(layer.mlp, "dense_4h_to_h"):
                down_name = "dense_4h_to_h"
                module = layer.mlp.dense_4h_to_h
            else:
                down_name = "fc2"
                module = layer.mlp.fc2
        elif hasattr(layer, "feed_forward"):
            base = "feed_forward"
            down_name = "output_dense"
            module = layer.feed_forward.output_dense
        else:
            raise ValueError(f"Unsupported MLP structure in layer: {dir(layer)}")
        
        return base, down_name, module
    
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
                layer = self.layers[sw.layer]
                _, _, module = self._get_mlp_component_info(layer)
                weight_matrix = module.weight
                
                if weight_matrix.device.type == 'meta':
                    self.logger.error(f"Weight matrix for layer {sw.layer} is on meta device. Skipping.")
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
                    layer = self.layers[sw.layer]
                    _, _, module = self._get_mlp_component_info(layer)
                    module.weight.data[sw.row, sw.column] = self.original_values[sw]
                    
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
            layer = self.layers[super_weight.layer]
            _, _, module = self._get_mlp_component_info(layer)
            return module.weight[super_weight.row, super_weight.column].clone()
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
    
    def __enter__(self):
        """Context manager entry - for backwards compatibility"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore all modifications"""
        self.restore_all()