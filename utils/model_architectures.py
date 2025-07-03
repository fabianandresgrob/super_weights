"""
Model architecture detection and utilities for super weight analysis.
Provides model-agnostic interfaces for detecting and working with different transformer architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
import logging


class MLPArchitectureType(Enum):
    """Supported MLP architecture types"""
    GATED_MLP = "gated_mlp"  # Gate + Up + Down (e.g., OLMo, LLaMA, Mistral)
    STANDARD_MLP = "standard_mlp"  # Hidden + Output (e.g., GPT-2)
    FUSED_GATED_MLP = "fused_gated_mlp"  # Fused Gate+Up + Down (e.g., Phi-4)
    MOE_MIXTRAL = "moe_mixtral"  # Mixtral-style MoE
    MOE_SWITCH = "moe_switch"  # Switch Transformer style
    MOE_GLM = "moe_glm"  # GLM-style MoE
    UNKNOWN = "unknown"


@dataclass
class MLPComponentInfo:
    """Information about MLP components"""
    component_name: str  # Actual attribute name in the model
    component_type: str  # Standardized type (gate, up, down, hidden, output)
    has_bias: bool
    weight_shape: Tuple[int, int]


@dataclass
class MoEExpertInfo:
    """Information about a single expert in MoE layer"""
    expert_id: int
    expert_path: str  # e.g., "experts.0"
    components: Dict[str, 'MLPComponentInfo']
    architecture_type: MLPArchitectureType


@dataclass 
class MoEArchitectureInfo:
    """Information about MoE architecture"""
    num_experts: int
    experts_per_token: int
    routing_method: str  # "top_k", "switch", etc.
    experts: List[MoEExpertInfo]
    router_path: str  # e.g., "gate"
    has_shared_expert: bool = False
    shared_expert_path: Optional[str] = None


@dataclass 
class MLPArchitectureInfo:
    """Complete information about an MLP architecture"""
    architecture_type: MLPArchitectureType
    activation_function: str
    components: Dict[str, MLPComponentInfo]
    has_gate: bool = False
    intermediate_size: Optional[int] = None
    hidden_size: Optional[int] = None
    
    # MoE-specific fields
    is_moe: bool = False
    moe_info: Optional[MoEArchitectureInfo] = None
    
    def __repr__(self):
        if self.is_moe:
            return f"MLPArchitectureInfo(type={self.architecture_type.value}, moe_experts={self.moe_info.num_experts if self.moe_info else 0})"
        return f"MLPArchitectureInfo(type={self.architecture_type.value}, activation={self.activation_function}, components={list(self.components.keys())})"


class ModelArchitectureRegistry:
    """Registry for different model architectures and their patterns"""
    
    # Common naming patterns for different model families
    MLP_COMPONENT_PATTERNS = {
        'gate': ['gate_proj', 'gate_projection', 'w1', 'gate_linear'],
        'up': ['up_proj', 'up_projection', 'w3', 'up_linear'],
        'down': ['down_proj', 'down_projection', 'w2', 'dense_4h_to_h', 'c_proj'],
        'hidden': ['c_fc', 'hidden', 'fc1', 'linear1', 'dense_h_to_4h'],
        'output': ['c_proj', 'output', 'fc2', 'linear2', 'dense_4h_to_h'],
        'gate_up_fused': ['gate_up_proj', 'gate_up_projection', 'w13']  # Fused gate+up for Phi-4
    }
    
    # Model path patterns for accessing layers
    LAYER_PATH_PATTERNS = [
        "model.layers",
        "layers", 
        "transformer.h",
        "transformer.layers",
        "model.decoder.layers",
        "decoder.layers",
        "encoder.layer",
        "model.encoder.layer"
    ]
    
    # MLP path patterns within layers
    MLP_PATH_PATTERNS = [
        "mlp",
        "feed_forward", 
        "mlp_block",
        "ffn"
    ]
    
    # Known activation functions by model family
    ACTIVATION_PATTERNS = {
        'silu': ['silu', 'swish'],
        'gelu': ['gelu', 'gelu_new'],
        'relu': ['relu'],
        'gelu_fast': ['gelu_fast'],
        'geglu': ['geglu'],
        'swiglu': ['swiglu']
    }
    
    # MoE-specific patterns
    MOE_PATH_PATTERNS = [
        "block_sparse_moe",  # Mixtral
        "moe",
        "experts_layer",
        "mixture_of_experts"
    ]
    
    MOE_EXPERT_PATTERNS = [
        "experts",
        "expert_layers"
    ]
    
    MOE_ROUTER_PATTERNS = [
        "gate",
        "router", 
        "switch"
    ]
    
    @classmethod
    def get_component_name(cls, mlp_module: nn.Module, component_type: str) -> Optional[str]:
        """Get the actual component name for a given component type"""
        if component_type not in cls.MLP_COMPONENT_PATTERNS:
            return None
        
        for pattern in cls.MLP_COMPONENT_PATTERNS[component_type]:
            if hasattr(mlp_module, pattern):
                return pattern
        return None
    
    @classmethod
    def find_layers(cls, model: nn.Module) -> Optional[List[nn.Module]]:
        """Find the layers container in a model"""
        for pattern in cls.LAYER_PATH_PATTERNS:
            try:
                layers = model
                for attr in pattern.split('.'):
                    layers = getattr(layers, attr)
                return layers
            except AttributeError:
                continue
        return None
    
    @classmethod
    def find_mlp(cls, layer: nn.Module) -> Optional[nn.Module]:
        """Find the MLP module within a layer"""
        for pattern in cls.MLP_PATH_PATTERNS:
            if hasattr(layer, pattern):
                return getattr(layer, pattern)
        return None
    
    @classmethod
    def find_moe_module(cls, layer: nn.Module) -> Optional[nn.Module]:
        """Find the MoE module within a layer"""
        for pattern in cls.MOE_PATH_PATTERNS:
            if hasattr(layer, pattern):
                return getattr(layer, pattern)
        return None
    
    @classmethod
    def find_experts(cls, moe_module: nn.Module) -> Optional[nn.ModuleList]:
        """Find experts within MoE module"""
        for pattern in cls.MOE_EXPERT_PATTERNS:
            if hasattr(moe_module, pattern):
                experts = getattr(moe_module, pattern)
                if isinstance(experts, (nn.ModuleList, list)):
                    return experts
        return None


class UniversalMLPHandler:
    """Universal handler for MLP components across different architectures"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.registry = ModelArchitectureRegistry()
        self.logger = logging.getLogger(f"UniversalMLPHandler_{id(self)}")
        
        # Get all layers once
        self.layers = self.registry.find_layers(self.model)
        if self.layers is None:
            raise ValueError("Could not find layers in model")
        
        # Detect and cache architecture for all layers upfront
        self._layer_architectures = self._detect_all_layer_architectures()
        
        self.logger.info(f"Detected architectures for {len(self._layer_architectures)} layers")
        self._log_architecture_summary()
    
    def _detect_all_layer_architectures(self) -> Dict[int, MLPArchitectureInfo]:
        """Detect architecture for all layers at initialization"""
        architectures = {}
        
        for layer_idx, layer in enumerate(self.layers):
            try:
                # Check if this is an MoE layer first
                moe_module = self.registry.find_moe_module(layer)
                if moe_module is not None:
                    arch_info = self._detect_moe_architecture(layer, moe_module)
                else:
                    # Regular MLP layer
                    mlp_module = self.registry.find_mlp(layer)
                    if mlp_module is not None:
                        arch_info = self._detect_mlp_architecture(mlp_module)
                    else:
                        # Unknown layer structure
                        arch_info = self._create_unknown_architecture()
                
                architectures[layer_idx] = arch_info
                
            except Exception as e:
                self.logger.warning(f"Failed to detect architecture for layer {layer_idx}: {e}")
                architectures[layer_idx] = self._create_unknown_architecture()
        
        return architectures
    
    def _create_unknown_architecture(self) -> MLPArchitectureInfo:
        """Create architecture info for unknown/unsupported layers"""
        return MLPArchitectureInfo(
            architecture_type=MLPArchitectureType.UNKNOWN,
            activation_function='unknown',
            components={},
            has_gate=False,
            intermediate_size=None,
            hidden_size=None,
            is_moe=False
        )
    
    def _log_architecture_summary(self):
        """Log summary of detected architectures"""
        arch_counts = {}
        moe_layers = []
        
        for layer_idx, arch_info in self._layer_architectures.items():
            arch_type = arch_info.architecture_type.value
            arch_counts[arch_type] = arch_counts.get(arch_type, 0) + 1
            
            if arch_info.is_moe:
                moe_layers.append(layer_idx)
        
        self.logger.info("Architecture summary:")
        for arch_type, count in arch_counts.items():
            self.logger.info(f"  {arch_type}: {count} layers")
        
        if moe_layers:
            self.logger.info(f"  MoE layers: {moe_layers}")
    
    # Simplified property methods
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer is MoE"""
        return self._layer_architectures.get(layer_idx, self._create_unknown_architecture()).is_moe
    
    def get_layer_architecture_type(self, layer_idx: int) -> MLPArchitectureType:
        """Get the architecture type for a specific layer"""
        return self._layer_architectures.get(layer_idx, self._create_unknown_architecture()).architecture_type
    
    def get_mlp_architecture(self, layer_idx: int) -> MLPArchitectureInfo:
        """Get architecture information for MLP at specified layer"""
        if layer_idx not in self._layer_architectures:
            raise ValueError(f"Layer index {layer_idx} out of range (model has {len(self.layers)} layers)")
        
        return self._layer_architectures[layer_idx]
    
    def get_mlp_components(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get all MLP components for a layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        
        if arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is MoE - use get_expert_components instead")
        
        mlp_module = self.registry.find_mlp(self.layers[layer_idx])
        if mlp_module is None:
            raise ValueError(f"Could not find MLP module in layer {layer_idx}")
        
        components = {}
        for comp_type, comp_info in arch_info.components.items():
            components[comp_type] = getattr(mlp_module, comp_info.component_name)
        
        return components
    
    def _detect_mlp_architecture(self, mlp_module: nn.Module) -> MLPArchitectureInfo:
        """Detect the architecture type and components of an MLP module"""
        components = {}
        
        # Check for fused gate+up architecture (Phi-4 style)
        gate_up_name = self.registry.get_component_name(mlp_module, 'gate_up_fused')
        down_name = self.registry.get_component_name(mlp_module, 'down')
        
        if gate_up_name and down_name:
            # Fused gated MLP architecture (Phi-4)
            gate_up_module = getattr(mlp_module, gate_up_name)
            down_module = getattr(mlp_module, down_name)
            
            components['gate_up_fused'] = MLPComponentInfo(
                component_name=gate_up_name,
                component_type='gate_up_fused',
                has_bias=gate_up_module.bias is not None,
                weight_shape=tuple(gate_up_module.weight.shape)
            )
            components['down'] = MLPComponentInfo(
                component_name=down_name,
                component_type='down',
                has_bias=down_module.bias is not None,
                weight_shape=tuple(down_module.weight.shape)
            )
            
            arch_type = MLPArchitectureType.FUSED_GATED_MLP
            has_gate = True
            # For fused gate+up, intermediate size is half the output dimension
            intermediate_size = gate_up_module.weight.shape[0] // 2
            hidden_size = gate_up_module.weight.shape[1]
            
        else:
            # Check for separate gated architecture components
            gate_name = self.registry.get_component_name(mlp_module, 'gate')
            up_name = self.registry.get_component_name(mlp_module, 'up') 
            down_name = self.registry.get_component_name(mlp_module, 'down')
            
            # Check for standard architecture components
            hidden_name = self.registry.get_component_name(mlp_module, 'hidden')
            output_name = self.registry.get_component_name(mlp_module, 'output')
            
            if gate_name and up_name and down_name:
                # Standard gated MLP architecture
                gate_module = getattr(mlp_module, gate_name)
                up_module = getattr(mlp_module, up_name)
                down_module = getattr(mlp_module, down_name)
                
                components['gate'] = MLPComponentInfo(
                    component_name=gate_name,
                    component_type='gate',
                    has_bias=gate_module.bias is not None,
                    weight_shape=tuple(gate_module.weight.shape)
                )
                components['up'] = MLPComponentInfo(
                    component_name=up_name,
                    component_type='up', 
                    has_bias=up_module.bias is not None,
                    weight_shape=tuple(up_module.weight.shape)
                )
                components['down'] = MLPComponentInfo(
                    component_name=down_name,
                    component_type='down',
                    has_bias=down_module.bias is not None,
                    weight_shape=tuple(down_module.weight.shape)
                )
                
                arch_type = MLPArchitectureType.GATED_MLP
                has_gate = True
                intermediate_size = gate_module.weight.shape[0]
                hidden_size = gate_module.weight.shape[1]
                
            elif hidden_name and output_name:
                # Standard MLP architecture
                hidden_module = getattr(mlp_module, hidden_name)
                output_module = getattr(mlp_module, output_name)
                
                components['hidden'] = MLPComponentInfo(
                    component_name=hidden_name,
                    component_type='hidden',
                    has_bias=hidden_module.bias is not None,
                    weight_shape=tuple(hidden_module.weight.shape)
                )
                components['output'] = MLPComponentInfo(
                    component_name=output_name,
                    component_type='output',
                    has_bias=output_module.bias is not None,
                    weight_shape=tuple(output_module.weight.shape)
                )
                
                arch_type = MLPArchitectureType.STANDARD_MLP
                has_gate = False
                intermediate_size = hidden_module.weight.shape[0]
                hidden_size = hidden_module.weight.shape[1]
                
            else:
                # Unknown architecture
                arch_type = MLPArchitectureType.UNKNOWN
                has_gate = False
                intermediate_size = None
                hidden_size = None
        
        # Detect activation function
        activation_fn = self._detect_activation_function(mlp_module)
        
        return MLPArchitectureInfo(
            architecture_type=arch_type,
            activation_function=activation_fn,
            components=components,
            has_gate=has_gate,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            is_moe=False
        )
    
    def _detect_moe_architecture(self, layer: nn.Module, moe_module: nn.Module) -> MLPArchitectureInfo:
        """Detect MoE architecture and expert structure"""
        # Find experts
        experts = self.registry.find_experts(moe_module)
        if experts is None:
            return self._create_unknown_architecture()
        
        num_experts = len(experts)
        expert_infos = []
        
        # Analyze each expert
        for expert_idx, expert in enumerate(experts):
            expert_arch = self._detect_mlp_architecture(expert)
            expert_info = MoEExpertInfo(
                expert_id=expert_idx,
                expert_path=f"experts.{expert_idx}",
                components=expert_arch.components,
                architecture_type=expert_arch.architecture_type
            )
            expert_infos.append(expert_info)
        
        # Create MoE info
        moe_info = MoEArchitectureInfo(
            num_experts=num_experts,
            experts_per_token=2,  # Default, could be detected from config
            routing_method="top_k",  # Default
            experts=expert_infos,
            router_path="gate"  # Default
        )
        
        # Use the first expert's architecture as the base
        base_arch = expert_infos[0].architecture_type if expert_infos else MLPArchitectureType.UNKNOWN
        
        return MLPArchitectureInfo(
            architecture_type=base_arch,
            activation_function=self._detect_activation_function(experts[0]) if experts else 'unknown',
            components={},  # MoE components are per-expert
            has_gate=base_arch in [MLPArchitectureType.GATED_MLP, MLPArchitectureType.FUSED_GATED_MLP],
            intermediate_size=None,
            hidden_size=None,
            is_moe=True,
            moe_info=moe_info
        )
    
    def extract_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract weight matrices from MLP layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        
        if arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is MoE - use expert-specific methods")
        
        # Handle fused gate+up architecture
        if arch_info.architecture_type == MLPArchitectureType.FUSED_GATED_MLP:
            return self._extract_fused_weights(layer_idx)
        
        # Standard weight extraction
        components = self.get_mlp_components(layer_idx)
        weights = {}
        for comp_type, module in components.items():
            if hasattr(module, 'weight'):
                weights[comp_type] = module.weight.detach()
        
        return weights
    
    def _extract_fused_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract weights from fused gate+up architecture"""
        components = self.get_mlp_components(layer_idx)
        weights = {}
        
        # Extract the fused gate+up weight and split it
        gate_up_module = components['gate_up_fused']
        gate_up_weight = gate_up_module.weight.detach()
        
        # Split the fused weight into gate and up components
        intermediate_size = gate_up_weight.shape[0] // 2
        weights['gate'] = gate_up_weight[:intermediate_size]
        weights['up'] = gate_up_weight[intermediate_size:]
        
        # Extract down weight normally
        down_module = components['down']
        weights['down'] = down_module.weight.detach()
        
        return weights
    
    def extract_biases(self, layer_idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Extract bias vectors from MLP layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        
        if arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is MoE - use expert-specific methods")
        
        # Handle fused gate+up architecture
        if arch_info.architecture_type == MLPArchitectureType.FUSED_GATED_MLP:
            return self._extract_fused_biases(layer_idx)
        
        # Standard bias extraction
        components = self.get_mlp_components(layer_idx)
        biases = {}
        for comp_type, module in components.items():
            if hasattr(module, 'bias'):
                bias = module.bias
                biases[comp_type] = bias.detach() if bias is not None else None
            else:
                biases[comp_type] = None
        
        return biases
    
    def _extract_fused_biases(self, layer_idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Extract biases from fused gate+up architecture"""
        components = self.get_mlp_components(layer_idx)
        biases = {}
        
        # Extract the fused gate+up bias and split it
        gate_up_module = components['gate_up_fused']
        if gate_up_module.bias is not None:
            gate_up_bias = gate_up_module.bias.detach()
            intermediate_size = gate_up_bias.shape[0] // 2
            biases['gate'] = gate_up_bias[:intermediate_size]
            biases['up'] = gate_up_bias[intermediate_size:]
        else:
            biases['gate'] = None
            biases['up'] = None
        
        # Extract down bias normally
        down_module = components['down']
        biases['down'] = down_module.bias.detach() if down_module.bias is not None else None
        
        return biases
    
    # Simplified MoE methods
    def get_moe_experts(self, layer_idx: int) -> List[nn.Module]:
        """Get all expert modules for an MoE layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        if not arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")
        
        layer = self.layers[layer_idx]
        moe_module = self.registry.find_moe_module(layer)
        experts = self.registry.find_experts(moe_module)
        
        if experts is None:
            raise ValueError(f"Could not find experts in MoE layer {layer_idx}")
        
        return list(experts)
    
    def get_expert_components(self, layer_idx: int, expert_idx: int) -> Dict[str, nn.Module]:
        """Get components for a specific expert"""
        arch_info = self.get_mlp_architecture(layer_idx)
        if not arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")
        
        if expert_idx >= len(arch_info.moe_info.experts):
            raise ValueError(f"Expert index {expert_idx} out of range")
        
        experts = self.get_moe_experts(layer_idx)
        expert = experts[expert_idx]
        
        # Get expert architecture info
        expert_arch = arch_info.moe_info.experts[expert_idx]
        
        components = {}
        for comp_type, comp_info in expert_arch.components.items():
            components[comp_type] = getattr(expert, comp_info.component_name)
        
        return components
    
    def _detect_activation_function(self, mlp_module: nn.Module) -> str:
        """Detect activation function used in MLP"""
        # Look for activation function attributes or modules
        activation_candidates = []
        
        # Check for activation function as an attribute
        for attr_name in dir(mlp_module):
            if 'act' in attr_name.lower() or 'activation' in attr_name.lower():
                attr = getattr(mlp_module, attr_name)
                if callable(attr) or isinstance(attr, nn.Module):
                    activation_candidates.append((attr_name, attr))
        
        # If we found activation functions, try to identify them
        for name, activation in activation_candidates:
            if hasattr(activation, '__class__'):
                class_name = activation.__class__.__name__.lower()
                
                # Map common activation function class names
                if 'silu' in class_name or 'swish' in class_name:
                    return 'silu'
                elif 'gelu' in class_name:
                    if 'new' in class_name:
                        return 'gelu_new'
                    elif 'fast' in class_name:
                        return 'gelu_fast'
                    else:
                        return 'gelu'
                elif 'relu' in class_name:
                    return 'relu'
                elif 'geglu' in class_name:
                    return 'geglu'
                elif 'swiglu' in class_name:
                    return 'swiglu'
        
        # Check for string activation functions in model config
        if hasattr(mlp_module, 'activation_fn'):
            activation_fn = mlp_module.activation_fn
            if isinstance(activation_fn, str):
                return activation_fn
        
        # Try to infer from model type patterns
        model_name = getattr(self.model, 'name_or_path', '').lower()
        
        if any(pattern in model_name for pattern in ['llama', 'mistral', 'olmo']):
            return 'silu'
        elif any(pattern in model_name for pattern in ['gpt', 'bert']):
            return 'gelu'
        elif 'phi' in model_name:
            return 'gelu_new'
        
        # Default fallback
        return 'silu'


def get_model_architecture_info(model: nn.Module, layer_idx: int = 0) -> MLPArchitectureInfo:
    """Convenience function to get architecture info for a specific layer"""
    handler = UniversalMLPHandler(model)
    return handler.get_mlp_architecture(layer_idx)


def is_gated_architecture(model: nn.Module, layer_idx: int = 0) -> bool:
    """Check if model uses gated MLP architecture"""
    try:
        arch_info = get_model_architecture_info(model, layer_idx)
        return arch_info.architecture_type == MLPArchitectureType.GATED_MLP
    except:
        return False


def create_mlp_handler(model: nn.Module) -> UniversalMLPHandler:
    """Factory function to create a universal MLP handler"""
    return UniversalMLPHandler(model)


# Add convenience functions for MoE
def is_moe_model(model: nn.Module) -> bool:
    """Check if model contains any MoE layers"""
    handler = UniversalMLPHandler(model)
    return any(handler.is_moe_layer(i) for i in range(len(handler.layers)))


def get_moe_layer_indices(model: nn.Module) -> List[int]:
    """Get indices of layers that contain MoE"""
    handler = UniversalMLPHandler(model)
    return [i for i in range(len(handler.layers)) if handler.is_moe_layer(i)]