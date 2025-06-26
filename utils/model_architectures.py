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
    """Information about a single expert in MoE"""
    expert_id: int
    expert_path: str  # e.g., "experts.0" or "experts[0]"
    components: Dict[str, MLPComponentInfo]
    architecture_type: MLPArchitectureType


@dataclass 
class MoEArchitectureInfo:
    """Information about MoE architecture"""
    num_experts: int
    experts_per_token: int  # How many experts are selected per token
    routing_method: str  # "top_k", "switch", etc.
    experts: List[MoEExpertInfo]
    router_path: str  # Path to router/gate
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
        'output': ['c_proj', 'output', 'fc2', 'linear2', 'dense_4h_to_h']
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
        
        # Cache for detected architecture info
        self._architecture_cache = {}
        
        # Detect if model has MoE layers
        self._moe_layers = self._detect_moe_layers()
    
    def _detect_moe_layers(self) -> Dict[int, bool]:
        """Detect which layers contain MoE modules"""
        moe_layers = {}
        layers = self.registry.find_layers(self.model)
        
        if layers:
            for i, layer in enumerate(layers):
                moe_module = self.registry.find_moe_module(layer)
                moe_layers[i] = moe_module is not None
        
        return moe_layers
    
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer is MoE"""
        return self._moe_layers.get(layer_idx, False)
    
    def get_moe_experts(self, layer_idx: int) -> List[nn.Module]:
        """Get all expert modules for an MoE layer"""
        if not self.is_moe_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")
        
        layers = self.registry.find_layers(self.model)
        layer = layers[layer_idx]
        moe_module = self.registry.find_moe_module(layer)
        experts = self.registry.find_experts(moe_module)
        
        if experts is None:
            raise ValueError(f"Could not find experts in MoE layer {layer_idx}")
        
        return list(experts)
    
    def get_moe_router(self, layer_idx: int) -> nn.Module:
        """Get router/gate module for MoE layer"""
        if not self.is_moe_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")
        
        layers = self.registry.find_layers(self.model)
        layer = layers[layer_idx]
        moe_module = self.registry.find_moe_module(layer)
        
        for pattern in self.registry.MOE_ROUTER_PATTERNS:
            if hasattr(moe_module, pattern):
                return getattr(moe_module, pattern)
        
        raise ValueError(f"Could not find router in MoE layer {layer_idx}")
    
    def get_expert_components(self, layer_idx: int, expert_idx: int) -> Dict[str, nn.Module]:
        """Get components for a specific expert"""
        experts = self.get_moe_experts(layer_idx)
        
        if expert_idx >= len(experts):
            raise ValueError(f"Expert index {expert_idx} out of range (layer has {len(experts)} experts)")
        
        expert = experts[expert_idx]
        
        # Detect expert architecture (similar to regular MLP detection)
        arch_info = self._detect_architecture(expert)
        
        components = {}
        for comp_type, comp_info in arch_info.components.items():
            components[comp_type] = getattr(expert, comp_info.component_name)
        
        return components
    
    def get_mlp_architecture(self, layer_idx: int) -> MLPArchitectureInfo:
        """Get architecture information for MLP at specified layer"""
        
        if layer_idx in self._architecture_cache:
            return self._architecture_cache[layer_idx]
        
        # Get the MLP module
        layers = self.registry.find_layers(self.model) 
        if layers is None:
            raise ValueError("Could not find layers in model")
        
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (model has {len(layers)} layers)")
        
        layer = layers[layer_idx]
        mlp_module = self.registry.find_mlp(layer)
        
        if mlp_module is None:
            raise ValueError(f"Could not find MLP module in layer {layer_idx}")
        
        # Detect architecture
        arch_info = self._detect_architecture(mlp_module)
        self._architecture_cache[layer_idx] = arch_info
        
        return arch_info
    
    def get_mlp_components(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get all MLP components for a layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        
        layers = self.registry.find_layers(self.model)
        mlp_module = self.registry.find_mlp(layers[layer_idx])
        
        components = {}
        for comp_type, comp_info in arch_info.components.items():
            components[comp_type] = getattr(mlp_module, comp_info.component_name)
        
        return components
    
    def extract_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract weight matrices from MLP layer"""
        components = self.get_mlp_components(layer_idx)
        
        weights = {}
        for comp_type, module in components.items():
            if hasattr(module, 'weight'):
                weights[comp_type] = module.weight.detach()
        
        return weights
    
    def extract_biases(self, layer_idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Extract bias vectors from MLP layer"""
        components = self.get_mlp_components(layer_idx)
        
        biases = {}
        for comp_type, module in components.items():
            if hasattr(module, 'bias'):
                bias = module.bias
                biases[comp_type] = bias.detach() if bias is not None else None
            else:
                biases[comp_type] = None
        
        return biases
    
    def get_layer_shapes(self, layer_idx: int) -> Dict[str, Tuple[int, int]]:
        """Get weight matrix shapes for all components"""
        weights = self.extract_weights(layer_idx)
        return {comp_type: tuple(weight.shape) for comp_type, weight in weights.items()}
    
    def _detect_architecture(self, mlp_module: nn.Module) -> MLPArchitectureInfo:
        """Detect the architecture type and components of an MLP module"""
        
        # First check if this is an MoE module
        moe_module = self.registry.find_moe_module(mlp_module) if hasattr(mlp_module, '__dict__') else None
        
        if moe_module is not None:
            return self._detect_moe_architecture(mlp_module, moe_module)
        
        components = {}
        
        # Check for gated architecture components
        gate_name = self.registry.get_component_name(mlp_module, 'gate')
        up_name = self.registry.get_component_name(mlp_module, 'up') 
        down_name = self.registry.get_component_name(mlp_module, 'down')
        
        # Check for standard architecture components
        hidden_name = self.registry.get_component_name(mlp_module, 'hidden')
        output_name = self.registry.get_component_name(mlp_module, 'output')
        
        if gate_name and up_name and down_name:
            # Gated MLP architecture
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
    
    def _detect_moe_architecture(self, layer_module: nn.Module, moe_module: nn.Module) -> MLPArchitectureInfo:
        """Detect MoE architecture details"""
        experts = self.registry.find_experts(moe_module)
        
        if experts is None:
            raise ValueError("Could not find experts in MoE module")
        
        num_experts = len(experts)
        
        # Analyze first expert to understand architecture
        sample_expert = experts[0] if experts else None
        expert_arch = self._detect_architecture(sample_expert) if sample_expert else None
        
        # Determine MoE type
        if hasattr(moe_module, 'experts') and hasattr(moe_module, 'gate'):
            moe_type = MLPArchitectureType.MOE_MIXTRAL
        else:
            moe_type = MLPArchitectureType.MOE_SWITCH  # Default
        
        # Build expert info
        expert_infos = []
        for i, expert in enumerate(experts):
            expert_arch = self._detect_architecture(expert)
            expert_info = MoEExpertInfo(
                expert_id=i,
                expert_path=f"experts.{i}",
                components=expert_arch.components,
                architecture_type=expert_arch.architecture_type
            )
            expert_infos.append(expert_info)
        
        # Determine routing info
        router_path = "gate"  # Default
        experts_per_token = 2  # Common default
        
        if hasattr(moe_module, 'num_experts_per_tok'):
            experts_per_token = moe_module.num_experts_per_tok
        
        moe_info = MoEArchitectureInfo(
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            routing_method="top_k",
            experts=expert_infos,
            router_path=router_path
        )
        
        return MLPArchitectureInfo(
            architecture_type=moe_type,
            activation_function=expert_arch.activation_function if expert_arch else 'silu',
            components={},  # No shared components
            has_gate=False,  # Different from MLP gate
            intermediate_size=expert_arch.intermediate_size if expert_arch else None,
            hidden_size=expert_arch.hidden_size if expert_arch else None,
            is_moe=True,
            moe_info=moe_info
        )
    
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
    return any(handler._moe_layers.values())


def get_moe_layer_indices(model: nn.Module) -> List[int]:
    """Get indices of layers that contain MoE"""
    handler = UniversalMLPHandler(model)
    return [i for i, is_moe in handler._moe_layers.items() if is_moe]