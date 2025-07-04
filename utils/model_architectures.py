"""
Model architecture detection and utilities for super weight analysis.
Provides model-agnostic interfaces for detecting and working with different transformer architectures.
"""

import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import logging


class MLPArchitectureType(Enum):
    """Supported MLP architecture types"""
    GATED_MLP = "gated_mlp"
    STANDARD_MLP = "standard_mlp"
    FUSED_GATED_MLP = "fused_gated_mlp"
    MOE_GATED = "moe_gated"
    MOE_STANDARD = "moe_standard"
    MOE_WITH_SHARED_EXPERT = "moe_with_shared_expert"
    UNKNOWN = "unknown"


@dataclass
class MLPComponentInfo:
    """Information about MLP components"""
    component_name: str
    component_type: str
    has_bias: bool
    weight_shape: Tuple[int, int]


class MoERoutingType(Enum):
    """Different MoE routing mechanisms"""
    TOP_K = "top_k"
    SWITCH = "switch"
    UNKNOWN = "unknown"


@dataclass
class MoERoutingInfo:
    """Information about MoE routing mechanism"""
    routing_type: MoERoutingType
    experts_per_token: int
    router_module_name: str
    router_module: Optional[nn.Module] = None


@dataclass 
class MoEArchitectureInfo:
    """Information about MoE architecture"""
    num_experts: int
    routing_info: MoERoutingInfo
    has_shared_expert: bool = False
    shared_expert_module: Optional[nn.Module] = None


@dataclass 
class MLPArchitectureInfo:
    """Complete information about an MLP architecture"""
    architecture_type: MLPArchitectureType
    activation_function: str
    components: Dict[str, MLPComponentInfo]
    has_gate: bool = False
    intermediate_size: Optional[int] = None
    hidden_size: Optional[int] = None
    is_moe: bool = False
    moe_info: Optional[MoEArchitectureInfo] = None


class ModelArchitectureRegistry:
    """Registry for different model architectures and their patterns"""
    
    MLP_COMPONENT_PATTERNS = {
        'gate': ['gate_proj', 'w1'],
        'up': ['up_proj', 'w3'],
        'down': ['down_proj', 'w2', 'c_proj'],
        'hidden': ['c_fc', 'fc1', 'dense_h_to_4h'],
        'output': ['c_proj', 'fc2', 'dense_4h_to_h'],
        'gate_up_fused': ['gate_up_proj']
    }
    
    LAYER_PATH_PATTERNS = [
        "model.layers", "layers", "transformer.h", "transformer.layers"
    ]
    
    MLP_PATH_PATTERNS = ["mlp", "feed_forward", "ffn"]
    
    MOE_PATH_PATTERNS = ["block_sparse_moe", "moe"]
    MOE_EXPERT_PATTERNS = ["experts"]
    MOE_ROUTER_PATTERNS = ["gate", "router"]
    
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
    def find_moe_module(cls, layer: nn.Module) -> Optional[nn.Module]:
        """Find the MoE module within a layer"""
        for pattern in cls.MOE_PATH_PATTERNS:
            if hasattr(layer, pattern):
                return getattr(layer, pattern)
        
        # Check if 'mlp' is actually an MoE block
        if hasattr(layer, 'mlp'):
            mlp_module = getattr(layer, 'mlp')
            if cls._is_moe_block(mlp_module):
                return mlp_module
        
        return None

    @classmethod
    def _is_moe_block(cls, module: nn.Module) -> bool:
        """Check if a module is actually an MoE block"""
        return hasattr(module, 'experts') and hasattr(module, 'gate')

    @classmethod
    def find_mlp(cls, layer: nn.Module) -> Optional[nn.Module]:
        """Find the MLP module within a layer"""
        for pattern in cls.MLP_PATH_PATTERNS:
            if hasattr(layer, pattern):
                mlp_module = getattr(layer, pattern)
                if not cls._is_moe_block(mlp_module):
                    return mlp_module
        return None
    
    @classmethod
    def find_experts(cls, moe_module: nn.Module) -> Optional[nn.ModuleList]:
        """Find experts within MoE module"""
        if hasattr(moe_module, 'experts'):
            return moe_module.experts
        return None
    
    @classmethod
    def detect_routing_mechanism(cls, moe_module: nn.Module) -> MoERoutingInfo:
        """Detect the routing mechanism used in MoE module"""
        router_module = getattr(moe_module, 'gate', None)
        if router_module is None:
            return MoERoutingInfo(
                routing_type=MoERoutingType.UNKNOWN,
                experts_per_token=1,
                router_module_name="unknown"
            )
        
        # Simple heuristic: most MoE models use top-k=2
        experts_per_token = getattr(moe_module, 'top_k', 2)
        
        return MoERoutingInfo(
            routing_type=MoERoutingType.TOP_K,
            experts_per_token=experts_per_token,
            router_module_name="gate",
            router_module=router_module
        )


class UniversalMLPHandler:
    """Universal handler for MLP components across different architectures"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.registry = ModelArchitectureRegistry()
        self.logger = logging.getLogger(f"UniversalMLPHandler_{id(self)}")
        
        self.layers = self.registry.find_layers(self.model)
        if self.layers is None:
            raise ValueError("Could not find layers in model")
        
        self._layer_architectures = self._detect_all_layer_architectures()
        self.logger.info(f"Detected architectures for {len(self._layer_architectures)} layers")
    
    def _detect_all_layer_architectures(self) -> Dict[int, MLPArchitectureInfo]:
        """Detect architecture for all layers at initialization"""
        architectures = {}
        
        for layer_idx, layer in enumerate(self.layers):
            try:
                moe_module = self.registry.find_moe_module(layer)
                if moe_module is not None:
                    arch_info = self._detect_moe_architecture(moe_module)
                else:
                    mlp_module = self.registry.find_mlp(layer)
                    if mlp_module is not None:
                        arch_info = self._detect_mlp_architecture(mlp_module)
                    else:
                        arch_info = self._create_unknown_architecture()
                
                architectures[layer_idx] = arch_info
                
            except Exception as e:
                self.logger.warning(f"Failed to detect architecture for layer {layer_idx}: {e}")
                architectures[layer_idx] = self._create_unknown_architecture()
        
        return architectures
    
    def _create_unknown_architecture(self) -> MLPArchitectureInfo:
        """Create architecture info for unknown layers"""
        return MLPArchitectureInfo(
            architecture_type=MLPArchitectureType.UNKNOWN,
            activation_function='unknown',
            components={},
            is_moe=False
        )
    
    def _detect_mlp_architecture(self, mlp_module: nn.Module) -> MLPArchitectureInfo:
        """Detect MLP architecture and components"""
        components = {}
        
        # Check for gated architecture
        gate_name = self.registry.get_component_name(mlp_module, 'gate')
        up_name = self.registry.get_component_name(mlp_module, 'up')
        down_name = self.registry.get_component_name(mlp_module, 'down')
        
        if gate_name and up_name and down_name:
            # Gated MLP
            components = self._extract_components(mlp_module, ['gate', 'up', 'down'])
            arch_type = MLPArchitectureType.GATED_MLP
            has_gate = True
        else:
            # Standard MLP
            hidden_name = self.registry.get_component_name(mlp_module, 'hidden')
            output_name = self.registry.get_component_name(mlp_module, 'output')
            
            if hidden_name and output_name:
                components = self._extract_components(mlp_module, ['hidden', 'output'])
                arch_type = MLPArchitectureType.STANDARD_MLP
                has_gate = False
            else:
                arch_type = MLPArchitectureType.UNKNOWN
                has_gate = False
        
        return MLPArchitectureInfo(
            architecture_type=arch_type,
            activation_function=self._detect_activation_function(mlp_module),
            components=components,
            has_gate=has_gate,
            is_moe=False
        )
    
    def _detect_moe_architecture(self, moe_module: nn.Module) -> MLPArchitectureInfo:
        """Detect MoE architecture"""
        experts = self.registry.find_experts(moe_module)
        if experts is None:
            return self._create_unknown_architecture()
        
        # Analyze first expert
        first_expert = experts[0]
        expert_arch = self._detect_mlp_architecture(first_expert)
        
        # Determine MoE type
        if expert_arch.architecture_type == MLPArchitectureType.GATED_MLP:
            if hasattr(moe_module, 'shared_expert'):
                moe_type = MLPArchitectureType.MOE_WITH_SHARED_EXPERT
            else:
                moe_type = MLPArchitectureType.MOE_GATED
        else:
            moe_type = MLPArchitectureType.MOE_STANDARD
        
        routing_info = self.registry.detect_routing_mechanism(moe_module)
        
        moe_info = MoEArchitectureInfo(
            num_experts=len(experts),
            routing_info=routing_info,
            has_shared_expert=hasattr(moe_module, 'shared_expert'),
            shared_expert_module=getattr(moe_module, 'shared_expert', None)
        )
        
        return MLPArchitectureInfo(
            architecture_type=moe_type,
            activation_function=expert_arch.activation_function,
            components={},
            has_gate=expert_arch.has_gate,
            is_moe=True,
            moe_info=moe_info
        )
    
    def _extract_components(self, mlp_module: nn.Module, component_types: List[str]) -> Dict[str, MLPComponentInfo]:
        """Extract component info for given types"""
        components = {}
        for comp_type in component_types:
            comp_name = self.registry.get_component_name(mlp_module, comp_type)
            if comp_name:
                module = getattr(mlp_module, comp_name)
                components[comp_type] = MLPComponentInfo(
                    component_name=comp_name,
                    component_type=comp_type,
                    has_bias=module.bias is not None,
                    weight_shape=tuple(module.weight.shape)
                )
        return components
    
    def _detect_activation_function(self, mlp_module: nn.Module) -> str:
        """Detect activation function"""
        for attr_name in ['act_fn', 'activation_fn']:
            if hasattr(mlp_module, attr_name):
                activation = getattr(mlp_module, attr_name)
                class_name = activation.__class__.__name__.lower()
                if 'silu' in class_name:
                    return 'silu'
                elif 'gelu' in class_name:
                    return 'gelu'
        return 'silu'  # Default
    
    # Public interface methods
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer is MoE"""
        return self._layer_architectures.get(layer_idx, self._create_unknown_architecture()).is_moe
    
    def get_mlp_architecture(self, layer_idx: int) -> MLPArchitectureInfo:
        """Get architecture information for layer"""
        if layer_idx not in self._layer_architectures:
            raise ValueError(f"Layer index {layer_idx} out of range")
        return self._layer_architectures[layer_idx]
    
    def get_moe_experts(self, layer_idx: int) -> List[nn.Module]:
        """Get expert modules for MoE layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        if not arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")
        
        moe_module = self.registry.find_moe_module(self.layers[layer_idx])
        experts = self.registry.find_experts(moe_module)
        return list(experts)
    
    def get_routing_info(self, layer_idx: int) -> Optional[MoERoutingInfo]:
        """Get routing information for MoE layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        if arch_info.is_moe and arch_info.moe_info:
            return arch_info.moe_info.routing_info
        return None
    
    def get_router_module(self, layer_idx: int) -> Optional[nn.Module]:
        """Get router module for MoE layer"""
        routing_info = self.get_routing_info(layer_idx)
        return routing_info.router_module if routing_info else None
    
    def get_expert_components(self, layer_idx: int, expert_idx: int) -> Dict[str, nn.Module]:
        """Get components of a specific expert in an MoE layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        if not arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")
        
        experts = self.get_moe_experts(layer_idx)
        if expert_idx >= len(experts):
            raise ValueError(f"Expert index {expert_idx} out of range (layer has {len(experts)} experts)")
        
        expert = experts[expert_idx]
        
        # Detect components for this expert (same as regular MLP)
        expert_arch = self._detect_mlp_architecture(expert)
        
        components = {}
        for comp_type, comp_info in expert_arch.components.items():
            components[comp_type] = getattr(expert, comp_info.component_name)
        
        return components
    
    def get_mlp_components(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get all MLP components for a layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        
        if arch_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is MoE - use get_moe_experts instead")
        
        mlp_module = self.registry.find_mlp(self.layers[layer_idx])
        if mlp_module is None:
            raise ValueError(f"Could not find MLP module in layer {layer_idx}")
        
        components = {}
        for comp_type, comp_info in arch_info.components.items():
            components[comp_type] = getattr(mlp_module, comp_info.component_name)
        
        return components
    
    def extract_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract weight tensors for a layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        
        if arch_info.is_moe:
            # For MoE, return first expert's weights as example
            expert_components = self.get_expert_components(layer_idx, 0)
            weights = {}
            for comp_type, module in expert_components.items():
                weights[comp_type] = module.weight
            return weights
        else:
            # Regular MLP
            components = self.get_mlp_components(layer_idx)
            weights = {}
            for comp_type, module in components.items():
                weights[comp_type] = module.weight
            return weights
    
    def extract_biases(self, layer_idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Extract bias tensors for a layer"""
        arch_info = self.get_mlp_architecture(layer_idx)
        
        if arch_info.is_moe:
            # For MoE, return first expert's biases as example
            expert_components = self.get_expert_components(layer_idx, 0)
            biases = {}
            for comp_type, module in expert_components.items():
                biases[comp_type] = module.bias
            return biases
        else:
            # Regular MLP
            components = self.get_mlp_components(layer_idx)
            biases = {}
            for comp_type, module in components.items():
                biases[comp_type] = module.bias
            return biases


# Convenience functions
def create_mlp_handler(model: nn.Module) -> UniversalMLPHandler:
    """Create a universal MLP handler"""
    return UniversalMLPHandler(model)

def is_moe_model(model: nn.Module) -> bool:
    """Check if model contains MoE layers"""
    handler = UniversalMLPHandler(model)
    return any(handler.is_moe_layer(i) for i in range(len(handler.layers)))