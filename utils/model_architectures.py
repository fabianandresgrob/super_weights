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

class AttentionArchitectureType(Enum):
    """Supported attention architecture types"""
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    GROUPED_QUERY_ATTENTION = "grouped_query_attention"
    MULTI_QUERY_ATTENTION = "multi_query_attention"
    SLIDING_WINDOW_ATTENTION = "sliding_window_attention"
    UNKNOWN = "unknown"


@dataclass
class AttentionComponentInfo:
    """Information about attention components"""
    component_name: str
    component_type: str
    has_bias: bool
    weight_shape: Tuple[int, int]
    head_dim: Optional[int] = None
    num_heads: Optional[int] = None


@dataclass
class AttentionArchitectureInfo:
    """Complete information about an attention architecture"""
    architecture_type: AttentionArchitectureType
    components: Dict[str, AttentionComponentInfo]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    sliding_window: Optional[int] = None
    rope_theta: Optional[float] = None


@dataclass
class LayerComponentInfo:
    """Information about all components in a layer"""
    attention_info: AttentionArchitectureInfo
    mlp_info: MLPArchitectureInfo
    normalization_components: Dict[str, str]  # component_type -> module_name


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
    
    ATTENTION_COMPONENT_PATTERNS = {
        'q_proj': ['q_proj', 'query', 'c_attn'],
        'k_proj': ['k_proj', 'key', 'c_attn'],
        'v_proj': ['v_proj', 'value', 'c_attn'],
        'o_proj': ['o_proj', 'dense', 'c_proj'],
        'qkv_proj': ['qkv_proj', 'c_attn'],  # Fused QKV
    }
    
    ATTENTION_PATH_PATTERNS = ["self_attn", "attn", "attention"]
    
    NORMALIZATION_PATTERNS = {
        'input_layernorm': ['input_layernorm', 'ln_1', 'attention_norm'],
        'post_attention_layernorm': ['post_attention_layernorm', 'ln_2', 'ffn_norm'],
    }

    # Add attention configuration patterns
    ATTENTION_CONFIG_PATTERNS = {
        'num_heads': ['num_attention_heads', 'num_heads', 'n_head', 'n_heads'],
        'num_key_value_heads': ['num_key_value_heads', 'num_kv_heads', 'n_kv_heads'],
        'head_dim': ['head_dim', 'head_size', 'd_head'],
        'hidden_size': ['hidden_size', 'd_model', 'embed_dim', 'embedding_size'],
    }

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
    
    @classmethod
    def get_attention_component_name(cls, attention_module: nn.Module, component_type: str) -> Optional[str]:
        """Get the actual attention component name for a given component type"""
        if component_type not in cls.ATTENTION_COMPONENT_PATTERNS:
            return None
        
        for pattern in cls.ATTENTION_COMPONENT_PATTERNS[component_type]:
            if hasattr(attention_module, pattern):
                return pattern
        return None
    
    @classmethod
    def find_attention_module(cls, layer: nn.Module) -> Optional[nn.Module]:
        """Find the attention module within a layer"""
        for pattern in cls.ATTENTION_PATH_PATTERNS:
            if hasattr(layer, pattern):
                return getattr(layer, pattern)
        return None
    
    @classmethod
    def find_normalization_components(cls, layer: nn.Module) -> Dict[str, str]:
        """Find normalization components in a layer"""
        components = {}
        for comp_type, patterns in cls.NORMALIZATION_PATTERNS.items():
            for pattern in patterns:
                if hasattr(layer, pattern):
                    components[comp_type] = pattern
                    break
        return components

    @classmethod
    def get_attention_config_value(cls, source, config_type: str, default_value=None):
        """Get attention configuration value from either module or config using various naming patterns"""
        if config_type not in cls.ATTENTION_CONFIG_PATTERNS:
            return default_value
        
        # Try to get from config first if it exists
        config = getattr(source, 'config', None)
        if config is not None:
            for pattern in cls.ATTENTION_CONFIG_PATTERNS[config_type]:
                if hasattr(config, pattern):
                    return getattr(config, pattern)
        
        # Fall back to checking the module directly
        for pattern in cls.ATTENTION_CONFIG_PATTERNS[config_type]:
            if hasattr(source, pattern):
                return getattr(source, pattern)
        
        return default_value

class UniversalLayerHandler:
    """Universal handler for complete transformer layers across different architectures"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.registry = ModelArchitectureRegistry()
        self.logger = logging.getLogger(f"UniversalLayerHandler_{id(self)}")
        
        self.layers = self.registry.find_layers(self.model)
        if self.layers is None:
            raise ValueError("Could not find layers in model")
        
        self._layer_architectures = self._detect_all_layer_architectures()
        self.logger.info(f"Detected architectures for {len(self._layer_architectures)} layers")
    
    def _detect_all_layer_architectures(self) -> Dict[int, LayerComponentInfo]:
        """Detect architecture for all layers at initialization"""
        architectures = {}
        
        for layer_idx, layer in enumerate(self.layers):
            try:
                # Detect attention architecture
                attention_module = self.registry.find_attention_module(layer)
                if attention_module is not None:
                    attention_info = self._detect_attention_architecture(attention_module, layer_idx)
                else:
                    attention_info = self._create_unknown_attention_architecture()
                
                # Detect MLP architecture (reuse existing logic)
                moe_module = self.registry.find_moe_module(layer)
                if moe_module is not None:
                    mlp_info = self._detect_moe_architecture(moe_module)
                else:
                    mlp_module = self.registry.find_mlp(layer)
                    if mlp_module is not None:
                        mlp_info = self._detect_mlp_architecture(mlp_module)
                    else:
                        mlp_info = self._create_unknown_mlp_architecture()
                
                # Detect normalization components
                norm_components = self.registry.find_normalization_components(layer)
                
                architectures[layer_idx] = LayerComponentInfo(
                    attention_info=attention_info,
                    mlp_info=mlp_info,
                    normalization_components=norm_components
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to detect architecture for layer {layer_idx}: {e}")
                architectures[layer_idx] = LayerComponentInfo(
                    attention_info=self._create_unknown_attention_architecture(),
                    mlp_info=self._create_unknown_mlp_architecture(),
                    normalization_components={}
                )
        
        return architectures
    
    def _detect_attention_architecture(self, attention_module: nn.Module, layer_idx: int) -> AttentionArchitectureInfo:
        """Detect attention architecture and components"""
        components = {}
        
        # Get architecture parameters first
        num_heads = self.registry.get_attention_config_value(attention_module, 'num_heads', 32)
        num_key_value_heads = self.registry.get_attention_config_value(attention_module, 'num_key_value_heads', num_heads)
        hidden_size = self.registry.get_attention_config_value(attention_module, 'hidden_size', 4096)
        head_dim = self.registry.get_attention_config_value(attention_module, 'head_dim', None)
        
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        # Check for fused QKV projection (like Phi-3's qkv_proj)
        qkv_fused_name = self.registry.get_attention_component_name(attention_module, 'qkv_proj')
        if qkv_fused_name:
            # Create wrapper modules for each component
            fused_module = getattr(attention_module, qkv_fused_name)
            
            for comp_type in ['q_proj', 'k_proj', 'v_proj']:
                wrapper = QKVSplitter(
                    fused_module=fused_module,
                    component_type=comp_type,
                    num_heads=num_heads,
                    num_kv_heads=num_key_value_heads,
                    head_dim=head_dim
                )
                
                components[comp_type] = AttentionComponentInfo(
                    component_name=f"{qkv_fused_name}_{comp_type}_wrapper",
                    component_type=comp_type,
                    has_bias=fused_module.bias is not None,
                    weight_shape=(wrapper.end_idx - wrapper.start_idx, fused_module.weight.shape[1])
                )
                
                # Store the wrapper module for retrieval
                setattr(self, f"_wrapper_{layer_idx}_{comp_type}", wrapper)
        else:
            # Handle separate Q, K, V projections as before
            for comp_type in ['q_proj', 'k_proj', 'v_proj']:
                comp_name = self.registry.get_attention_component_name(attention_module, comp_type)
                if comp_name:
                    module = getattr(attention_module, comp_name)
                    
                    # Handle legacy fused QKV (like GPT-2's c_attn)
                    if comp_name == 'c_attn' and comp_type in ['q_proj', 'k_proj', 'v_proj']:
                        # For legacy fused QKV, we need to handle this specially
                        weight_shape = tuple(module.weight.shape)
                    else:
                        weight_shape = tuple(module.weight.shape)
                    
                    components[comp_type] = AttentionComponentInfo(
                        component_name=comp_name,
                        component_type=comp_type,
                        has_bias=module.bias is not None,
                        weight_shape=weight_shape
                    )
        
        # Always check for output projection separately
        o_proj_name = self.registry.get_attention_component_name(attention_module, 'o_proj')
        if o_proj_name:
            o_proj_module = getattr(attention_module, o_proj_name)
            components['o_proj'] = AttentionComponentInfo(
                component_name=o_proj_name,
                component_type='o_proj',
                has_bias=o_proj_module.bias is not None,
                weight_shape=tuple(o_proj_module.weight.shape)
            )
        
        # Detect attention type and parameters using the registry helper
        num_heads = self.registry.get_attention_config_value(attention_module, 'num_heads', 32)
        num_key_value_heads = self.registry.get_attention_config_value(attention_module, 'num_key_value_heads', num_heads)
        hidden_size = self.registry.get_attention_config_value(attention_module, 'hidden_size', 4096)
        
        # Try to get head_dim from config first, then calculate from weights
        head_dim = self.registry.get_attention_config_value(attention_module, 'head_dim', None)
        
        if head_dim is None and 'q_proj' in components:
            # Calculate head_dim from q_proj weight dimensions
            q_proj_out_features = components['q_proj'].weight_shape[0]
            
            # For fused QKV, the output dimension includes Q, K, V
            if qkv_fused_name:
                # For Phi-3 style fused QKV: output_dim = (num_heads + 2 * num_kv_heads) * head_dim
                # So head_dim = output_dim / (num_heads + 2 * num_kv_heads)
                head_dim = q_proj_out_features // (num_heads + 2 * num_key_value_heads)
            else:
                # For separate projections
                head_dim = q_proj_out_features // num_heads
        elif head_dim is None:
            # Fallback calculation
            head_dim = hidden_size // num_heads
        
        # Determine attention type
        if num_key_value_heads == 1:
            arch_type = AttentionArchitectureType.MULTI_QUERY_ATTENTION
        elif num_key_value_heads < num_heads:
            arch_type = AttentionArchitectureType.GROUPED_QUERY_ATTENTION
        else:
            arch_type = AttentionArchitectureType.MULTI_HEAD_ATTENTION
        
        # Check for sliding window
        sliding_window = getattr(attention_module, 'sliding_window', None)
        if sliding_window is None:
            # Also check config
            config = getattr(attention_module, 'config', None)
            if config is not None:
                sliding_window = getattr(config, 'sliding_window_size', None)
                if sliding_window is None:
                    sliding_window = getattr(config, 'sliding_window', None)
        
        if sliding_window is not None:
            arch_type = AttentionArchitectureType.SLIDING_WINDOW_ATTENTION
        
        # Get RoPE theta
        rope_theta = None
        config = getattr(attention_module, 'config', None)
        if config is not None:
            rope_theta = getattr(config, 'rope_theta', None)
            if rope_theta is None:
                rope_theta = getattr(config, 'rotary_emb_base', None)
        
        return AttentionArchitectureInfo(
            architecture_type=arch_type,
            components=components,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            sliding_window=sliding_window,
            rope_theta=rope_theta
        )
    
    def _create_unknown_attention_architecture(self) -> AttentionArchitectureInfo:
        """Create architecture info for unknown attention"""
        return AttentionArchitectureInfo(
            architecture_type=AttentionArchitectureType.UNKNOWN,
            components={},
            num_attention_heads=0,
            num_key_value_heads=0,
            head_dim=0,
            hidden_size=0
        )
    
    def _create_unknown_mlp_architecture(self) -> MLPArchitectureInfo:
        """Create architecture info for unknown MLP"""
        return MLPArchitectureInfo(
            architecture_type=MLPArchitectureType.UNKNOWN,
            activation_function='unknown',
            components={},
            is_moe=False
        )
    
    # Reuse existing MLP detection methods
    def _detect_mlp_architecture(self, mlp_module: nn.Module) -> MLPArchitectureInfo:
        """Detect MLP architecture and components"""
        components = {}
        
        # Check for fused gated architecture first (like Phi-3)
        gate_up_fused_name = self.registry.get_component_name(mlp_module, 'gate_up_fused')
        down_name = self.registry.get_component_name(mlp_module, 'down')
        
        if gate_up_fused_name and down_name:
            components = self._extract_mlp_components(mlp_module, ['gate_up_fused', 'down'])
            arch_type = MLPArchitectureType.FUSED_GATED_MLP
            has_gate = True
        else:
            # Check for separate gated architecture
            gate_name = self.registry.get_component_name(mlp_module, 'gate')
            up_name = self.registry.get_component_name(mlp_module, 'up')
            down_name = self.registry.get_component_name(mlp_module, 'down')
            
            if gate_name and up_name and down_name:
                components = self._extract_mlp_components(mlp_module, ['gate', 'up', 'down'])
                arch_type = MLPArchitectureType.GATED_MLP
                has_gate = True
            else:
                # Check for standard MLP architecture
                hidden_name = self.registry.get_component_name(mlp_module, 'hidden')
                output_name = self.registry.get_component_name(mlp_module, 'output')
                
                if hidden_name and output_name:
                    components = self._extract_mlp_components(mlp_module, ['hidden', 'output'])
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
        # Reuse existing logic from UniversalMLPHandler
        experts = self.registry.find_experts(moe_module)
        if experts is None:
            return self._create_unknown_mlp_architecture()
        
        first_expert = experts[0]
        expert_arch = self._detect_mlp_architecture(first_expert)
        
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
    
    def _extract_mlp_components(self, mlp_module: nn.Module, component_types: List[str]) -> Dict[str, MLPComponentInfo]:
        """Extract MLP component info for given types"""
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
        return 'silu'
    
    # Public interface methods
    def get_layer_architecture(self, layer_idx: int) -> LayerComponentInfo:
        """Get complete architecture information for layer"""
        if layer_idx not in self._layer_architectures:
            raise ValueError(f"Layer index {layer_idx} out of range")
        return self._layer_architectures[layer_idx]
    
    def get_attention_architecture(self, layer_idx: int) -> AttentionArchitectureInfo:
        """Get attention architecture information for layer"""
        return self.get_layer_architecture(layer_idx).attention_info
    
    def get_mlp_architecture(self, layer_idx: int) -> MLPArchitectureInfo:
        """Get MLP architecture information for layer"""
        return self.get_layer_architecture(layer_idx).mlp_info

    def get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the complete attention module for a layer"""
        attention_module = self.registry.find_attention_module(self.layers[layer_idx])
        if attention_module is None:
            raise ValueError(f"Could not find attention module in layer {layer_idx}")
        return attention_module
    
    def get_attention_components(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get all attention components for a layer"""
        attention_info = self.get_attention_architecture(layer_idx)
        attention_module = self.registry.find_attention_module(self.layers[layer_idx])
    
        components = {}
        for comp_type, comp_info in attention_info.components.items():
            if comp_info.component_name.endswith('_wrapper'):
                # This is a wrapper module we created
                wrapper_attr = f"_wrapper_{layer_idx}_{comp_type}"
                if hasattr(self, wrapper_attr):
                    components[comp_type] = getattr(self, wrapper_attr)
                else:
                    raise RuntimeError(f"Wrapper {wrapper_attr} not found")
            else:
                # Regular component
                components[comp_type] = getattr(attention_module, comp_info.component_name)
    
        return components
    
    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """Get the complete MLP module for a layer"""
        mlp_info = self.get_mlp_architecture(layer_idx)
        
        if mlp_info.is_moe:
            moe_module = self.registry.find_moe_module(self.layers[layer_idx])
            if moe_module is None:
                raise ValueError(f"Could not find MoE module in layer {layer_idx}")
            return moe_module
        else:
            mlp_module = self.registry.find_mlp(self.layers[layer_idx])
            if mlp_module is None:
                raise ValueError(f"Could not find MLP module in layer {layer_idx}")
            return mlp_module

    def get_mlp_components(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get all MLP components for a layer"""
        mlp_info = self.get_mlp_architecture(layer_idx)
        
        if mlp_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is MoE - use get_moe_experts instead")
        
        mlp_module = self.registry.find_mlp(self.layers[layer_idx])
        if mlp_module is None:
            raise ValueError(f"Could not find MLP module in layer {layer_idx}")
        
        components = {}
        for comp_type, comp_info in mlp_info.components.items():
            components[comp_type] = getattr(mlp_module, comp_info.component_name)
        
        return components
    
    def get_normalization_components(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get normalization components for a layer"""
        layer_info = self.get_layer_architecture(layer_idx)
        layer = self.layers[layer_idx]
        
        components = {}
        for comp_type, comp_name in layer_info.normalization_components.items():
            components[comp_type] = getattr(layer, comp_name)
        
        return components

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the complete layer module"""
        if layer_idx >= len(self.layers):
            raise ValueError(f"Layer index {layer_idx} out of range")
        return self.layers[layer_idx]
    
    def get_all_layer_modules(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get all top-level modules of a layer organized by type"""
        modules = {
            'layer': self.get_layer_module(layer_idx),
            'attention': self.get_attention_module(layer_idx),
            'mlp': self.get_mlp_module(layer_idx),
        }
        
        # Add normalization modules
        norm_components = self.get_normalization_components(layer_idx)
        for comp_type, module in norm_components.items():
            modules[comp_type] = module
        
        return modules

    def get_complete_layer_info(self, layer_idx: int) -> Dict[str, Dict[str, nn.Module]]:
        """Get complete layer information with both top-level modules and subcomponents"""
        return {
            'modules': self.get_all_layer_modules(layer_idx),
            'components': self.get_all_layer_components(layer_idx)
        }

    def get_attention_hierarchy(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get complete attention hierarchy - both module and components"""
        attention_module = self.get_attention_module(layer_idx)
        attention_components = self.get_attention_components(layer_idx)
        
        return {
            'module': attention_module,
            'components': attention_components
        }
    
    def get_mlp_hierarchy(self, layer_idx: int) -> Dict[str, nn.Module]:
        """Get complete MLP hierarchy - both module and components"""
        mlp_module = self.get_mlp_module(layer_idx)
        mlp_info = self.get_mlp_architecture(layer_idx)
        
        hierarchy = {
            'module': mlp_module,
        }
        
        if mlp_info.is_moe:
            hierarchy['experts'] = self.get_moe_experts(layer_idx)
            # Add routing module if available
            if mlp_info.moe_info and mlp_info.moe_info.routing_info.router_module:
                hierarchy['router'] = mlp_info.moe_info.routing_info.router_module
        else:
            hierarchy['components'] = self.get_mlp_components(layer_idx)
        
        return hierarchy

     # Enhanced version of get_all_layer_components to include modules
    def get_all_layer_components(self, layer_idx: int) -> Dict[str, Dict[str, nn.Module]]:
        """Get all components of a layer organized by type"""
        try:
            attention_hierarchy = self.get_attention_hierarchy(layer_idx)
        except ValueError:
            attention_hierarchy = {}

        try:
            if not self.get_mlp_architecture(layer_idx).is_moe:
                mlp_hierarchy = self.get_mlp_hierarchy(layer_idx)
            else:
                mlp_hierarchy = {}
        except ValueError:
            mlp_hierarchy = {}

        return {
            'attention': attention_hierarchy,
            'mlp': mlp_hierarchy,
            'normalization': self.get_normalization_components(layer_idx)
        }
    
    def extract_attention_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract attention weight tensors for a layer"""
        components = self.get_attention_components(layer_idx)
        weights = {}
        for comp_type, module in components.items():
            weights[comp_type] = module.weight
        return weights
    
    def extract_attention_biases(self, layer_idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """Extract attention bias tensors for a layer"""
        components = self.get_attention_components(layer_idx)
        biases = {}
        for comp_type, module in components.items():
            biases[comp_type] = module.bias
        return biases
    
    # Backward compatibility - delegate to existing MLP methods
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer is MoE"""
        return self.get_mlp_architecture(layer_idx).is_moe
    
    def get_moe_experts(self, layer_idx: int) -> List[nn.Module]:
        """Get expert modules for MoE layer"""
        mlp_info = self.get_mlp_architecture(layer_idx)
        if not mlp_info.is_moe:
            raise ValueError(f"Layer {layer_idx} is not an MoE layer")
        
        moe_module = self.registry.find_moe_module(self.layers[layer_idx])
        experts = self.registry.find_experts(moe_module)
        return list(experts)

    def get_layer_summary(self, layer_idx: int) -> Dict[str, Any]:
        """Get a summary of layer structure"""
        layer_info = self.get_layer_architecture(layer_idx)
        
        summary = {
            'layer_idx': layer_idx,
            'layer_class': self.get_layer_module(layer_idx).__class__.__name__,
            'attention': {
                'module_class': self.get_attention_module(layer_idx).__class__.__name__,
                'architecture_type': layer_info.attention_info.architecture_type.value,
                'num_heads': layer_info.attention_info.num_attention_heads,
                'num_kv_heads': layer_info.attention_info.num_key_value_heads,
                'components': list(layer_info.attention_info.components.keys())
            },
            'mlp': {
                'module_class': self.get_mlp_module(layer_idx).__class__.__name__,
                'architecture_type': layer_info.mlp_info.architecture_type.value,
                'is_moe': layer_info.mlp_info.is_moe,
                'activation': layer_info.mlp_info.activation_function,
                'components': list(layer_info.mlp_info.components.keys()) if not layer_info.mlp_info.is_moe else []
            },
            'normalization': list(layer_info.normalization_components.keys())
        }
        
        if layer_info.mlp_info.is_moe and layer_info.mlp_info.moe_info:
            summary['mlp']['num_experts'] = layer_info.mlp_info.moe_info.num_experts
            summary['mlp']['routing_type'] = layer_info.mlp_info.moe_info.routing_info.routing_type.value
        
        return summary

    def get_attention_constraints(self, layer_idx: int) -> Dict[str, Any]:
        """Get attention pattern constraints for a specific layer."""
        attention_info = self.get_attention_architecture(layer_idx)
        
        # Check for sliding window from architecture info
        if attention_info.sliding_window is not None:
            return {
                'type': 'sliding_window',
                'window_size': attention_info.sliding_window,
                'has_constraints': True
            }
        
        # Check model config as fallback
        config = getattr(self.model, 'config', None)
        if config is not None:
            sliding_window = getattr(config, 'sliding_window', None)
            if sliding_window is not None:
                return {
                    'type': 'sliding_window', 
                    'window_size': sliding_window,
                    'has_constraints': True
                }
        
        return {
            'type': 'full_attention',
            'window_size': None,
            'has_constraints': False
        }

# Convenience functions
def create_layer_handler(model: nn.Module) -> UniversalLayerHandler:
    """Create a universal layer handler"""
    return UniversalLayerHandler(model)


# Keep existing MLP handler for backward compatibility
class UniversalMLPHandler(UniversalLayerHandler):
    """Backward compatibility wrapper - delegates to UniversalLayerHandler"""
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.logger = logging.getLogger(f"UniversalMLPHandler_{id(self)}")
    
    def extract_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Extract MLP weight tensors for a layer"""
        mlp_info = self.get_mlp_architecture(layer_idx)
        
        if mlp_info.is_moe:
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
        """Extract MLP bias tensors for a layer"""
        mlp_info = self.get_mlp_architecture(layer_idx)
        
        if mlp_info.is_moe:
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
    
    def get_expert_components(self, layer_idx: int, expert_idx: int) -> Dict[str, nn.Module]:
        """Get components of a specific expert in an MoE layer"""
        mlp_info = self.get_mlp_architecture(layer_idx)
        if not mlp_info.is_moe:
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
    
    def get_routing_info(self, layer_idx: int) -> Optional[MoERoutingInfo]:
        """Get routing information for an MoE layer"""
        if not self.is_moe_layer(layer_idx):
            return None
        
        arch_info = self.get_mlp_architecture(layer_idx)
        if arch_info.is_moe and arch_info.moe_info:
            return arch_info.moe_info.routing_info
        return None
    
    def get_router_module(self, layer_idx: int) -> Optional[nn.Module]:
        """Get the router module for an MoE layer"""
        if not self.is_moe_layer(layer_idx):
            return None
        
        moe_module = self.get_mlp_module(layer_idx)
        if moe_module is None:
            return None
        
        # Try common router names
        for router_name in ['gate', 'router', 'switch']:
            if hasattr(moe_module, router_name):
                return getattr(moe_module, router_name)
        
        return None
    
    def get_expert_module(self, layer_idx: int, expert_idx: int) -> Optional[nn.Module]:
        """Get a specific expert module"""
        if not self.is_moe_layer(layer_idx):
            return None
        
        experts = self.get_moe_experts(layer_idx)
        if expert_idx < len(experts):
            return experts[expert_idx]
        return None
    
    def get_expert_component_name(self, layer_idx: int, expert_idx: int, component_type: str) -> Optional[str]:
        """Get the actual module name for an expert component"""
        expert_module = self.get_expert_module(layer_idx, expert_idx)
        if expert_module is None:
            return None
        
        # Use the registry to find the component name
        component_name = self.registry.get_component_name(expert_module, component_type)
        if component_name:
            return f"experts.{expert_idx}.{component_name}"
        return None

class QKVSplitter(nn.Module):
    """Wrapper to split fused QKV projection into individual Q, K, V components"""
    
    def __init__(self, fused_module: nn.Module, component_type: str, 
                 num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.fused_module = fused_module
        self.component_type = component_type
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads  
        self.head_dim = head_dim
        
        # Calculate split indices for Q, K, V
        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim  
        v_size = num_kv_heads * head_dim
        
        if component_type == 'q_proj':
            self.start_idx = 0
            self.end_idx = q_size
        elif component_type == 'k_proj':
            self.start_idx = q_size
            self.end_idx = q_size + k_size
        elif component_type == 'v_proj':
            self.start_idx = q_size + k_size
            self.end_idx = q_size + k_size + v_size
        else:
            raise ValueError(f"Unknown component_type: {component_type}")
    
    @property
    def weight(self):
        """Return the slice of the fused weight matrix for this component"""
        return self.fused_module.weight[self.start_idx:self.end_idx, :]
    
    @property 
    def bias(self):
        """Return the slice of the fused bias vector for this component"""
        if self.fused_module.bias is None:
            return None
        return self.fused_module.bias[self.start_idx:self.end_idx]
    
    def forward(self, x):
        """Forward pass through the fused module, then slice the output"""
        fused_output = self.fused_module(x)  # [batch, seq, total_dim]
        return fused_output[..., self.start_idx:self.end_idx]
