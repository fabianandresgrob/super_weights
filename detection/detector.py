import dataclasses
import torch
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .super_weight import SuperWeight, MoESuperWeight
from utils.model_architectures import UniversalMLPHandler


class BaseSuperWeightDetector:
    """Base class for all super weight detectors"""
    
    def __init__(self, model, tokenizer, log_level=logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        self.iteration_data: List[Dict] = []
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Initialize model architecture handler
        self.mlp_handler = UniversalMLPHandler(model)
        
        # Get model info from utilities
        self.model_name = getattr(self.model, 'name_or_path', 'unknown').replace('/', '-')
        self.layers = self.mlp_handler.registry.find_layers(self.model)
        if self.layers is None:
            raise ValueError("Could not find layers in model using universal handler")
        self.num_layers = len(self.layers)
        
        # Device info
        self.device = next(model.parameters()).device
        
        # Model architecture info for detector selection
        self.model_info = self._analyze_model_architecture()
        
        self.logger.info(f"SuperWeightDetector initialized for {self.model_name}")
        self.logger.info(f"Model has {self.num_layers} layers")
    
    def _analyze_model_architecture(self) -> Dict:
        """Analyze model architecture to determine type"""
        # Add logic to detect MoE vs regular models
        sample_layer = self.layers[0] if self.layers else None
        
        is_moe = False
        architecture = 'standard'
        
        if sample_layer:
            # Check for Mixtral-style MoE
            if hasattr(sample_layer, 'block_sparse_moe'):
                is_moe = True
                architecture = 'mixtral'
            # Check for Switch Transformer style
            elif hasattr(sample_layer, 'mlp') and hasattr(sample_layer.mlp, 'experts'):
                is_moe = True  
                architecture = 'switch'
            # Add more MoE architecture checks as needed
        
        return {
            'is_moe': is_moe,
            'architecture': architecture,
            'num_layers': self.num_layers
        }
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the detector"""
        logger = logging.getLogger(f"SuperWeightDetector_{id(self)}")
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
    
    def _get_mlp_component_info(self, layer_idx: int) -> Tuple[str, str, torch.nn.Module]:
        """Get MLP component information for a layer using universal handler"""
        # Get architecture info and components
        arch_info = self.mlp_handler.get_mlp_architecture(layer_idx)
        components = self.mlp_handler.get_mlp_components(layer_idx)
        
        # Find the down/output projection component (the one we hook for detection)
        if 'down' in components:
            # Gated architecture (LLaMA, OLMo, Mistral, etc.)
            down_component = components['down']
            down_info = arch_info.components['down']
            mlp_base = self.mlp_handler.registry.find_mlp(self.layers[layer_idx])
            base_name = self._get_mlp_base_name(mlp_base)
            return base_name, down_info.component_name, down_component
        elif 'output' in components:
            # Standard architecture (GPT-2, etc.)
            output_component = components['output']
            output_info = arch_info.components['output']
            mlp_base = self.mlp_handler.registry.find_mlp(self.layers[layer_idx])
            base_name = self._get_mlp_base_name(mlp_base)
            return base_name, output_info.component_name, output_component
        else:
            raise ValueError(f"No down/output projection found in layer {layer_idx}")
    
    def _get_mlp_base_name(self, mlp_module) -> str:
        """Get the base name of the MLP module (mlp, feed_forward, etc.)"""
        # Check common MLP names
        if hasattr(mlp_module, '__class__'):
            class_name = mlp_module.__class__.__name__.lower()
            if 'mlp' in class_name:
                return 'mlp'
            elif 'feed' in class_name:
                return 'feed_forward'
        
        # Default fallback
        return 'mlp'
    
    def detect_super_weights(self, 
                           input_text: str = "Apple Inc. is a worldwide tech company.",
                           spike_threshold: float = 50.0,
                           max_iterations: int = 10) -> List[SuperWeight]:
        """
        Main method to detect super weights using iterative removal.
        
        Args:
            input_text: Text to use for detection
            spike_threshold: Threshold for detecting activation spikes
            max_iterations: Maximum number of iterations
            
        Returns:
            List of detected super weights
        """
        self.logger.info("Starting super weight detection")
        self.logger.info(f"Parameters: threshold={spike_threshold}, max_iterations={max_iterations}")
        
        # Reset state
        self._reset_detection_state()
        
        # Tokenize input
        input_tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        detected_super_weights = []
        processed_coordinates = set()
        
        # Iterative detection
        for iteration in range(max_iterations):
            self.logger.info(f"=== Iteration {iteration + 1} ===")
            
            # Detect super weights in this iteration
            new_super_weights, activation_data = self._detect_single_iteration(
                input_tokens, spike_threshold, iteration
            )
            
            # Filter out duplicates based on coordinates
            unique_super_weights = []
            for sw in new_super_weights:
                if sw.weight_key not in processed_coordinates:
                    unique_super_weights.append(sw)
                    processed_coordinates.add(sw.weight_key)
            
            if not unique_super_weights:
                self.logger.info("No new super weights found. Stopping detection.")
                break
            
            # Add to detected list
            detected_super_weights.extend(unique_super_weights)
            
            # Log results for this iteration
            self.logger.info(f"Found {len(unique_super_weights)} new super weights:")
            for i, sw in enumerate(unique_super_weights):
                self.logger.info(f"  {i+1}. {sw} - Input: {sw.input_value:.2f}, Output: {sw.output_value:.2f}")
            
            # Check termination condition
            max_output_mag = max(abs(val) for val in activation_data['max_output_values'])
            if max_output_mag < spike_threshold:
                self.logger.info(f"Maximum output magnitude ({max_output_mag:.2f}) below threshold. Stopping.")
                break
        
        # Log final results
        self._log_final_results(detected_super_weights)
        
        return detected_super_weights
    
    def _reset_detection_state(self):
        """Reset detection state for a new run"""
        self.iteration_data.clear()
    
    def _detect_single_iteration(self, input_tokens, spike_threshold: float, iteration: int):
        """Run detection for a single iteration"""
        # Storage for this iteration
        max_input_values = []
        max_input_indices = []
        max_output_values = []
        max_output_indices = []
        hooks = []
        
        def create_hook(layer_idx: int, base: str, down_name: str):
            def hook(module, inputs, output):
                # Process input tensor
                act_input = inputs[0].detach()
                abs_input = torch.abs(act_input)
                max_mag_input, max_idx_input = torch.max(abs_input.reshape(-1), dim=0)
                actual_input_value = act_input.reshape(-1)[max_idx_input].item()
                
                # Process output tensor
                act_output = output.detach()
                abs_output = torch.abs(act_output)
                max_mag_output, max_idx_output = torch.max(abs_output.reshape(-1), dim=0)
                actual_output_value = act_output.reshape(-1)[max_idx_output].item()
                
                # Extract channel indices
                if len(act_input.shape) == 3:  # [batch, seq, hidden]
                    h_input = max_idx_input.item() % act_input.shape[2]
                    max_input_values.append(actual_input_value)
                    max_input_indices.append(h_input)
                
                if len(act_output.shape) == 3:  # [batch, seq, hidden]
                    h_output = max_idx_output.item() % act_output.shape[2]
                    max_output_values.append(actual_output_value)
                    max_output_indices.append(h_output)
                
                self.logger.debug(f"Layer {layer_idx} {base}.{down_name} - "
                                f"Input: {actual_input_value:.2f}@{h_input}, "
                                f"Output: {actual_output_value:.2f}@{h_output}")
            
            return hook
        
        # Register hooks
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            base, down_name, module = self._get_mlp_component_info(layer_idx)
            hook = module.register_forward_hook(create_hook(layer_idx, base, down_name))
            hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            self.model(**input_tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Store activation data
        activation_data = {
            'max_input_values': max_input_values,
            'max_output_values': max_output_values,
            'max_input_indices': max_input_indices,
            'max_output_indices': max_output_indices
        }
        
        self.iteration_data.append({
            'iteration': iteration + 1,
            **activation_data
        })
        
        # Identify super weights
        super_weights = []
        for i in range(len(max_output_values)):
            if abs(max_output_values[i]) > spike_threshold and abs(max_input_values[i]) > spike_threshold:
                layer = self.layers[i]
                base, down_name, down_module = self._get_mlp_component_info(i)
                original_value = down_module.weight.data[max_output_indices[i], max_input_indices[i]].item()
                
                super_weights.append(SuperWeight(
                    layer=i,
                    row=max_output_indices[i],
                    column=max_input_indices[i],
                    component=f"{base}.{down_name}",
                    input_value=max_input_values[i],
                    output_value=max_output_values[i],
                    iteration_found=iteration + 1,
                    original_value=original_value
                ))
        
        self.logger.info(f"Found {len(super_weights)} potential super weights in iteration {iteration + 1}")
        
        return super_weights, activation_data
    
    def _log_final_results(self, detected_super_weights: List[SuperWeight]):
        """Log final detection results"""
        self.logger.info("=== DETECTION COMPLETE ===")
        self.logger.info(f"Found {len(detected_super_weights)} super weights:")
        
        for i, sw in enumerate(detected_super_weights):
            self.logger.info(f"{i+1}. {sw} (Iteration {sw.iteration_found})")
            self.logger.info(f"   Input: {sw.input_value:.2f}, Output: {sw.output_value:.2f}")
        
        # Analyze patterns
        self._analyze_patterns(detected_super_weights)
    
    def _analyze_patterns(self, detected_super_weights: List[SuperWeight]):
        """Analyze patterns in detected super weights"""
        if not detected_super_weights:
            return
        
        # Group by layer
        layer_groups = defaultdict(list)
        for sw in detected_super_weights:
            layer_groups[sw.layer].append(sw)
        
        self.logger.info(f"Super weights found in {len(layer_groups)} layers:")
        for layer, weights in layer_groups.items():
            self.logger.info(f"  Layer {layer}: {len(weights)} super weights")
        
        # Group by input channel
        input_channel_groups = defaultdict(list)
        for sw in detected_super_weights:
            key = (sw.layer, sw.column)
            input_channel_groups[key].append(sw)
        
        multi_output_channels = {k: v for k, v in input_channel_groups.items() if len(v) > 1}
        if multi_output_channels:
            self.logger.info("Input channels with multiple super weights:")
            for (layer, input_ch), group in multi_output_channels.items():
                output_channels = [sw.row for sw in group]
                self.logger.info(f"  Layer {layer}, Input channel {input_ch}: outputs {output_channels}")
    
    def get_iteration_data(self) -> List[Dict]:
        """Get raw activation data from all iterations"""
        return self.iteration_data.copy()
    


class SuperWeightDetector(BaseSuperWeightDetector):
    """
    Pure detection of super weights in transformer models.
    Does not manage weight states - only detects based on activation patterns.
    """
    
    def __init__(self, model, tokenizer, log_level=logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        self.iteration_data: List[Dict] = []
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Initialize model architecture handler
        self.mlp_handler = UniversalMLPHandler(model)
        
        # Get model info from utilities
        self.model_name = getattr(self.model, 'name_or_path', 'unknown').replace('/', '-')
        self.layers = self.mlp_handler.registry.find_layers(self.model)
        if self.layers is None:
            raise ValueError("Could not find layers in model using universal handler")
        self.num_layers = len(self.layers)
        
        self.logger.info(f"SuperWeightDetector initialized for {self.model_name}")
        self.logger.info(f"Model has {self.num_layers} layers")
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the detector"""
        logger = logging.getLogger(f"SuperWeightDetector_{id(self)}")
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
    
    def _get_mlp_component_info(self, layer_idx: int) -> Tuple[str, str, torch.nn.Module]:
        """Get MLP component information for a layer using universal handler"""
        # Get architecture info and components
        arch_info = self.mlp_handler.get_mlp_architecture(layer_idx)
        components = self.mlp_handler.get_mlp_components(layer_idx)
        
        # Find the down/output projection component (the one we hook for detection)
        if 'down' in components:
            # Gated architecture (LLaMA, OLMo, Mistral, etc.)
            down_component = components['down']
            down_info = arch_info.components['down']
            mlp_base = self.mlp_handler.registry.find_mlp(self.layers[layer_idx])
            base_name = self._get_mlp_base_name(mlp_base)
            return base_name, down_info.component_name, down_component
        elif 'output' in components:
            # Standard architecture (GPT-2, etc.)
            output_component = components['output']
            output_info = arch_info.components['output']
            mlp_base = self.mlp_handler.registry.find_mlp(self.layers[layer_idx])
            base_name = self._get_mlp_base_name(mlp_base)
            return base_name, output_info.component_name, output_component
        else:
            raise ValueError(f"No down/output projection found in layer {layer_idx}")
    
    def _get_mlp_base_name(self, mlp_module) -> str:
        """Get the base name of the MLP module (mlp, feed_forward, etc.)"""
        # Check common MLP names
        if hasattr(mlp_module, '__class__'):
            class_name = mlp_module.__class__.__name__.lower()
            if 'mlp' in class_name:
                return 'mlp'
            elif 'feed' in class_name:
                return 'feed_forward'
        
        # Default fallback
        return 'mlp'
    
    def detect_super_weights(self, 
                           input_text: str = "Apple Inc. is a worldwide tech company.",
                           spike_threshold: float = 50.0,
                           max_iterations: int = 10) -> List[SuperWeight]:
        """
        Main method to detect super weights using iterative removal.
        
        Args:
            input_text: Text to use for detection
            spike_threshold: Threshold for detecting activation spikes
            max_iterations: Maximum number of iterations
            
        Returns:
            List of detected super weights
        """
        self.logger.info("Starting super weight detection")
        self.logger.info(f"Parameters: threshold={spike_threshold}, max_iterations={max_iterations}")
        
        # Reset state
        self._reset_detection_state()
        
        # Tokenize input
        input_tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        detected_super_weights = []
        processed_coordinates = set()
        
        # Iterative detection
        for iteration in range(max_iterations):
            self.logger.info(f"=== Iteration {iteration + 1} ===")
            
            # Detect super weights in this iteration
            new_super_weights, activation_data = self._detect_single_iteration(
                input_tokens, spike_threshold, iteration
            )
            
            # Filter out duplicates based on coordinates
            unique_super_weights = []
            for sw in new_super_weights:
                if sw.weight_key not in processed_coordinates:
                    unique_super_weights.append(sw)
                    processed_coordinates.add(sw.weight_key)
            
            if not unique_super_weights:
                self.logger.info("No new super weights found. Stopping detection.")
                break
            
            # Add to detected list
            detected_super_weights.extend(unique_super_weights)
            
            # Log results for this iteration
            self.logger.info(f"Found {len(unique_super_weights)} new super weights:")
            for i, sw in enumerate(unique_super_weights):
                self.logger.info(f"  {i+1}. {sw} - Input: {sw.input_value:.2f}, Output: {sw.output_value:.2f}")
            
            # Check termination condition
            max_output_mag = max(abs(val) for val in activation_data['max_output_values'])
            if max_output_mag < spike_threshold:
                self.logger.info(f"Maximum output magnitude ({max_output_mag:.2f}) below threshold. Stopping.")
                break
        
        # Log final results
        self._log_final_results(detected_super_weights)
        
        return detected_super_weights
    
    def _reset_detection_state(self):
        """Reset detection state for a new run"""
        self.iteration_data.clear()
    
    def _detect_single_iteration(self, input_tokens, spike_threshold: float, iteration: int):
        """Run detection for a single iteration"""
        # Storage for this iteration
        max_input_values = []
        max_input_indices = []
        max_output_values = []
        max_output_indices = []
        hooks = []
        
        def create_hook(layer_idx: int, base: str, down_name: str):
            def hook(module, inputs, output):
                # Process input tensor
                act_input = inputs[0].detach()
                abs_input = torch.abs(act_input)
                max_mag_input, max_idx_input = torch.max(abs_input.reshape(-1), dim=0)
                actual_input_value = act_input.reshape(-1)[max_idx_input].item()
                
                # Process output tensor
                act_output = output.detach()
                abs_output = torch.abs(act_output)
                max_mag_output, max_idx_output = torch.max(abs_output.reshape(-1), dim=0)
                actual_output_value = act_output.reshape(-1)[max_idx_output].item()
                
                # Extract channel indices
                if len(act_input.shape) == 3:  # [batch, seq, hidden]
                    h_input = max_idx_input.item() % act_input.shape[2]
                    max_input_values.append(actual_input_value)
                    max_input_indices.append(h_input)
                
                if len(act_output.shape) == 3:  # [batch, seq, hidden]
                    h_output = max_idx_output.item() % act_output.shape[2]
                    max_output_values.append(actual_output_value)
                    max_output_indices.append(h_output)
                
                self.logger.debug(f"Layer {layer_idx} {base}.{down_name} - "
                                f"Input: {actual_input_value:.2f}@{h_input}, "
                                f"Output: {actual_output_value:.2f}@{h_output}")
            
            return hook
        
        # Register hooks
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            base, down_name, module = self._get_mlp_component_info(layer_idx)
            hook = module.register_forward_hook(create_hook(layer_idx, base, down_name))
            hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            self.model(**input_tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Store activation data
        activation_data = {
            'max_input_values': max_input_values,
            'max_output_values': max_output_values,
            'max_input_indices': max_input_indices,
            'max_output_indices': max_output_indices
        }
        
        self.iteration_data.append({
            'iteration': iteration + 1,
            **activation_data
        })
        
        # Identify super weights
        super_weights = []
        for i in range(len(max_output_values)):
            if abs(max_output_values[i]) > spike_threshold and abs(max_input_values[i]) > spike_threshold:
                layer = self.layers[i]
                base, down_name, down_module = self._get_mlp_component_info(i)
                original_value = down_module.weight.data[max_output_indices[i], max_input_indices[i]].item()
                
                super_weights.append(SuperWeight(
                    layer=i,
                    row=max_output_indices[i],
                    column=max_input_indices[i],
                    component=f"{base}.{down_name}",
                    input_value=max_input_values[i],
                    output_value=max_output_values[i],
                    iteration_found=iteration + 1,
                    original_value=original_value
                ))
        
        self.logger.info(f"Found {len(super_weights)} potential super weights in iteration {iteration + 1}")
        
        return super_weights, activation_data
    
    def _log_final_results(self, detected_super_weights: List[SuperWeight]):
        """Log final detection results"""
        self.logger.info("=== DETECTION COMPLETE ===")
        self.logger.info(f"Found {len(detected_super_weights)} super weights:")
        
        for i, sw in enumerate(detected_super_weights):
            self.logger.info(f"{i+1}. {sw} (Iteration {sw.iteration_found})")
            self.logger.info(f"   Input: {sw.input_value:.2f}, Output: {sw.output_value:.2f}")
        
        # Analyze patterns
        self._analyze_patterns(detected_super_weights)
    
    def _analyze_patterns(self, detected_super_weights: List[SuperWeight]):
        """Analyze patterns in detected super weights"""
        if not detected_super_weights:
            return
        
        # Group by layer
        layer_groups = defaultdict(list)
        for sw in detected_super_weights:
            layer_groups[sw.layer].append(sw)
        
        self.logger.info(f"Super weights found in {len(layer_groups)} layers:")
        for layer, weights in layer_groups.items():
            self.logger.info(f"  Layer {layer}: {len(weights)} super weights")
        
        # Group by input channel
        input_channel_groups = defaultdict(list)
        for sw in detected_super_weights:
            key = (sw.layer, sw.column)
            input_channel_groups[key].append(sw)
        
        multi_output_channels = {k: v for k, v in input_channel_groups.items() if len(v) > 1}
        if multi_output_channels:
            self.logger.info("Input channels with multiple super weights:")
            for (layer, input_ch), group in multi_output_channels.items():
                output_channels = [sw.row for sw in group]
                self.logger.info(f"  Layer {layer}, Input channel {input_ch}: outputs {output_channels}")
    
    def get_iteration_data(self) -> List[Dict]:
        """Get raw activation data from all iterations"""
        return self.iteration_data.copy()
    


class MoESuperWeightDetector(BaseSuperWeightDetector):
    """
    Super weight detector specialized for MoE models
    
    This class handles the complexity of MoE architectures where super weights
    can appear in individual expert modules and interact with routing mechanisms.
    """
    
    def __init__(self, model, tokenizer, architecture_type: str = None, **kwargs):
        """Initialize MoE detector"""
        super().__init__(model, tokenizer)
        
        # MoE-specific initialization
        self.architecture_type = architecture_type or self.model_info['architecture']
        self.expert_info = self._analyze_expert_structure()
        
        self.logger.info(f"Initialized MoE detector for {self.architecture_type} architecture")
        self.logger.info(f"Expert structure: {self.expert_info}")
    
    def _analyze_expert_structure(self) -> Dict:
        """Analyze the structure of experts in this model"""
        expert_info = {
            'experts_per_layer': 0,
            'expert_hidden_size': None,
            'routing_method': 'unknown',
            'has_shared_experts': False,
            'expert_components': []
        }
        
        if not self.layers:
            return expert_info
        
        # Analyze first layer to understand structure
        sample_layer = self.layers[0]
        
        if self.architecture_type == 'mixtral':
            expert_info.update(self._analyze_mixtral_structure(sample_layer))
        elif self.architecture_type == 'switch':
            expert_info.update(self._analyze_switch_structure(sample_layer))
        else:
            self.logger.warning(f"Unknown MoE architecture: {self.architecture_type}")
        
        return expert_info
    
    def _analyze_mixtral_structure(self, layer) -> Dict:
        """Analyze Mixtral-specific structure"""
        info = {}
        
        if hasattr(layer, 'block_sparse_moe'):
            moe = layer.block_sparse_moe
            
            # Count experts
            if hasattr(moe, 'experts'):
                info['experts_per_layer'] = len(moe.experts)
                
                # Analyze expert components
                if moe.experts:
                    sample_expert = moe.experts[0]
                    components = []
                    for attr in ['w1', 'w2', 'w3', 'gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(sample_expert, attr):
                            components.append(attr)
                    info['expert_components'] = components
                    
                    # Get hidden size from w2 (down projection)
                    if hasattr(sample_expert, 'w2'):
                        info['expert_hidden_size'] = sample_expert.w2.in_features
            
            # Routing method
            if hasattr(moe, 'gate'):
                info['routing_method'] = 'top_k_gating'
        
        return info
    
    def _analyze_switch_structure(self, layer) -> Dict:
        """Analyze Switch Transformer structure"""
        info = {}
        
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            experts = layer.mlp.experts
            info['experts_per_layer'] = len(experts)
            info['routing_method'] = 'switch_routing'
            
            # Analyze components
            if experts:
                sample_expert = experts[0]
                components = []
                for attr in ['wi', 'wo', 'dense']:
                    if hasattr(sample_expert, attr):
                        components.append(attr)
                info['expert_components'] = components
        
        return info
    
    def detect_super_weights(self, 
                           input_text: str = "Apple Inc. is a worldwide tech company.",
                           spike_threshold: float = 20.0,  # Lower threshold for MoE
                           max_iterations: int = 5) -> List[MoESuperWeight]:
        """
        Detect super weights in MoE model across all expert components
        """
        self.logger.info("Starting MoE super weight detection")
        self.logger.info(f"Architecture: {self.architecture_type}")
        self.logger.info(f"Parameters: threshold={spike_threshold}, max_iterations={max_iterations}")
        
        self._reset_detection_state()
        
        detected_super_weights = []
        processed_coordinates = set()
        
        # For MoE, we need to check all components, not just down_proj
        for iteration in range(max_iterations):
            self.logger.info(f"=== MoE Iteration {iteration + 1} ===")
            
            new_super_weights, activation_data = self.detect_super_weights_single_iteration(
                input_text, spike_threshold, iteration
            )
            
            # Filter duplicates
            unique_super_weights = []
            for sw in new_super_weights:
                # For MoE, include expert_id in coordinate key
                coord_key = (sw.layer, sw.row, sw.column, sw.component, getattr(sw, 'expert_id', None))
                if coord_key not in processed_coordinates:
                    unique_super_weights.append(sw)
                    processed_coordinates.add(coord_key)
            
            if not unique_super_weights:
                self.logger.info("No new MoE super weights found. Stopping detection.")
                break
            
            detected_super_weights.extend(unique_super_weights)
            
            # Log results
            self.logger.info(f"Found {len(unique_super_weights)} new MoE super weights:")
            for i, sw in enumerate(unique_super_weights):
                expert_info = f" (Expert {sw.expert_id})" if hasattr(sw, 'expert_id') else ""
                self.logger.info(f"  {i+1}. {sw}{expert_info}")
        
        # Analyze MoE-specific patterns
        if detected_super_weights:
            self._analyze_moe_patterns(detected_super_weights)
        
        return detected_super_weights
    
    def detect_super_weights_single_iteration(self, input_text: str, spike_threshold: float = 20.0, iteration: int = 0):
        """Detect super weights across all experts in MoE model"""
        
        detected_super_weights = []
        
        for layer_idx in range(self.num_layers):
            if not self.mlp_handler.is_moe_layer(layer_idx):
                continue  # Skip non-MoE layers
            
            # Get all experts for this layer
            num_experts = self.mlp_handler.get_mlp_architecture(layer_idx).moe_info.num_experts
            
            for expert_idx in range(num_experts):
                # Check each component of each expert
                expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_idx)
                
                for component_type, component_module in expert_components.items():
                    # Hook and detect spikes in this expert component
                    spikes = self._detect_spikes_in_component(
                        layer_idx, component_module, component_type, 
                        input_text, spike_threshold,
                        expert_id=expert_idx
                    )
                    detected_super_weights.extend(spikes)
        
        return detected_super_weights, {}
    
    def _analyze_moe_patterns(self, super_weights: List[MoESuperWeight]):
        """Analyze MoE-specific patterns"""
        # Expert distribution analysis
        expert_distribution = {}
        for sw in super_weights:
            if hasattr(sw, 'expert_id'):
                expert_distribution[sw.expert_id] = expert_distribution.get(sw.expert_id, 0) + 1
        
        self.logger.info(f"Super weights found across {len(expert_distribution)} experts:")
        for expert_id, count in sorted(expert_distribution.items()):
            self.logger.info(f"  Expert {expert_id}: {count} super weights")