import dataclasses
import torch
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .super_weight import SuperWeight, MoESuperWeight
from utils.model_architectures import UniversalMLPHandler


class BaseSuperWeightDetector:
    """Base class for all super weight detectors"""
    
    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler, log_level=logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        self.mlp_handler = mlp_handler
        self.iteration_data: List[Dict] = []
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Get basic model info
        self.model_name = getattr(self.model, 'name_or_path', 'unknown').replace('/', '-')
        self.layers = self.mlp_handler.registry.find_layers(self.model)
        if self.layers is None:
            raise ValueError("Could not find layers in model using universal handler")
        self.num_layers = len(self.layers)
        
        # Device info
        self.device = next(model.parameters()).device
        
        self.logger.info(f"{self.__class__.__name__} initialized for {self.model_name}")
        self.logger.info(f"Model has {self.num_layers} layers")
    
    def _setup_logger(self, log_level) -> logging.Logger:
        """Setup logging for the detector"""
        logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
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
    
    def _reset_detection_state(self):
        """Reset detection state for a new run"""
        self.iteration_data.clear()
    
    def get_iteration_data(self) -> List[Dict]:
        """Get raw activation data from all iterations"""
        return self.iteration_data.copy()
    
    def detect_super_weights(self, **kwargs):
        """Abstract method - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement detect_super_weights")


class SuperWeightDetector(BaseSuperWeightDetector):
    """
    Detection of super weights in standard transformer models.
    Focuses on down/output projection components in MLP layers.
    """
    
    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler, log_level=logging.INFO):
        super().__init__(model, tokenizer, mlp_handler, log_level)
    
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


class MoESuperWeightDetector(BaseSuperWeightDetector):
    """
    Super weight detector specialized for MoE models.
    Handles expert-specific detection across all components.
    """
    
    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler, 
                 architecture_type: str, log_level=logging.INFO):
        super().__init__(model, tokenizer, mlp_handler, log_level)
        
        # MoE-specific initialization
        self.architecture_type = architecture_type
        self.expert_info = self._analyze_expert_structure()
        
        self.logger.info(f"Initialized MoE detector for {self.architecture_type} architecture")
    
    def _analyze_expert_structure(self) -> Dict:
        """Analyze expert structure using the MLP handler"""
        if not self.layers:
            return {}
        
        # Get info from first MoE layer using the new architecture detection
        first_moe_layer = next(
            (idx for idx in range(len(self.layers)) 
             if self.mlp_handler.is_moe_layer(idx)),
            None
        )
        
        if first_moe_layer is None:
            return {}
        
        arch_info = self.mlp_handler.get_mlp_architecture(first_moe_layer)
        if not arch_info.is_moe or not arch_info.moe_info:
            return {}
        
        return {
            'experts_per_layer': arch_info.moe_info.num_experts,
            'routing_method': arch_info.moe_info.routing_method,
            'experts_per_token': arch_info.moe_info.experts_per_token,
            'expert_components': [list(expert.components.keys()) for expert in arch_info.moe_info.experts],
            'first_moe_layer': first_moe_layer
        }
    
    def detect_super_weights(self, 
                       input_text: str = "Apple Inc. is a worldwide tech company.",
                       spike_threshold: float = 50.0,
                       max_iterations: int = 10) -> List[MoESuperWeight]:
        """
        Detect super weights in MoE model.
        """
        self.logger.info(f"Starting MoE super weight detection with threshold {spike_threshold}")
        
        # Reset detection state
        self._reset_detection_state()
        
        # Get MoE layers
        moe_layers = [i for i in range(len(self.layers)) if self.mlp_handler.is_moe_layer(i)]
        
        if not moe_layers:
            self.logger.warning("No MoE layers found in model")
            return []
        
        super_weights = []
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        for iteration in range(max_iterations):
            self.logger.info(f"MoE Detection iteration {iteration + 1}/{max_iterations}")
            
            iteration_super_weights = []
            
            # Process each MoE layer
            for layer_idx in moe_layers:
                layer_super_weights = self._detect_moe_layer_super_weights(
                    tokens, layer_idx, spike_threshold, iteration
                )
                iteration_super_weights.extend(layer_super_weights)
            
            if not iteration_super_weights:
                self.logger.info(f"No super weights found in iteration {iteration + 1}")
                break
            
            super_weights.extend(iteration_super_weights)
            self.logger.info(f"Found {len(iteration_super_weights)} super weights in iteration {iteration + 1}")
        
        self.logger.info(f"MoE detection complete. Total super weights: {len(super_weights)}")
        return super_weights

    def _detect_moe_layer_super_weights(self, tokens, layer_idx: int, spike_threshold: float, iteration: int) -> List[MoESuperWeight]:
        """Detect super weights in a specific MoE layer"""
        
        arch_info = self.mlp_handler.get_mlp_architecture(layer_idx)
        if not arch_info.is_moe:
            return []
        
        experts = self.mlp_handler.get_moe_experts(layer_idx)
        layer_super_weights = []
        
        for expert_idx, expert in enumerate(experts):
            # Get expert components
            expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_idx)
            
            # Check each output component (down/output projection)
            for comp_type, module in expert_components.items():
                if comp_type in ['down', 'output']:  # Output projection components
                    expert_super_weights = self._detect_expert_component_super_weights(
                        tokens, layer_idx, expert_idx, comp_type, module, spike_threshold, iteration
                    )
                    layer_super_weights.extend(expert_super_weights)
        
        return layer_super_weights

    def _detect_expert_component_super_weights(self, tokens, layer_idx: int, expert_idx: int, 
                                             component_type: str, module, spike_threshold: float, 
                                             iteration: int) -> List[MoESuperWeight]:
        """Detect super weights in a specific expert component"""
        
        super_weights = []
        
        # Hook to capture activations
        activations = []
        
        def capture_activation(module, input, output):
            if isinstance(input, tuple):
                activations.append(input[0].detach().cpu())
            else:
                activations.append(input.detach().cpu())
        
        hook = module.register_forward_hook(capture_activation)
        
        try:
            with torch.no_grad():
                self.model(**tokens)
            
            if activations:
                activation = activations[0]  # [batch, seq, hidden]
                
                # Find super activations (high magnitude inputs to this component)
                # Take first token, first batch
                input_vector = activation[0, 0, :]  # [hidden_dim]
                
                # Find channels with extreme activations
                abs_activations = torch.abs(input_vector)
                threshold_value = torch.quantile(abs_activations, 0.99)  # Top 1%
                
                if threshold_value > spike_threshold:
                    super_channels = torch.where(abs_activations > spike_threshold)[0]
                    
                    for channel_idx in super_channels:
                        channel_idx = int(channel_idx)
                        input_value = float(input_vector[channel_idx])
                        
                        # Create MoESuperWeight for each row in this component
                        weight_matrix = module.weight
                        for row_idx in range(weight_matrix.shape[0]):
                            output_value = float(weight_matrix[row_idx, channel_idx] * input_value)
                            
                            if abs(output_value) > spike_threshold:
                                super_weight = MoESuperWeight(
                                    layer=layer_idx,
                                    expert=expert_idx,
                                    component=f"experts.{expert_idx}.{self._get_component_name(module)}",
                                    row=row_idx,
                                    column=channel_idx,
                                    input_value=input_value,
                                    output_value=output_value,
                                    iteration=iteration
                                )
                                super_weights.append(super_weight)
        
        finally:
            hook.remove()
        
        return super_weights

    def _get_component_name(self, module) -> str:
        """Get the component name from module"""
        class_name = module.__class__.__name__.lower()
        if 'down' in class_name or 'proj' in class_name:
            return 'down_proj'
        elif 'output' in class_name:
            return 'output'
        else:
            return 'unknown'