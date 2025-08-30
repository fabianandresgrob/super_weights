import dataclasses
import torch
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .super_weight import SuperWeight, MoESuperWeight
from utils.model_architectures import UniversalMLPHandler
from management.manager import SuperWeightManager


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
    
    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler, 
                 manager: SuperWeightManager, log_level=logging.INFO):
        super().__init__(model, tokenizer, mlp_handler, log_level)

        # Initialize or use provided manager
        if manager is None:
            self.manager = SuperWeightManager(model, mlp_handler, log_level)
            self.owns_manager = True
        else:
            self.manager = manager
            self.owns_manager = False
        
        self.logger.info("SuperWeightDetector initialized with manager integration")
    
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
                           max_iterations: int = 10,
                           zero_detected_weights: bool = True) -> List[SuperWeight]:
        """
        Main method to detect super weights using iterative removal.
        
        Args:
            input_text: Text to use for detection
            spike_threshold: Threshold for detecting activation spikes
            max_iterations: Maximum number of iterations
            zero_detected_weights: Whether to zero out detected weights between iterations
            
        Returns:
            List of detected super weights
        """
        self.logger.info("Starting super weight detection")
        self.logger.info(f"Parameters: threshold={spike_threshold}, max_iterations={max_iterations}")
        self.logger.info(f"Zero detected weights: {zero_detected_weights}")
        
        # Reset state
        self._reset_detection_state()
        
        # Tokenize input
        input_tokens = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        detected_super_weights = []
        processed_coordinates = set()
        
        try:
            # Iterative detection
            for iteration in range(max_iterations):
                self.logger.info(f"=== Iteration {iteration + 1} ===")

                # Assert for each already detected super weight, that it's weight is set to 0
                for sw in detected_super_weights:
                    base, down_name, module = self._get_mlp_component_info(sw.layer)
                    current_weight = module.weight[sw.row, sw.column].item()
                    assert current_weight == 0, f"Super weight {sw} is not zeroed out."

                # Detect super weights in this iteration
                new_super_weights = self._detect_single_iteration(
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
                
                # Zero out detected weights for next iteration
                if zero_detected_weights and iteration < max_iterations - 1:  # Don't zero on last iteration
                    self.logger.info(f"Zeroing {len(unique_super_weights)} detected super weights...")
                    success = self.manager.zero_super_weights(unique_super_weights)
                    if not success:
                        self.logger.warning("Some weights could not be zeroed. Detection may be affected.")
                
                # Log results for this iteration
                self.logger.info(f"Found {len(unique_super_weights)} new super weights:")
                for i, sw in enumerate(unique_super_weights):
                    self.logger.info(f"  {i+1}. {sw} - Input: {sw.input_value:.2f}, Output: {sw.output_value:.2f}")
            
            # Log final results
            self._log_final_results(detected_super_weights)
            
            return detected_super_weights
            
        finally:
            # Always restore weights when detection is complete
            if zero_detected_weights and detected_super_weights:
                self.logger.info("Restoring all modified weights...")
                self.manager.restore_super_weights(detected_super_weights)
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        if self.owns_manager:
            self.manager.restore_all()

    def _safely_get_weight_column(self, module, h_input: int, device):
        """Safely get weight column, handling device mapping"""
        try:
            if hasattr(module.weight, 'device') and module.weight.device.type == 'meta':
                # Weight is on meta device, we can't access it
                return None
            
            # Ensure weight is on the same device as the input
            weight = module.weight
            if weight.device != device:
                # Move weight to target device temporarily
                weight_column = weight[:, h_input].to(device).detach().cpu()
            else:
                weight_column = weight[:, h_input].detach().cpu()
            
            return weight_column
        except Exception as e:
            self.logger.warning(f"Could not access weight column: {e}")
            return None
    
    def _detect_single_iteration(self, input_tokens, spike_threshold: float, iteration: int):
        """Run detection for a single iteration"""
        # Storage for this iteration
        max_input_values = [0.0] * self.num_layers
        max_input_indices = [0] * self.num_layers
        max_output_values = [0.0] * self.num_layers
        max_output_indices = [0] * self.num_layers
        weight_columns = [None] * self.num_layers
        weight_values = [0.0] * self.num_layers
        hooks = []

        def create_pre_hook(layer_idx: int):
            def pre_hook(module, inputs):
                act_input = inputs[0].detach()
                abs_input = torch.abs(act_input)
                max_mag_input, max_idx_input = torch.max(abs_input.reshape(-1), dim=0)
                actual_input_value = act_input.reshape(-1)[max_idx_input].item()

                if len(act_input.shape) == 3:
                    h_input = max_idx_input.item() % act_input.shape[2]
                    max_input_values[layer_idx] = actual_input_value
                    max_input_indices[layer_idx] = h_input
                    weight_columns[layer_idx] = self._safely_get_weight_column(module, h_input, device=self.device)

            return pre_hook

        def create_post_hook(layer_idx: int, base: str, down_name: str):
            def hook(module, inputs, output):
                act_output = output.detach()
                abs_output = torch.abs(act_output)
                max_mag_output, max_idx_output = torch.max(abs_output.reshape(-1), dim=0)
                actual_output_value = act_output.reshape(-1)[max_idx_output].item()

                h_output = None
                if len(act_output.shape) == 3:
                    h_output = max_idx_output.item() % act_output.shape[2]
                    max_output_values[layer_idx] = actual_output_value
                    max_output_indices[layer_idx] = h_output
                    if weight_columns[layer_idx] is not None:
                        weight_values[layer_idx] = weight_columns[layer_idx][h_output].item()

                h_input = max_input_indices[layer_idx]
                self.logger.debug(
                    f"Layer {layer_idx} {base}.{down_name} - "
                    f"Input: {max_input_values[layer_idx]:.2f}@{h_input}, "
                    f"Output: {actual_output_value:.2f}@{h_output}"
                )

            return hook

        # Register hooks
        for layer_idx in range(self.num_layers):
            base, down_name, module = self._get_mlp_component_info(layer_idx)
            pre_hook = module.register_forward_pre_hook(create_pre_hook(layer_idx))
            post_hook = module.register_forward_hook(create_post_hook(layer_idx, base, down_name))
            hooks.extend([pre_hook, post_hook])
        
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
            'max_output_indices': max_output_indices,
            'weight_values': weight_values,
        }
        
        self.iteration_data.append({
            'iteration': iteration + 1,
            **activation_data
        })
        
        # Identify super weights
        super_weights = []
        for i in range(self.num_layers):
            if abs(max_output_values[i]) > spike_threshold and abs(max_input_values[i]) > spike_threshold:
                base, down_name, _ = self._get_mlp_component_info(i)
                original_value = weight_values[i]
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
        
        return super_weights
    
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
    Uses a two-phase approach: router analysis + active expert analysis.
    """
    
    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler, 
                 architecture_type: str, log_level=logging.INFO):
        super().__init__(model, tokenizer, mlp_handler, log_level)
        
        # MoE-specific initialization
        self.architecture_type = architecture_type
        
        # Cache for expert activation patterns
        self.expert_activation_cache = {}

        self._temp_modified_weights = {}  # Store original values for restoration
        
        self.logger.info(f"Initialized MoE detector for {self.architecture_type} architecture")
    
    def detect_super_weights(self, 
                       input_text: str = "Apple Inc. is a worldwide tech company.",
                       spike_threshold: float = 50.0,
                       max_iterations: int = 10,
                       router_analysis_samples: int = 5) -> List[MoESuperWeight]:
        """
        Detect super weights in MoE model using two-phase approach.
        
        Args:
            input_text: Text for detection
            spike_threshold: Threshold for spike detection
            max_iterations: Max iterations for super weight detection
            router_analysis_samples: Number of samples for router analysis
        """
        self.logger.info(f"Starting MoE super weight detection with threshold {spike_threshold}")
        
        # Reset detection state
        self._reset_detection_state()
        
        # Phase 1: Analyze router patterns to find frequently activated experts
        self.logger.info("Phase 1: Analyzing router patterns...")
        active_experts = self._analyze_router_patterns(
            input_text, router_analysis_samples
        )
        
        if not active_experts:
            self.logger.warning("No consistently active experts found")
            return []
        
        # Phase 2: Detect super weights in active experts only
        self.logger.info(f"Phase 2: Analyzing active experts in {len(active_experts)} layers...")
        super_weights = self._detect_active_expert_super_weights(
            input_text, active_experts, spike_threshold, max_iterations
        )
        
        self.logger.info(f"MoE detection complete. Total super weights: {len(super_weights)}")
        return super_weights
    
    def _analyze_router_patterns(self, input_text: str, num_samples: int) -> Dict[int, List[int]]:
        """
        Analyze router patterns to identify frequently activated experts.
        
        Returns:
            Dict mapping layer_idx -> list of frequently activated expert indices
        """
        # Get sample inputs (could be variations of the input or different texts)
        sample_inputs = self._generate_sample_inputs(input_text, num_samples)
        
        # Track expert activations across samples
        expert_activations = defaultdict(lambda: defaultdict(int))
        
        for sample_idx, sample_text in enumerate(sample_inputs):
            self.logger.debug(f"Processing sample {sample_idx + 1}/{num_samples}")
            tokens = self.tokenizer(sample_text, return_tensors='pt').to(self.model.device)
            
            # Hook routers to capture expert selections
            router_hooks = []
            
            def create_router_hook(layer_idx: int):
                def hook(module, input, output):
                    # Extract expert selections from router output
                    routing_info = self.mlp_handler.get_routing_info(layer_idx)
                    if routing_info:
                        selected_experts = self._extract_selected_experts(output, routing_info)
                        for expert_idx in selected_experts:
                            expert_activations[layer_idx][expert_idx] += 1
                return hook
            
            # Hook routers in MoE layers - check each layer dynamically
            for layer_idx in range(self.num_layers):
                if self.mlp_handler.is_moe_layer(layer_idx):
                    router_module = self.mlp_handler.get_router_module(layer_idx)
                    if router_module:
                        hook = router_module.register_forward_hook(create_router_hook(layer_idx))
                        router_hooks.append(hook)
                    else:
                        self.logger.warning(f"No router module found for MoE layer {layer_idx}")

            # Run forward pass
            with torch.no_grad():
                self.model(**tokens)
            
            # Clean up hooks
            for hook in router_hooks:
                hook.remove()
        
        # Identify frequently activated experts (activated in >50% of samples)
        activation_threshold = num_samples * 0.5
        active_experts = {}
        
        for layer_idx, expert_counts in expert_activations.items():
            active_expert_list = [
                expert_idx for expert_idx, count in expert_counts.items()
                if count >= activation_threshold
            ]
            if active_expert_list:
                active_experts[layer_idx] = active_expert_list
        
        # Log results
        for layer_idx, experts in active_experts.items():
            self.logger.info(f"Layer {layer_idx}: Active experts {experts}")
        
        return active_experts
    
    def _generate_sample_inputs(self, base_text: str, num_samples: int) -> List[str]:
        """Generate sample inputs for router analysis"""
        # Simple approach: use the same text multiple times
        # Could be enhanced with paraphrases, variations, etc.
        samples = [base_text]
        
        # Add some variations if needed
        variations = [
            "Apple Inc. is a technology company.",
            "Apple is a major tech corporation.",
            "The company Apple Inc. develops technology.",
            "Apple develops consumer electronics.",
        ]
        
        for i in range(min(num_samples - 1, len(variations))):
            samples.append(variations[i])
        
        # Fill remaining with base text
        while len(samples) < num_samples:
            samples.append(base_text)
        
        return samples[:num_samples]
    
    def _extract_selected_experts(self, router_output, routing_info, layer_idx: int) -> List[int]:
        """Extract selected expert indices from router output with robust parsing"""
        try:
            # Detect model type for specific handling
            model_config = getattr(self.model, 'config', None)
            model_type = getattr(model_config, 'model_type', '').lower()
            
            if 'mixtral' in model_type or 'mistral' in model_type:
                # Mixtral-specific handling
                if hasattr(router_output, 'indices'):
                    return router_output.indices.flatten().unique().tolist()
                elif isinstance(router_output, tuple) and len(router_output) >= 2:
                    # router_output might be (router_logits, indices)
                    indices = router_output[1]
                    if torch.is_tensor(indices):
                        return indices.flatten().unique().tolist()
            
            # Generic tensor handling
            if torch.is_tensor(router_output):
                # Assume it's router logits, get top-k
                k = getattr(routing_info, 'experts_per_token', 2) if routing_info else 2
                
                if len(router_output.shape) == 2:  # [seq_len, num_experts]
                    # Get top-k across all sequence positions
                    _, indices = torch.topk(router_output, k, dim=-1)
                    return torch.unique(indices.flatten()).tolist()
                elif len(router_output.shape) == 3:  # [batch, seq_len, num_experts]
                    # Flatten batch and sequence dimensions
                    flat_output = router_output.view(-1, router_output.shape[-1])
                    _, indices = torch.topk(flat_output, k, dim=-1)
                    return torch.unique(indices.flatten()).tolist()
            
            # Fallback: assume it's already indices
            if isinstance(router_output, (list, tuple)):
                return list(router_output)
            
            self.logger.warning(f"Could not parse router output format for layer {layer_idx}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error parsing router output for layer {layer_idx}: {e}")
            return []

    def _detect_active_expert_super_weights(self, input_text: str, 
                                          active_experts: Dict[int, List[int]], 
                                          spike_threshold: float, 
                                          max_iterations: int) -> List[MoESuperWeight]:
        """Detect super weights in active experts with iterative suppression"""
        detected_super_weights = []
        processed_coordinates = set()
        
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        
        for iteration in range(max_iterations):
            self.logger.info(f"MoE Detection iteration {iteration + 1}/{max_iterations}")
            
            iteration_super_weights = []
            
            for layer_idx, expert_indices in active_experts.items():
                layer_super_weights = self._detect_layer_expert_super_weights(
                    tokens, layer_idx, expert_indices, spike_threshold, iteration
                )
                
                # Filter out already processed coordinates
                new_super_weights = []
                for sw in layer_super_weights:
                    coord = (sw.layer, sw.expert_id, sw.row, sw.column)
                    if coord not in processed_coordinates:
                        new_super_weights.append(sw)
                        processed_coordinates.add(coord)
                
                iteration_super_weights.extend(new_super_weights)
            
            if not iteration_super_weights:
                self.logger.info(f"No new super weights found in iteration {iteration + 1}")
                break
            
            detected_super_weights.extend(iteration_super_weights)
            
            # Zero the detected weights for next iteration
            if iteration < max_iterations - 1:  # Don't zero on last iteration
                self._temporarily_zero_weights(iteration_super_weights)
        
        # Restore all weights after detection
        self._restore_all_weights()
        
        return detected_super_weights
    
    def _temporarily_zero_weights(self, super_weights: List[MoESuperWeight]):
        """Temporarily zero detected super weights"""
        for sw in super_weights:
            try:
                # Get the expert module and component
                expert_module = self.mlp_handler.get_expert_module(sw.layer, sw.expert_id)
                if expert_module:
                    component_parts = sw.component.split('.')[-1]  # Get last part (e.g., 'down_proj')
                    if hasattr(expert_module, component_parts):
                        component_module = getattr(expert_module, component_parts)
                        if hasattr(component_module, 'weight'):
                            # ✅ Store original value for restoration
                            coord_key = (sw.layer, sw.expert_id, sw.row, sw.column)
                            original_value = component_module.weight.data[sw.row, sw.column].clone()
                            self._temp_modified_weights[coord_key] = original_value
                            
                            # Zero the weight
                            component_module.weight.data[sw.row, sw.column] = 0.0
            except Exception as e:
                self.logger.warning(f"Could not zero super weight {sw}: {e}")
    
    def _restore_all_weights(self):
        """Restore all temporarily modified weights"""
        for (layer_idx, expert_id, row, col), original_value in self._temp_modified_weights.items():
            try:
                expert_module = self.mlp_handler.get_expert_module(layer_idx, expert_id)
                if expert_module:
                    # Find the component - this is tricky, we need to reverse-engineer it
                    for comp_type in ['down', 'output']:  # Try common component types
                        expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_id)
                        if comp_type in expert_components:
                            component_module = expert_components[comp_type]
                            if hasattr(component_module, 'weight'):
                                if (component_module.weight.shape[0] > row and 
                                    component_module.weight.shape[1] > col):
                                    component_module.weight.data[row, col] = original_value
                                    break
            except Exception as e:
                self.logger.warning(f"Could not restore weight at ({layer_idx}, {expert_id}, {row}, {col}): {e}")
        
        # Clear the storage
        self._temp_modified_weights.clear()
    
    def _detect_layer_expert_super_weights(self, tokens, layer_idx: int, 
                                         expert_indices: List[int], 
                                         spike_threshold: float, 
                                         iteration: int) -> List[MoESuperWeight]:
        """Detect super weights in specific experts of a layer"""
        
        super_weights = []
        hooks = []
        
        # Storage for activations
        expert_activations = {}
        
        def create_expert_hook(expert_idx: int, component_type: str):
            def hook(module, input, output):
                # Store input/output for this expert
                if isinstance(input, tuple):
                    input_tensor = input[0].detach()
                else:
                    input_tensor = input.detach()
                
                output_tensor = output.detach()
                
                expert_activations[expert_idx] = {
                    'input': input_tensor,
                    'output': output_tensor,
                    'component_type': component_type,
                    'module': module
                }
            return hook
        
        # Hook only the down/output projection of active experts
        for expert_idx in expert_indices:
            try:
                expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_idx)
                
                # Hook the output projection component (down or output)
                for comp_type, module in expert_components.items():
                    if comp_type in ['down', 'output']:
                        hook = module.register_forward_hook(
                            create_expert_hook(expert_idx, comp_type)
                        )
                        hooks.append(hook)
                        break  # Only hook one component per expert
            except Exception as e:
                self.logger.warning(f"Failed to hook expert {expert_idx} in layer {layer_idx}: {e}")
        
        # Run forward pass
        with torch.no_grad():
            self.model(**tokens)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activations for super weights
        for expert_idx, activation_data in expert_activations.items():
            input_tensor = activation_data['input']
            output_tensor = activation_data['output']
            component_type = activation_data['component_type']
            module = activation_data['module']
            
            # Find super weights using similar logic to standard detector
            expert_super_weights = self._find_expert_super_weights(
                layer_idx, expert_idx, component_type, module,
                input_tensor, output_tensor, spike_threshold, iteration
            )
            super_weights.extend(expert_super_weights)
        
        return super_weights
    
    def _find_expert_super_weights(self, layer_idx: int, expert_idx: int, 
                                 component_type: str, module, 
                                 input_tensor, output_tensor, 
                                 spike_threshold: float, iteration: int) -> List[MoESuperWeight]:
        """Find super weights in a specific expert component"""
        
        super_weights = []

        # Get the actual component name for this expert
        component_name = self.mlp_handler.get_expert_component_name(layer_idx, expert_idx, component_type)
        if not component_name:
            self.logger.warning(f"Could not get component name for expert {expert_idx} in layer {layer_idx}")
            return super_weights
        
        # Process tensors (similar to standard detector)
        if len(input_tensor.shape) == 3:  # [batch, seq, hidden]
            input_flat = input_tensor.reshape(-1)
            abs_input = torch.abs(input_flat)
            max_input_mag, max_input_idx = torch.max(abs_input, dim=0)
            
            if max_input_mag > spike_threshold:
                # Find corresponding output spike
                output_flat = output_tensor.reshape(-1)
                abs_output = torch.abs(output_flat)
                max_output_mag, max_output_idx = torch.max(abs_output, dim=0)
                
                if max_output_mag > spike_threshold:
                    # Extract indices
                    input_channel = max_input_idx.item() % input_tensor.shape[2]
                    output_channel = max_output_idx.item() % output_tensor.shape[2]
                    
                    # Get actual values
                    input_value = input_flat[max_input_idx].item()
                    output_value = output_flat[max_output_idx].item()
                    
                    # Get original weight value
                    original_value = module.weight.data[output_channel, input_channel].item()
                    
                    # Create MoESuperWeight
                    super_weight = MoESuperWeight(
                        layer=layer_idx,
                        expert_id=expert_idx,
                        component=component_name,
                        row=output_channel,
                        column=input_channel,
                        input_value=input_value,
                        output_value=output_value,
                        iteration_found=iteration + 1,
                        original_value=original_value
                    )
                    super_weights.append(super_weight)
        
        return super_weights