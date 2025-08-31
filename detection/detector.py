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
            layer = self.layers[layer_idx]
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
    Uses a two-phase approach: router analysis + active expert analysis.
    """

    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler,
                 architecture_type: str, log_level=logging.INFO,
                 p_active_floor: float = 0.01,
                 routing_entropy_factor: float = 0.7,
                 co_spike_tau: float = 0.1):
        super().__init__(model, tokenizer, mlp_handler, log_level)

        # MoE-specific initialization
        self.architecture_type = architecture_type

        # Thresholds and configuration
        self.p_active_floor = p_active_floor
        self.routing_entropy_factor = routing_entropy_factor
        self.co_spike_tau = co_spike_tau

        # Cache for router statistics
        # layer_idx -> {"p_active": {...}, "pos_entropy": {...}, "low_entropy_positions": [...],
        #                "overflow": float}
        self.expert_activation_cache = {}

        # Storage for temporary weight modifications
        self._temp_modified_weights = {}

        self.logger.info(
            f"Initialized MoE detector for {self.architecture_type} architecture"
        )
    
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
        """Gather routing statistics and select active experts.

        This replaces the previous hard ``>50%`` activation filter.  For each
        MoE layer we compute the probability that an expert is active
        (``p_active``) and the position wise routing entropy.  Experts are kept if
        their ``p_active`` exceeds ``p_active_floor`` *or* if they appear in any
        low entropy token position.

        Returns:
            Mapping layer_idx -> list of selected expert indices.
        """

        sample_inputs = self._generate_sample_inputs(input_text, num_samples)

        # Collect per-layer statistics
        layer_stats: Dict[int, Dict] = defaultdict(
            lambda: {
                "usage": defaultdict(float),
                "pos_counts": defaultdict(lambda: defaultdict(int)),
                "total_tokens": 0,
            }
        )

        for sample_idx, sample_text in enumerate(sample_inputs):
            self.logger.debug(f"Processing sample {sample_idx + 1}/{num_samples}")
            tokens = self.tokenizer(sample_text, return_tensors="pt").to(self.device)

            router_hooks = []

            def make_router_hook(layer_idx: int):
                def hook(module, inp, out):
                    routing_info = self.mlp_handler.get_routing_info(layer_idx)
                    k = getattr(routing_info, "experts_per_token", 2) if routing_info else 2

                    if not torch.is_tensor(out):
                        return

                    # Assume out is router logits; convert to probabilities
                    logits = out
                    if logits.dim() == 2:  # [seq, experts]
                        probs = torch.softmax(logits, dim=-1)
                        probs = probs.unsqueeze(0)  # add batch dim
                    else:  # [batch, seq, experts]
                        probs = torch.softmax(logits, dim=-1)

                    top_p, top_idx = torch.topk(probs, k, dim=-1)
                    batch, seq_len, _ = top_idx.shape
                    stats = layer_stats[layer_idx]
                    for b in range(batch):
                        for pos in range(seq_len):
                            stats["total_tokens"] += 1
                            for j in range(k):
                                e = int(top_idx[b, pos, j])
                                stats["usage"][e] += 1
                                stats["pos_counts"][pos][e] += 1

                return hook

            for layer_idx in range(self.num_layers):
                if not self.mlp_handler.is_moe_layer(layer_idx):
                    continue
                router_module = self.mlp_handler.get_router_module(layer_idx)
                if router_module is None:
                    continue
                hook = router_module.register_forward_hook(make_router_hook(layer_idx))
                router_hooks.append(hook)

            with torch.no_grad():
                self.model(**tokens)

            for h in router_hooks:
                h.remove()

        active_experts: Dict[int, List[int]] = {}

        for layer_idx, stats in layer_stats.items():
            total_tokens = max(stats["total_tokens"], 1)
            usage = stats["usage"]
            p_active = {e: cnt / total_tokens for e, cnt in usage.items()}

            # Position-wise entropy
            pos_entropy = {}
            for pos, counts in stats["pos_counts"].items():
                total = sum(counts.values())
                if total == 0:
                    continue
                probs = torch.tensor([c / total for c in counts.values()], dtype=torch.float)
                entropy = (-probs * torch.log(probs + 1e-8)).sum().item()
                pos_entropy[pos] = entropy

            # Determine low entropy positions
            if pos_entropy:
                median_entropy = torch.median(torch.tensor(list(pos_entropy.values()))).item()
                H_thr = self.routing_entropy_factor * median_entropy
                low_entropy_pos = {
                    pos for pos, H in pos_entropy.items() if H <= H_thr
                }
            else:
                H_thr = 0.0
                low_entropy_pos = set()

            # Select experts
            selected = []
            for e in usage.keys():
                if p_active.get(e, 0.0) >= self.p_active_floor:
                    selected.append(e)
                    continue
                # check if expert routes through any low entropy positions
                for pos in low_entropy_pos:
                    if stats["pos_counts"][pos].get(e, 0) > 0:
                        selected.append(e)
                        break

            if selected:
                active_experts[layer_idx] = selected

            # Store stats for later use
            self.expert_activation_cache[layer_idx] = {
                "p_active": p_active,
                "pos_entropy": pos_entropy,
                "low_entropy_positions": list(low_entropy_pos),
                "overflow": 0.0,  # capacity overflow tracking not implemented
            }

            self.logger.info(
                f"Layer {layer_idx}: selected experts {selected} (p_active floor {self.p_active_floor})"
            )

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
        """Detect super weights in specific experts of a layer.

        This function now records routing information so that activations are
        computed only on tokens that were actually routed to a given expert.
        """

        super_weights: List[MoESuperWeight] = []
        hooks = []

        expert_activations: Dict[int, Dict[str, torch.Tensor]] = {}
        routing_map: Dict[int, List[int]] = defaultdict(list)

        def create_expert_hook(expert_idx: int, component_type: str):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    input_tensor = input[0].detach()
                else:
                    input_tensor = input.detach()
                output_tensor = output.detach()
                expert_activations[expert_idx] = {
                    "input": input_tensor,
                    "output": output_tensor,
                    "component_type": component_type,
                    "module": module,
                }

            return hook

        # Hook router to capture token -> expert mapping
        router_module = self.mlp_handler.get_router_module(layer_idx)
        routing_info = self.mlp_handler.get_routing_info(layer_idx)
        router_hook = None
        if router_module is not None:
            def router_hook_fn(module, inp, out):
                k = getattr(routing_info, "experts_per_token", 2) if routing_info else 2
                if not torch.is_tensor(out):
                    return
                logits = out
                if logits.dim() == 2:
                    probs = torch.softmax(logits, dim=-1).unsqueeze(0)
                else:
                    probs = torch.softmax(logits, dim=-1)
                _, idx = torch.topk(probs, k, dim=-1)
                batch, seq_len, _ = idx.shape
                for b in range(batch):
                    for pos in range(seq_len):
                        token_idx = b * seq_len + pos
                        for j in range(k):
                            e = int(idx[b, pos, j])
                            routing_map[e].append(token_idx)

            router_hook = router_module.register_forward_hook(router_hook_fn)
            hooks.append(router_hook)

        # Hook only the down/output projection of active experts
        for expert_idx in expert_indices:
            try:
                expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_idx)
                for comp_type, module in expert_components.items():
                    if comp_type in ["down", "output"]:
                        hook = module.register_forward_hook(
                            create_expert_hook(expert_idx, comp_type)
                        )
                        hooks.append(hook)
                        break
            except Exception as e:
                self.logger.warning(
                    f"Failed to hook expert {expert_idx} in layer {layer_idx}: {e}"
                )

        with torch.no_grad():
            self.model(**tokens)

        for hook in hooks:
            hook.remove()

        # Analyze activations for super weights
        for expert_idx, activation_data in expert_activations.items():
            positions = routing_map.get(expert_idx, [])
            if not positions:
                continue

            input_tensor = activation_data["input"].reshape(-1, activation_data["input"].shape[-1])
            output_tensor = activation_data["output"].reshape(-1, activation_data["output"].shape[-1])
            routed_inputs = input_tensor[positions]
            routed_outputs = output_tensor[positions]

            component_type = activation_data["component_type"]
            module = activation_data["module"]

            expert_super_weights = self._find_expert_super_weights(
                layer_idx,
                expert_idx,
                component_type,
                module,
                routed_inputs,
                routed_outputs,
                spike_threshold,
                iteration,
            )
            super_weights.extend(expert_super_weights)

        return super_weights

    def _find_expert_super_weights(
        self,
        layer_idx: int,
        expert_idx: int,
        component_type: str,
        module,
        input_tensor,
        output_tensor,
        spike_threshold: float,
        iteration: int,
    ) -> List[MoESuperWeight]:
        """Find super weights in a specific expert component using co-spike scoring."""

        super_weights: List[MoESuperWeight] = []

        component_name = self.mlp_handler.get_expert_component_name(
            layer_idx, expert_idx, component_type
        )
        if not component_name:
            self.logger.warning(
                f"Could not get component name for expert {expert_idx} in layer {layer_idx}"
            )
            return super_weights

        if input_tensor.numel() == 0 or output_tensor.numel() == 0:
            return super_weights

        try:
            r_star, c_star, score = self._argmax_co_spike(input_tensor, output_tensor)
        except Exception as e:
            self.logger.warning(f"Co-spike computation failed: {e}")
            return super_weights

        if score < self.co_spike_tau:
            return super_weights

        input_value = input_tensor[:, r_star].max().item()
        output_value = output_tensor[:, c_star].max().item()
        original_value = module.weight.data[c_star, r_star].item()

        delta_energy, delta_stop = self._run_micro_ablation(
            layer_idx, expert_idx, module, r_star, c_star, input_tensor, output_tensor
        )

        if not self._passes_proxy_thresholds(delta_energy, delta_stop):
            return super_weights

        layer_cache = self.expert_activation_cache.get(layer_idx, {})
        p_active = layer_cache.get("p_active", {}).get(expert_idx)
        low_entropy_positions = layer_cache.get("low_entropy_positions")
        overflow = layer_cache.get("overflow")

        I_nat = self._eval_weighted_metric(
            layer_idx, expert_idx, module, r_star, c_star, p_active
        )
        I_int = self._eval_metric_with_routing_intervention(
            layer_idx, expert_idx, module, r_star, c_star
        )

        sw = MoESuperWeight(
            layer=layer_idx,
            expert_id=expert_idx,
            component=component_name,
            row=c_star,
            column=r_star,
            input_value=input_value,
            output_value=output_value,
            iteration_found=iteration + 1,
            original_value=original_value,
            score_co_spike=score,
            p_active=p_active,
            low_entropy_positions=low_entropy_positions,
            capacity_overflow_rate=overflow,
            proxies={"energy": delta_energy, "stop": delta_stop},
            I_nat=I_nat,
            I_int=I_int,
        )
        super_weights.append(sw)

        return super_weights

    # ------------------------------------------------------------------
    # Helper utilities for upgraded MoE detection

    def _argmax_co_spike(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8):
        """Compute co-spike score matrix and return argmax indices and score."""

        # Flatten batch dimension if present
        if X.dim() == 3:
            X_flat = X.reshape(-1, X.shape[-1])
            Y_flat = Y.reshape(-1, Y.shape[-1])
        else:
            X_flat, Y_flat = X, Y

        x_norm = torch.sqrt((X_flat ** 2).sum(0) + eps)
        y_norm = torch.sqrt((Y_flat ** 2).sum(0) + eps)
        score_matrix = torch.abs(X_flat.T @ Y_flat) / (x_norm[:, None] * y_norm[None, :])
        max_idx = torch.argmax(score_matrix)
        r_star = int(max_idx // score_matrix.shape[1])
        c_star = int(max_idx % score_matrix.shape[1])
        score = score_matrix[r_star, c_star].item()
        return r_star, c_star, score

    def _run_micro_ablation(
        self,
        layer_idx: int,
        expert_idx: int,
        module,
        r_star: int,
        c_star: int,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> Tuple[float, float]:
        """Compute micro-ablation proxies for a candidate weight."""

        # Baseline energy
        baseline_energy = (output_tensor[:, c_star] ** 2).mean().item()

        original_val = module.weight.data[c_star, r_star].item()
        try:
            module.weight.data[c_star, r_star] = 0.0
            with torch.no_grad():
                # Recompute output for ablated weight using stored input
                ablated_output = input_tensor @ module.weight.T
                ablated_energy = (ablated_output[:, c_star] ** 2).mean().item()
        finally:
            module.weight.data[c_star, r_star] = original_val

        delta_energy = ablated_energy - baseline_energy

        # Simple stopword proxy: placeholder using tokenizer to decode probabilities
        delta_stop = 0.0
        try:
            stop_ids = getattr(self.tokenizer, "stop_token_ids", [])
            if stop_ids:
                baseline_logits = output_tensor
                ablated_logits = ablated_output
                baseline_probs = torch.softmax(baseline_logits, dim=-1)
                ablated_probs = torch.softmax(ablated_logits, dim=-1)
                baseline_mass = baseline_probs[:, stop_ids].sum().item()
                ablated_mass = ablated_probs[:, stop_ids].sum().item()
                delta_stop = ablated_mass - baseline_mass
        except Exception:
            pass

        return delta_energy, delta_stop

    def _passes_proxy_thresholds(self, delta_energy: float, delta_stop: float) -> bool:
        """Basic thresholding for proxy metrics."""

        energy_drop_ok = delta_energy < -0.05 * abs(delta_energy)
        # stopword skew can be any sign; require magnitude above small threshold
        stop_ok = abs(delta_stop) > 0.01
        return energy_drop_ok and stop_ok

    def _eval_weighted_metric(
        self,
        layer_idx: int,
        expert_idx: int,
        module,
        r_star: int,
        c_star: int,
        p_active: Optional[float],
    ) -> float:
        """Placeholder for routing-aware metric evaluation."""

        if p_active is None:
            return 0.0
        # simple proxy: scaled by magnitude of weight
        weight_val = module.weight.data[c_star, r_star].item()
        return p_active * abs(weight_val)

    def _eval_metric_with_routing_intervention(
        self,
        layer_idx: int,
        expert_idx: int,
        module,
        r_star: int,
        c_star: int,
    ) -> float:
        """Placeholder for interventional routing metric."""

        weight_val = module.weight.data[c_star, r_star].item()
        return abs(weight_val) * 0.1