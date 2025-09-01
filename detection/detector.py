import dataclasses
import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
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
                if zero_detected_weights and iteration < max_iterations:
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
    Enhanced super weight detector specialized for MoE models.
    Uses routing statistics, per-expert co-spike analysis, and causal scoring.
    Implements the mathematical framework from the prompt.
    """
    
    def __init__(self, model, tokenizer, mlp_handler: UniversalMLPHandler, 
                 architecture_type: str, log_level=logging.INFO):
        super().__init__(model, tokenizer, mlp_handler, log_level)
        
        # MoE-specific initialization
        self.architecture_type = architecture_type
        
        # Enhanced routing analysis
        self.routing_statistics = {}  # Store p_active, entropy, etc.
        self.expert_activation_history = defaultdict(lambda: defaultdict(list))
        self.position_routing_entropy = {}  # H^(l)(pos)

        # Check for hybrid architecture
        self.hybrid_info = mlp_handler.get_hybrid_info()
        if self.hybrid_info:
            self.logger.info(f"Detected hybrid architecture: {self.hybrid_info}")
        
        # Cache for expert activation patterns
        self.expert_activation_cache = {}
        self.routed_token_cache = {}  # Cache routed tokens per expert
        
        # Temporary weight modifications for iterative suppression
        self._temp_modified_weights = {}
        
        # Causal scoring configuration
        self.stopwords = {"the", "and", "a", "an", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        self.logger.info(f"Enhanced MoE detector initialized for {self.architecture_type} architecture")
    
    def detect_super_weights(self, 
                       input_text: str = "Apple Inc. is a worldwide tech company.",
                       spike_threshold: float = 50.0,
                       max_iterations: int = 10,
                       router_analysis_samples: int = 5,
                       p_active_floor: float = 0.01,
                       co_spike_threshold: float = 0.12,
                       enable_causal_scoring: bool = True) -> List[MoESuperWeight]:
        """
        Enhanced MoE super weight detection with routing statistics and causal analysis.
        
        Args:
            input_text: Text for detection
            spike_threshold: Legacy threshold (kept for compatibility)  
            max_iterations: Max iterations for super weight detection
            router_analysis_samples: Number of samples for router analysis
            p_active_floor: Minimum p_active for expert consideration (default 1%)
            co_spike_threshold: Threshold for co-spike score S^(l,e)(r,c)
            enable_causal_scoring: Whether to compute causal impact scores
        """
        # Use hybrid detection for Deepseek-style models
        if self.hybrid_info and self.hybrid_info['is_hybrid']:
            return self.detect_hybrid_super_weights(
                input_text=input_text,
                spike_threshold=spike_threshold,
                max_iterations=max_iterations,
                router_analysis_samples=router_analysis_samples,
                p_active_floor=p_active_floor,
                co_spike_threshold=co_spike_threshold,
                enable_causal_scoring=enable_causal_scoring
            )
        
        self.logger.info(f"Starting enhanced MoE super weight detection")
        self.logger.info(f"Parameters: co_spike_threshold={co_spike_threshold}, p_active_floor={p_active_floor}")
        
        # Reset detection state
        self._reset_detection_state()
        
        # Phase 1: Enhanced routing instrumentation & statistics
        self.logger.info("Phase 1: Enhanced routing analysis...")
        routing_stats = self._enhanced_routing_analysis(
            input_text, router_analysis_samples, p_active_floor
        )
        
        if not routing_stats['candidate_experts']:
            self.logger.warning("No candidate experts found meeting p_active threshold")
            return []
        
        # Phase 2: Per-expert co-spike detection with iterative suppression  
        self.logger.info("Phase 2: Per-expert co-spike detection...")
        super_weights = self._per_expert_co_spike_detection(
            input_text, routing_stats, co_spike_threshold, max_iterations
        )
        
        # Phase 3: Causal impact scoring (if enabled)
        if enable_causal_scoring and super_weights:
            self.logger.info("Phase 3: Causal impact scoring...")
            self._compute_causal_impact_scores(super_weights, input_text)
        
        # Phase 4: Fast proxy evaluation
        self.logger.info("Phase 4: Fast proxy metrics...")
        self._compute_fast_proxies(super_weights, input_text)
        
        self.logger.info(f"Enhanced MoE detection complete. Total super weights: {len(super_weights)}")
        return super_weights
    
    def _enhanced_routing_analysis(self, input_text: str, num_samples: int, 
                                 p_active_floor: float) -> Dict[str, Any]:
        """
        Enhanced routing analysis with statistics computation.
        
        Computes:
        - p_active^(l)(e): Expert usage probability  
        - H^(l)(pos): Position-wise routing entropy
        - Capacity overflow rates
        
        Returns:
            Dict with routing statistics and candidate experts
        """
        sample_inputs = self._generate_sample_inputs(input_text, num_samples)
        
        # Enhanced tracking structures
        expert_activations = defaultdict(lambda: defaultdict(int))  # [layer][expert] -> count
        position_expert_selections = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # [layer][pos][expert] -> count
        overflow_counts = defaultdict(int)  # [layer] -> overflow_count
        total_tokens_per_layer = defaultdict(int)
        
        # Process each sample
        for sample_idx, sample_text in enumerate(sample_inputs):
            self.logger.info(f"Processing routing sample {sample_idx + 1}/{num_samples}")
            tokens = self.tokenizer(sample_text, return_tensors='pt').to(self.device)
            seq_len = tokens['input_ids'].shape[1]
            
            def create_enhanced_router_hook(layer_idx: int):
                def hook(module, input, output):
                    try:
                        # Extract routing information with position awareness
                        routing_info = self.mlp_handler.get_routing_info(layer_idx)
                        selected_experts, expert_probs, overflow_flag = self._extract_enhanced_routing_info(
                            output, routing_info, layer_idx
                        )
                        
                        # Store per-position expert selections
                        for pos in range(len(selected_experts)):
                            if pos < seq_len:  # Ensure we don't exceed sequence length
                                experts_at_pos = selected_experts[pos] if isinstance(selected_experts[pos], list) else [selected_experts[pos]]
                                for expert_idx in experts_at_pos:
                                    expert_activations[layer_idx][expert_idx] += 1
                                    position_expert_selections[layer_idx][pos][expert_idx] += 1
                        
                        # Track overflow
                        if overflow_flag:
                            overflow_counts[layer_idx] += 1
                        
                        total_tokens_per_layer[layer_idx] += seq_len
                        
                    except Exception as e:
                        self.logger.warning(f"Router hook error for layer {layer_idx}: {e}")
                
                return hook
            
            # Register hooks for MoE layers
            router_hooks = []
            for layer_idx in range(self.num_layers):
                if self.mlp_handler.is_moe_layer(layer_idx):
                    router_module = self.mlp_handler.get_router_module(layer_idx)
                    if router_module:
                        hook = router_module.register_forward_hook(create_enhanced_router_hook(layer_idx))
                        router_hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                self.model(**tokens)
            
            # Clean up hooks
            for hook in router_hooks:
                hook.remove()
        
        # Compute routing statistics
        routing_stats = {
            'p_active': {},  # [layer][expert] -> probability
            'position_entropy': {},  # [layer][position] -> entropy  
            'overflow_rates': {},  # [layer] -> rate
            'candidate_experts': {},  # [layer] -> list of expert indices
            'total_samples': num_samples
        }
        
        # Calculate p_active^(l)(e) for each expert
        for layer_idx, expert_counts in expert_activations.items():
            routing_stats['p_active'][layer_idx] = {}
            total_activations = sum(expert_counts.values())
            
            for expert_idx, count in expert_counts.items():
                p_active = count / (num_samples * total_tokens_per_layer.get(layer_idx, 1))
                routing_stats['p_active'][layer_idx][expert_idx] = p_active
        
        # Calculate position-wise routing entropy H^(l)(pos)
        for layer_idx, pos_expert_data in position_expert_selections.items():
            routing_stats['position_entropy'][layer_idx] = {}
            
            for pos, expert_counts in pos_expert_data.items():
                if sum(expert_counts.values()) > 0:
                    total_selections = sum(expert_counts.values())
                    entropy = 0.0
                    
                    for expert_idx, count in expert_counts.items():
                        if count > 0:
                            p_pos_expert = count / total_selections
                            entropy -= p_pos_expert * torch.log(torch.tensor(p_pos_expert)).item()
                    
                    routing_stats['position_entropy'][layer_idx][pos] = entropy
        
        # Calculate overflow rates
        for layer_idx in overflow_counts:
            routing_stats['overflow_rates'][layer_idx] = overflow_counts[layer_idx] / num_samples
        
        # Identify candidate experts using p_active threshold and entropy considerations
        for layer_idx, expert_p_active in routing_stats['p_active'].items():
            candidates = []
            layer_entropies = routing_stats['position_entropy'].get(layer_idx, {})
            low_entropy_positions = [pos for pos, h in layer_entropies.items() if h < 1.0]  # Low entropy positions
            
            for expert_idx, p_active in expert_p_active.items():
                # Include expert if above p_active threshold OR if it appears in low-entropy positions
                if p_active >= p_active_floor:
                    candidates.append(expert_idx)
                elif low_entropy_positions and any(
                    expert_idx in position_expert_selections[layer_idx].get(pos, {})
                    for pos in low_entropy_positions
                ):
                    candidates.append(expert_idx)
                    self.logger.debug(f"Including expert {expert_idx} in layer {layer_idx} due to low-entropy position activity")
            
            routing_stats['candidate_experts'][layer_idx] = candidates
        
        # Store routing statistics for later use
        self.routing_statistics = routing_stats
        
        # Log routing analysis results
        self._log_routing_statistics(routing_stats)
        
        return routing_stats
    
    def _generate_sample_inputs(self, base_text: str, num_samples: int) -> List[str]:
        """Generate diverse sample inputs for router analysis using dataset samples"""
        try:
            from utils.datasets import DatasetLoader
            
            # Initialize dataset loader
            loader = DatasetLoader(seed=42)
            
            # Load diverse samples from wikitext
            try:
                wikitext_samples = loader.load_perplexity_dataset(
                    dataset_name='wikitext',
                    config='wikitext-2-raw-v1', 
                    split='validation',
                    n_samples=num_samples,
                    min_length=20  # Ensure reasonable length
                )
                
                # Filter out empty or very short samples
                valid_samples = [text.strip() for text in wikitext_samples if len(text.strip()) > 10]
                
                if len(valid_samples) >= num_samples:
                    self.logger.info(f"Using {num_samples} diverse WikiText samples for routing analysis")
                    return valid_samples[:num_samples]
                else:
                    self.logger.warning(f"Only got {len(valid_samples)} valid WikiText samples, padding with variations")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load WikiText samples: {e}, falling back to custom samples")
                
            # Fallback: Try custom text samples
            try:
                custom_samples = []
                domains = ['general', 'scientific', 'conversational']
                samples_per_domain = max(1, num_samples // len(domains))
                
                for domain in domains:
                    domain_samples = loader.load_custom_text_samples(
                        domain=domain,
                        n_samples=samples_per_domain
                    )
                    custom_samples.extend(domain_samples)
                
                # Add extra samples if needed
                while len(custom_samples) < num_samples:
                    extra_samples = loader.load_custom_text_samples(
                        domain='general',
                        n_samples=num_samples - len(custom_samples)
                    )
                    custom_samples.extend(extra_samples)
                
                if len(custom_samples) >= num_samples:
                    self.logger.info(f"Using {num_samples} custom domain samples for routing analysis")
                    return custom_samples[:num_samples]
                    
            except Exception as e:
                self.logger.warning(f"Failed to load custom samples: {e}, using simple variations")
        
        except ImportError:
            self.logger.warning("DatasetLoader not available!")
    
    def _extract_enhanced_routing_info(self, router_output, routing_info, layer_idx: int) -> Tuple[List, List, bool]:
        """
        Extract enhanced routing information including expert probabilities and overflow detection.
        
        Returns:
            Tuple of (selected_experts_per_position, expert_probs_per_position, overflow_flag)
        """
        try:
            selected_experts = []
            expert_probs = []
            overflow_flag = False
            
            # Get number of experts per token (top-k)
            k = getattr(routing_info, 'experts_per_token', 2) if routing_info else 2
            
            if torch.is_tensor(router_output):
                if len(router_output.shape) == 2:  # [seq_len, num_experts]
                    seq_len, num_experts = router_output.shape
                    
                    # Apply softmax to get probabilities
                    probs = torch.softmax(router_output, dim=-1)
                    
                    # Get top-k for each position
                    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
                    
                    for pos in range(seq_len):
                        pos_experts = topk_indices[pos].tolist()
                        pos_probs = topk_probs[pos].tolist()
                        
                        selected_experts.append(pos_experts)
                        expert_probs.append(pos_probs)
                    
                elif len(router_output.shape) == 3:  # [batch, seq_len, num_experts]
                    batch_size, seq_len, num_experts = router_output.shape
                    
                    # Take first batch element for simplicity
                    batch_output = router_output[0]  # [seq_len, num_experts]
                    probs = torch.softmax(batch_output, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
                    
                    for pos in range(seq_len):
                        pos_experts = topk_indices[pos].tolist()
                        pos_probs = topk_probs[pos].tolist()
                        
                        selected_experts.append(pos_experts)
                        expert_probs.append(pos_probs)
                        
            # Detect capacity overflow (simplified heuristic)
            if expert_probs and len(expert_probs) > 0:
                # If routing probabilities are very uniform, might indicate overflow
                avg_entropy = sum(
                    -sum(p * torch.log(torch.tensor(p + 1e-8)).item() for p in pos_probs if p > 0)
                    for pos_probs in expert_probs
                ) / len(expert_probs)
                
                # High entropy might indicate capacity overflow forcing uniform routing
                if avg_entropy > torch.log(torch.tensor(float(k) * 0.8)):
                    overflow_flag = True
            
            return selected_experts, expert_probs, overflow_flag
            
        except Exception as e:
            self.logger.warning(f"Error extracting enhanced routing info for layer {layer_idx}: {e}")
            # Fallback to simple extraction
            simple_experts = self._extract_selected_experts(router_output, routing_info, layer_idx)
            return [simple_experts], [[1.0] * len(simple_experts)], False
    
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
    
    def _log_routing_statistics(self, routing_stats: Dict[str, Any]):
        """Log detailed routing statistics"""
        self.logger.info("=== ROUTING STATISTICS ===")
        
        for layer_idx, expert_p_active in routing_stats['p_active'].items():
            candidates = routing_stats['candidate_experts'].get(layer_idx, [])
            overflow_rate = routing_stats['overflow_rates'].get(layer_idx, 0.0)
            
            self.logger.info(f"Layer {layer_idx}: {len(candidates)} candidate experts, overflow_rate={overflow_rate:.3f}")
            
            # Show top experts by p_active
            top_experts = sorted(expert_p_active.items(), key=lambda x: x[1], reverse=True)[:5]
            for expert_idx, p_active in top_experts:
                is_candidate = expert_idx in candidates
                status = "[CANDIDATE]" if is_candidate else "[FILTERED]"
                self.logger.info(f"  {status} Expert {expert_idx}: p_active={p_active:.4f}")
            
            # Show entropy statistics  
            layer_entropies = routing_stats['position_entropy'].get(layer_idx, {})
            if layer_entropies:
                avg_entropy = sum(layer_entropies.values()) / len(layer_entropies)
                low_entropy_positions = len([h for h in layer_entropies.values() if h < 1.0])
                self.logger.info(f"  Routing entropy: avg={avg_entropy:.3f}, low_entropy_positions={low_entropy_positions}")
    
    def _per_expert_co_spike_detection(self, input_text: str, routing_stats: Dict[str, Any],
                                     co_spike_threshold: float, max_iterations: int) -> List[MoESuperWeight]:
        """
        Per-expert co-spike detection with iterative suppression.
        
        Implements the co-spike alignment score:
        S^(l,e)(r,c) = sum_t |X^(l,e)_t,r * Y^(l,e)_t,c| / (||X_r|| * ||Y_c|| + ε)
        
        Where X^(l,e) is expert input, Y^(l,e) is expert output, conditioned on routing.
        """
        detected_super_weights = []
        processed_coordinates = set()
        
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        candidate_experts = routing_stats['candidate_experts']
        
        for iteration in range(max_iterations):
            self.logger.info(f"Co-spike detection iteration {iteration + 1}/{max_iterations}")
            
            iteration_super_weights = []
            
            # Process each layer with candidate experts
            for layer_idx, expert_indices in candidate_experts.items():
                if not expert_indices:
                    continue
                    
                layer_super_weights = self._detect_layer_co_spikes(
                    tokens, layer_idx, expert_indices, routing_stats,
                    co_spike_threshold, iteration
                )
                
                # Filter out already processed coordinates 
                for sw in layer_super_weights:
                    coord = (sw.layer, sw.expert_id, sw.row, sw.column)
                    if coord not in processed_coordinates:
                        iteration_super_weights.append(sw)
                        processed_coordinates.add(coord)
            
            if not iteration_super_weights:
                self.logger.info(f"No new super weights found in iteration {iteration + 1}")
                break
            
            detected_super_weights.extend(iteration_super_weights)
            self.logger.info(f"Found {len(iteration_super_weights)} super weights in iteration {iteration + 1}")
            
            # Zero detected weights for next iteration (iterative suppression)
            if iteration < max_iterations - 1:
                self._temporarily_zero_weights(iteration_super_weights)
        
        # Restore all weights after detection
        self._restore_all_weights()
        
        return detected_super_weights
    
    def _detect_layer_co_spikes(self, tokens, layer_idx: int, expert_indices: List[int],
                          routing_stats: Dict[str, Any], co_spike_threshold: float,
                          iteration: int) -> List[MoESuperWeight]:
        """Detect co-spikes in a specific layer's experts using routed tokens only"""
        
        super_weights = []
        
        # Check if this layer has shared experts
        shared_expert_module = self.mlp_handler.get_shared_expert_module(layer_idx)
        has_shared_expert = shared_expert_module is not None
        
        # Collect routed activations for each expert
        expert_activations = {}  # expert_id -> {'X': input_tensor, 'Y': output_tensor, 'positions': [...]}
        shared_expert_activations = {}  # For shared expert (processes all tokens)
        
        def create_routing_aware_hook(expert_idx: int, component_type: str, is_shared: bool = False):
            def hook(module, input, output):
                # Store activations - we'll filter by routing later (except for shared expert)
                if isinstance(input, tuple):
                    input_tensor = input[0].detach()
                else:
                    input_tensor = input.detach()
                
                output_tensor = output.detach()
                
                # Store activations
                if is_shared:
                    shared_expert_activations[expert_idx] = {
                        'X': input_tensor,  
                        'Y': output_tensor, 
                        'component_type': component_type,
                        'module': module
                    }
                else:
                    expert_activations[expert_idx] = {
                        'X': input_tensor,  
                        'Y': output_tensor, 
                        'component_type': component_type,
                        'module': module
                    }
            return hook
        
        # Hook regular experts
        hooks = []
        for expert_idx in expert_indices:
            try:
                expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_idx)
                
                # Hook the down/output projection component
                for comp_type, module in expert_components.items():
                    if comp_type in ['down', 'output']:
                        hook = module.register_forward_hook(
                            create_routing_aware_hook(expert_idx, comp_type, is_shared=False)
                        )
                        hooks.append(hook)
                        break
            except Exception as e:
                self.logger.warning(f"Failed to hook expert {expert_idx} in layer {layer_idx}: {e}")
        
        # Hook shared expert if it exists
        if has_shared_expert:
            try:
                shared_expert_components = self.mlp_handler.get_shared_expert_components(layer_idx)
                
                for comp_type, module in shared_expert_components.items():
                    if comp_type in ['down', 'output']:
                        # Use a special expert_idx for shared expert (e.g., -1)
                        hook = module.register_forward_hook(
                            create_routing_aware_hook(-1, comp_type, is_shared=True)  # -1 indicates shared
                        )
                        hooks.append(hook)
                        self.logger.debug(f"Hooked shared expert in layer {layer_idx}")
                        break
            except Exception as e:
                self.logger.warning(f"Failed to hook shared expert in layer {layer_idx}: {e}")
        
        # Also hook the router to get routing decisions
        routing_decisions = {}  # position -> [selected_expert_indices]
        
        def router_hook(module, input, output):
            try:
                routing_info = self.mlp_handler.get_routing_info(layer_idx)
                selected_experts, expert_probs, _ = self._extract_enhanced_routing_info(
                    output, routing_info, layer_idx
                )
                routing_decisions.update({
                    pos: experts for pos, experts in enumerate(selected_experts)
                })
            except Exception as e:
                self.logger.warning(f"Router hook failed for layer {layer_idx}: {e}")
        
        router_module = self.mlp_handler.get_router_module(layer_idx)
        if router_module:
            router_hook_handle = router_module.register_forward_hook(router_hook)
            hooks.append(router_hook_handle)
        
        # Forward pass to collect activations
        with torch.no_grad():
            self.model(**tokens)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process regular expert activations (existing logic)
        for expert_idx, activation_data in expert_activations.items():
            X = activation_data['X']  # [batch, seq_len, d_ff] 
            Y = activation_data['Y']  # [batch, seq_len, d_model]
            module = activation_data['module']
            component_type = activation_data['component_type']
            
            # Filter to only routed tokens for this expert
            routed_positions = [
                pos for pos, selected_experts in routing_decisions.items()
                if expert_idx in selected_experts
            ]
            
            if not routed_positions:
                self.logger.debug(f"No routed tokens found for expert {expert_idx} in layer {layer_idx}")
                continue
            
            # Extract routed token activations
            if X.dim() == 3:  # [batch, seq, hidden]
                batch_size, seq_len, _ = X.shape
                routed_positions = [p for p in routed_positions if p < seq_len]
                
                if not routed_positions:
                    continue
                    
                # Get activations only for routed positions
                X_routed = X[0, routed_positions, :]  # [routed_tokens, d_ff]
                Y_routed = Y[0, routed_positions, :]  # [routed_tokens, d_model]
                
                # Apply co-spike detection to routed tokens
                co_spike_results = self._compute_co_spike_scores(
                    X_routed, Y_routed, co_spike_threshold, layer_idx, expert_idx
                )
                
                # Convert to MoESuperWeight objects
                for result in co_spike_results:
                    r_star, c_star, score = result['r'], result['c'], result['score']
                    
                    # Get routing statistics for this expert
                    p_active = routing_stats['p_active'][layer_idx].get(expert_idx, 0.0)
                    
                    # Get component name  
                    component_name = self.mlp_handler.get_expert_component_name(
                        layer_idx, expert_idx, component_type
                    )
                    
                    # Extract values and original weight
                    input_value = result.get('input_value', 0.0)
                    output_value = result.get('output_value', 0.0)
                    original_value = module.weight.data[c_star, r_star].item()
                    
                    super_weight = MoESuperWeight(
                        layer=layer_idx,
                        expert_id=expert_idx,
                        component=component_name or f"expert_{expert_idx}.{component_type}",
                        row=c_star,  # output channel
                        column=r_star,  # input channel  
                        input_value=input_value,
                        output_value=output_value,
                        iteration_found=iteration + 1,
                        original_value=original_value,
                        # Enhanced MoE fields
                        p_active=p_active,
                        co_spike_score=score,
                        routed_tokens_count=len(routed_positions),
                        position_indices=routed_positions,
                        detection_iteration=iteration + 1
                    )
                    super_weights.append(super_weight)
        
        # Process shared expert activations (NEW)
        if shared_expert_activations:
            for shared_expert_idx, activation_data in shared_expert_activations.items():
                X = activation_data['X']  # [batch, seq_len, d_ff] 
                Y = activation_data['Y']  # [batch, seq_len, d_model]
                module = activation_data['module']
                component_type = activation_data['component_type']
                
                # Shared expert processes ALL tokens (no routing filter needed)
                if X.dim() == 3:  # [batch, seq, hidden]
                    batch_size, seq_len, _ = X.shape
                    
                    # Use all tokens for shared expert
                    X_all = X[0, :, :]  # [seq_len, d_ff] - all tokens
                    Y_all = Y[0, :, :]  # [seq_len, d_model] - all tokens
                    
                    # Apply co-spike detection to all tokens
                    co_spike_results = self._compute_co_spike_scores(
                        X_all, Y_all, co_spike_threshold, layer_idx, shared_expert_idx
                    )
                    
                    # Convert to MoESuperWeight objects for shared expert
                    for result in co_spike_results:
                        r_star, c_star, score = result['r'], result['c'], result['score']
                        
                        # Shared expert has p_active = 1.0 (always active)
                        p_active = 1.0
                        
                        # Get component name for shared expert
                        component_name = f"shared_experts.{component_type}"
                        
                        # Extract values and original weight
                        input_value = result.get('input_value', 0.0)
                        output_value = result.get('output_value', 0.0)
                        original_value = module.weight.data[c_star, r_star].item()
                        
                        super_weight = MoESuperWeight(
                            layer=layer_idx,
                            expert_id=shared_expert_idx,  # -1 for shared expert
                            component=component_name,
                            row=c_star,  # output channel
                            column=r_star,  # input channel  
                            input_value=input_value,
                            output_value=output_value,
                            iteration_found=iteration + 1,
                            original_value=original_value,
                            # Enhanced MoE fields
                            p_active=p_active,
                            co_spike_score=score,
                            routed_tokens_count=seq_len,  # All tokens
                            position_indices=list(range(seq_len)),  # All positions
                            detection_iteration=iteration + 1,
                            # Mark as shared expert
                            is_shared_expert=True
                        )
                        super_weights.append(super_weight)
    
        return super_weights
    
    def _compute_co_spike_scores(self, X_routed: torch.Tensor, Y_routed: torch.Tensor, 
                               threshold: float, layer_idx: int, expert_idx: int) -> List[Dict]:
        """
        Compute co-spike alignment scores S^(l,e)(r,c) for routed tokens.
        
        S^(l,e)(r,c) = sum_t |X^(l,e)_t,r * Y^(l,e)_t,c| / (sqrt(sum_t (X_t,r)^2) * sqrt(sum_t (Y_t,c)^2) + ε)
        
        Args:
            X_routed: [routed_tokens, d_ff] - input to down projection
            Y_routed: [routed_tokens, d_model] - output from down projection  
            threshold: minimum score threshold
        """
        if X_routed.shape[0] == 0:  # No routed tokens
            return []
        
        T, d_ff = X_routed.shape
        _, d_model = Y_routed.shape
        
        eps = 1e-8
        results = []
        
        # First, identify candidate input channels (r) with high energy
        X_energy = torch.sum(X_routed ** 2, dim=0)  # [d_ff] 
        top_k_input = min(10, d_ff)  # Limit search space
        _, top_input_channels = torch.topk(X_energy, top_k_input)
        
        # For each candidate input channel, find best output channel
        for r_idx in range(top_k_input):
            r = top_input_channels[r_idx].item()
            X_r = X_routed[:, r]  # [T]
            
            # Compute correlation with all output channels
            # This is more efficient than nested loop
            X_r_expanded = X_r.unsqueeze(1)  # [T, 1]
            
            # Co-spike numerator: sum_t |X_t,r * Y_t,c|
            numerator = torch.sum(torch.abs(X_r_expanded * Y_routed), dim=0)  # [d_model]
            
            # Denominators
            X_r_norm = torch.sqrt(torch.sum(X_r ** 2) + eps)
            Y_norms = torch.sqrt(torch.sum(Y_routed ** 2, dim=0) + eps)  # [d_model]
            
            # Co-spike scores for this input channel r
            scores = numerator / (X_r_norm * Y_norms)  # [d_model]
            
            # Find best output channel c for this input channel r
            max_score, best_c = torch.max(scores, dim=0)
            
            if max_score.item() > threshold:
                # Get actual values for the winning pair
                c = best_c.item()
                input_value = torch.max(torch.abs(X_r)).item()
                output_value = torch.max(torch.abs(Y_routed[:, c])).item()
                
                results.append({
                    'r': r,
                    'c': c,  
                    'score': max_score.item(),
                    'input_value': input_value,
                    'output_value': output_value,
                    'routed_tokens': T
                })
        
        # Sort by score and return top candidates
        results.sort(key=lambda x: x['score'], reverse=True)
        
        if results:
            self.logger.debug(f"Layer {layer_idx}, Expert {expert_idx}: Found {len(results)} co-spike candidates, "
                           f"top score: {results[0]['score']:.4f}")
        
        return results[:3]  # Return top 3 candidates per expert
    
    def _compute_causal_impact_scores(self, super_weights: List[MoESuperWeight], input_text: str):
        """
        Compute causal impact scores: I_nat and I_int
        
        I_nat(w*, e, l) = E[p_active^(l)(e | prompt) * ΔMetric(zero(w*))]
        I_int(w*, e, l) = E[ΔMetric(zero(w*))]_forced_to_e
        """
        if not super_weights:
            return
        
        self.logger.info(f"Computing causal impact scores for {len(super_weights)} super weights")
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        
        # Compute baseline perplexity
        baseline_loss = self._compute_loss(tokens)
        
        for sw in super_weights:
            try:
                # Natural routing impact
                sw.impact_natural = self._compute_natural_impact(sw, tokens, baseline_loss)
                
                # Interventional routing impact  
                sw.impact_interventional = self._compute_interventional_impact(sw, tokens, baseline_loss)
                
                self.logger.debug(f"{sw}: I_nat={sw.impact_natural:.4f}, I_int={sw.impact_interventional:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to compute causal scores for {sw}: {e}")
    
    def _compute_natural_impact(self, sw: MoESuperWeight, tokens, baseline_loss: float) -> float:
        """Compute impact under natural routing weighted by p_active"""
        try:
            # Temporarily zero the weight
            self._temporarily_zero_weights([sw])
            
            # Measure change in loss
            ablated_loss = self._compute_loss(tokens)
            impact = ablated_loss - baseline_loss
            
            # Weight by p_active (expert usage probability)
            if sw.p_active is not None:
                impact *= sw.p_active
            
            # Restore weight
            self._restore_all_weights()
            
            return impact
            
        except Exception as e:
            self._restore_all_weights()  # Ensure cleanup
            self.logger.warning(f"Error computing natural impact for {sw}: {e}")
            return 0.0
    
    def _compute_interventional_impact(self, sw: MoESuperWeight, tokens, baseline_loss: float) -> float:
        """Compute impact under interventional routing (forced to expert e)"""
        try:
            # For simplicity, we'll use a routing bias approach
            # In practice, this could be more sophisticated
            
            # Temporarily zero the weight
            self._temporarily_zero_weights([sw])
            
            # Apply routing bias toward this expert (simplified intervention)
            router_module = self.mlp_handler.get_router_module(sw.layer)
            original_bias = None
            
            if router_module and hasattr(router_module, 'weight'):
                # Add bias to favor this expert (simple intervention)
                if hasattr(router_module, 'bias') and router_module.bias is not None:
                    original_bias = router_module.bias.data[sw.expert_id].clone()
                    router_module.bias.data[sw.expert_id] += 2.0  # Boost this expert
                
                # Measure change in loss under biased routing
                ablated_loss = self._compute_loss(tokens)
                impact = ablated_loss - baseline_loss
                
                # Restore router bias
                if original_bias is not None:
                    router_module.bias.data[sw.expert_id] = original_bias
            else:
                # Fallback: same as natural impact if can't intervene on routing
                impact = self._compute_natural_impact(sw, tokens, baseline_loss)
            
            # Restore weight
            self._restore_all_weights()
            
            return impact
            
        except Exception as e:
            self._restore_all_weights()  # Ensure cleanup
            self.logger.warning(f"Error computing interventional impact for {sw}: {e}")
            return 0.0
    
    def _compute_loss(self, tokens) -> float:
        """Compute language modeling loss"""
        try:
            with torch.no_grad():
                outputs = self.model(**tokens, labels=tokens['input_ids'])
                return outputs.loss.item() if hasattr(outputs, 'loss') else 0.0
        except Exception as e:
            self.logger.warning(f"Error computing loss: {e}")
            return 0.0
    
    def _compute_fast_proxies(self, super_weights: List[MoESuperWeight], input_text: str):
        """
        Compute fast proxy metrics:
        - Energy reduction: E_c* = (1/T) * sum_t (H^(l+1)_t,c*)^2  
        - Stopword skew: change in stopword probability mass
        """
        if not super_weights:
            return
        
        self.logger.info(f"Computing fast proxy metrics for {len(super_weights)} super weights")
        tokens = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        
        for sw in super_weights:
            try:
                # Compute baseline proxies
                baseline_energy = self._measure_post_layer_energy(sw, tokens)
                baseline_stopword_mass = self._measure_stopword_mass(tokens)
                
                # Zero weight and measure change
                self._temporarily_zero_weights([sw])
                
                ablated_energy = self._measure_post_layer_energy(sw, tokens)
                ablated_stopword_mass = self._measure_stopword_mass(tokens)
                
                # Compute proxy metrics
                sw.energy_reduction = baseline_energy - ablated_energy
                sw.stopword_skew = ablated_stopword_mass - baseline_stopword_mass
                
                # Restore weight
                self._restore_all_weights()
                
                self.logger.debug(f"{sw}: energy_reduction={sw.energy_reduction:.4f}, stopword_skew={sw.stopword_skew:.4f}")
                
            except Exception as e:
                self._restore_all_weights()  # Ensure cleanup
                self.logger.warning(f"Error computing fast proxies for {sw}: {e}")
    
    def _measure_post_layer_energy(self, sw: MoESuperWeight, tokens) -> float:
        """Measure energy in suspect output channel after layer processing"""
        try:
            c_star = sw.row  # output channel
            layer_idx = sw.layer
            
            # Hook to capture post-layer hidden states
            post_layer_hidden = {}
            
            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]  # Usually (hidden_states, attention_weights, ...)
                else:
                    hidden = output
                post_layer_hidden['states'] = hidden.detach()
            
            # Hook the layer output (after residual connection)
            layer_module = self.mlp_handler.get_layer_module(layer_idx)
            if layer_module:
                hook_handle = layer_module.register_forward_hook(capture_hook)
                
                # Forward pass
                with torch.no_grad():
                    self.model(**tokens)
                
                hook_handle.remove()
                
                # Compute energy in channel c_star
                if 'states' in post_layer_hidden:
                    hidden_states = post_layer_hidden['states']  # [batch, seq, d_model]
                    if c_star < hidden_states.shape[-1]:
                        channel_energy = torch.mean(hidden_states[0, :, c_star] ** 2).item()
                        return channel_energy
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Error measuring post-layer energy: {e}")
            return 0.0
    
    def _measure_stopword_mass(self, tokens) -> float:
        """Measure total probability mass on stopwords"""
        try:
            with torch.no_grad():
                outputs = self.model(**tokens)
                logits = outputs.logits[0, -1, :]  # Last token logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get stopword token IDs
                stopword_mass = 0.0
                for stopword in self.stopwords:
                    try:
                        stopword_ids = self.tokenizer.encode(stopword, add_special_tokens=False)
                        if stopword_ids:
                            stopword_id = stopword_ids[0]  # Take first token
                            if stopword_id < probs.shape[0]:
                                stopword_mass += probs[stopword_id].item()
                    except:
                        continue
                
                return stopword_mass
                
        except Exception as e:
            self.logger.warning(f"Error measuring stopword mass: {e}")
            return 0.0
    
    
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
    
    def detect_hybrid_super_weights(self,
                                   input_text: str = "Apple Inc. is a worldwide tech company.",
                                   spike_threshold: float = 50.0,
                                   max_iterations: int = 10,
                                   router_analysis_samples: int = 5,
                                   p_active_floor: float = 0.01,
                                   co_spike_threshold: float = 0.12,
                                   enable_causal_scoring: bool = True) -> List[MoESuperWeight]:
        """
        Hybrid super weight detection for models like Deepseek.
        Handles both regular MLP layers and MoE layers in the same model.
        """
        self.logger.info(f"Starting hybrid MoE super weight detection")
        self.logger.info(f"Hybrid info: {self.hybrid_info}")
        
        # Reset detection state
        self._reset_detection_state()
        
        all_super_weights = []
        moe_start_layer = self.hybrid_info['moe_start_layer']
        
        # Phase 1: Handle regular MLP layers (before MoE layers)
        if moe_start_layer > 0:
            self.logger.info(f"Phase 1: Detecting super weights in regular MLP layers (0 to {moe_start_layer-1})")
            # Create manager
            regular_manager = SuperWeightManager(self.model, self.mlp_handler)
            regular_detector = SuperWeightDetector(self.model, self.tokenizer, self.mlp_handler, regular_manager)
            
            # Temporarily limit the detector to only regular layers
            original_layers = regular_detector.layers
            original_num_layers = regular_detector.num_layers
            
            regular_detector.layers = original_layers[:moe_start_layer]
            regular_detector.num_layers = moe_start_layer
            
            try:
                regular_super_weights = regular_detector.detect_super_weights(
                    input_text=input_text,
                    spike_threshold=spike_threshold,
                    max_iterations=max_iterations
                )
                
                # Convert to MoESuperWeight for consistency
                for sw in regular_super_weights:
                    moe_sw = MoESuperWeight(
                        layer=sw.layer,
                        expert_id=0,  # Regular MLP acts like single expert
                        component=sw.component,
                        row=sw.row,
                        column=sw.column,
                        input_value=sw.input_value,
                        output_value=sw.output_value,
                        iteration_found=sw.iteration_found,
                        original_value=sw.original_value,
                        # MoE-specific fields with default values
                        p_active=1.0,  # Regular MLP is always "active"
                        co_spike_score=0.0,  # Not applicable
                        routed_tokens_count=1,  # All tokens go through regular MLP
                        position_indices=[],
                        detection_iteration=sw.iteration_found
                    )
                    all_super_weights.append(moe_sw)
                    
                self.logger.info(f"Found {len(regular_super_weights)} super weights in regular MLP layers")
                
            finally:
                # Restore original layer info
                regular_detector.layers = original_layers
                regular_detector.num_layers = original_num_layers
        
        # Phase 2: Handle MoE layers (from moe_start_layer onwards)
        if moe_start_layer < self.num_layers:
            self.logger.info(f"Phase 2: Enhanced routing analysis for MoE layers ({moe_start_layer} to {self.num_layers-1})")
            
            # Only analyze routing for MoE layers
            routing_stats = self._enhanced_routing_analysis_hybrid(
                input_text, router_analysis_samples, p_active_floor, moe_start_layer
            )
            
            if routing_stats['candidate_experts']:
                self.logger.info("Phase 3: Per-expert co-spike detection for MoE layers")
                moe_super_weights = self._per_expert_co_spike_detection(
                    input_text, routing_stats, co_spike_threshold, max_iterations
                )
                
                # Phase 4: Causal impact scoring for MoE layers
                if enable_causal_scoring and moe_super_weights:
                    self.logger.info("Phase 4: Causal impact scoring for MoE layers")
                    self._compute_causal_impact_scores(moe_super_weights, input_text)
                
                # Phase 5: Fast proxy evaluation for MoE layers
                self.logger.info("Phase 5: Fast proxy metrics for MoE layers")
                self._compute_fast_proxies(moe_super_weights, input_text)
                
                all_super_weights.extend(moe_super_weights)
                self.logger.info(f"Found {len(moe_super_weights)} super weights in MoE layers")
            else:
                self.logger.warning("No candidate experts found in MoE layers")
        
        self.logger.info(f"Hybrid detection complete. Total super weights: {len(all_super_weights)}")
        return all_super_weights
    
    def _enhanced_routing_analysis_hybrid(self, input_text: str, num_samples: int, 
                                        p_active_floor: float, moe_start_layer: int) -> Dict[str, Any]:
        """Enhanced routing analysis but only for MoE layers in hybrid architecture"""
        self.logger.info(f"Starting hybrid routing analysis from layer {moe_start_layer}")
    
        sample_inputs = self._generate_sample_inputs(input_text, num_samples)
        
        # Enhanced tracking structures
        expert_activations = defaultdict(lambda: defaultdict(int))
        position_expert_selections = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        overflow_counts = defaultdict(int)
        total_tokens_per_layer = defaultdict(int)
        
        # Process each sample - only hook MoE layers
        for sample_idx, sample_text in enumerate(sample_inputs):
            self.logger.debug(f"Processing routing sample {sample_idx + 1}/{num_samples}")
            tokens = self.tokenizer(sample_text, return_tensors='pt').to(self.device)
            seq_len = tokens['input_ids'].shape[1]
            
            def create_enhanced_router_hook(layer_idx: int):
                def hook(module, input, output):
                    try:
                        # Extract routing information with position awareness
                        routing_info = self.mlp_handler.get_routing_info(layer_idx)
                        selected_experts, expert_probs, overflow_flag = self._extract_enhanced_routing_info(
                            output, routing_info, layer_idx
                        )
                        
                        self.logger.debug(f"Layer {layer_idx}: selected_experts={selected_experts[:5]}...")  # Show first 5
                        
                        # Store per-position expert selections
                        for pos in range(min(len(selected_experts), seq_len)):
                            experts_at_pos = selected_experts[pos] if isinstance(selected_experts[pos], list) else [selected_experts[pos]]
                            for expert_idx in experts_at_pos:
                                if expert_idx is not None and isinstance(expert_idx, int):
                                    expert_activations[layer_idx][expert_idx] += 1
                                    position_expert_selections[layer_idx][pos][expert_idx] += 1
                        
                        # Track overflow
                        if overflow_flag:
                            overflow_counts[layer_idx] += 1
                        
                        total_tokens_per_layer[layer_idx] += seq_len
                        
                    except Exception as e:
                        self.logger.warning(f"Router hook error for layer {layer_idx}: {e}")
                        import traceback
                        self.logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                return hook
            
            # Register hooks ONLY for MoE layers (starting from moe_start_layer)
            router_hooks = []
            hooks_registered = 0
            
            for layer_idx in range(moe_start_layer, self.num_layers):
                # Debug: Check if layer is detected as MoE
                is_moe = self.mlp_handler.is_moe_layer(layer_idx)
                self.logger.debug(f"Layer {layer_idx}: is_moe={is_moe}")
                
                if is_moe:
                    router_module = self.mlp_handler.get_router_module(layer_idx)
                    self.logger.debug(f"Layer {layer_idx}: router_module={router_module}")
                    
                    if router_module:
                        hook = router_module.register_forward_hook(create_enhanced_router_hook(layer_idx))
                        router_hooks.append(hook)
                        hooks_registered += 1
                        self.logger.debug(f"Registered hook for layer {layer_idx}")
                    else:
                        self.logger.warning(f"Layer {layer_idx} is MoE but has no router module")
                else:
                    self.logger.debug(f"Layer {layer_idx} is not MoE, skipping")
            
            self.logger.info(f"Registered {hooks_registered} router hooks for MoE layers")
            
            if hooks_registered == 0:
                self.logger.error("No router hooks registered! This will cause empty routing statistics.")
                # Let's debug the MoE detection
                for layer_idx in range(moe_start_layer, min(moe_start_layer + 3, self.num_layers)):
                    layer_module = self.mlp_handler.get_layer_module(layer_idx)
                    mlp_module = self.mlp_handler.get_mlp_module(layer_idx)
                    self.logger.debug(f"Layer {layer_idx}: layer_module={type(layer_module).__name__}, mlp_module={type(mlp_module).__name__}")
                    
                    # Check for MoE attributes
                    if hasattr(mlp_module, 'experts'):
                        self.logger.debug(f"Layer {layer_idx}: has experts={len(mlp_module.experts)}")
                    if hasattr(mlp_module, 'gate'):
                        self.logger.debug(f"Layer {layer_idx}: has gate={type(mlp_module.gate).__name__}")
                    if hasattr(mlp_module, 'shared_experts'):
                        self.logger.debug(f"Layer {layer_idx}: has shared_experts={type(mlp_module.shared_experts).__name__}")
            
            # Forward pass
            with torch.no_grad():
                self.model(**tokens)
            
            # Clean up hooks
            for hook in router_hooks:
                hook.remove()
        
        # Debug: Check if we collected any data
        self.logger.info(f"Collected expert activations for {len(expert_activations)} layers")
        for layer_idx, experts in expert_activations.items():
            self.logger.info(f"  Layer {layer_idx}: {len(experts)} experts activated")
        
        # Compute routing statistics (same as before but only for MoE layers)
        routing_stats = {
            'p_active': {},  # [layer][expert] -> probability
            'position_entropy': {},  # [layer][position] -> entropy  
            'overflow_rates': {},  # [layer] -> rate
            'candidate_experts': {},  # [layer] -> list of expert indices
            'total_samples': num_samples,
            'moe_start_layer': moe_start_layer  # track where MoE starts
        }
        
        # Calculate p_active^(l)(e) for each expert (only MoE layers)
        for layer_idx, expert_counts in expert_activations.items():
            if layer_idx >= moe_start_layer:  # Only process MoE layers
                routing_stats['p_active'][layer_idx] = {}
                total_activations = sum(expert_counts.values())
                
                for expert_idx, count in expert_counts.items():
                    if expert_idx is not None:  # Filter out None values
                        p_active = count / (num_samples * total_tokens_per_layer.get(layer_idx, 1))
                        routing_stats['p_active'][layer_idx][expert_idx] = p_active
        
        # Calculate position-wise routing entropy H^(l)(pos) (only MoE layers)
        for layer_idx, pos_expert_data in position_expert_selections.items():
            if layer_idx >= moe_start_layer:  # Only process MoE layers
                routing_stats['position_entropy'][layer_idx] = {}
                
                for pos, expert_counts in pos_expert_data.items():
                    if sum(expert_counts.values()) > 0:
                        total_selections = sum(expert_counts.values())
                        entropy = 0.0
                        
                        for expert_idx, count in expert_counts.items():
                            if count > 0 and expert_idx is not None:
                                p_pos_expert = count / total_selections
                                entropy -= p_pos_expert * torch.log(torch.tensor(p_pos_expert)).item()
                        
                        routing_stats['position_entropy'][layer_idx][pos] = entropy
        
        # Calculate overflow rates (only MoE layers)
        for layer_idx in overflow_counts:
            if layer_idx >= moe_start_layer:
                routing_stats['overflow_rates'][layer_idx] = overflow_counts[layer_idx] / num_samples
        
        # Identify candidate experts using p_active threshold and entropy considerations (only MoE layers)
        for layer_idx, expert_p_active in routing_stats['p_active'].items():
            if layer_idx >= moe_start_layer:  # Only process MoE layers
                candidates = []
                layer_entropies = routing_stats['position_entropy'].get(layer_idx, {})
                low_entropy_positions = [pos for pos, h in layer_entropies.items() if h < 1.0]
                
                for expert_idx, p_active in expert_p_active.items():
                    if expert_idx is not None:  # Filter out None values
                        # Include expert if above p_active threshold OR if it appears in low-entropy positions
                        if p_active >= p_active_floor:
                            candidates.append(expert_idx)
                        elif low_entropy_positions and any(
                            expert_idx in position_expert_selections[layer_idx].get(pos, {})
                            for pos in low_entropy_positions
                        ):
                            candidates.append(expert_idx)
                            self.logger.debug(f"Including expert {expert_idx} in layer {layer_idx} due to low-entropy position activity")
                
                routing_stats['candidate_experts'][layer_idx] = candidates
        
        # Store routing statistics for later use
        self.routing_statistics = routing_stats
        
        # Log routing analysis results
        self._log_routing_statistics(routing_stats)
        
        return routing_stats
    
    def _temporarily_zero_weights(self, super_weights: List[MoESuperWeight]):
        """Temporarily zero detected super weights with enhanced tracking"""
        for sw in super_weights:
            try:
                if sw.is_shared_expert:
                    # Handle shared expert
                    shared_expert_module = self.mlp_handler.get_shared_expert_module(sw.layer)
                    if shared_expert_module:
                        # Find the component (down_proj, etc.)
                        shared_components = self.mlp_handler.get_shared_expert_components(sw.layer)
                        for comp_type, component_module in shared_components.items():
                            if comp_type in ['down', 'output'] and hasattr(component_module, 'weight'):
                                coord_key = (sw.layer, sw.expert_id, sw.row, sw.column, 'shared')
                                if coord_key not in self._temp_modified_weights:
                                    original_value = component_module.weight.data[sw.row, sw.column].clone()
                                    self._temp_modified_weights[coord_key] = original_value
                                
                                component_module.weight.data[sw.row, sw.column] = 0.0
                                self.logger.debug(f"Zeroed shared expert weight {coord_key}")
                                break
                else:
                    # Get the expert module and component
                    expert_module = self.mlp_handler.get_expert_module(sw.layer, sw.expert_id)
                    if expert_module:
                        # Get component name and find the module
                        component_parts = sw.component.split('.')[-1]  # Get last part (e.g., 'down_proj')
                        component_module = None
                        
                        # Try to find the component
                        if hasattr(expert_module, component_parts):
                            component_module = getattr(expert_module, component_parts)
                        else:
                            # Try alternative names
                            expert_components = self.mlp_handler.get_expert_components(sw.layer, sw.expert_id)
                            for comp_type, module in expert_components.items():
                                if comp_type in ['down', 'output'] and hasattr(module, 'weight'):
                                    component_module = module
                                    break
                        
                        if component_module and hasattr(component_module, 'weight'):
                            # Store original value for restoration
                            coord_key = (sw.layer, sw.expert_id, sw.row, sw.column)
                            if coord_key not in self._temp_modified_weights:  # Avoid overwriting
                                original_value = component_module.weight.data[sw.row, sw.column].clone()
                                self._temp_modified_weights[coord_key] = original_value
                            
                            # Zero the weight
                            component_module.weight.data[sw.row, sw.column] = 0.0
                            self.logger.debug(f"Zeroed weight {coord_key}")
                        
            except Exception as e:
                self.logger.warning(f"Could not zero super weight {sw}: {e}")
    
    def _restore_all_weights(self):
        """Restore all temporarily modified weights with enhanced error handling"""
        restored_count = 0
        
        for (layer_idx, expert_id, row, col), original_value in self._temp_modified_weights.items():
            try:
                expert_module = self.mlp_handler.get_expert_module(layer_idx, expert_id)
                if expert_module:
                    # Try to find the right component
                    component_found = False
                    
                    # Try common component names
                    for comp_name in ['down_proj', 'output', 'c_proj', 'w2']:
                        if hasattr(expert_module, comp_name):
                            component_module = getattr(expert_module, comp_name)
                            if (hasattr(component_module, 'weight') and 
                                component_module.weight.shape[0] > row and 
                                component_module.weight.shape[1] > col):
                                component_module.weight.data[row, col] = original_value
                                restored_count += 1
                                component_found = True
                                break
                    
                    if not component_found:
                        # Fallback: try all components
                        expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_id)
                        for comp_type, component_module in expert_components.items():
                            if (hasattr(component_module, 'weight') and 
                                component_module.weight.shape[0] > row and 
                                component_module.weight.shape[1] > col):
                                component_module.weight.data[row, col] = original_value
                                restored_count += 1
                                break
                        
            except Exception as e:
                self.logger.warning(f"Could not restore weight at ({layer_idx}, {expert_id}, {row}, {col}): {e}")
        
        self.logger.debug(f"Restored {restored_count}/{len(self._temp_modified_weights)} weights")
        
        # Clear the storage
        self._temp_modified_weights.clear()
    
    # Keep old methods for backward compatibility but mark as legacy
    def _detect_active_expert_super_weights(self, input_text: str, 
                                          active_experts: Dict[int, List[int]], 
                                          spike_threshold: float, 
                                          max_iterations: int) -> List[MoESuperWeight]:
        """Legacy method - use _per_expert_co_spike_detection instead"""
        self.logger.warning("Using legacy detection method. Consider using enhanced co-spike detection.")
        
        # Convert to new format for compatibility
        routing_stats = {
            'candidate_experts': active_experts,
            'p_active': {layer_idx: {expert_idx: 1.0 for expert_idx in experts} 
                        for layer_idx, experts in active_experts.items()}
        }
        
        # Use new method with default threshold
        return self._per_expert_co_spike_detection(input_text, routing_stats, 0.1, max_iterations)