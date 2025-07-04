import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Literal
from tqdm.auto import tqdm
from dataclasses import dataclass

from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalLayerHandler

# --- Configuration ---

@dataclass
class SuperWeightTarget:
    """Defines the target super weight and any other necessary info for an attack."""
    super_weight: SuperWeight
    head_idx: Optional[int] = None
    subsequent_token_idx: int = 1

@dataclass
class SuperWeightAttackConfig:
    """Configuration for the GCG attack on super weights."""
    target: SuperWeightTarget
    hypothesis: Literal['A', 'B', 'C', 'D', 'E']
    num_steps: int = 200
    adv_string_init: str = "! ! ! ! ! ! ! ! ! !"
    search_width: int = 512
    batch_size: int = 256
    top_k_search: int = 256  # Renamed to avoid conflict
    allow_non_ascii: bool = True
    prompt_text: str = "The quick brown fox jumps over the lazy dog."
    placement: Literal["prefix", "suffix"] = "suffix"
    # Optional weighted combination of hypotheses
    loss_weights: Optional[Dict[str, float]] = None
    # Explicit content positioning
    content_start_idx: Optional[int] = None
    content_end_idx: Optional[int] = None
    # Flags for Hypothesis D
    target_all_content_tokens: bool = True
    head_reduction: str = "single"  # Options: "single", "mean", "weighted", "topk"
    top_k_heads: int = 4  # For topk head reduction
    tau: float = 0.5

# --- Core Abstractions ---

class PartialModel(torch.nn.Module):
    """Runs the base model only up to (and including) a target transformer block."""
    def __init__(self, model: torch.nn.Module, target_layer_idx: int):
        super().__init__()
        # Store reference to the full model for rotary embeddings
        self.full_model = model
        # Typical HF path: model.model.layers is a ModuleList of blocks
        self.model_layers = model.model.layers[:target_layer_idx + 1]

    def forward(self, inputs_embeds: torch.Tensor, *, output_attentions: bool = False) -> torch.Tensor:
        """Forward pass through blocks, optionally requesting attention weights."""
        hidden_states = inputs_embeds
        
        # Generate position embeddings like the full model does
        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        
        # Create position_ids for the sequence
        position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)
        
        # Get position embeddings from the full model's rotary embedding
        position_embeddings = self.full_model.model.rotary_emb(hidden_states, position_ids)
        
        # Create a dummy cache_position (required by some layers)
        cache_position = torch.arange(0, seq_len, device=device)
        
        # Process through layers
        for layer in self.model_layers:
            # OLMo layers expect position_embeddings parameter
            outputs = layer(
                hidden_states, 
                use_cache=False, 
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                cache_position=cache_position
            )
            # HF blocks typically return (hidden_states, attn_weights, present, ...)
            hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
        return hidden_states

class Loss:
    """Abstract base for hypothesis loss functions with hook management."""
    def __init__(self, config: SuperWeightAttackConfig, layer_handler: UniversalLayerHandler):
        self.config = config
        self.target = config.target
        self.layer_handler = layer_handler
        self.captured_tensor = None
        # Explicit content positioning (set by attacker during initialization)
        self.content_start_idx: int = 0
        self.content_end_idx: int = 0
        self.prompt_len: int = 0
        self.adv_len: int = 0
        self.adv_start: int = 0
        self.placement: Literal["prefix", "suffix"] = "suffix"

    def set_content_layout(self, content_start_idx: int, content_end_idx: int, prompt_len: int, adv_len: int, adv_start: int, placement: Literal["prefix", "suffix"]):
        """Set explicit content positioning and other layout info."""
        self.content_start_idx = content_start_idx
        self.content_end_idx = content_end_idx
        self.prompt_len = prompt_len
        self.adv_len = adv_len
        self.adv_start = adv_start
        self.placement = placement

    def first_content_idx(self) -> int:
        """Return the explicitly calculated start of content tokens."""
        return self.content_start_idx

    # --- Hook API used by the attacker ---
    def needs_attentions(self) -> bool:
        """Override for losses that require attention probabilities."""
        return False

    def install_hooks(self) -> List[Any]:
        """Register the forward hook needed to capture tensors for this loss."""
        module = self.get_target_module()
        handle = module.register_forward_hook(self.hook_fn)
        return [handle]

    def remove_hooks(self, handles: List[Any]) -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # --- To be implemented by subclasses ---
    def get_target_module(self) -> torch.nn.Module:
        raise NotImplementedError

    def hook_fn(self, module, inp, outp):
        raise NotImplementedError

    def compute_loss(self) -> torch.Tensor:
        """Consume the captured tensor and return a per-example loss tensor."""
        if self.captured_tensor is None:
            raise RuntimeError("Hook did not capture any tensor.")
        loss = self._compute_loss_from_tensor(self.captured_tensor)
        self.captured_tensor = None
        return loss
        
    def _compute_loss_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# --- Hypothesis Implementations ---

class HypothesisA(Loss):
    """Minimize the input magnitude to the down_proj on the super-weight column."""
    
    def get_target_module(self) -> torch.nn.Module:
        mlp_components = self.layer_handler.get_mlp_components(self.target.super_weight.layer)
        return mlp_components['down']
    
    def hook_fn(self, module, inp, outp):
        # Capture input to down_proj
        self.captured_tensor = inp[0]

    def _compute_loss_from_tensor(self, t: torch.Tensor) -> torch.Tensor:
        # t: down_proj input tensor, shape [bs, seq, hidden]
        # Minimize the specific input channel at sink position
        pos = self.config.sink_position
        col = self.target.super_weight.column
        x_col = t[:, pos, col]
        return x_col.abs()

class HypothesisB(Loss):
    """Minimize the L2 norm of the gate projection output at target position."""
    
    def get_target_module(self) -> torch.nn.Module:
        mlp_components = self.layer_handler.get_mlp_components(self.target.super_weight.layer)
        return mlp_components['gate']

    def hook_fn(self, module, inp, outp):
        # Capture gate output
        self.captured_tensor = outp

    def _compute_loss_from_tensor(self, t: torch.Tensor) -> torch.Tensor:
        # t: gate_proj output at target layer, shape [bs, seq, hidden]
        # Minimize the whole vector norm at sink position
        pos = self.config.sink_position
        return torch.linalg.norm(t[:, pos, :], dim=-1)

class HypothesisC(Loss):
    """Minimize the L2 norm of the up projection output at target position."""
    
    def get_target_module(self) -> torch.nn.Module:
        mlp_components = self.layer_handler.get_mlp_components(self.target.super_weight.layer)
        return mlp_components['up']
    
    def hook_fn(self, module, inp, outp):
        # Capture up output
        self.captured_tensor = outp

    def _compute_loss_from_tensor(self, t: torch.Tensor) -> torch.Tensor:
        # t: up_proj output at target layer, shape [bs, seq, hidden]
        # Minimize the whole vector norm at sink position
        pos = self.config.sink_position
        return torch.linalg.norm(t[:, pos, :], dim=-1)

class HypothesisD(Loss):
    """
    Anti-align q at adversarial position with k at the sink for attention heads.

    Head reduction options:
        - "single": Use specified head_idx only
        - "mean": Uniform mean over heads
        - "weighted": Softmax-weighted by per-head sink strength
        - "topk": Mean over top-k heads by sink strength
    """
    
    def __init__(self, config: SuperWeightAttackConfig, layer_handler: UniversalLayerHandler):
        super().__init__(config, layer_handler)
        # Extract head reduction parameters from config
        self.head_reduction = getattr(config, 'head_reduction', 'single')
        self.topk = getattr(config, 'top_k_heads', 4)
        self.tau = getattr(config, 'tau', 2.0)
        self._latest_attn_probs = None  # optional local cache
        
        # Add logging to see what mode we're in
        print(f"HypothesisD initialized with head_reduction='{self.head_reduction}', topk={self.topk}, tau={self.tau}")

    def get_target_module(self) -> torch.nn.Module:
        return self.layer_handler.get_layer_module(self.target.super_weight.layer)

    def needs_attentions(self) -> bool:
        return getattr(self, "head_reduction", "single") in ("weighted", "topk")

    def hook_fn(self, module, inp, outp):
        # Capture layer input (pre-attn norm hidden states)
        self.captured_tensor = inp[0]

    def set_latest_attn_probs(self, attn_probs: torch.Tensor):
        self._latest_attn_probs = attn_probs

    def _get_latest_attn_probs(self) -> Optional[torch.Tensor]:
        attn = self._latest_attn_probs
        if attn is None:
            attn = getattr(self.config, "_latest_attn_probs", None)
        return attn

    def _per_head_sink_weights(self, attn_probs: torch.Tensor, rows: slice, col: int) -> torch.Tensor:
        # attn_probs: [bs, h, q, k] -> per-head sink strength s_h on content rows to sink column
        s = attn_probs[:, :, rows, col].mean(dim=2)     # [bs, h]
        w = torch.softmax(self.tau * s, dim=-1)         # [bs, h]
        return w

    def _reduce_over_heads(self, cos_per_head: torch.Tensor) -> torch.Tensor:
        """
        cos_per_head: [bs, h] cosine per head.
        Returns per-example [bs] according to self.head_reduction.
        """
        bs, h = cos_per_head.shape
        mode = self.head_reduction

        if mode == "single":
            if self.target.head_idx is None:
                raise ValueError("Hypothesis D with head_reduction='single' requires head_idx.")
            idx = self.target.head_idx
            return cos_per_head.gather(dim=1, index=torch.full((bs, 1), idx, device=cos_per_head.device)).squeeze(1)

        if mode == "mean":
            return cos_per_head.mean(dim=1)

        attn = self._get_latest_attn_probs()
        if attn is None:
            # Fallbacks if attentions not cached
            return cos_per_head.mean(dim=1) if mode == "weighted" else \
                   cos_per_head.gather(dim=1, index=torch.full((bs, 1),
                        (self.target.head_idx if self.target.head_idx is not None else 0),
                        device=cos_per_head.device)).squeeze(1)

        rows = slice(self.content_start_idx, self.content_end_idx)
        col = self.config.sink_position if hasattr(self.config, 'sink_position') else 0
        w = self._per_head_sink_weights(attn, rows, col)  # [bs, h]

        if mode == "weighted":
            return (cos_per_head * w).sum(dim=1)          # [bs]
        elif mode == "topk":
            topk_vals, topk_idx = w.topk(self.topk, dim=1)      # [bs, k]
            gathered = cos_per_head.gather(dim=1, index=topk_idx)
            return gathered.mean(dim=1)                         # [bs]
        else:
            # unknown -> safe default
            return cos_per_head.mean(dim=1)

    def _compute_loss_from_tensor(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Get attention architecture info and projections
        attn_info = self.layer_handler.get_attention_architecture(self.target.super_weight.layer)
        comps = self.layer_handler.get_attention_components(self.target.super_weight.layer)
        q_proj, k_proj = comps['q_proj'], comps['k_proj']
        norm = self.layer_handler.get_normalization_components(self.target.super_weight.layer)

        if 'input_layernorm' in norm:
            hidden_states = norm['input_layernorm'](hidden_states)

        # Compute Q/K and reshape to [bs, heads, seq, d_head]
        queries = q_proj(hidden_states)
        keys = k_proj(hidden_states)
        bs, sl, _ = queries.shape
        num_heads = attn_info.num_attention_heads
        head_dim = attn_info.head_dim

        queries = queries.view(bs, sl, num_heads, head_dim).transpose(1, 2)  # [bs, h, seq, d]
        keys    = keys.view(bs, sl, num_heads, head_dim).transpose(1, 2)     # [bs, h, seq, d]

        sink_col = self.config.sink_position if hasattr(self.config, 'sink_position') else 0
        
        if getattr(self.config, 'target_all_content_tokens', False):
            content_start = self.content_start_idx
            content_end = self.content_end_idx
            content_queries = queries[:, :, content_start:content_end, :]           # [bs, h, Lc, d]
            k_sink_all = keys[:, :, sink_col, :].unsqueeze(2)                       # [bs, h, 1, d]
            # cosine per head (and per content token) -> [bs, h, Lc]
            cos = F.cosine_similarity(k_sink_all, content_queries, dim=-1)
            # average across content tokens -> [bs, h]
            cos_per_head = cos.mean(dim=2)
        else:
            content_pos = self.content_start_idx
            qt_all = queries[:, :, content_pos, :]                                  # [bs, h, d]
            k_sink_all = keys[:, :, sink_col, :]                                    # [bs, h, d]
            cos_per_head = F.cosine_similarity(k_sink_all, qt_all, dim=-1)          # [bs, h]

        # Reduce over heads according to the selected strategy (returns [bs])
        return self._reduce_over_heads(cos_per_head)


class HypothesisE(Loss):
    """Reduce attention mass from adversarial rows to the sink (first content) column."""
    def get_target_module(self) -> torch.nn.Module:
        # Hook the whole layer to capture attention weights from its outputs
        return self.layer_handler.get_layer_module(self.target.super_weight.layer)

    def needs_attentions(self) -> bool:
        return True

    def hook_fn(self, module, inp, outp):
        # Common HF order: (hidden_states, self_attn_weights, present, ...)
        if isinstance(outp, (tuple, list)) and len(outp) >= 2:
            self.captured_tensor = outp[1]
        elif hasattr(outp, 'attentions') and outp.attentions is not None:
            self.captured_tensor = outp.attentions
        else:
            raise RuntimeError("Attention probabilities not available. Ensure output_attentions=True.")

    def _compute_loss_from_tensor(self, attn_probs: torch.Tensor) -> torch.Tensor:
        col = self.config.sink_position  # Use sink position
        rows = slice(self.content_start_idx, self.content_end_idx)
        # mean over heads and content rows toward sink col
        return attn_probs[:, :, rows, col].mean(dim=(1, 2))  # [bs]

# --- Composite Loss ---

class CompositeLoss(Loss):
    """Combine multiple hypothesis losses with weights."""
    def __init__(self, config: SuperWeightAttackConfig, layer_handler: UniversalLayerHandler, parts: List[Tuple[Loss, float]]):
        super().__init__(config, layer_handler)
        self.parts: List[Tuple[Loss, float]] = parts

    def needs_attentions(self) -> bool:
        return any(subloss.needs_attentions() for subloss, _ in self.parts)

    def install_hooks(self) -> List[Any]:
        # Install hooks for all sub-losses
        handles: List[Any] = []
        for subloss, _ in self.parts:
            handles.extend(subloss.install_hooks())
        return handles

    def compute_loss(self) -> torch.Tensor:
        # Sum weighted sub-losses. Each subloss consumes its captured_tensor internally.
        total = 0.0
        for subloss, w in self.parts:
            if w != 0.0:
                total = total + w * subloss.compute_loss()
        return total

    def set_content_layout(self, content_start_idx: int, content_end_idx: int, prompt_len: int, adv_len: int, adv_start: int, placement: Literal["prefix", "suffix"]):
        # Propagate to all sub-losses
        super().set_content_layout(content_start_idx, content_end_idx, prompt_len, adv_len, adv_start, placement)
        for subloss, _ in self.parts:
            subloss.set_content_layout(content_start_idx, content_end_idx, prompt_len, adv_len, adv_start, placement)

    # These are unused but must exist; CompositeLoss manages hooks via sublosses.
    def get_target_module(self) -> torch.nn.Module:
        raise NotImplementedError("CompositeLoss does not expose a single target module.")
    def hook_fn(self, module, inp, outp):
        raise NotImplementedError("CompositeLoss does not use a single hook.")

# --- GCG Attacker ---

class SuperWeightAttacker:
    """Runs a GCG-style search to craft an adversarial prefix/suffix that functionally disables a super weight."""
    def __init__(self, model: torch.nn.Module, tokenizer, config: SuperWeightAttackConfig, log_level: int = logging.INFO):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        self.layer_handler = UniversalLayerHandler(model)
        self.embedding_layer = model.get_input_embeddings()
        self.partial_model = PartialModel(model, config.target.super_weight.layer)

        # Logger
        self.logger = self._setup_logger(log_level)
        
        # Calculate explicit content positioning during initialization
        self._calculate_content_layout()

        # AUTO-SELECT HEAD if not provided and needed for D/E hypotheses
        if self._needs_head_selection():
            if self.config.target.head_idx is None:
                self.logger.info("No head_idx provided for hypothesis %s - auto-selecting best head", self.config.hypothesis)
                auto_selected_head = self.pick_head_for_attack()
                self.config.target.head_idx = auto_selected_head
                self.logger.info("Auto-selected head %d for hypothesis %s", auto_selected_head, self.config.hypothesis)
            else:
                self.logger.info("Using user-provided head_idx=%d for hypothesis %s", 
                                self.config.target.head_idx, self.config.hypothesis)

        self.logger.info("Content layout: content_tokens[%d:%d], adv_start=%d, prompt_len=%d, adv_len=%d", 
                        self.content_start_idx, self.content_end_idx, self.adv_start, self.prompt_len, self.adv_len)
        
        # Initialize loss functions (now head_idx is guaranteed to be set if needed)
        loss_map = {'A': HypothesisA, 'B': HypothesisB, 'C': HypothesisC, 'D': HypothesisD, 'E': HypothesisE}

        if config.loss_weights and any(w > 0 for w in config.loss_weights.values()):
            parts: List[Tuple[Loss, float]] = []
            for key, w in config.loss_weights.items():
                if w == 0:
                    continue
                if key not in loss_map:
                    raise ValueError(f"Unknown loss key '{key}' in loss_weights.")
                parts.append((loss_map[key](config, self.layer_handler), float(w)))
            self.loss_fn: Loss = CompositeLoss(config, self.layer_handler, parts)
        else:
            self.loss_fn = loss_map[config.hypothesis](config, self.layer_handler)

        # Set the calculated layout in the loss function
        self.loss_fn.set_content_layout(
            content_start_idx=self.content_start_idx,
            content_end_idx=self.content_end_idx,
            prompt_len=self.prompt_len,
            adv_len=self.adv_len,
            adv_start=self.adv_start,
            placement=self.config.placement
        )

        self.not_allowed_ids = None if config.allow_non_ascii else self._get_nonascii_toks()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _needs_head_selection(self) -> bool:
        """Check if the current hypothesis configuration requires head selection."""
        # Check single hypothesis
        if self.config.hypothesis in ['D']:
            return True
        
        # Check composite loss weights
        if self.config.loss_weights:
            return any(key in ['D'] and weight > 0 for key, weight in self.config.loss_weights.items())
        
        return False

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Create a module-specific logger similar to other components."""
        logger = logging.getLogger("super_weights.attack")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(log_level)
        return logger

    def _has_bos_token(self) -> bool:
        """Check if tokenizer has a valid BOS token."""
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(self.model.config, "bos_token_id", None)
        return bos_id is not None

    def _get_bos_token_id(self) -> Optional[int]:
        """Get BOS token ID if available."""
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(self.model.config, "bos_token_id", None)
        return bos_id

    def _calculate_content_layout(self):
        """Calculate explicit content token positions, handling both BOS and non-BOS tokenizers."""
        has_bos = self._has_bos_token()
        self.sink_position = 0  # ALWAYS position 0 (first token in sequence)

        # Tokenize prompt and adversarial text
        if has_bos:
            # With BOS: tokenizer will add BOS automatically
            prompt_ids = self.tokenizer(self.config.prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
            # BOS + content tokens
            self.prompt_len = prompt_ids.shape[1]
        else:
            # No BOS: tokenizer won't add any special tokens even with add_special_tokens=True
            prompt_ids = self.tokenizer(self.config.prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
            self.prompt_len = prompt_ids.shape[1]

        adv_ids = self.tokenizer(self.config.adv_string_init, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
        self.adv_len = adv_ids.shape[1]
        
        if has_bos:
            if self.config.placement == "prefix":
                # Layout: [BOS][adv_tokens][content_tokens_without_BOS]
                self.adv_start = 1  # After BOS
                self.content_start_idx = 1 + self.adv_len  # After BOS + adversarial
                self.content_end_idx = 1 + self.adv_len + (self.prompt_len - 1)  # Exclude BOS from prompt_len
            else:  # suffix
                # Layout: [BOS][content_tokens_without_BOS][adv_tokens]
                self.adv_start = self.prompt_len  # After BOS + content
                self.content_start_idx = 1  # After BOS
                self.content_end_idx = self.prompt_len  # Before adversarial
        else:
            # No BOS token case
            if self.config.placement == "prefix":
                # Layout: [adv_tokens][content_tokens]
                self.adv_start = 0  # Start at position 0
                self.content_start_idx = self.adv_len  # After adversarial
                self.content_end_idx = self.adv_len + self.prompt_len
            else:  # suffix  
                # Layout: [content_tokens][adv_tokens]
                self.adv_start = self.prompt_len  # After content
                self.content_start_idx = 0  # Start at position 0
                self.content_end_idx = self.prompt_len
        
        # Store in config for reference (including sink_position for loss functions)
        self.config.content_start_idx = self.content_start_idx
        self.config.content_end_idx = self.content_end_idx
        self.config.sink_position = self.sink_position

        # Log the layout
        total_len = self.adv_len + self.prompt_len - (1 if has_bos else 0)  # Adjust for BOS
        layout_desc = []
        for i in range(total_len + (1 if has_bos else 0)):
            if has_bos and i == 0:
                layout_desc.append("B")  # BOS
            elif self.content_start_idx <= i < self.content_end_idx:
                layout_desc.append("C")  # Content
            elif self.adv_start <= i < self.adv_start + self.adv_len:
                layout_desc.append("A")  # Adversarial
            else:
                layout_desc.append("?")  # Unknown
        
        self.logger.info("Tokenizer has BOS: %s", has_bos)
        self.logger.info("Content layout: content[%d:%d], adv_start=%d, sink_pos=%d", 
                        self.content_start_idx, self.content_end_idx, self.adv_start, self.sink_position)
        self.logger.debug("Token layout: %s (B=BOS, C=content, A=adversarial)", "".join(layout_desc))

    def attack(self) -> Dict[str, Any]:
        """
        Run GCG-style adversarial attack to optimize adversarial string.
        
        Performs iterative optimization using gradient-based token substitution:
        1. Compute gradients with respect to adversarial token embeddings
        2. Sample candidate token replacements based on gradients  
        3. Evaluate candidates and select the best one
        4. Repeat until convergence or max steps
        
        Returns:
            Dictionary containing loss history, adversarial string history, 
            final adversarial string, and final loss value.
        """
        has_bos = self._has_bos_token()
        
        if has_bos:
            prompt_ids = self.tokenizer(self.config.prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
            bos_id = prompt_ids[:, 0:1]  # First token (BOS)
            content_ids = prompt_ids[:, 1:]  # Content tokens (excluding BOS)
        else:
            prompt_ids = self.tokenizer(self.config.prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
            bos_id = None
            content_ids = prompt_ids  # All tokens are content
        
        adv_ids = self.tokenizer(self.config.adv_string_init, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
        
        self.logger.info("Starting GCG attack (has_bos=%s, hypotheses=%s, placement=%s, steps=%d)", 
                         has_bos,
                         (list(self.config.loss_weights.keys()) if self.config.loss_weights else self.config.hypothesis),
                         self.config.placement, self.config.num_steps)

        results = {'loss_history': [], 'adv_string_history': []}
        pbar = tqdm(range(self.config.num_steps), desc=f"Attacking Hypothesis {self.config.hypothesis}")
        for step in pbar:
            grad = self._get_gradients(bos_id, content_ids, adv_ids, has_bos)
            with torch.no_grad():
                candidate_ids = self._sample_candidates(adv_ids, grad)
                losses = self._evaluate_candidates(bos_id, content_ids, candidate_ids, has_bos)
                best_idx = losses.argmin()
                adv_ids = candidate_ids[best_idx].unsqueeze(0)
                best_loss = losses[best_idx].item()
                results['loss_history'].append(best_loss)
                current_adv = self.tokenizer.decode(adv_ids.squeeze(0), skip_special_tokens=True)
                results['adv_string_history'].append(current_adv)
                pbar.set_postfix({"loss": f"{best_loss:.4f}", "adv_string": f"'{current_adv}'"})

        final_adv = self.tokenizer.decode(adv_ids.squeeze(0), skip_special_tokens=True)
        results['final_adv_string'] = final_adv
        results['final_loss'] = results['loss_history'][-1] if results['loss_history'] else float('nan')
        self.logger.info("Attack finished. final_loss=%.6f, final_adv='%s'", results['final_loss'], final_adv)
        return results

    def _get_gradients(self, bos_id: Optional[torch.Tensor], content_ids: torch.Tensor, adv_ids: torch.Tensor, has_bos: bool) -> torch.Tensor:
        """Differentiate the loss wrt the one-hot adv tokens to obtain vocab gradients."""
        adv_one_hot = F.one_hot(adv_ids, num_classes=self.embedding_layer.weight.shape[0]).float()
        adv_one_hot.requires_grad_()
        
        adv_embeds = adv_one_hot @ self.embedding_layer.weight
        content_embeds = self.embedding_layer(content_ids)
        
        if has_bos:
            bos_embeds = self.embedding_layer(bos_id)
            if self.config.placement == "prefix":
                # [BOS][adv_tokens][content_tokens]
                inputs_embeds = torch.cat([bos_embeds, adv_embeds, content_embeds], dim=1)
            else:  # suffix
                # [BOS][content_tokens][adv_tokens]
                inputs_embeds = torch.cat([bos_embeds, content_embeds, adv_embeds], dim=1)
        else:
            if self.config.placement == "prefix":
                # [adv_tokens][content_tokens]
                inputs_embeds = torch.cat([adv_embeds, content_embeds], dim=1)
            else:  # suffix
                # [content_tokens][adv_tokens]
                inputs_embeds = torch.cat([content_embeds, adv_embeds], dim=1)

        handles = self.loss_fn.install_hooks()
        try:
            self.partial_model(inputs_embeds, output_attentions=self.loss_fn.needs_attentions())
            loss = self.loss_fn.compute_loss().mean()
            loss.backward()
        finally:
            self.loss_fn.remove_hooks(handles)

        grad = adv_one_hot.grad.clone()
        return grad

    def _sample_candidates(self, adv_ids: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """GCG coordinate step: replace one random position with top-k token suggestion."""
        top_indices = (-grad).topk(self.config.top_k_search, dim=-1).indices  # Use renamed field
        token_to_replace = torch.randint(0, adv_ids.shape[1], (self.config.search_width,)).to(self.device)
        replacement_tokens = torch.randint(0, self.config.top_k_search, (self.config.search_width,)).to(self.device)  # Use renamed field
        new_token_values = top_indices[0, token_to_replace, replacement_tokens]
        candidate_ids = adv_ids.repeat(self.config.search_width, 1)
        candidate_ids[torch.arange(self.config.search_width), token_to_replace] = new_token_values

        # Optionally filter to ASCII/printable tokens
        if self.not_allowed_ids is not None:
            before = candidate_ids.clone()
            disallowed_mask = torch.isin(candidate_ids, self.not_allowed_ids)
            original_tokens = adv_ids.repeat(self.config.search_width, 1)
            candidate_ids[disallowed_mask] = original_tokens[disallowed_mask]
            num_filtered = (before != candidate_ids).sum().item()
            if num_filtered:
                self.logger.debug("Filtered %d disallowed token placements.", num_filtered)
        return candidate_ids

    def _evaluate_candidates(self, bos_id: Optional[torch.Tensor], content_ids: torch.Tensor, candidate_ids: torch.Tensor, has_bos: bool) -> torch.Tensor:
        """Evaluate loss for a batch of candidate adversarial strings."""
        all_losses = []
        num_candidates = candidate_ids.shape[0]
        
        for i in range(0, num_candidates, self.config.batch_size):
            batch_ids = candidate_ids[i : i + self.config.batch_size]
            current_batch_size = batch_ids.shape[0]
            
            content_embeds = self.embedding_layer(content_ids).repeat(current_batch_size, 1, 1)
            adv_embeds = self.embedding_layer(batch_ids)
            
            if has_bos:
                bos_embeds = self.embedding_layer(bos_id).repeat(current_batch_size, 1, 1)
                if self.config.placement == "prefix":
                    # [BOS][adv_tokens][content_tokens]
                    inputs_embeds = torch.cat([bos_embeds, adv_embeds, content_embeds], dim=1)
                else:  # suffix
                    # [BOS][content_tokens][adv_tokens]
                    inputs_embeds = torch.cat([bos_embeds, content_embeds, adv_embeds], dim=1)
            else:
                if self.config.placement == "prefix":
                    # [adv_tokens][content_tokens]
                    inputs_embeds = torch.cat([adv_embeds, content_embeds], dim=1)
                else:  # suffix
                    # [content_tokens][adv_tokens]
                    inputs_embeds = torch.cat([content_embeds, adv_embeds], dim=1)

            handles = self.loss_fn.install_hooks()
            try:
                with torch.no_grad():
                    self.partial_model(inputs_embeds, output_attentions=self.loss_fn.needs_attentions())
                    batch_loss = self.loss_fn.compute_loss()
                all_losses.append(batch_loss)
            finally:
                self.loss_fn.remove_hooks(handles)
        
        losses = torch.cat(all_losses)
        return losses

    def _get_nonascii_toks(self) -> torch.Tensor:
        """Build a mask of tokens that are non-ASCII, non-printable, or special."""
        non_ascii_toks = []
        for i in range(self.tokenizer.vocab_size):
            try:
                token = self.tokenizer.decode([i])
                if not token.isascii() or not token.isprintable():
                    non_ascii_toks.append(i)
            except Exception:
                non_ascii_toks.append(i)
        for token_id in self.tokenizer.all_special_ids:
            if token_id not in non_ascii_toks:
                non_ascii_toks.append(token_id)
        return torch.tensor(non_ascii_toks, device=self.device)

    def eval_metrics(self, adv_text: str = "") -> Dict[str, Any]:
        """
        Evaluate comprehensive attack effectiveness metrics.
        
        Computes attention patterns, super-activation path metrics, and stopword
        mass for the given adversarial text. Uses consistent layout calculation 
        with the main attack method.
        
        Args:
            adv_text: Adversarial string to evaluate (empty for baseline)
            
        Returns:
            Dictionary containing:
            - Attention metrics (entropy, sink attention mass)
            - Super-activation metrics (down_proj in/out, gate/up norms)
            - Stopword mass from next-token distribution
            - Layout information for debugging
        """
        has_bos = self._has_bos_token()
        
        # Get the components we need
        prompt_ids = self.tokenizer(self.config.prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
        
        if has_bos:
            bos_id = prompt_ids[:, 0:1]
            content_ids = prompt_ids[:, 1:]
            original_content_len = content_ids.shape[1]
        else:
            bos_id = None
            content_ids = prompt_ids  # All tokens are content
            original_content_len = content_ids.shape[1]
        
        # Handle adversarial text
        if adv_text.strip() == "":
            # Baseline: no adversarial tokens
            adv_ids = torch.empty(1, 0, dtype=torch.long, device=self.device)
            current_adv_len = 0
        else:
            adv_ids = self.tokenizer(adv_text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
            current_adv_len = adv_ids.shape[1]
        
        # Calculate layout for this specific evaluation
        if current_adv_len == 0:
            # Baseline case
            if has_bos:
                # [BOS][content_tokens]
                inputs_embeds = torch.cat([self.embedding_layer(bos_id), self.embedding_layer(content_ids)], dim=1)
                eval_content_start = 1  # After BOS
                eval_content_end = 1 + original_content_len
                eval_adv_start = 1 + original_content_len
                eval_adv_len = 0
                eval_prompt_len = 1 + original_content_len
                sink_pos = 0  # BOS - ALWAYS position 0
            else:
                # [content_tokens]
                inputs_embeds = self.embedding_layer(content_ids)
                eval_content_start = 0  # Start at position 0
                eval_content_end = original_content_len
                eval_adv_start = original_content_len
                eval_adv_len = 0
                eval_prompt_len = original_content_len
                sink_pos = 0  # First token - ALWAYS position 0
        else:
            # Attack case: use the configured placement
            content_embeds = self.embedding_layer(content_ids)
            adv_embeds = self.embedding_layer(adv_ids)
            sink_pos = 0  # ALWAYS position 0 (first token in sequence)
            if has_bos:
                bos_embeds = self.embedding_layer(bos_id)
                if self.config.placement == "prefix":
                    # [BOS][adv_tokens][content_tokens]
                    inputs_embeds = torch.cat([bos_embeds, adv_embeds, content_embeds], dim=1)
                    eval_content_start = 1 + current_adv_len
                    eval_content_end = 1 + current_adv_len + original_content_len
                    eval_adv_start = 1
                    eval_adv_len = current_adv_len
                    eval_prompt_len = 1 + current_adv_len + original_content_len
                else:  # suffix
                    # [BOS][content_tokens][adv_tokens]
                    inputs_embeds = torch.cat([bos_embeds, content_embeds, adv_embeds], dim=1)
                    eval_content_start = 1
                    eval_content_end = 1 + original_content_len
                    eval_adv_start = 1 + original_content_len
                    eval_adv_len = current_adv_len
                    eval_prompt_len = 1 + original_content_len + current_adv_len
            else:
                # No BOS case
                if self.config.placement == "prefix":
                    # [adv_tokens][content_tokens]
                    inputs_embeds = torch.cat([adv_embeds, content_embeds], dim=1)
                    eval_content_start = current_adv_len
                    eval_content_end = current_adv_len + original_content_len
                    eval_adv_start = 0
                    eval_adv_len = current_adv_len
                    eval_prompt_len = current_adv_len + original_content_len
                else:  # suffix
                    # [content_tokens][adv_tokens]
                    inputs_embeds = torch.cat([content_embeds, adv_embeds], dim=1)
                    eval_content_start = 0
                    eval_content_end = original_content_len
                    eval_adv_start = original_content_len
                    eval_adv_len = current_adv_len
                    eval_prompt_len = original_content_len + current_adv_len

        # ---------------------------
        # 1) Attention-to-sink metrics
        # ---------------------------
        content_rows = slice(eval_content_start, eval_content_end)
        if eval_adv_len > 0:
            adv_rows = slice(eval_adv_start, eval_adv_start + eval_adv_len)
        else:
            adv_rows = slice(0, 0)  # empty

        with torch.no_grad():
            out = self.model(inputs_embeds=inputs_embeds, output_attentions=True)
            # HF commonly returns a tuple per layer: [bs, heads, q, k] or [heads, q, k]
            attn_layer = out.attentions[self.config.target.super_weight.layer][0]  # [heads, seq, seq]
            # head-specific (if provided)
            if self.config.target.head_idx is not None:
                h = self.config.target.head_idx
                if eval_content_end > eval_content_start:
                    content_attn_to_sink_head = attn_layer[h, content_rows, sink_pos].mean().item()
                else:
                    content_attn_to_sink_head = 0.0
                if eval_adv_len > 0:
                    adv_attn_to_sink_head = attn_layer[h, adv_rows, sink_pos].mean().item()
                else:
                    adv_attn_to_sink_head = 0.0
                total_attn_to_sink_head = attn_layer[h, 1:eval_prompt_len, sink_pos].mean().item()
            else:
                content_attn_to_sink_head = 0.0
                adv_attn_to_sink_head = 0.0
                total_attn_to_sink_head = 0.0

            # all-heads means
            if eval_content_end > eval_content_start:
                content_attn_to_sink_all_heads_mean = attn_layer[:, content_rows, sink_pos].mean().item()
            else:
                content_attn_to_sink_all_heads_mean = 0.0
            total_attn_to_sink_all_heads_mean = attn_layer[:, 1:eval_prompt_len, sink_pos].mean().item()

            # sink rate over heads (threshold ~0.3 is common), fraction of heads whose mean attention (content -> sink) exceeds threshold
            sink_thresh = getattr(self.config, "sink_threshold", 0.3)
            if eval_content_end > eval_content_start:
                per_head_content = attn_layer[:, content_rows, sink_pos].mean(dim=1)  # [heads]
                sink_rate = (per_head_content > sink_thresh).float().mean().item()
            else:
                sink_rate = 0.0

            # optional: entropy over keys for content queries (higher => less peaky sink)
            # NOTE: attn_layer is [heads, q, k] and rows can be long; compute mean entropy across rows & heads
            if eval_content_end > eval_content_start:
                probs = attn_layer[:, content_rows, :]  # [heads, Lc, k]
                eps = 1e-8
                attn_entropy_content = (- (probs.clamp_min(eps) * (probs.clamp_min(eps).log())).sum(dim=-1)).mean().item()
            else:
                attn_entropy_content = 0.0

        # --- Super-activation path: down_proj in/out at sink ---
        # Also collect gate/up norms at sink and stopword mass from next-token distribution
        
        mlp_components = self.layer_handler.get_mlp_components(self.config.target.super_weight.layer)
        down_proj = mlp_components['down']
        gate_proj = mlp_components.get('gate', None)
        up_proj   = mlp_components.get('up',   None)

        captured = {}

        def hook_down(module, inp, outp):
            # inp[0]: [bs, seq, d_mlp_in]; outp: [bs, seq, d_mlp_out]
            captured['down_input'] = inp[0].detach()
            captured['down_output'] = outp.detach()

        def hook_gate(module, inp, outp):
            # outp: gate_proj(hidden) BEFORE SiLU (architecture-dependent).
            # If your layer_handler returns a post-activation module instead, this will be post-SiLU.
            captured['gate_out'] = outp.detach()

        def hook_up(module, inp, outp):
            captured['up_out'] = outp.detach()

        # Register hooks
        handles = []
        handles.append(down_proj.register_forward_hook(hook_down))
        if gate_proj is not None:
            handles.append(gate_proj.register_forward_hook(hook_gate))
        if up_proj is not None:
            handles.append(up_proj.register_forward_hook(hook_up))

        try:
            with torch.no_grad():
                out2 = self.model(inputs_embeds=inputs_embeds)
        finally:
            for h in handles:
                h.remove()

        # --- Sink-centric measures ---
        down_proj_in_col_at_sink = float('nan')     # X[sink, k*]
        down_proj_out_row_at_sink = float('nan')    # Y[sink, j*]
        gate_norm_at_sink = float('nan')            # ||gate[sink,:]||_2
        up_norm_at_sink   = float('nan')            # ||up[sink,:]||_2

        if 'down_input' in captured:
            din = captured['down_input']  # [1, seq, d_in]
            if sink_pos < din.shape[1]:
                assert sink_pos == 0
                down_proj_in_col_at_sink = torch.as_tensor(
                    din[0, sink_pos, self.config.target.super_weight.column]
                ).abs().item()

        if 'down_output' in captured:
            dout = captured['down_output']  # [1, seq, d_out]
            if sink_pos < dout.shape[1]:
                down_proj_out_row_at_sink = torch.as_tensor(
                    dout[0, sink_pos, self.config.target.super_weight.row]
                ).abs().item()

        if 'gate_out' in captured:
            gout = captured['gate_out']  # [1, seq, d_gate]
            if sink_pos < gout.shape[1]:
                gate_norm_at_sink = torch.linalg.norm(gout[0, sink_pos, :]).item()

        if 'up_out' in captured:
            uout = captured['up_out']  # [1, seq, d_up]
            if sink_pos < uout.shape[1]:
                up_norm_at_sink = torch.linalg.norm(uout[0, sink_pos, :]).item()

        # --- Stopword mass from next-token distribution ---
        # Build a robust stopword set with strings that map to single token ids
        _stopword_strings = [
            " the", " a", " an", " of", " and", " to", " in", " is", ",", ".", ":", ";", "!", "?", "\"", "'"
        ]
        stopword_ids = []
        for s in _stopword_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                stopword_ids.append(ids[0])

        stopword_mass = float('nan')
        if len(stopword_ids) > 0 and hasattr(out2, "logits"):
            # Next token distribution at the last prompt position
            last_pos = eval_prompt_len - 1
            logits_last = out2.logits[0, last_pos]
            probs_last = torch.softmax(logits_last, dim=-1)
            stopword_mass = probs_last[torch.tensor(stopword_ids, device=probs_last.device)].sum().item()

        # Average stopword mass across content rows
        stopword_mass_content_mean = float('nan')
        if hasattr(out2, "logits") and eval_content_end > eval_content_start:
            logits_content = out2.logits[0, eval_content_start:eval_content_end, :]
            probs_content = torch.softmax(logits_content, dim=-1)
            stopword_mass_content_mean = probs_content[:, torch.tensor(stopword_ids, device=probs_content.device)].sum(dim=1).mean().item()

        extra_metrics = {
            # super-activation path
            'down_proj_in_col_at_sink': down_proj_in_col_at_sink,      # X[sink, k*]
            'down_proj_out_row_at_sink': down_proj_out_row_at_sink,    # Y[sink, j*]
            'gate_norm_at_sink': gate_norm_at_sink,                    # ||gate[sink,:]||_2
            'up_norm_at_sink': up_norm_at_sink,                        # ||up[sink,:]||_2

            # distributional signal
            'stopword_mass_next_token': stopword_mass,
            'stopword_mass_content_mean': stopword_mass_content_mean
        }

        return {
            # attention to sink (head-specific + aggregate)
            'content_attn_to_sink_head': content_attn_to_sink_head,
            'adv_attn_to_sink_head': adv_attn_to_sink_head,
            'total_attn_to_sink_head': total_attn_to_sink_head,
            'content_attn_to_sink_all_heads_mean': content_attn_to_sink_all_heads_mean,
            'total_attn_to_sink_all_heads_mean': total_attn_to_sink_all_heads_mean,
            'sink_rate': sink_rate,
            'attn_entropy_content': attn_entropy_content,

            # super-activation path
            **extra_metrics,

            # layout echo (unchanged)
            'layout': {
                'content_start': eval_content_start,
                'content_end': eval_content_end,
                'adv_start': eval_adv_start,
                'adv_len': eval_adv_len,
                'prompt_len': eval_prompt_len,
                'placement': self.config.placement,
                'sink_pos': sink_pos,
                'has_bos': has_bos
            }
        }

    def pick_head_for_attack(self, prompt_texts: Optional[List[str]] = None) -> int:
        """
        Pick the best head for the current attack hypothesis, automatically choosing the right method.
        
        For Hypothesis D:
            - If target_all_content_tokens=True: Uses attention-based method (content->BOS attention)
            - If target_all_content_tokens=False: Uses key-bias alignment method (k0·q1 similarity)
        For Hypothesis E:
            - Always uses attention-based method (content->BOS attention)
        For other hypotheses:
            - Falls back to attention-based method
            
        Args:
            prompt_texts: List of prompts to analyze. Defaults to [config.prompt_text].
            
        Returns:
            Head index best suited for the attack hypothesis.
        """
        if prompt_texts is None:
            prompt_texts = [self.config.prompt_text]
        
        # Determine which method to use based on hypothesis and config
        if self.config.hypothesis == 'D' and not self.config.target_all_content_tokens:
            # Hypothesis D targeting only first content token -> use key-bias alignment
            return self._pick_head_by_key_bias_alignment(prompt_texts)
        else:
            # Hypothesis E, or Hypothesis D targeting all content tokens -> use attention-based
            return self._pick_head_by_attention_sink(prompt_texts)

    def _pick_head_by_attention_sink(self, prompt_texts: List[str]) -> int:
        """Pick head with strongest actual attention: content -> sink"""
        has_bos = self._has_bos_token()
        all_scores = []
        
        for prompt_text in prompt_texts:
            prompt_ids = self.tokenizer(prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
            inputs_embeds = self.embedding_layer(prompt_ids)
            
            with torch.no_grad():
                outputs = self.model(inputs_embeds=inputs_embeds, output_attentions=True)
            
            attn = outputs.attentions[self.config.target.super_weight.layer][0]  # [heads, seq, seq]
            
            if attn.shape[-1] > 1:
                if has_bos:
                    # BOS case: sum attention from positions 1: -> position 0 (BOS)
                    scores = attn[:, 1:, 0].sum(dim=1)  # [num_heads]
                else:
                    # No BOS case: sum attention from positions 1: -> position 0 (first content)
                    scores = attn[:, 1:, 0].sum(dim=1) if attn.shape[-1] > 1 else attn[:, :, 0].sum(dim=1)
                all_scores.append(scores.cpu())
        
        if not all_scores:
            self.logger.warning("No valid prompts for attention analysis, returning head 0")
            return 0
        
        # Average scores across all prompts
        avg_scores = torch.stack(all_scores).mean(dim=0)  # [num_heads]
        best_head = int(torch.argmax(avg_scores).item())
        best_score = avg_scores[best_head].item()
        
        self.logger.info("Selected attention-based head %d (layer %d) with sink score %.4f (has_bos=%s)", 
                        best_head, self.config.target.super_weight.layer, best_score, has_bos)

        return best_head

    def _pick_head_by_key_bias_alignment(self, prompt_texts: List[str]) -> int:
        """Pick head with strongest key-bias alignment: k_sink·q_content similarity"""
        has_bos = self._has_bos_token()
        all_scores = []
        
        for prompt_text in prompt_texts:
            prompt_ids = self.tokenizer(prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
            
            # Need at least 2 tokens for key-bias alignment
            min_tokens = 2 if has_bos else 1
            if prompt_ids.shape[1] < min_tokens:
                continue
                
            with torch.no_grad():
                outputs = self.model(input_ids=prompt_ids, output_hidden_states=True)

            hidden_states = outputs.hidden_states[self.config.target.super_weight.layer]

            # Get attention components
            attn_info = self.layer_handler.get_attention_architecture(self.config.target.super_weight.layer)
            q_proj = self.layer_handler.get_attention_components(self.config.target.super_weight.layer)['q_proj']
            k_proj = self.layer_handler.get_attention_components(self.config.target.super_weight.layer)['k_proj']

            norm_comps = self.layer_handler.get_normalization_components(self.config.target.super_weight.layer)
            if 'input_layernorm' in norm_comps:
                hidden_states = norm_comps['input_layernorm'](hidden_states)

            queries = q_proj(hidden_states)
            keys = k_proj(hidden_states)

            batch_size, seq_len, _ = queries.shape
            if seq_len < min_tokens:
                continue

            num_heads = attn_info.num_attention_heads
            head_dim = attn_info.head_dim
            queries = queries.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            keys = keys.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            if has_bos:
                # k_0: Key of the BOS token, q_1: Query of the first content token
                k_sink = keys[:, :, 0, :]  # [bs, heads, d_head]
                q_content = queries[:, :, 1, :]  # [bs, heads, d_head]
            else:
                # k_0: Key of the first token, q_1: Query of the second token (if available)
                # or just use k_0 and q_0 if only one token
                k_sink = keys[:, :, 0, :]  # [bs, heads, d_head]
                if seq_len > 1:
                    q_content = queries[:, :, 1, :]  # [bs, heads, d_head]
                else:
                    q_content = queries[:, :, 0, :]  # [bs, heads, d_head] - same as k_sink
            
            alignment = F.cosine_similarity(k_sink, q_content, dim=-1)  # [bs, heads]
            scores = alignment.mean(dim=0)  # [heads] - average over batch
            all_scores.append(scores.cpu())
        
        if not all_scores:
            self.logger.warning("No valid prompts for key-bias analysis, returning head 0")
            return 0
        
        # Average scores across all prompts
        avg_scores = torch.stack(all_scores).mean(dim=0)  # [num_heads]
        best_head = int(torch.argmax(avg_scores).item())
        best_score = avg_scores[best_head].item()
        
        self.logger.info("Selected key-bias head %d (layer %d) with alignment %.4f (has_bos=%s)", 
                        best_head, self.config.target.super_weight.layer, best_score, has_bos)
        
        return best_head

class MultiPromptSuperWeightAttacker(SuperWeightAttacker):
    """Extended attacker that optimizes adversarial strings across multiple prompts."""
    
    def __init__(self, model, tokenizer, config: SuperWeightAttackConfig, prompt_texts: List[str], **kwargs):
        # Use the first prompt for base initialization
        original_prompt = config.prompt_text
        super().__init__(model, tokenizer, config, **kwargs)
        
        # Store all prompts and validate them
        self.prompt_texts = prompt_texts
        self.original_prompt = original_prompt
        
        # Pre-tokenize all prompts and cache embeddings for efficiency
        self._prepare_multi_prompt_data()
        
        self.logger.info(f"Multi-prompt attacker initialized with {len(prompt_texts)} prompts")
    
    def _prepare_multi_prompt_data(self):
        """Pre-tokenize and cache embeddings for all prompts."""
        has_bos = self._has_bos_token()
        self.prompt_data = []
        
        with torch.no_grad():  # Cache embeddings without gradients
            for i, prompt_text in enumerate(self.prompt_texts):
                if has_bos:
                    prompt_ids = self.tokenizer(prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
                    bos_id = prompt_ids[:, 0:1]
                    content_ids = prompt_ids[:, 1:]
                    original_content_len = content_ids.shape[1]
                    
                    # Cache embeddings
                    bos_embeds = self.embedding_layer(bos_id)
                    content_embeds = self.embedding_layer(content_ids)
                else:
                    prompt_ids = self.tokenizer(prompt_text, return_tensors='pt', add_special_tokens=True)['input_ids'].to(self.device)
                    bos_id = None
                    content_ids = prompt_ids
                    original_content_len = content_ids.shape[1]
                    
                    # Cache embeddings
                    bos_embeds = None
                    content_embeds = self.embedding_layer(content_ids)
                
                self.prompt_data.append({
                    'prompt_text': prompt_text,
                    'prompt_ids': prompt_ids,
                    'bos_id': bos_id,
                    'content_ids': content_ids,
                    'content_len': original_content_len,
                    # Cached embeddings (major speedup!)
                    'bos_embeds': bos_embeds,
                    'content_embeds': content_embeds
                })

    def _needs_attentions_for_config(self) -> bool:
        """Check if current config actually needs attention weights."""
        if hasattr(self.loss_fn, 'needs_attentions'):
            return self.loss_fn.needs_attentions()
        return False

    def robust_reduce(self, loss_vec, mode="lse", tau=0.5, q=0.8):
        # loss_vec: shape [B]
        if mode == "mean":
            return loss_vec.mean()
        if mode == "max":
            return loss_vec.max()
        if mode == "quantile":
            k = max(1, int(q * loss_vec.numel()))
            return loss_vec.kthvalue(k).values
        # default: log-sum-exp (soft-max)
        return torch.logsumexp(loss_vec / tau, dim=0) * tau


    def _get_gradients_multi_prompt(self, adv_ids: torch.Tensor) -> torch.Tensor:
        """Get gradients averaged across all prompts using cached embeddings."""
        has_bos = self._has_bos_token()
        grads = []
        needs_attentions = self._needs_attentions_for_config()

        for prompt_data in self.prompt_data:
            # one-hot for this prompt
            adv_one_hot = F.one_hot(adv_ids, num_classes=self.embedding_layer.weight.shape[0]).float().to(self.device)
            adv_one_hot.requires_grad_()

            adv_embeds = adv_one_hot @ self.embedding_layer.weight  # [1, L_adv, d]
            
            # Use cached embeddings instead of recomputing
            content_embeds = prompt_data['content_embeds']

            if has_bos:
                bos_embeds = prompt_data['bos_embeds']
                inputs_embeds = (
                    torch.cat([bos_embeds, adv_embeds, content_embeds], dim=1)
                    if self.config.placement == "prefix"
                    else torch.cat([bos_embeds, content_embeds, adv_embeds], dim=1)
                )
            else:
                inputs_embeds = (
                    torch.cat([adv_embeds, content_embeds], dim=1)
                    if self.config.placement == "prefix"
                    else torch.cat([content_embeds, adv_embeds], dim=1)
                )

            # per-prompt layout
            self._update_loss_layout_for_prompt(prompt_data, adv_ids.shape[1])

            handles = self.loss_fn.install_hooks()
            try:
                # IMPORTANT: allow grads; no torch.no_grad() here
                self.model.zero_grad(set_to_none=True)  # keep param grads empty
                outs = self.partial_model(inputs_embeds, output_attentions=needs_attentions)
                
                # Only handle attentions if actually needed
                if needs_attentions and hasattr(outs, "attentions"):
                    attn = outs.attentions[self.config.target.super_weight.layer]
                    # attn can be [heads, q, k] (bs=1 elided) or [bs, heads, q, k]
                    if isinstance(attn, (list, tuple)):
                        attn = attn[0]
                    if attn.dim() == 3:  # [h, q, k] -> add batch
                        attn = attn.unsqueeze(0)
                    # make available to all D instances
                    setattr(self.config, "_latest_attn_probs", attn)
                    # optional: also push directly to D if you're not using CompositeLoss
                    if hasattr(self.loss_fn, "losses"):
                        for l in self.loss_fn.losses:
                            if isinstance(l, HypothesisD):
                                l.set_latest_attn_probs(attn)
                    elif isinstance(self.loss_fn, HypothesisD):
                        self.loss_fn.set_latest_attn_probs(attn)
                
                loss_vec = self.loss_fn.compute_loss()      # shape [B]
                loss = self.robust_reduce(loss_vec, mode="lse", tau=self.config.tau)

                # Get grads only for adv_one_hot (no param grads kept)
                (grad,) = torch.autograd.grad(loss, adv_one_hot, retain_graph=False, create_graph=False)
                grads.append(grad.detach())
            finally:
                self.loss_fn.remove_hooks(handles)

        # Average across prompts
        return torch.stack(grads, dim=0).mean(dim=0)

    
    def _update_loss_layout_for_prompt(self, prompt_data: Dict, adv_len: int):
        """Update loss function layout for current prompt."""
        has_bos = self._has_bos_token()
        
        if has_bos:
            if self.config.placement == "prefix":
                content_start = 1 + adv_len
                content_end = 1 + adv_len + prompt_data['content_len']
                adv_start = 1
                prompt_len = 1 + adv_len + prompt_data['content_len']
            else:
                content_start = 1
                content_end = 1 + prompt_data['content_len']
                adv_start = 1 + prompt_data['content_len']
                prompt_len = 1 + prompt_data['content_len'] + adv_len
        else:
            if self.config.placement == "prefix":
                content_start = adv_len
                content_end = adv_len + prompt_data['content_len']
                adv_start = 0
                prompt_len = adv_len + prompt_data['content_len']
            else:
                content_start = 0
                content_end = prompt_data['content_len']
                adv_start = prompt_data['content_len']
                prompt_len = prompt_data['content_len'] + adv_len
        
        self.loss_fn.set_content_layout(
            content_start_idx=content_start,
            content_end_idx=content_end,
            prompt_len=prompt_len,
            adv_len=adv_len,
            adv_start=adv_start,
            placement=self.config.placement
        )
    
    def _evaluate_candidates_multi_prompt(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate candidates across all prompts using cached embeddings."""
        has_bos = self._has_bos_token()
        num_candidates = candidate_ids.shape[0]
        all_losses_per_prompt = []
        needs_attentions = self._needs_attentions_for_config()
        
        # Evaluate each prompt separately
        for prompt_data in self.prompt_data:
            prompt_losses = []
            
            for i in range(0, num_candidates, self.config.batch_size):
                batch_ids = candidate_ids[i : i + self.config.batch_size]
                current_batch_size = batch_ids.shape[0]
                
                # Use cached embeddings and expand for batch
                content_embeds = prompt_data['content_embeds'].expand(current_batch_size, -1, -1)
                adv_embeds = self.embedding_layer(batch_ids)
                
                if has_bos:
                    bos_embeds = prompt_data['bos_embeds'].expand(current_batch_size, -1, -1)
                    if self.config.placement == "prefix":
                        inputs_embeds = torch.cat([bos_embeds, adv_embeds, content_embeds], dim=1)
                    else:
                        inputs_embeds = torch.cat([bos_embeds, content_embeds, adv_embeds], dim=1)
                else:
                    if self.config.placement == "prefix":
                        inputs_embeds = torch.cat([adv_embeds, content_embeds], dim=1)
                    else:
                        inputs_embeds = torch.cat([content_embeds, adv_embeds], dim=1)
                
                # Update layout for this prompt and batch
                self._update_loss_layout_for_prompt(prompt_data, batch_ids.shape[1])
                
                handles = self.loss_fn.install_hooks()
                try:
                    with torch.no_grad():
                        outs = self.partial_model(inputs_embeds, output_attentions=needs_attentions)
                        
                        # Only handle attentions if actually needed
                        if needs_attentions and hasattr(outs, "attentions"):
                            attn = outs.attentions[self.config.target.super_weight.layer]
                            if isinstance(attn, (list, tuple)): attn = attn[0]
                            if attn.dim() == 3: attn = attn.unsqueeze(0)
                            setattr(self.config, "_latest_attn_probs", attn)
                            if hasattr(self.loss_fn, "losses"):
                                for l in self.loss_fn.losses:
                                    if isinstance(l, HypothesisD):
                                        l.set_latest_attn_probs(attn)
                            elif isinstance(self.loss_fn, HypothesisD):
                                self.loss_fn.set_latest_attn_probs(attn)

                        batch_loss = self.loss_fn.compute_loss().detach()
                    prompt_losses.append(batch_loss)
                finally:
                    self.loss_fn.remove_hooks(handles)
            
            # Concatenate losses for this prompt
            all_losses_per_prompt.append(torch.cat(prompt_losses))
        
        # Stack losses: [num_prompts, num_candidates]
        stacked_losses = torch.stack(all_losses_per_prompt)
        
        # Return average loss across prompts for each candidate
        avg_losses = stacked_losses.mean(dim=0)
        return avg_losses
    
    def attack_multi_prompt(self) -> Dict[str, Any]:
        """Main multi-prompt GCG loop with early stopping."""
        has_bos = self._has_bos_token()
        
        # Use first prompt's tokenization for initialization
        adv_ids = self.tokenizer(self.config.adv_string_init, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
        
        self.logger.info(f"Starting multi-prompt GCG attack (has_bos={has_bos}, hypothesis={self.config.hypothesis}, placement={self.config.placement}, steps={self.config.num_steps})")
        self.logger.info(f"Optimizing over {len(self.prompt_texts)} prompts")
        
        results = {
            'loss_history': [], 
            'adv_string_history': [],
            'per_prompt_losses': {i: [] for i in range(len(self.prompt_texts))},
            'prompt_texts': self.prompt_texts
        }
        
        # Early stopping parameters
        patience = getattr(self.config, 'early_stopping_patience', 20)
        min_improvement = getattr(self.config, 'min_improvement', 1e-6)
        best_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(self.config.num_steps), desc=f"Multi-Prompt Attack Hypothesis {self.config.hypothesis}")
        
        for step in pbar:
            grad = self._get_gradients_multi_prompt(adv_ids)
            
            with torch.no_grad():
                candidate_ids = self._sample_candidates(adv_ids, grad)
                losses = self._evaluate_candidates_multi_prompt(candidate_ids)
                best_idx = losses.argmin()
                adv_ids = candidate_ids[best_idx].unsqueeze(0)
                best_loss_step = losses[best_idx].item()
                
                results['loss_history'].append(best_loss_step)
                current_adv = self.tokenizer.decode(adv_ids.squeeze(0), skip_special_tokens=True)
                results['adv_string_history'].append(current_adv)
                
                # Optional: track per-prompt losses for analysis
                if step % 10 == 0:  # Every 10 steps to save computation
                    per_prompt_losses = self._get_per_prompt_losses(adv_ids)
                    for i, loss_val in enumerate(per_prompt_losses):
                        results['per_prompt_losses'][i].append(loss_val)
                
                # Early stopping check
                if best_loss - best_loss_step > min_improvement:
                    best_loss = best_loss_step
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at step {step} - no improvement for {patience} steps")
                    break
                
                pbar.set_postfix({
                    "avg_loss": f"{best_loss_step:.4f}", 
                    "adv_string": f"'{current_adv[:30]}{'...' if len(current_adv) > 30 else ''}'"
                })
        
        final_adv = self.tokenizer.decode(adv_ids.squeeze(0), skip_special_tokens=True)
        results['final_adv_string'] = final_adv
        results['final_loss'] = results['loss_history'][-1] if results['loss_history'] else float('nan')
        results['early_stopped'] = patience_counter >= patience
        results['steps_completed'] = len(results['loss_history'])
        
        self.logger.info(f"Multi-prompt attack finished. final_avg_loss={results['final_loss']:.6f}, final_adv='{final_adv}'")
        return results
    
    def _get_per_prompt_losses(self, adv_ids: torch.Tensor) -> List[float]:
        """Get individual loss for each prompt using cached embeddings."""
        has_bos = self._has_bos_token()
        per_prompt_losses = []
        needs_attentions = self._needs_attentions_for_config()
        
        for prompt_data in self.prompt_data:
            # Use cached embeddings
            content_embeds = prompt_data['content_embeds']
            adv_embeds = self.embedding_layer(adv_ids)
            
            if has_bos:
                bos_embeds = prompt_data['bos_embeds']
                if self.config.placement == "prefix":
                    inputs_embeds = torch.cat([bos_embeds, adv_embeds, content_embeds], dim=1)
                else:
                    inputs_embeds = torch.cat([bos_embeds, content_embeds, adv_embeds], dim=1)
            else:
                if self.config.placement == "prefix":
                    inputs_embeds = torch.cat([adv_embeds, content_embeds], dim=1)
                else:
                    inputs_embeds = torch.cat([content_embeds, adv_embeds], dim=1)
            
            self._update_loss_layout_for_prompt(prompt_data, adv_ids.shape[1])
            
            handles = self.loss_fn.install_hooks()
            try:
                with torch.no_grad():
                    self.partial_model(inputs_embeds, output_attentions=needs_attentions)
                    loss_vec = self.loss_fn.compute_loss()      # shape [B]
                    loss = self.robust_reduce(loss_vec, mode="lse", tau=self.config.tau)

                per_prompt_losses.append(loss.item())
            finally:
                self.loss_fn.remove_hooks(handles)
        
        return per_prompt_losses
    
    def eval_metrics_multi_prompt(self, adv_text: str = "") -> Dict[str, Any]:
        """Evaluate metrics across all prompts."""
        metrics_per_prompt = []
        
        # Temporarily update config for each prompt evaluation
        original_prompt = self.config.prompt_text
        
        for i, prompt_data in enumerate(self.prompt_data):
            self.config.prompt_text = prompt_data['prompt_text']
            metrics = self.eval_metrics(adv_text=adv_text)
            metrics['prompt_idx'] = i
            metrics['prompt_text'] = prompt_data['prompt_text']
            metrics_per_prompt.append(metrics)
        
        # Restore original prompt
        self.config.prompt_text = original_prompt
        
        # Compute aggregate metrics
        aggregate_metrics = self._aggregate_metrics(metrics_per_prompt)
        
        return {
            'per_prompt': metrics_per_prompt,
            'aggregate': aggregate_metrics,
            'num_prompts': len(self.prompt_texts)
        }
    
    def _aggregate_metrics(self, metrics_per_prompt: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across prompts."""
        numeric_keys = [
            'content_attn_to_sink_all_heads_mean', 'total_attn_to_sink_all_heads_mean',
            'sink_rate', 'attn_entropy_content', 'down_proj_in_col_at_sink',
            'down_proj_out_row_at_sink', 'gate_norm_at_sink', 'up_norm_at_sink',
            'stopword_mass_next_token', 'stopword_mass_content_mean'
        ]
        
        aggregated = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics_per_prompt if not np.isnan(m[key])]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
            else:
                aggregated[f'{key}_mean'] = float('nan')
                aggregated[f'{key}_std'] = float('nan')
                aggregated[f'{key}_min'] = float('nan')
                aggregated[f'{key}_max'] = float('nan')
        
        return aggregated
