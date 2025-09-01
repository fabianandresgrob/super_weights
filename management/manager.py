import torch
import logging
from typing import Dict, List, Set, Tuple, Optional
from contextlib import contextmanager

# Public Accelerate utilities
try:
    from accelerate.utils import load_offloaded_weights, set_module_tensor_to_device
except Exception:  # pragma: no cover
    load_offloaded_weights = None
    set_module_tensor_to_device = None

from detection.super_weight import SuperWeight
from utils.model_architectures import UniversalMLPHandler


class SuperWeightManager:
    """
    Manages super weight modifications with automatic backup/restore.
    Works with Accelerate/Transformers sharded/offloaded models where some
    parameters may be on the 'meta' device. Uses Accelerate hooks to
    materialize just the tensors we need.
    """

    def __init__(self, model, mlp_handler: UniversalMLPHandler, log_level=logging.INFO):
        self.model = model
        self.mlp_handler = mlp_handler
        self.original_values: Dict[SuperWeight, torch.Tensor] = {}
        self.current_scales: Dict[SuperWeight, float] = {}
        self.currently_modified: Set[SuperWeight] = set()

        # Setup logging
        self.logger = self._setup_logger(log_level)
        self.logger.info("SuperWeightManager initialized with shared MLP handler")

    # Logging setup
    def _setup_logger(self, log_level) -> logging.Logger:
        logger = logging.getLogger(f"SuperWeightManager_{id(self)}")
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    # Accelerate-aware helpers
    def _named_parameters_dict(self) -> Dict[str, torch.nn.Parameter]:
        # Cached conversion helps when repeatedly looking up by name
        return dict(self.model.named_parameters())

    def _named_modules_dict(self) -> Dict[str, torch.nn.Module]:
        return dict(self.model.named_modules())

    def _resolve_module_fqname(self, target: torch.nn.Module) -> Optional[str]:
        """Find fully-qualified module name for a module instance within self.model."""
        for name, mod in self._named_modules_dict().items():
            if mod is target:
                return name
        return None

    def _resolve_param_fqname(self, module: torch.nn.Module, param_name: str = "weight") -> Optional[str]:
        fq_mod = self._resolve_module_fqname(module)
        if fq_mod is None:
            return None
        return f"{fq_mod}.{param_name}" if fq_mod else param_name

    def _find_accelerate_hook(self, module: torch.nn.Module) -> Optional[object]:
        """
        Try to find the AlignDevicesHook that Accelerate attaches. Prefer module-level
        hook, then fall back to root model. Attribute names are part of the public surface
        in recent Accelerate versions.
        """
        hook = getattr(module, "_hf_hook", None)
        if hook is not None:
            return hook
        return getattr(self.model, "_hf_hook", None)

    def _materialize_param_if_meta(self, fq_param_name: str) -> torch.nn.Parameter:
        """
        If the parameter lives on 'meta', materialize only this tensor using one of
        the Accelerate pathways:
          1) CPU offload via `weights_map` (most common when using device_map="auto").
          2) Disk offload via (offload_dir + offload_index).
        Returns the (possibly replaced) Parameter.
        """
        params = self._named_parameters_dict()
        if fq_param_name not in params:
            raise KeyError(f"Parameter {fq_param_name!r} not found in model.named_parameters().")
        p = params[fq_param_name]
        if p.device.type != "meta":
            return p

        owner = self._owner_module_for_param(fq_param_name)
        hook = self._find_accelerate_hook(owner)
        if hook is None:
            raise RuntimeError(
                f"{fq_param_name!r} is on 'meta' but no Accelerate hook was found. "
                "The model might not have been dispatched with accelerate."
            )

        # Path A: CPU offload (weights_map)
        weights_map = getattr(hook, "weights_map", None)
        fq_param_name_prefix = fq_param_name.rstrip('weight')
        if weights_map is not None and fq_param_name_prefix == weights_map.prefix:
            if set_module_tensor_to_device is None:
                raise RuntimeError(
                    "accelerate.utils.set_module_tensor_to_device is unavailable. Update Accelerate to materialize meta tensors."
                )
            local_name = fq_param_name.split(".")[-1]
            value = weights_map[local_name]
            # Some accelerate versions store lazy callables, resolve if needed
            try:
                value = value() if callable(value) else value
            except Exception:
                pass
            # Place on CPU (hook will stream to GPU when needed)
            set_module_tensor_to_device(owner, local_name, torch.device("cpu"), value=value)
            return self._named_parameters_dict()[fq_param_name]

        # Path B: Disk offload (offload_dir + offload_index)
        offload_folder = getattr(hook, "offload_folder", None) or getattr(hook, "offload_dir", None)
        offload_index = getattr(hook, "offload_index", None)
        if offload_folder is not None and offload_index is not None and fq_param_name in offload_index:
            if load_offloaded_weights is None:
                raise RuntimeError(
                    "accelerate.utils.load_offloaded_weights is unavailable. Update Accelerate to load offloaded tensors."
                )
            tiny_index = {fq_param_name: offload_index[fq_param_name]}
            load_offloaded_weights(self.model, tiny_index, offload_folder)
            return self._named_parameters_dict()[fq_param_name]

        raise RuntimeError(
            "Accelerate hook present, but neither `weights_map` (CPU offload) nor `offload_dir/index` (disk offload) "
            f"contain an entry for {fq_param_name!r}. If the tensor is quantized or tied, edit before quantization or "
            "ensure the model was loaded via Transformers with device_map/offload so hooks carry metadata."
        )

    def _owner_module_for_param(self, fq_param_name: str) -> torch.nn.Module:
        parts = fq_param_name.split(".")
        owner = self.model
        for p in parts[:-1]:
            owner = getattr(owner, p)
        return owner

    def _ensure_editable_weight(self, weight_module: torch.nn.Module) -> Tuple[str, torch.nn.Parameter]:
        """
        Ensure the weight tensor is materialized (not on 'meta') and return its fully-qualified
        name and Parameter. Only materializes the specific tensor we need.
        """
        fq_param = self._resolve_param_fqname(weight_module, "weight")
        if fq_param is None:
            raise RuntimeError(
                "Could not resolve fully-qualified name for weight module; is it attached to the provided model?"
            )
        p = self._materialize_param_if_meta(fq_param)
        return fq_param, p

    # Model architecture resolution
    def _get_weight_module(self, super_weight: SuperWeight) -> torch.nn.Module:
        """Get the weight module for a super weight using the MLP handler."""
        layer_idx = super_weight.layer

        # MoE-aware path
        if self.mlp_handler.is_moe_layer(layer_idx):
            if hasattr(super_weight, 'expert_id'):
                expert_idx = super_weight.expert_id
                component_path = super_weight.component

                # Shared expert case
                if getattr(super_weight, 'is_shared_expert', False):
                    shared_components = self.mlp_handler.get_shared_expert_components(layer_idx)
                    for comp_type, module in shared_components.items():
                        if comp_type in ['down', 'output']:
                            return module
                    raise ValueError(f"No shared expert output projection found in layer {layer_idx}")

                # Parse component path like 'experts.0.down_proj'
                parts = component_path.split('.') if component_path else []
                if len(parts) >= 3 and parts[0] == 'experts':
                    component_name = parts[2]  # e.g., 'down_proj'
                    expert_components = self.mlp_handler.get_expert_components(layer_idx, expert_idx)

                    # Try to match by name using architecture hints
                    arch_info = self.mlp_handler.get_mlp_architecture(layer_idx)
                    if arch_info.is_moe and arch_info.moe_info and len(self.mlp_handler.get_moe_experts(layer_idx)) > expert_idx:
                        first_expert = self.mlp_handler.get_moe_experts(layer_idx)[0]
                        expert_arch = self.mlp_handler._detect_mlp_architecture(first_expert)
                        for comp_type, module in expert_components.items():
                            if comp_type in expert_arch.components:
                                comp_info = expert_arch.components[comp_type]
                                if comp_info.component_name == component_name:
                                    return module
                    raise ValueError(f"Component {component_name} not found in expert {expert_idx}")
                else:
                    raise ValueError(f"Invalid MoE component path: {component_path}")
            else:
                raise ValueError(f"MoE super weight missing expert_id: {super_weight}")
        else:
            # Regular MLP layer
            components = self.mlp_handler.get_mlp_components(layer_idx)
            if 'down' in components:
                return components['down']
            if 'output' in components:
                return components['output']
            raise ValueError(f"No output projection found in layer {layer_idx}")

    # Public operations
    def scale_super_weights(self, super_weights: List[SuperWeight], scale_factor: float) -> bool:
        """
        Scale super weights by a given factor.

        Args:
            super_weights: List of SuperWeight objects to scale
            scale_factor: Factor to scale by (0.0 = zero, 1.0 = no change, 2.0 = double)
        Returns:
            True if all weights were successfully scaled
        """
        success_count = 0

        for sw in super_weights:
            try:
                weight_module = self._get_weight_module(sw)
                fq_param, param = self._ensure_editable_weight(weight_module)

                # Backup original value once
                if sw not in self.original_values:
                    orig = param[sw.row, sw.column].detach().clone()
                    self.original_values[sw] = orig
                    sw.original_value = orig

                # Apply scaling on the parameter's own device/dtype
                with torch.no_grad():
                    new_val = self.original_values[sw] * scale_factor
                    param.data[sw.row, sw.column] = new_val.to(param.dtype)

                self.current_scales[sw] = scale_factor
                self.currently_modified.add(sw)
                success_count += 1
                self.logger.debug(
                    f"Scaled {sw} by {scale_factor:.3f}: {self.original_values[sw].item():.4f} -> {new_val.item():.4f} (param={fq_param})"
                )
            except Exception as e:
                self.logger.error(f"Error scaling weight {sw}: {e}")

        self.logger.info(
            f"Successfully scaled {success_count}/{len(super_weights)} super weights by {scale_factor:.3f}"
        )
        return success_count == len(super_weights)

    def zero_super_weights(self, super_weights: List[SuperWeight]) -> bool:
        """Zero out super weights (convenience method for scale_factor=0.0)."""
        return self.scale_super_weights(super_weights, 0.0)

    def restore_super_weights(self, super_weights: List[SuperWeight]) -> bool:
        """
        Restore specific super weights to their original values.
        """
        success_count = 0

        for sw in super_weights:
            try:
                if sw not in self.original_values:
                    self.logger.warning(f"No original value stored for {sw}")
                    continue

                weight_module = self._get_weight_module(sw)
                fq_param, param = self._ensure_editable_weight(weight_module)

                with torch.no_grad():
                    param.data[sw.row, sw.column] = self.original_values[sw].to(param.dtype)

                # Update tracking
                self.current_scales.pop(sw, None)
                self.currently_modified.discard(sw)
                success_count += 1
                self.logger.debug(
                    f"Restored {sw} to {self.original_values[sw].item():.4f} (param={fq_param})"
                )
            except Exception as e:
                self.logger.error(f"Failed to restore {sw}: {e}")

        self.logger.info(f"Restored {success_count}/{len(super_weights)} weights")
        return success_count == len(super_weights)

    def restore_all(self) -> bool:
        """Restore all currently modified super weights."""
        if not self.currently_modified:
            self.logger.info("No weights to restore")
            return True
        return self.restore_super_weights(list(self.currently_modified))

    def is_modified(self, super_weight: SuperWeight) -> bool:
        return super_weight in self.currently_modified

    def is_zeroed(self, super_weight: SuperWeight) -> bool:
        return (super_weight in self.current_scales and abs(self.current_scales[super_weight]) < 1e-6)

    def get_current_scale(self, super_weight: SuperWeight) -> float:
        return self.current_scales.get(super_weight, 1.0)

    def get_current_value(self, super_weight: SuperWeight) -> Optional[torch.Tensor]:
        """Get the current value of a super weight safely (materializing if needed)."""
        try:
            weight_module = self._get_weight_module(super_weight)
            _, param = self._ensure_editable_weight(weight_module)
            return param[super_weight.row, super_weight.column].detach().clone()
        except Exception as e:
            self.logger.error(f"Error getting current value for {super_weight}: {e}")
            return None

    # Context managers
    @contextmanager
    def temporary_scale(self, super_weights: List[SuperWeight], scale_factor: float):
        """Temporarily scale a set of weights; automatically restore on exit."""
        previously_modified = {sw for sw in super_weights if sw in self.currently_modified}
        newly_modified = [sw for sw in super_weights if sw not in previously_modified]
        try:
            self.scale_super_weights(super_weights, scale_factor)
            yield
        finally:
            if newly_modified:
                self.restore_super_weights(newly_modified)

    @contextmanager
    def temporary_zero(self, super_weights: List[SuperWeight]):
        with self.temporary_scale(super_weights, 0.0):
            yield

    # Validation and summary
    def get_modification_summary(self) -> Dict:
        return {
            'total_modified': len(self.currently_modified),
            'total_zeroed': sum(1 for sw in self.currently_modified if self.is_zeroed(sw)),
            'scale_distribution': {
                scale: len([sw for sw, s in self.current_scales.items() if abs(s - scale) < 1e-6])
                for scale in sorted(set(self.current_scales.values()))
            },
            'modified_weights': [
                {
                    'super_weight': str(sw),
                    'original_value': float(self.original_values[sw].item()),
                    'current_scale': float(self.current_scales[sw]),
                    'current_value': (val.item() if (val := self.get_current_value(sw)) is not None else None),
                }
                for sw in self.currently_modified
            ],
        }

    def validate_super_weights(self, super_weights: List[SuperWeight]) -> List[SuperWeight]:
        """Validate that super weights are compatible with the current model architecture."""
        valid_weights: List[SuperWeight] = []
        for sw in super_weights:
            try:
                weight_module = self._get_weight_module(sw)
                _, param = self._ensure_editable_weight(weight_module)
                shape = param.shape
                if 0 <= sw.row < shape[0] and 0 <= sw.column < shape[1]:
                    valid_weights.append(sw)
                else:
                    self.logger.warning(f"Super weight {sw} coordinates out of bounds for shape {tuple(shape)}")
            except Exception as e:
                self.logger.warning(f"Super weight {sw} is invalid: {e}")
        self.logger.info(f"Validated {len(valid_weights)}/{len(super_weights)} super weights")
        return valid_weights

    # Context manager protocol
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore_all()
