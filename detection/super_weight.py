"""Data models for representing detected super weights."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SuperWeight:
    """Data class to represent a detected super weight in dense models."""

    layer: int
    row: int  # output channel
    column: int  # input channel
    component: str
    input_value: float
    output_value: float
    iteration_found: int
    original_value: Optional[torch.Tensor] = None  # Store original weight value

    @property
    def coordinates(self) -> Tuple[int, int]:
        return (self.row, self.column)

    @property
    def weight_key(self) -> Tuple[int, int, int]:
        return (self.layer, self.row, self.column)

    @property
    def magnitude_product(self) -> float:
        return abs(self.input_value * self.output_value)

    def __str__(self) -> str:  # pragma: no cover - simple repr
        return f"Layer {self.layer} {self.component}.weight[{self.row}, {self.column}]"

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"SuperWeight(layer={self.layer}, coords=[{self.row}, {self.column}], "
            f"input={self.input_value:.2f}, output={self.output_value:.2f})"
        )

    def __hash__(self) -> int:
        return hash(self.weight_key)

    def __eq__(self, other) -> bool:  # pragma: no cover - trivial equality
        if not isinstance(other, SuperWeight):
            return False
        return self.weight_key == other.weight_key


@dataclass
class MoESuperWeight(SuperWeight):
    """Extended super weight representation for Mixture-of-Experts models."""

    expert_id: Optional[int] = None
    routing_weight: Optional[float] = None
    expert_activation_rank: Optional[int] = None  # Rank among activated experts
    router_confidence: Optional[float] = None  # Router confidence for this expert

    # Routing-aware metadata
    score_co_spike: Optional[float] = None
    p_active: Optional[float] = None
    low_entropy_positions: Optional[List[int]] = None
    capacity_overflow_rate: Optional[float] = None
    proxies: Optional[Dict[str, float]] = None
    I_nat: Optional[float] = None
    I_int: Optional[float] = None

    def __str__(self) -> str:  # pragma: no cover - simple repr
        base_str = super().__str__()
        if self.expert_id is not None:
            return f"{base_str} (Expert {self.expert_id})"
        return base_str

