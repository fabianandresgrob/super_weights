import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SuperWeight:
    """Data class to represent a detected super weight"""
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
    
    def __str__(self) -> str:
        return f"Layer {self.layer} {self.component}.weight[{self.row}, {self.column}]"
    
    def __repr__(self) -> str:
        return f"SuperWeight(layer={self.layer}, coords=[{self.row}, {self.column}], input={self.input_value:.2f}, output={self.output_value:.2f})"
    
    def __hash__(self) -> int:
        return hash(self.weight_key)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, SuperWeight):
            return False
        return self.weight_key == other.weight_key


@dataclass
class MoESuperWeight(SuperWeight):
    """Extended super weight for MoE models"""
    expert_id: Optional[int] = None
    routing_weight: Optional[float] = None
    expert_activation_rank: Optional[int] = None  # Rank among activated experts
    router_confidence: Optional[float] = None  # Router confidence for this expert

    def __str__(self) -> str:
        base_str = super().__str__()
        if self.expert_id is not None:
            return f"{base_str} (Expert {self.expert_id})"
        return base_str