import torch
from typing import Tuple, Optional, List
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
    """Enhanced super weight for MoE models with routing statistics and causal scoring"""
    expert_id: Optional[int] = None
    routing_weight: Optional[float] = None
    expert_activation_rank: Optional[int] = None  # Rank among activated experts
    router_confidence: Optional[float] = None  # Router confidence for this expert
    
    # Enhanced routing statistics
    p_active: Optional[float] = None  # Expert usage probability p_active^(l)(e)
    routing_entropy: Optional[float] = None  # Position-wise routing entropy
    capacity_overflow_rate: Optional[float] = None  # Overflow rate for this expert
    
    # Co-spike scoring
    co_spike_score: Optional[float] = None  # S^(l,e)(r,c) alignment score
    routed_tokens_count: Optional[int] = None  # Number of tokens routed to this expert
    
    # Causal impact scoring
    impact_natural: Optional[float] = None  # I_nat(w*, e, l) - natural routing impact
    impact_interventional: Optional[float] = None  # I_int(w*, e, l) - interventional impact
    
    # Fast proxy metrics
    energy_reduction: Optional[float] = None  # Post-layer energy reduction E_c*
    stopword_skew: Optional[float] = None  # Change in stopword probability mass
    
    # Detection metadata
    detection_iteration: Optional[int] = None  # Which iteration this was found
    position_indices: Optional[List[int]] = None  # Token positions that contributed to detection
    
    # for shared experts
    is_shared_expert: bool = False  # Flag for shared experts
    
    @property
    def causal_agreement(self) -> Optional[float]:
        """Agreement between natural and interventional impact scores"""
        if self.impact_natural is not None and self.impact_interventional is not None:
            if abs(self.impact_interventional) < 1e-8:
                return 0.0
            return self.impact_natural / self.impact_interventional
        return None
    
    @property
    def routing_stability(self) -> Optional[float]:
        """Measure of routing stability (inverse of entropy)"""
        if self.routing_entropy is not None:
            return 1.0 / (1.0 + self.routing_entropy)
        return None
    
    def __str__(self) -> str:
        base_str = super().__str__()
        if self.expert_id is not None:
            additional_info = f" (Expert {self.expert_id}"
            if self.p_active is not None:
                additional_info += f", p_active={self.p_active:.3f}"
            if self.co_spike_score is not None:
                additional_info += f", score={self.co_spike_score:.3f}"
            additional_info += ")"
            return f"{base_str}{additional_info}"
        return base_str