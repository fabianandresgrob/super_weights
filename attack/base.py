"""
Base classes and interfaces for super weight attacks
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import torch
import numpy as np

class AttackType(Enum):
    """Types of attacks supported"""
    ZERO_ACTIVATION = "zero_activation"
    GRADIENT_BASED = "gradient_based"
    PERTURBATION = "perturbation"
    SATURATION = "saturation"

@dataclass
class AttackStrategy:
    """Container for a specific attack strategy"""
    name: str
    attack_type: AttackType
    target_component: str  # 'gate', 'up', 'activation', etc.
    feasibility_score: float
    perturbation_magnitude: float
    success_probability: float
    computational_cost: str  # 'low', 'medium', 'high'
    description: str
    parameters: Dict[str, Any]

@dataclass
class AttackResult:
    """Container for attack execution results"""
    success: bool
    perturbed_input: torch.Tensor
    original_activation: float
    final_activation: float
    reduction_achieved: float  # Percentage reduction in activation magnitude
    perturbation_magnitude: float
    strategy_used: AttackStrategy
    constraint_violations: List[str]
    execution_time: float
    metadata: Dict[str, Any]
    
    @property
    def activation_reduction_percentage(self) -> float:
        """Calculate percentage reduction in activation magnitude"""
        if abs(self.original_activation) < 1e-8:
            return 0.0
        return abs(self.original_activation - self.final_activation) / abs(self.original_activation) * 100

@dataclass 
class AttackConstraints:
    """Constraints for realistic attacks"""
    input_bounds: Tuple[float, float] = (-3.0, 3.0)
    max_perturbation_l2: Optional[float] = None
    max_perturbation_linf: Optional[float] = None
    preserve_token_boundaries: bool = True
    semantic_similarity_threshold: Optional[float] = None
    allowed_token_substitutions: Optional[List[str]] = None

class SuperWeightAttack(ABC):
    """
    Base class for all super weight attacks
    
    This abstract class defines the interface that all super weight attacks
    must implement, providing a consistent API for attack execution and analysis.
    """
    
    def __init__(self, model, tokenizer, constraints: Optional[AttackConstraints] = None):
        """
        Initialize attack system
        
        Args:
            model: The transformer model to attack
            tokenizer: Associated tokenizer
            constraints: Optional constraints for realistic attacks
        """
        self.model = model
        self.tokenizer = tokenizer
        self.constraints = constraints or AttackConstraints()
        self.attack_history: List[AttackResult] = []
    
    @abstractmethod
    def attack(self, super_weight, input_vector: torch.Tensor, **kwargs) -> AttackResult:
        """
        Execute the attack on a super weight
        
        Args:
            super_weight: SuperWeight object to attack
            input_vector: Input tensor to perturb
            **kwargs: Additional attack-specific parameters
            
        Returns:
            AttackResult containing results of the attack
        """
        pass
    
    @abstractmethod
    def analyze_feasibility(self, super_weight, input_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze attack feasibility before execution
        
        Args:
            super_weight: SuperWeight object to analyze
            input_vector: Input tensor to analyze
            
        Returns:
            Dictionary containing feasibility analysis
        """
        pass
    
    @abstractmethod
    def get_available_strategies(self, super_weight, input_vector: torch.Tensor) -> List[AttackStrategy]:
        """
        Get list of available attack strategies for this super weight
        
        Args:
            super_weight: SuperWeight object to analyze
            input_vector: Input tensor to analyze
            
        Returns:
            List of available AttackStrategy objects
        """
        pass
    
    def validate_constraints(self, perturbed_input: torch.Tensor, 
                           original_input: torch.Tensor) -> List[str]:
        """
        Validate that attack results respect constraints
        
        Args:
            perturbed_input: The perturbed input tensor
            original_input: The original input tensor
            
        Returns:
            List of constraint violations (empty if all constraints satisfied)
        """
        violations = []
        
        # Check input bounds
        if torch.any(perturbed_input < self.constraints.input_bounds[0]):
            violations.append("Input values below minimum bound")
        if torch.any(perturbed_input > self.constraints.input_bounds[1]):
            violations.append("Input values above maximum bound")
        
        # Check L2 perturbation limit
        if self.constraints.max_perturbation_l2 is not None:
            l2_norm = torch.norm(perturbed_input - original_input).item()
            if l2_norm > self.constraints.max_perturbation_l2:
                violations.append(f"L2 perturbation {l2_norm:.4f} exceeds limit {self.constraints.max_perturbation_l2}")
        
        # Check Linf perturbation limit
        if self.constraints.max_perturbation_linf is not None:
            linf_norm = torch.norm(perturbed_input - original_input, p=float('inf')).item()
            if linf_norm > self.constraints.max_perturbation_linf:
                violations.append(f"Linf perturbation {linf_norm:.4f} exceeds limit {self.constraints.max_perturbation_linf}")
        
        return violations
    
    def calculate_perturbation_metrics(self, perturbed_input: torch.Tensor, 
                                     original_input: torch.Tensor) -> Dict[str, float]:
        """Calculate various perturbation metrics"""
        diff = perturbed_input - original_input
        
        return {
            'l1_norm': torch.norm(diff, p=1).item(),
            'l2_norm': torch.norm(diff, p=2).item(),
            'linf_norm': torch.norm(diff, p=float('inf')).item(),
            'relative_l2': (torch.norm(diff, p=2) / torch.norm(original_input, p=2)).item() if torch.norm(original_input) > 0 else float('inf'),
            'num_changed_elements': (diff != 0).sum().item(),
            'max_absolute_change': torch.abs(diff).max().item(),
            'mean_absolute_change': torch.abs(diff).mean().item()
        }
    
    def log_attack(self, result: AttackResult):
        """Log attack result for analysis"""
        self.attack_history.append(result)
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get statistics about attack history"""
        if not self.attack_history:
            return {"total_attacks": 0}
        
        successes = [r for r in self.attack_history if r.success]
        
        return {
            "total_attacks": len(self.attack_history),
            "successful_attacks": len(successes),
            "success_rate": len(successes) / len(self.attack_history),
            "average_reduction": np.mean([r.activation_reduction_percentage for r in successes]) if successes else 0,
            "average_perturbation": np.mean([r.perturbation_magnitude for r in self.attack_history]),
            "most_successful_strategy": max(successes, key=lambda x: x.activation_reduction_percentage).strategy_used.name if successes else None
        }