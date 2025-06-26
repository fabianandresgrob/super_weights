"""
Constraint handling and optimization for super weight attacks
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from .base import AttackConstraints, AttackStrategy, AttackResult

class ConstraintHandler:
    """
    Handles constraints during attack execution to ensure realistic perturbations
    """
    
    def __init__(self, constraints: AttackConstraints):
        self.constraints = constraints
    
    def project_to_constraints(self, perturbed_input: torch.Tensor, 
                             original_input: torch.Tensor) -> torch.Tensor:
        """
        Project perturbed input back to constraint space
        
        Args:
            perturbed_input: Input that may violate constraints
            original_input: Original input for reference
            
        Returns:
            Projected input that satisfies constraints
        """
        projected = perturbed_input.clone()
        
        # Apply bound constraints
        projected = torch.clamp(
            projected,
            self.constraints.input_bounds[0],
            self.constraints.input_bounds[1]
        )
        
        # Apply L2 constraint if specified
        if self.constraints.max_perturbation_l2 is not None:
            diff = projected - original_input
            diff_norm = torch.norm(diff)
            if diff_norm > self.constraints.max_perturbation_l2:
                # Scale down the perturbation
                scaling_factor = self.constraints.max_perturbation_l2 / diff_norm
                projected = original_input + diff * scaling_factor
        
        # Apply Linf constraint if specified
        if self.constraints.max_perturbation_linf is not None:
            diff = projected - original_input
            diff = torch.clamp(
                diff,
                -self.constraints.max_perturbation_linf,
                self.constraints.max_perturbation_linf
            )
            projected = original_input + diff
        
        # Final bounds check after all projections
        projected = torch.clamp(
            projected,
            self.constraints.input_bounds[0],
            self.constraints.input_bounds[1]
        )
        
        return projected
    
    def calculate_constraint_room(self, input_vector: torch.Tensor) -> Dict[str, float]:
        """
        Calculate how much room we have for perturbations given constraints
        
        Args:
            input_vector: Current input vector
            
        Returns:
            Dictionary with constraint room metrics
        """
        room_metrics = {}
        
        # Bound constraints room
        room_lower = (input_vector - self.constraints.input_bounds[0]).min().item()
        room_upper = (self.constraints.input_bounds[1] - input_vector).min().item()
        room_metrics['bound_room_lower'] = room_lower
        room_metrics['bound_room_upper'] = room_upper
        room_metrics['bound_room_symmetric'] = min(abs(room_lower), abs(room_upper))
        
        # L2 perturbation room
        if self.constraints.max_perturbation_l2 is not None:
            room_metrics['l2_budget'] = self.constraints.max_perturbation_l2
        else:
            # Estimate reasonable L2 budget as fraction of input norm
            room_metrics['l2_budget'] = 0.1 * torch.norm(input_vector).item()
        
        # Linf perturbation room
        if self.constraints.max_perturbation_linf is not None:
            room_metrics['linf_budget'] = self.constraints.max_perturbation_linf
        else:
            # Use minimum bound room as Linf budget
            room_metrics['linf_budget'] = room_metrics['bound_room_symmetric']
        
        return room_metrics
    
    def optimize_multi_dimensional_perturbation(self, 
                                              target_dimensions: List[int],
                                              target_weights: List[float],
                                              target_change: float,
                                              input_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Optimize perturbation across multiple dimensions to achieve target change
        while respecting constraints
        
        Args:
            target_dimensions: List of input dimensions to modify
            target_weights: Corresponding weights (importance) for each dimension
            target_change: Target total change in output
            input_vector: Current input vector
            
        Returns:
            Dictionary with optimization results
        """
        
        perturbations = {}
        remaining_change = target_change
        total_perturbation_magnitude = 0.0
        
        # Sort dimensions by weight magnitude (most influential first)
        dim_weight_pairs = list(zip(target_dimensions, target_weights))
        dim_weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for dim, weight in dim_weight_pairs:
            if abs(remaining_change) < 1e-6:  # Close enough to target
                break
            
            current_value = input_vector[dim].item()
            
            # Calculate maximum allowed change for this dimension
            max_decrease = self.constraints.input_bounds[0] - current_value
            max_increase = self.constraints.input_bounds[1] - current_value
            
            # Calculate ideal change for this dimension
            if weight != 0:
                ideal_input_change = remaining_change / weight
            else:
                continue
            
            # Clamp to bounds
            actual_input_change = np.clip(
                ideal_input_change,
                max_decrease,
                max_increase
            )
            
            # Calculate contribution to target change
            contribution = weight * actual_input_change
            
            # If this would overshoot, scale it down
            if abs(contribution) > abs(remaining_change):
                actual_input_change = remaining_change / weight
                contribution = remaining_change
            
            # Apply additional constraints (L2, Linf)
            if self.constraints.max_perturbation_linf is not None:
                actual_input_change = np.clip(
                    actual_input_change,
                    -self.constraints.max_perturbation_linf,
                    self.constraints.max_perturbation_linf
                )
                contribution = weight * actual_input_change
            
            new_value = current_value + actual_input_change
            
            perturbations[dim] = {
                'original_value': current_value,
                'perturbation': actual_input_change,
                'new_value': new_value,
                'contribution': contribution,
                'weight': weight
            }
            
            remaining_change -= contribution
            total_perturbation_magnitude += abs(actual_input_change)
        
        # Create perturbed input
        perturbed_input = input_vector.clone()
        for dim, info in perturbations.items():
            perturbed_input[dim] = info['new_value']
        
        # Final constraint projection
        perturbed_input = self.project_to_constraints(perturbed_input, input_vector)
        
        # Calculate actual achieved change
        achieved_change = target_change - remaining_change
        success_rate = abs(achieved_change) / abs(target_change) if target_change != 0 else 1.0
        
        return {
            'perturbed_input': perturbed_input,
            'perturbations': perturbations,
            'target_change': target_change,
            'achieved_change': achieved_change,
            'remaining_change': remaining_change,
            'success_rate': success_rate,
            'total_perturbation_magnitude': total_perturbation_magnitude,
            'dimensions_modified': len(perturbations),
            'constraint_violations': self._check_violations(perturbed_input, input_vector)
        }
    
    def execute_constrained_attack(self, super_weight, input_vector: torch.Tensor, 
                                 strategy: AttackStrategy) -> Dict[str, Any]:
        """
        Execute an attack strategy while respecting all constraints
        
        Args:
            super_weight: SuperWeight object being attacked
            input_vector: Input to perturb
            strategy: Attack strategy to execute
            
        Returns:
            Dictionary with attack execution results
        """
        
        if strategy.attack_type.value == "zero_activation":
            return self._execute_zero_activation_constrained(super_weight, input_vector, strategy)
        else:
            raise NotImplementedError(f"Attack type {strategy.attack_type} not implemented")
    
    def _execute_zero_activation_constrained(self, super_weight, input_vector: torch.Tensor, 
                                           strategy: AttackStrategy) -> Dict[str, Any]:
        """Execute zero activation attack with constraints"""
        
        # Extract strategy parameters
        params = strategy.parameters
        target_component = strategy.target_component
        
        if target_component == "gate":
            return self._execute_gate_zeroing_constrained(super_weight, input_vector, params)
        elif target_component == "up":
            return self._execute_up_zeroing_constrained(super_weight, input_vector, params)
        elif target_component == "activation_saturation":
            return self._execute_saturation_constrained(super_weight, input_vector, params)
        else:
            raise ValueError(f"Unknown target component: {target_component}")
    
    def _execute_gate_zeroing_constrained(self, super_weight, input_vector: torch.Tensor, 
                                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gate zeroing with constraints"""
        
        # Get gate weights and current output from params
        target_dimensions = params.get('target_dimensions', [803])
        target_weights = params.get('target_weights', [0.317138671875])
        current_gate_output = params.get('current_gate_output', 15.2421875)
        
        # Use multi-dimensional optimization
        result = self.optimize_multi_dimensional_perturbation(
            target_dimensions=target_dimensions,
            target_weights=target_weights, 
            target_change=-current_gate_output,  # Want to zero the gate output
            input_vector=input_vector
        )
        
        # Calculate final activation (approximation)
        gate_reduction = abs(result['achieved_change']) / abs(current_gate_output) if current_gate_output != 0 else 1.0
        up_output = params.get('up_output', -26.34375)
        
        # Approximate final super activation
        remaining_gate_output = current_gate_output + result['achieved_change']
        # SiLU approximation for small values
        activated_gate = remaining_gate_output if remaining_gate_output > 0 else remaining_gate_output * np.exp(remaining_gate_output)
        final_super_activation = activated_gate * up_output
        
        return {
            'success': result['success_rate'] > 0.8,
            'perturbed_input': result['perturbed_input'],
            'final_activation': final_super_activation,
            'perturbation_magnitude': result['total_perturbation_magnitude'],
            'violations': result['constraint_violations'],
            'gate_reduction_achieved': gate_reduction,
            'optimization_details': result
        }
    
    def _execute_up_zeroing_constrained(self, super_weight, input_vector: torch.Tensor, 
                                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute up projection zeroing with constraints"""
        
        target_dimensions = params.get('target_dimensions', [1764])
        target_weights = params.get('target_weights', [0.275390625])
        current_up_output = params.get('current_up_output', -26.34375)
        
        result = self.optimize_multi_dimensional_perturbation(
            target_dimensions=target_dimensions,
            target_weights=target_weights,
            target_change=-current_up_output,  # Want to zero the up output
            input_vector=input_vector
        )
        
        # Calculate final activation
        gate_output = params.get('gate_output', 15.2421875)
        remaining_up_output = current_up_output + result['achieved_change']
        final_super_activation = gate_output * remaining_up_output  # Simplified
        
        return {
            'success': result['success_rate'] > 0.8,
            'perturbed_input': result['perturbed_input'],
            'final_activation': final_super_activation,
            'perturbation_magnitude': result['total_perturbation_magnitude'],
            'violations': result['constraint_violations'],
            'up_reduction_achieved': abs(result['achieved_change']) / abs(current_up_output) if current_up_output != 0 else 1.0,
            'optimization_details': result
        }
    
    def _execute_saturation_constrained(self, super_weight, input_vector: torch.Tensor, 
                                      params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute activation saturation attack with constraints"""
        
        target_dimensions = params.get('target_dimensions', [803])
        target_weights = params.get('target_weights', [0.317138671875])
        current_gate_output = params.get('current_gate_output', 15.2421875)
        saturation_target = params.get('saturation_target', -10.0)
        
        result = self.optimize_multi_dimensional_perturbation(
            target_dimensions=target_dimensions,
            target_weights=target_weights,
            target_change=saturation_target - current_gate_output,
            input_vector=input_vector
        )
        
        # Calculate final activation with SiLU saturation
        final_gate_output = current_gate_output + result['achieved_change']
        if final_gate_output < -5:  # SiLU approximately zero
            activated_gate = 0.0
        else:
            # SiLU approximation
            activated_gate = final_gate_output if final_gate_output > 0 else final_gate_output * np.exp(final_gate_output)
        
        up_output = params.get('up_output', -26.34375)
        final_super_activation = activated_gate * up_output
        
        return {
            'success': final_gate_output < -5,  # Successfully saturated
            'perturbed_input': result['perturbed_input'],
            'final_activation': final_super_activation,
            'perturbation_magnitude': result['total_perturbation_magnitude'],
            'violations': result['constraint_violations'],
            'saturation_achieved': final_gate_output < -5,
            'final_gate_output': final_gate_output,
            'optimization_details': result
        }
    
    def _check_violations(self, perturbed_input: torch.Tensor, 
                         original_input: torch.Tensor) -> List[str]:
        """Check for constraint violations"""
        violations = []
        
        # Check bounds
        if torch.any(perturbed_input < self.constraints.input_bounds[0]):
            violations.append("Values below lower bound")
        if torch.any(perturbed_input > self.constraints.input_bounds[1]):
            violations.append("Values above upper bound")
        
        # Check L2 constraint
        if self.constraints.max_perturbation_l2 is not None:
            l2_norm = torch.norm(perturbed_input - original_input).item()
            if l2_norm > self.constraints.max_perturbation_l2:
                violations.append(f"L2 perturbation exceeds limit: {l2_norm:.4f} > {self.constraints.max_perturbation_l2}")
        
        # Check Linf constraint
        if self.constraints.max_perturbation_linf is not None:
            linf_norm = torch.norm(perturbed_input - original_input, p=float('inf')).item()
            if linf_norm > self.constraints.max_perturbation_linf:
                violations.append(f"Linf perturbation exceeds limit: {linf_norm:.4f} > {self.constraints.max_perturbation_linf}")
        
        return violations