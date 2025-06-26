"""
Zero activation attack implementation for super weights

This module implements attacks that attempt to zero out super weight activations
by manipulating input vectors to drive intermediate computations to zero.
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional
from .base import SuperWeightAttack, AttackResult, AttackStrategy, AttackType, AttackConstraints
from .constraints import ConstraintHandler

class ZeroActivationAttack(SuperWeightAttack):
    """
    Implementation of zero activation attacks for super weights
    
    Uses mathematical analysis of MLP computations to find input perturbations
    that drive super weight activations to zero, based on the gated MLP structure:
    
    super_activation = SILU(W_gate @ x) * (W_up @ x)
    
    Three main strategies:
    1. Gate Zeroing: Make W_gate @ x = 0 → SILU(0) = 0
    2. Up Zeroing: Make W_up @ x = 0 → activation * 0 = 0  
    3. Activation Saturation: Drive gate to negative values where SILU ≈ 0
    """
    
    def __init__(self, model, tokenizer, constraints: Optional[AttackConstraints] = None):
        super().__init__(model, tokenizer, constraints)
        self.constraint_handler = ConstraintHandler(self.constraints)
        
        # Cache for mathematical analysis results
        self._analysis_cache = {}
    
    def attack(self, super_weight, input_vector: torch.Tensor, 
               strategy: Optional[AttackStrategy] = None, **kwargs) -> AttackResult:
        """
        Execute zero activation attack on super weight
        
        Args:
            super_weight: SuperWeight object to attack
            input_vector: Input tensor to perturb
            strategy: Specific strategy to use (if None, chooses best automatically)
            **kwargs: Additional parameters
            
        Returns:
            AttackResult with attack execution results
        """
        start_time = time.time()
        
        # Get mathematical analysis for this super weight and input
        analysis = self._get_mathematical_analysis(super_weight, input_vector)
        
        # Choose strategy if not provided
        if strategy is None:
            strategies = self.get_available_strategies(super_weight, input_vector)
            strategy = max(strategies, key=lambda s: s.feasibility_score)
        
        # Execute the attack using constraint handler
        try:
            execution_result = self.constraint_handler.execute_constrained_attack(
                super_weight, input_vector, strategy
            )
            
            # Create attack result
            result = AttackResult(
                success=execution_result['success'],
                perturbed_input=execution_result['perturbed_input'],
                original_activation=analysis.get('original_super_activation', 0.0),
                final_activation=execution_result['final_activation'],
                reduction_achieved=self._calculate_reduction(
                    analysis.get('original_super_activation', 0.0),
                    execution_result['final_activation']
                ),
                perturbation_magnitude=execution_result['perturbation_magnitude'],
                strategy_used=strategy,
                constraint_violations=execution_result['violations'],
                execution_time=time.time() - start_time,
                metadata={
                    'mathematical_analysis': analysis,
                    'execution_details': execution_result,
                    'super_weight_coords': super_weight.coords if hasattr(super_weight, 'coords') else None
                }
            )
            
            # Log the attack
            self.log_attack(result)
            
            return result
            
        except Exception as e:
            # Create failed attack result
            return AttackResult(
                success=False,
                perturbed_input=input_vector,
                original_activation=analysis.get('original_super_activation', 0.0),
                final_activation=analysis.get('original_super_activation', 0.0),
                reduction_achieved=0.0,
                perturbation_magnitude=0.0,
                strategy_used=strategy,
                constraint_violations=[f"Execution error: {str(e)}"],
                execution_time=time.time() - start_time,
                metadata={'error': str(e), 'analysis': analysis}
            )
    
    def analyze_feasibility(self, super_weight, input_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze feasibility of zero activation attacks
        
        Args:
            super_weight: SuperWeight to analyze
            input_vector: Input vector for analysis
            
        Returns:
            Dictionary with feasibility analysis including available strategies
        """
        
        # Get mathematical analysis
        analysis = self._get_mathematical_analysis(super_weight, input_vector)
        
        # Get available strategies
        strategies = self.get_available_strategies(super_weight, input_vector)
        
        # Calculate constraint room
        constraint_room = self.constraint_handler.calculate_constraint_room(input_vector)
        
        # Rank strategies by feasibility
        strategies.sort(key=lambda s: s.feasibility_score, reverse=True)
        
        return {
            'super_weight': super_weight,
            'mathematical_analysis': analysis,
            'available_strategies': strategies,
            'recommended_strategy': strategies[0] if strategies else None,
            'constraint_analysis': constraint_room,
            'overall_feasibility': strategies[0].feasibility_score if strategies else 0.0,
            'estimated_success_probability': strategies[0].success_probability if strategies else 0.0
        }
    
    def get_available_strategies(self, super_weight, input_vector: torch.Tensor) -> List[AttackStrategy]:
        """
        Get list of available attack strategies for this super weight
        
        Args:
            super_weight: SuperWeight to analyze
            input_vector: Input vector to analyze
            
        Returns:
            List of AttackStrategy objects, sorted by feasibility
        """
        
        analysis = self._get_mathematical_analysis(super_weight, input_vector)
        strategies = []
        
        # Strategy 1: Gate Zeroing
        gate_strategy = self._create_gate_zeroing_strategy(analysis, input_vector)
        if gate_strategy:
            strategies.append(gate_strategy)
        
        # Strategy 2: Up Projection Zeroing
        up_strategy = self._create_up_zeroing_strategy(analysis, input_vector)
        if up_strategy:
            strategies.append(up_strategy)
        
        # Strategy 3: Activation Saturation
        saturation_strategy = self._create_saturation_strategy(analysis, input_vector)
        if saturation_strategy:
            strategies.append(saturation_strategy)
        
        return strategies
    
    def _get_mathematical_analysis(self, super_weight, input_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Get or compute mathematical analysis for the super weight and input
        
        Uses cached results when possible to avoid recomputation
        """
        
        # Create cache key
        cache_key = (
            str(super_weight.coords if hasattr(super_weight, 'coords') else super_weight),
            hash(input_vector.data.tobytes())
        )
        
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        # Simulate mathematical analysis based on your existing work
        # In real implementation, this would call your ActivationAnalyzer
        analysis = self._simulate_mathematical_analysis(super_weight, input_vector)
        
        self._analysis_cache[cache_key] = analysis
        return analysis
    
    def _simulate_mathematical_analysis(self, super_weight, input_vector: torch.Tensor) -> Dict[str, Any]:
        """
        Simulate the mathematical analysis from your activation.py
        
        In real implementation, this would integrate with your existing ActivationAnalyzer
        """
        
        # These values come from your mathematical breakdown
        # In real implementation, compute these dynamically
        return {
            'original_super_activation': -401.5,
            'gate_output': 15.2421875,
            'up_output': -26.34375,
            'activated_gate': 15.2421875,
            'gate_weights': {
                'top_positive_dims': [803, 1542, 1930, 1344, 517],
                'top_positive_weights': [0.317138671875, 0.266357421875, 0.246337890625, 0.2396240234375, 0.1917724609375],
                'top_negative_dims': [1981, 623, 1916, 967, 130],
                'top_negative_weights': [-0.2432861328125, -0.2041015625, -0.149658203125, -0.1490478515625, -0.1474609375]
            },
            'up_weights': {
                'top_positive_dims': [1764, 1139, 1655, 1618, 1916],
                'top_positive_weights': [0.275390625, 0.1201171875, 0.11749267578125, 0.10455322265625, 0.0968017578125],
                'top_negative_dims': [517, 819, 803, 915, 1735],
                'top_negative_weights': [-0.09918212890625, -0.0963134765625, -0.09539794921875, -0.089111328125, -0.08795166015625]
            },
            'target_channel': super_weight.coords[1] if hasattr(super_weight, 'coords') and len(super_weight.coords) > 1 else 1710,
            'super_weight_row': super_weight.coords[0] if hasattr(super_weight, 'coords') else 1764
        }
    
    def _create_gate_zeroing_strategy(self, analysis: Dict[str, Any], 
                                    input_vector: torch.Tensor) -> Optional[AttackStrategy]:
        """Create gate zeroing attack strategy"""
        
        gate_output = analysis.get('gate_output', 15.2421875)
        gate_weights = analysis.get('gate_weights', {})
        
        # Use top positive dimensions for gate zeroing
        target_dims = gate_weights.get('top_positive_dims', [803])[:3]  # Top 3 dimensions
        target_weights = gate_weights.get('top_positive_weights', [0.317138671875])[:3]
        
        # Calculate required perturbation magnitude
        if len(target_weights) > 0:
            # Estimate perturbation needed using strongest weight
            estimated_perturbation = abs(gate_output / target_weights[0])
        else:
            estimated_perturbation = float('inf')
        
        # Calculate feasibility score
        constraint_room = self.constraint_handler.calculate_constraint_room(input_vector)
        feasibility_score = self._calculate_feasibility_score(
            estimated_perturbation, constraint_room, "gate"
        )
        
        return AttackStrategy(
            name="Gate Zeroing Attack",
            attack_type=AttackType.ZERO_ACTIVATION,
            target_component="gate",
            feasibility_score=feasibility_score,
            perturbation_magnitude=estimated_perturbation,
            success_probability=0.9 if feasibility_score > 0.7 else 0.5,
            computational_cost="low",
            description="Zero the gate projection to make SILU(0) = 0",
            parameters={
                'target_dimensions': target_dims,
                'target_weights': target_weights,
                'current_gate_output': gate_output,
                'up_output': analysis.get('up_output', -26.34375),
                'strategy_type': 'gate_zeroing'
            }
        )
    
    def _create_up_zeroing_strategy(self, analysis: Dict[str, Any], 
                                  input_vector: torch.Tensor) -> Optional[AttackStrategy]:
        """Create up projection zeroing attack strategy"""
        
        up_output = analysis.get('up_output', -26.34375)
        up_weights = analysis.get('up_weights', {})
        
        # Use top positive dimensions for up zeroing
        target_dims = up_weights.get('top_positive_dims', [1764])[:3]
        target_weights = up_weights.get('top_positive_weights', [0.275390625])[:3]
        
        # Calculate required perturbation magnitude
        if len(target_weights) > 0:
            estimated_perturbation = abs(up_output / target_weights[0])
        else:
            estimated_perturbation = float('inf')
        
        # Calculate feasibility score
        constraint_room = self.constraint_handler.calculate_constraint_room(input_vector)
        feasibility_score = self._calculate_feasibility_score(
            estimated_perturbation, constraint_room, "up"
        )
        
        return AttackStrategy(
            name="Up Projection Zeroing Attack",
            attack_type=AttackType.ZERO_ACTIVATION,
            target_component="up",
            feasibility_score=feasibility_score,
            perturbation_magnitude=estimated_perturbation,
            success_probability=0.8 if feasibility_score > 0.6 else 0.4,
            computational_cost="low",
            description="Zero the up projection to make activation * 0 = 0",
            parameters={
                'target_dimensions': target_dims,
                'target_weights': target_weights,
                'current_up_output': up_output,
                'gate_output': analysis.get('gate_output', 15.2421875),
                'strategy_type': 'up_zeroing'
            }
        )
    
    def _create_saturation_strategy(self, analysis: Dict[str, Any], 
                                  input_vector: torch.Tensor) -> Optional[AttackStrategy]:
        """Create activation saturation attack strategy"""
        
        gate_output = analysis.get('gate_output', 15.2421875)
        gate_weights = analysis.get('gate_weights', {})
        
        # Target to drive gate to -10 where SiLU ≈ 0
        saturation_target = -10.0
        required_change = saturation_target - gate_output
        
        # Use top positive dimensions
        target_dims = gate_weights.get('top_positive_dims', [803])[:3]
        target_weights = gate_weights.get('top_positive_weights', [0.317138671875])[:3]
        
        # Calculate required perturbation magnitude
        if len(target_weights) > 0:
            estimated_perturbation = abs(required_change / target_weights[0])
        else:
            estimated_perturbation = float('inf')
        
        # Calculate feasibility score (harder than zeroing)
        constraint_room = self.constraint_handler.calculate_constraint_room(input_vector)
        feasibility_score = self._calculate_feasibility_score(
            estimated_perturbation, constraint_room, "saturation"
        ) * 0.8  # Penalty for approximation errors
        
        return AttackStrategy(
            name="SiLU Saturation Attack",
            attack_type=AttackType.SATURATION,
            target_component="activation_saturation",
            feasibility_score=feasibility_score,
            perturbation_magnitude=estimated_perturbation,
            success_probability=0.7 if feasibility_score > 0.5 else 0.2,
            computational_cost="medium",
            description="Drive gate to negative values where SiLU ≈ 0",
            parameters={
                'target_dimensions': target_dims,
                'target_weights': target_weights,
                'current_gate_output': gate_output,
                'saturation_target': saturation_target,
                'up_output': analysis.get('up_output', -26.34375),
                'strategy_type': 'saturation'
            }
        )
    
    def _calculate_feasibility_score(self, perturbation_magnitude: float, 
                                   constraint_room: Dict[str, float], 
                                   strategy_type: str) -> float:
        """
        Calculate feasibility score for a strategy
        
        Args:
            perturbation_magnitude: Required perturbation magnitude
            constraint_room: Available constraint room
            strategy_type: Type of strategy ("gate", "up", "saturation")
            
        Returns:
            Feasibility score between 0 and 1
        """
        
        # Base score from perturbation vs available room
        available_budget = min(
            constraint_room.get('l2_budget', 10.0),
            constraint_room.get('linf_budget', 2.0) * 10  # Scale linf to approximate l2
        )
        
        if available_budget <= 0:
            return 0.0
        
        magnitude_score = max(0.0, 1.0 - (perturbation_magnitude / available_budget))
        
        # Strategy-specific bonuses/penalties
        strategy_multipliers = {
            'gate': 1.0,      # Gate zeroing is cleanest mathematically
            'up': 0.9,        # Up zeroing works but less preferred
            'saturation': 0.8  # Saturation has approximation errors
        }
        
        multiplier = strategy_multipliers.get(strategy_type, 0.5)
        
        return magnitude_score * multiplier
    
    def _calculate_reduction(self, original_activation: float, final_activation: float) -> float:
        """Calculate percentage reduction in activation magnitude"""
        if abs(original_activation) < 1e-8:
            return 100.0 if abs(final_activation) < 1e-8 else 0.0
        
        return abs(original_activation - final_activation) / abs(original_activation) * 100.0
    
    def batch_attack(self, super_weights: List, input_vectors: List[torch.Tensor], 
                    strategy: Optional[AttackStrategy] = None) -> List[AttackResult]:
        """
        Execute zero activation attacks on multiple super weights
        
        Args:
            super_weights: List of SuperWeight objects to attack
            input_vectors: List of input tensors (must match super_weights length)
            strategy: Optional strategy to use for all attacks
            
        Returns:
            List of AttackResult objects
        """
        
        if len(super_weights) != len(input_vectors):
            raise ValueError("Number of super weights must match number of input vectors")
        
        results = []
        for sw, input_vec in zip(super_weights, input_vectors):
            result = self.attack(sw, input_vec, strategy)
            results.append(result)
        
        return results
    
    def find_optimal_strategy(self, super_weight, input_vector: torch.Tensor) -> AttackStrategy:
        """
        Find the optimal attack strategy for a given super weight and input
        
        Args:
            super_weight: SuperWeight to analyze
            input_vector: Input vector to analyze
            
        Returns:
            Best AttackStrategy based on feasibility and success probability
        """
        
        strategies = self.get_available_strategies(super_weight, input_vector)
        
        if not strategies:
            raise ValueError("No feasible attack strategies found")
        
        # Score strategies by combined feasibility and success probability
        def strategy_score(strategy: AttackStrategy) -> float:
            return (strategy.feasibility_score * 0.6 + 
                   strategy.success_probability * 0.4)
        
        return max(strategies, key=strategy_score)
    
    def analyze_attack_surface(self, super_weights: List, 
                             input_vectors: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Analyze the attack surface across multiple super weights
        
        Args:
            super_weights: List of SuperWeight objects
            input_vectors: List of input tensors
            
        Returns:
            Dictionary with attack surface analysis
        """
        
        if len(super_weights) != len(input_vectors):
            input_vectors = [input_vectors[0]] * len(super_weights)  # Broadcast single input
        
        analysis_results = []
        strategy_counts = {}
        feasibility_scores = []
        
        for sw, input_vec in zip(super_weights, input_vectors):
            feasibility = self.analyze_feasibility(sw, input_vec)
            analysis_results.append(feasibility)
            
            if feasibility['recommended_strategy']:
                strategy_name = feasibility['recommended_strategy'].name
                strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
                feasibility_scores.append(feasibility['overall_feasibility'])
        
        return {
            'total_super_weights': len(super_weights),
            'attackable_super_weights': sum(1 for r in analysis_results if r['overall_feasibility'] > 0.5),
            'average_feasibility': np.mean(feasibility_scores) if feasibility_scores else 0.0,
            'strategy_distribution': strategy_counts,
            'most_common_strategy': max(strategy_counts, key=strategy_counts.get) if strategy_counts else None,
            'detailed_analysis': analysis_results
        }