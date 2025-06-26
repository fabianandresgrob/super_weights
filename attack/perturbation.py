"""
Perturbation optimization methods for super weight attacks

This module provides advanced optimization techniques for finding minimal
perturbations that achieve target activation reductions while respecting constraints.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from scipy.optimize import minimize
from .base import AttackConstraints

class PerturbationOptimizer:
    """
    Advanced optimization methods for finding minimal perturbations
    that achieve target effects on super weight activations
    """
    
    def __init__(self, constraints: AttackConstraints):
        self.constraints = constraints
        
    def minimize_perturbation_for_target_reduction(self, 
                                                 super_weight,
                                                 input_vector: torch.Tensor,
                                                 target_reduction: float,
                                                 activation_function: Callable) -> Dict[str, Any]:
        """
        Find minimal perturbation that achieves target activation reduction
        
        Args:
            super_weight: SuperWeight object being attacked
            input_vector: Original input vector
            target_reduction: Target reduction percentage (0.0 to 1.0)
            activation_function: Function that computes activation from input
            
        Returns:
            Dictionary with optimization results
        """
        
        x_orig = input_vector.detach().cpu().numpy()
        original_activation = activation_function(input_vector).item()
        target_activation = original_activation * (1.0 - target_reduction)
        
        def objective(x_pert):
            """Minimize L2 norm of perturbation"""
            return np.linalg.norm(x_pert - x_orig)
        
        def activation_constraint(x_pert):
            """Constraint: achieve target activation reduction"""
            x_tensor = torch.from_numpy(x_pert).float()
            current_activation = activation_function(x_tensor).item()
            return target_activation - current_activation  # Should be close to 0
        
        def bound_constraints(x_pert):
            """Ensure perturbation stays within input bounds"""
            violations = []
            
            # Lower bound constraints
            violations.extend(x_pert - self.constraints.input_bounds[0])
            # Upper bound constraints  
            violations.extend(self.constraints.input_bounds[1] - x_pert)
            
            return np.array(violations)
        
        # Set up constraints for scipy
        constraints = [
            {'type': 'eq', 'fun': activation_constraint},
            {'type': 'ineq', 'fun': bound_constraints}
        ]
        
        # Add L2 constraint if specified
        if self.constraints.max_perturbation_l2 is not None:
            def l2_constraint(x_pert):
                return self.constraints.max_perturbation_l2 - np.linalg.norm(x_pert - x_orig)
            constraints.append({'type': 'ineq', 'fun': l2_constraint})
        
        # Initial guess: small random perturbation
        x0 = x_orig + np.random.normal(0, 0.01, x_orig.shape)
        x0 = np.clip(x0, self.constraints.input_bounds[0], self.constraints.input_bounds[1])
        
        try:
            # Run optimization
            result = minimize(
                objective, 
                x0, 
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                perturbed_input = torch.from_numpy(result.x).float()
                final_activation = activation_function(perturbed_input).item()
                
                return {
                    'success': True,
                    'perturbed_input': perturbed_input,
                    'perturbation_magnitude': np.linalg.norm(result.x - x_orig),
                    'original_activation': original_activation,
                    'final_activation': final_activation,
                    'target_activation': target_activation,
                    'reduction_achieved': abs(original_activation - final_activation) / abs(original_activation) * 100,
                    'optimization_result': result,
                    'constraint_violations': self._check_constraint_violations(
                        torch.from_numpy(result.x).float(), input_vector
                    )
                }
            else:
                return {
                    'success': False,
                    'error': result.message,
                    'optimization_result': result
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def gradient_based_attack(self, 
                            super_weight,
                            input_vector: torch.Tensor,
                            target_reduction: float,
                            model,
                            learning_rate: float = 0.01,
                            max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Gradient-based attack using backpropagation to find perturbations
        
        Args:
            super_weight: SuperWeight object being attacked
            input_vector: Original input vector
            target_reduction: Target reduction percentage
            model: The transformer model
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with attack results
        """
        
        # Clone input and enable gradients
        x_pert = input_vector.clone().detach().requires_grad_(True)
        x_orig = input_vector.clone().detach()
        
        # Get original activation
        with torch.no_grad():
            original_activation = self._compute_super_weight_activation(model, super_weight, x_orig)
        
        target_activation = original_activation * (1.0 - target_reduction)
        
        # Optimization loop
        history = []
        best_result = None
        best_loss = float('inf')
        
        optimizer = torch.optim.Adam([x_pert], lr=learning_rate)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Compute current activation
            current_activation = self._compute_super_weight_activation(model, super_weight, x_pert)
            
            # Loss function: combination of activation target and perturbation magnitude
            activation_loss = (current_activation - target_activation) ** 2
            perturbation_loss = torch.norm(x_pert - x_orig, p=2) ** 2
            
            # Weighted combination
            total_loss = activation_loss + 0.1 * perturbation_loss
            
            total_loss.backward()
            
            # Apply gradients
            optimizer.step()
            
            # Project to constraints
            with torch.no_grad():
                x_pert.data = torch.clamp(
                    x_pert.data,
                    self.constraints.input_bounds[0],
                    self.constraints.input_bounds[1]
                )
                
                # Apply L2 constraint if specified
                if self.constraints.max_perturbation_l2 is not None:
                    perturbation = x_pert.data - x_orig
                    pert_norm = torch.norm(perturbation)
                    if pert_norm > self.constraints.max_perturbation_l2:
                        x_pert.data = x_orig + perturbation * (self.constraints.max_perturbation_l2 / pert_norm)
            
            # Record progress
            current_loss = total_loss.item()
            history.append({
                'iteration': iteration,
                'total_loss': current_loss,
                'activation_loss': activation_loss.item(),
                'perturbation_loss': perturbation_loss.item(),
                'current_activation': current_activation.item(),
                'perturbation_norm': torch.norm(x_pert - x_orig).item()
            })
            
            # Track best result
            if current_loss < best_loss:
                best_loss = current_loss
                best_result = {
                    'perturbed_input': x_pert.clone().detach(),
                    'activation': current_activation.item(),
                    'perturbation_norm': torch.norm(x_pert - x_orig).item(),
                    'iteration': iteration
                }
            
            # Early stopping if target achieved
            if abs(current_activation.item() - target_activation) < abs(target_activation) * 0.01:
                break
        
        final_activation = best_result['activation']
        reduction_achieved = abs(original_activation - final_activation) / abs(original_activation) * 100
        
        return {
            'success': reduction_achieved > target_reduction * 80,  # 80% of target
            'perturbed_input': best_result['perturbed_input'],
            'original_activation': original_activation,
            'final_activation': final_activation,
            'target_activation': target_activation,
            'reduction_achieved': reduction_achieved,
            'perturbation_magnitude': best_result['perturbation_norm'],
            'iterations_used': best_result['iteration'],
            'optimization_history': history,
            'constraint_violations': self._check_constraint_violations(
                best_result['perturbed_input'], input_vector
            )
        }
    
    def evolutionary_optimization(self, 
                                super_weight,
                                input_vector: torch.Tensor,
                                target_reduction: float,
                                activation_function: Callable,
                                population_size: int = 50,
                                generations: int = 100) -> Dict[str, Any]:
        """
        Evolutionary algorithm for finding perturbations
        
        Args:
            super_weight: SuperWeight object being attacked
            input_vector: Original input vector
            target_reduction: Target reduction percentage
            activation_function: Function to compute activation
            population_size: Size of evolutionary population
            generations: Number of generations to evolve
            
        Returns:
            Dictionary with optimization results
        """
        
        x_orig = input_vector.detach().cpu().numpy()
        original_activation = activation_function(input_vector).item()
        target_activation = original_activation * (1.0 - target_reduction)
        
        def fitness_function(x):
            """Fitness function: minimize distance to target activation + perturbation magnitude"""
            x_tensor = torch.from_numpy(x).float()
            current_activation = activation_function(x_tensor).item()
            
            activation_error = abs(current_activation - target_activation)
            perturbation_magnitude = np.linalg.norm(x - x_orig)
            
            # Penalty for constraint violations
            penalty = 0
            if np.any(x < self.constraints.input_bounds[0]) or np.any(x > self.constraints.input_bounds[1]):
                penalty += 1000
            
            if self.constraints.max_perturbation_l2 is not None:
                if perturbation_magnitude > self.constraints.max_perturbation_l2:
                    penalty += 1000
            
            # Minimize: activation error + small perturbation penalty + constraint penalties
            return activation_error + 0.1 * perturbation_magnitude + penalty
        
        # Initialize population
        population = []
        for _ in range(population_size):
            # Random perturbation around original input
            noise_scale = min(0.1, self.constraints.max_perturbation_l2 or 0.5)
            individual = x_orig + np.random.normal(0, noise_scale, x_orig.shape)
            
            # Ensure bounds
            individual = np.clip(
                individual,
                self.constraints.input_bounds[0],
                self.constraints.input_bounds[1]
            )
            population.append(individual)
        
        best_individual = None
        best_fitness = float('inf')
        fitness_history = []
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitnesses = [fitness_function(ind) for ind in population]
            
            # Track best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            fitness_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses)
            })
            
            # Selection: tournament selection
            new_population = []
            for _ in range(population_size):
                # Tournament
                tournament_size = 3
                tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
                tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
                
                # Mutation
                parent = population[winner_idx].copy()
                mutation_strength = 0.01 * (1.0 - generation / generations)  # Decay over time
                offspring = parent + np.random.normal(0, mutation_strength, parent.shape)
                
                # Ensure constraints
                offspring = np.clip(
                    offspring,
                    self.constraints.input_bounds[0],
                    self.constraints.input_bounds[1]
                )
                
                new_population.append(offspring)
            
            population = new_population
        
        # Evaluate best result
        best_tensor = torch.from_numpy(best_individual).float()
        final_activation = activation_function(best_tensor).item()
        reduction_achieved = abs(original_activation - final_activation) / abs(original_activation) * 100
        
        return {
            'success': reduction_achieved > target_reduction * 80,
            'perturbed_input': best_tensor,
            'original_activation': original_activation,
            'final_activation': final_activation,
            'target_activation': target_activation,
            'reduction_achieved': reduction_achieved,
            'perturbation_magnitude': np.linalg.norm(best_individual - x_orig),
            'generations_used': generations,
            'best_fitness': best_fitness,
            'evolution_history': fitness_history,
            'constraint_violations': self._check_constraint_violations(best_tensor, input_vector)
        }
    
    def _compute_super_weight_activation(self, model, super_weight, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute the activation of a super weight for given input
        
        This is a placeholder - in real implementation, this would use your
        existing infrastructure to compute super weight activations
        """
        
        # Placeholder: return a differentiable computation
        # In real implementation, this would do a forward pass through the model
        # and extract the specific super weight activation
        
        # For now, simulate with a simple computation
        if hasattr(super_weight, 'coords'):
            layer, channel = super_weight.coords
            # Simulate some computation that depends on the input
            return torch.sum(input_vector) * 0.001  # Placeholder
        else:
            return torch.sum(input_vector) * 0.001  # Placeholder
    
    def _check_constraint_violations(self, perturbed_input: torch.Tensor, 
                                   original_input: torch.Tensor) -> List[str]:
        """Check for constraint violations"""
        violations = []
        
        # Check bounds
        if torch.any(perturbed_input < self.constraints.input_bounds[0]):
            violations.append("Input values below minimum bound")
        if torch.any(perturbed_input > self.constraints.input_bounds[1]):
            violations.append("Input values above maximum bound")
        
        # Check L2 constraint
        if self.constraints.max_perturbation_l2 is not None:
            l2_norm = torch.norm(perturbed_input - original_input).item()
            if l2_norm > self.constraints.max_perturbation_l2:
                violations.append(f"L2 perturbation {l2_norm:.4f} exceeds limit {self.constraints.max_perturbation_l2}")
        
        # Check Linf constraint
        if self.constraints.max_perturbation_linf is not None:
            linf_norm = torch.norm(perturbed_input - original_input, p=float('inf')).item()
            if linf_norm > self.constraints.max_perturbation_linf:
                violations.append(f"Linf perturbation {linf_norm:.4f} exceeds limit {self.constraints.max_perturbation_linf}")
        
        return violations