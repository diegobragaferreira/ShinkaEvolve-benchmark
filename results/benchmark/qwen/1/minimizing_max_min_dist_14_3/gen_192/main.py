# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random
from typing import Tuple, Optional, List
import time
from sobol_seq import i4_sobol_generate  # Using Sobol sequence for better space filling

class PointConfiguration:
    """Configuration class for point dispersion optimization parameters."""
    
    def __init__(self):
        self.n_points = 14
        self.dim = 3
        self.seed = 42
        self.base_perturbation = 0.01
        self.perturbation_decay = 0.97
        self.min_perturbation = 0.001
        self.max_starts = 30
        self.initial_optimization_options = {'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        self.refinement_options = {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
        self.final_refinement_options = {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
        self.regularization_weight = 0.15  # Weight for distance variance regularization

class PointInitializer:
    """Handles generation and normalization of initial point configurations."""
    
    @staticmethod
    def sobol_points(samples: int = 14) -> np.ndarray:
        """Generate points using 3D Sobol sequence for better space filling."""
        # Generate Sobol points in [0,1]^3
        sobol_points = i4_sobol_generate(3, samples)
        # Scale to [-1,1]^3 then map to unit sphere
        scaled_points = sobol_points * 2 - 1
        # Normalize to unit sphere
        norms = np.linalg.norm(scaled_points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized_points = scaled_points / norms
        return normalized_points
    
    @staticmethod
    def fibonacci_sphere(samples: int = 14) -> np.ndarray:
        """Generate points on unit sphere using Fibonacci spiral."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    @staticmethod
    def normalize_to_cube(points: np.ndarray) -> np.ndarray:
        """Normalize points to fit in [0,1]^3 cube."""
        # Center around origin and scale appropriately
        centered = points - np.mean(points, axis=0)
        max_coord = np.max(np.abs(centered))
        if max_coord > 0:
            scaled = centered / max_coord * 0.5
        else:
            scaled = centered
        # Shift to [0,1]^3
        normalized = scaled + 0.5
        return normalized

class PointOptimizer:
    """Core optimization engine for point dispersion problem."""
    
    def __init__(self, config: PointConfiguration):
        self.config = config
        np.random.seed(config.seed)
        random.seed(config.seed)
        
    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio and distances."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        # Filter out near-zero distances to avoid numerical issues
        distances = distances[distances > 1e-12]
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max > 0:
            ratio = d_min / d_max
        else:
            ratio = 0.0
            
        return ratio, d_min, d_max
    
    def objective_function(self, x_flat: np.ndarray) -> float:
        """Objective function to maximize min/max distance ratio with regularization."""
        points = x_flat.reshape((self.config.n_points, self.config.dim))
        distances = pdist(points)
        # Filter out near-zero distances
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -np.inf
            
        # Add distance variance regularization to encourage uniform distribution
        if len(distances) > 1:
            distance_variance = np.var(distances)
            regularization = self.config.regularization_weight * distance_variance
        else:
            regularization = 0
            
        # Return negative because we're minimizing
        return -(d_min / d_max) + regularization
    
    def constraint_sphere(self, x):
        """Constraint to ensure points stay within unit sphere."""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0
    
    def constraint_bounds(self, x):
        """Constraint to ensure all points are within [0,1]^3 bounds."""
        points = x.reshape(-1, 3)
        return np.concatenate([
            points.flatten() - 0.0,      # lower bound
            1.0 - points.flatten()       # upper bound
        ])

class MultiStartOptimizer:
    """Manages multiple optimization starts with adaptive strategies."""
    
    def __init__(self, config: PointConfiguration, optimizer: PointOptimizer):
        self.config = config
        self.optimizer = optimizer
        self.initializer = PointInitializer()
        
    def run_adaptive_multi_start(self) -> np.ndarray:
        """Perform adaptive multi-start optimization with refined perturbation strategy."""
        # Generate base initialization using Sobol sequence for better space filling
        base_points = self.initializer.sobol_points(self.config.n_points)
        base_points = self.initializer.normalize_to_cube(base_points)
        
        best_ratio = -np.inf
        best_points = base_points.copy()
        
        # Multi-start optimization loop with adaptive perturbation
        for start_iteration in range(self.config.max_starts):
            # Adaptive perturbation scaling
            current_perturbation = max(
                self.config.base_perturbation * (self.config.perturbation_decay ** start_iteration), 
                self.config.min_perturbation
            )
            
            # Generate starting points with diversity
            if start_iteration == 0:
                # First start: use base Sobol points
                current_points = base_points.copy()
            elif start_iteration <= 5:
                # First few starts: perturbed Sobol points
                perturbation = np.random.normal(0, current_perturbation, base_points.shape)
                current_points = base_points + perturbation
                current_points = np.clip(current_points, 0, 1)
            else:
                # Later starts: alternate between Fibonacci and Sobol variations
                if start_iteration % 2 == 0:
                    # Fibonacci variant
                    fib_points = self.initializer.fibonacci_sphere(self.config.n_points)
                    fib_points = self.initializer.normalize_to_cube(fib_points)
                    perturbation = np.random.normal(0, current_perturbation * 2, fib_points.shape)
                    current_points = fib_points + perturbation
                    current_points = np.clip(current_points, 0, 1)
                else:
                    # Another Sobol variant
                    sobol_points = self.initializer.sobol_points(self.config.n_points)
                    sobol_points = self.initializer.normalize_to_cube(sobol_points)
                    perturbation = np.random.normal(0, current_perturbation * 1.5, sobol_points.shape)
                    current_points = sobol_points + perturbation
                    current_points = np.clip(current_points, 0, 1)
            
            # Run optimization
            optimized_points, ratio = self.optimizer.optimize_single_start(current_points)
            
            # Update best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
            # Early termination if we're getting close to good solutions
            if best_ratio > 0.48 and start_iteration > 10:
                break
                
        return best_points

    def optimize_single_start(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run optimization from given initial points with constraint tightening."""
        initial_flat = initial_points.flatten()
        bounds = [(0, 1) for _ in range(self.config.n_points * self.config.dim)]
        
        # First phase with relaxed constraints (allowing slight violation)
        try:
            # Use L-BFGS-B for the first phase with relaxed tolerances
            result = minimize(
                self.optimizer.objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
            )
        except Exception:
            return initial_points.copy(), -np.inf
            
        if not result.success:
            return initial_points.copy(), -np.inf
            
        optimized_points = result.x.reshape((self.config.n_points, self.config.dim))
        # Ensure bounds are respected
        optimized_points = np.clip(optimized_points, 0, 1)
        
        # Second phase with tightened constraints
        try:
            result = minimize(
                self.optimizer.objective_function,
                optimized_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': self.optimizer.constraint_sphere},
                           {'type': 'ineq', 'fun': self.optimizer.constraint_bounds}],
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
            )
        except Exception:
            pass
            
        if result.success:
            optimized_points = result.x.reshape((self.config.n_points, self.config.dim))
            optimized_points = np.clip(optimized_points, 0, 1)
        
        # Final validation
        ratio, _, _ = self.optimizer.calculate_ratio(optimized_points)
        return optimized_points, ratio

class SolutionRefiner:
    """Handles solution refinement and validation."""
    
    def __init__(self, config: PointConfiguration, optimizer: PointOptimizer):
        self.config = config
        self.optimizer = optimizer
    
    def refine_solution(self, points: np.ndarray) -> np.ndarray:
        """Apply sequential refinement to improve solution quality."""
        current_points = points.copy()
        bounds = [(0, 1) for _ in range(self.config.n_points * self.config.dim)]
        
        # Try SLSQP for better constraint handling
        try:
            initial_flat = current_points.flatten()
            result = minimize(
                self.optimizer.objective_function,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': self.optimizer.constraint_sphere},
                           {'type': 'ineq', 'fun': self.optimizer.constraint_bounds}],
                options=self.config.refinement_options
            )
            
            if result.success:
                refined_points = result.x.reshape((self.config.n_points, self.config.dim))
                refined_points = np.clip(refined_points, 0, 1)
                
                # Validate improvement
                _, old_min, old_max = self.optimizer.calculate_ratio(current_points)
                _, new_min, new_max = self.optimizer.calculate_ratio(refined_points)
                
                if new_min > old_min and new_max <= old_max:
                    current_points = refined_points
        except Exception:
            pass
        
        # Apply final aggressive refinement
        try:
            initial_flat = current_points.flatten()
            result = minimize(
                self.optimizer.objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options=self.config.final_refinement_options
            )
            
            if result.success:
                candidate_points = result.x.reshape((self.config.n_points, self.config.dim))
                candidate_points = np.clip(candidate_points, 0, 1)
                
                # Validate improvement
                _, old_min, old_max = self.optimizer.calculate_ratio(current_points)
                _, new_min, new_max = self.optimizer.calculate_ratio(candidate_points)
                
                if new_min > old_min and new_max <= old_max:
                    current_points = candidate_points
        except Exception:
            pass
            
        return current_points
    
    def validate_and_correct(self, points: np.ndarray) -> np.ndarray:
        """Validate solution and correct any issues."""
        # Check for degenerate cases
        distances = pdist(points)
        if len(distances) > 0:
            min_distance = np.min(distances)
            if min_distance < 1e-12:
                # Return fallback to base points if degenerate
                initializer = PointInitializer()
                base_points = initializer.sobol_points(self.config.n_points)
                return initializer.normalize_to_cube(base_points)
                
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize configuration
    config = PointConfiguration()
    
    # Initialize components
    optimizer = PointOptimizer(config)
    multi_start_optimizer = MultiStartOptimizer(config, optimizer)
    refiner = SolutionRefiner(config, optimizer)
    
    # Main optimization pipeline
    best_points = multi_start_optimizer.run_adaptive_multi_start()
    refined_points = refiner.refine_solution(best_points)
    final_points = refiner.validate_and_correct(refined_points)
    
    return final_points

# EVOLVE-BLOCK-END