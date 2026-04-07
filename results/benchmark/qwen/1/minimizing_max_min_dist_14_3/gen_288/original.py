# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random
import time
from typing import Tuple, Optional

class PointDispersionOptimizer:
    """Advanced optimizer for 3D point dispersion problem combining multiple strategies."""
    
    def __init__(self, n_points: int = 14, dim: int = 3, seed: int = 42):
        self.n_points = n_points
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def fibonacci_sphere(self, samples: int = 14) -> np.ndarray:
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
    
    def normalize_to_cube(self, points: np.ndarray) -> np.ndarray:
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
        """Objective function to maximize min/max distance ratio."""
        points = x_flat.reshape((self.n_points, self.dim))
        distances = pdist(points)
        # Filter out near-zero distances
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -np.inf
            
        # Return negative because we're minimizing
        return -d_min / d_max
    
    def run_single_optimization(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run optimization from given initial points."""
        initial_flat = initial_points.flatten()
        
        # First optimization with L-BFGS-B (coarse)
        bounds = [(0, 1) for _ in range(self.n_points * self.dim)]
        result = minimize(
            self.objective_function,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        if not result.success:
            # Fallback to initial points if first optimization fails
            optimized_points = initial_points.copy()
        else:
            optimized_points = result.x.reshape((self.n_points, self.dim))
            # Ensure bounds are respected
            optimized_points = np.clip(optimized_points, 0, 1)
            
        # Final validation
        ratio, _, _ = self.calculate_ratio(optimized_points)
        return optimized_points, ratio
    
    def adaptive_multi_start(self, max_iterations: int = 30) -> np.ndarray:
        """Perform adaptive multi-start optimization."""
        best_ratio = -np.inf
        best_points = None
        
        # Generate base Fibonacci points
        base_points = self.fibonacci_sphere(self.n_points)
        base_points = self.normalize_to_cube(base_points)
        
        # Adaptive perturbation strategy inspired by successful approaches
        base_perturbation = 0.02
        perturbation_decay = 0.95
        min_perturbation = 0.001
        max_starts = 25  # Increased from 15 for better exploration
        
        # Multi-start optimization loop
        for start_iteration in range(max_starts):
            # Adaptive perturbation scaling
            current_perturbation = max(base_perturbation * (perturbation_decay ** start_iteration), 
                                     min_perturbation)
            
            # Generate perturbed starting points
            if start_iteration == 0:
                # First start: use base points
                current_points = base_points.copy()
            else:
                # Subsequent starts: perturb base points with adaptive scaling
                perturbation = np.random.normal(0, current_perturbation, base_points.shape)
                current_points = base_points + perturbation
                current_points = np.clip(current_points, 0, 1)
            
            # Run optimization
            optimized_points, ratio = self.run_single_optimization(current_points)
            
            # Update best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
            # Early termination if we're getting good results and have explored enough
            if best_ratio > 0.45 and start_iteration > 5:
                break
                
        return best_points if best_points is not None else base_points
    
    def refine_solution(self, points: np.ndarray, max_refinements: int = 3) -> np.ndarray:
        """Apply sequential refinement to improve solution quality."""
        current_points = points.copy()
        
        for refinement_step in range(max_refinements):
            # Try different optimization methods for better refinement
            initial_flat = current_points.flatten()
            bounds = [(0, 1) for _ in range(self.n_points * self.dim)]
            
            # Try SLSQP for better constraint handling
            try:
                result = minimize(
                    self.objective_function,
                    initial_flat,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success:
                    refined_points = result.x.reshape((self.n_points, self.dim))
                    refined_points = np.clip(refined_points, 0, 1)
                    
                    # Validate improvement
                    _, old_min, old_max = self.calculate_ratio(current_points)
                    _, new_min, new_max = self.calculate_ratio(refined_points)
                    
                    if new_min > old_min and new_max <= old_max:
                        current_points = refined_points
            except:
                pass
                
        return current_points
    
    def validate_and_correct(self, points: np.ndarray) -> np.ndarray:
        """Validate solution and correct any issues."""
        # Check for degenerate cases
        distances = pdist(points)
        if len(distances) > 0:
            min_distance = np.min(distances)
            if min_distance < 1e-12:
                # Return fallback to initial points if degenerate
                base_points = self.fibonacci_sphere(self.n_points)
                return self.normalize_to_cube(base_points)
                
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointDispersionOptimizer(n_points=14, dim=3, seed=42)
    
    # Main optimization loop with adaptive strategy
    best_points = optimizer.adaptive_multi_start(max_iterations=25)
    
    # Apply refinement with SLSQP for final improvement
    refined_points = optimizer.refine_solution(best_points, max_refinements=2)
    
    # Final validation
    final_points = optimizer.validate_and_correct(refined_points)
    
    return final_points

# EVOLVE-BLOCK-END