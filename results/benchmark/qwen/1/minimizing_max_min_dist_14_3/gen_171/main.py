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
        
    def sobol_like_distribution(self, samples: int = 14) -> np.ndarray:
        """Generate points using Sobol-like distribution for better space-filling properties."""
        points = []
        
        # Generate points using modified Fibonacci approach with Sobol-inspired properties
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(samples):
            # Distribute points more uniformly across the sphere
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Add controlled perturbations to create better 3D distribution
            theta = phi * i + np.sin(i * 0.7) * 0.1 + np.cos(i * 0.3) * 0.05
            
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
        """Enhanced objective function with distance variance regularization."""
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
            
        # Add regularization term to penalize extreme distance variations
        # This helps achieve more uniform distribution
        distance_variance_penalty = 0.05 * (d_max - d_min) / (d_max + 1e-12)
        
        # Return negative because we're minimizing
        return -(d_min / d_max) + distance_variance_penalty
    
    def run_single_optimization(self, initial_points: np.ndarray, method: str = 'L-BFGS-B') -> Tuple[np.ndarray, float]:
        """Run optimization from given initial points with specific method."""
        initial_flat = initial_points.flatten()
        
        # Optimization bounds: [0,1] for all coordinates
        bounds = [(0, 1) for _ in range(self.n_points * self.dim)]
        
        # Different optimization parameters for different methods
        options = {'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
        
        # Perform optimization
        result = minimize(
            self.objective_function,
            initial_flat,
            method=method,
            bounds=bounds,
            options=options
        )
        
        if not result.success:
            # Fallback to initial points if optimization fails
            optimized_points = initial_points.copy()
        else:
            optimized_points = result.x.reshape((self.n_points, self.dim))
            # Ensure bounds are respected
            optimized_points = np.clip(optimized_points, 0, 1)
            
        # Final validation
        ratio, _, _ = self.calculate_ratio(optimized_points)
        return optimized_points, ratio
    
    def adaptive_multi_start(self, max_iterations: int = 35) -> np.ndarray:
        """Perform adaptive multi-start optimization with improved diversity."""
        best_ratio = -np.inf
        best_points = None
        
        # Generate base Sobol-like points
        base_points = self.sobol_like_distribution(self.n_points)
        base_points = self.normalize_to_cube(base_points)
        
        # Adaptive perturbation strategy with faster decay
        base_perturbation = 0.03
        perturbation_decay = 0.93
        min_perturbation = 0.0005
        max_starts = 35  # Increased for better exploration
        
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
            
            # Alternate optimization methods for better exploration
            method = 'L-BFGS-B' if start_iteration % 2 == 0 else 'SLSQP'
            
            # Run optimization
            optimized_points, ratio = self.run_single_optimization(current_points, method)
            
            # Update best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
            # Early termination if we're getting good results and have explored enough
            if best_ratio > 0.47 and start_iteration > 8:
                break
                
        return best_points if best_points is not None else base_points
    
    def refine_solution(self, points: np.ndarray, max_refinements: int = 4) -> np.ndarray:
        """Apply sequential refinement with multiple optimization methods."""
        current_points = points.copy()
        
        for refinement_step in range(max_refinements):
            # Try different optimization methods for better refinement
            initial_flat = current_points.flatten()
            bounds = [(0, 1) for _ in range(self.n_points * self.dim)]
            
            # Alternate between L-BFGS-B and SLSQP for comprehensive refinement
            method = 'L-BFGS-B' if refinement_step % 2 == 0 else 'SLSQP'
            
            try:
                result = minimize(
                    self.objective_function,
                    initial_flat,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    refined_points = result.x.reshape((self.n_points, self.dim))
                    refined_points = np.clip(refined_points, 0, 1)
                    
                    # Validate improvement
                    _, old_min, old_max = self.calculate_ratio(current_points)
                    _, new_min, new_max = self.calculate_ratio(refined_points)
                    
                    # Only accept improvements in min distance
                    if new_min > old_min:
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
                base_points = self.sobol_like_distribution(self.n_points)
                return self.normalize_to_cube(base_points)
                
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = PointDispersionOptimizer(n_points=14, dim=3, seed=42)
    
    # Main optimization loop with improved strategy
    best_points = optimizer.adaptive_multi_start(max_iterations=35)
    
    # Apply aggressive refinement with multiple passes
    refined_points = optimizer.refine_solution(best_points, max_refinements=4)
    
    # Final validation
    final_points = optimizer.validate_and_correct(refined_points)
    
    return final_points

# EVOLVE-BLOCK-END