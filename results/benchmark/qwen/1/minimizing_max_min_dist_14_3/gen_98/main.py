# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
from typing import Tuple, Optional, List
import warnings

class PointDispersionOptimizer:
    """
    An advanced optimizer for distributing N points in 3D space to maximize 
    the ratio of minimum to maximum pairwise distances.
    """
    
    def __init__(self, n_points: int = 14, dim: int = 3, seed: int = 42):
        self.n_points = n_points
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        
        # Performance and optimization parameters
        self.max_initial_optimizations = 15
        self.max_refinement_stages = 3
        self.max_time_limit = 350.0  # seconds
        self.base_perturbation = 0.05
        self.perturbation_decay = 0.95
        self.min_perturbation = 0.001
        self.initial_tolerance = 1e-9
        self.refined_tolerance = 1e-12
        
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
        if len(points) == 0:
            return points
            
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
    
    def calculate_ratio_stats(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate ratio and distance statistics."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
            
        distances = pdist(points)
        
        # Remove zero distances that might occur due to numerical errors
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
        try:
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
        except Exception:
            return -np.inf
    
    def optimize_single_start(self, initial_points: np.ndarray, 
                            tolerance: float = 1e-9, maxiter: int = 1000) -> Tuple[np.ndarray, float]:
        """Run a single optimization from given starting points."""
        try:
            initial_flat = initial_points.flatten()
            
            bounds = [(0, 1) for _ in range(self.n_points * self.dim)]
            
            # Primary optimization
            result = minimize(
                self.objective_function,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': tolerance, 'gtol': tolerance}
            )
            
            if result.success:
                optimized_points = result.x.reshape((self.n_points, self.dim))
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio, _, _ = self.calculate_ratio_stats(optimized_points)
                return optimized_points, ratio
            else:
                # Return initial points if optimization fails
                ratio, _, _ = self.calculate_ratio_stats(initial_points)
                return initial_points.copy(), ratio
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            ratio, _, _ = self.calculate_ratio_stats(initial_points)
            return initial_points.copy(), ratio
    
    def adaptive_multi_start_strategy(self, start_time: float) -> np.ndarray:
        """Perform adaptive multi-start optimization with intelligent perturbation."""
        best_ratio = -np.inf
        best_points = None
        
        # Generate base Fibonacci points
        base_points = self.fibonacci_sphere(self.n_points)
        base_points = self.normalize_to_cube(base_points)
        
        # Track optimization progress
        current_perturbation = self.base_perturbation
        
        for iteration in range(self.max_initial_optimizations):
            # Check time limit
            if time.time() - start_time > self.max_time_limit:
                break
                
            # Adaptive perturbation
            if iteration > 0:
                perturbation = np.random.normal(0, current_perturbation, base_points.shape)
                current_points = base_points + perturbation
                current_points = np.clip(current_points, 0, 1)
            else:
                current_points = base_points.copy()
            
            # Optimize with decreasing tolerance for better accuracy
            tolerance = self.initial_tolerance * (0.5 ** iteration)
            maxiter = max(200, 1000 // (iteration + 1))
            
            optimized_points, ratio = self.optimize_single_start(
                current_points, tolerance, maxiter
            )
            
            # Update best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
            
            # Update perturbation for next iteration
            current_perturbation = max(
                current_perturbation * self.perturbation_decay,
                self.min_perturbation
            )
            
            # Early stopping condition
            if best_ratio > 0.45 and iteration > 3:
                break
        
        # Return best solution or fallback to base points
        if best_points is not None:
            return best_points
        else:
            return base_points
    
    def sequential_refinement(self, points: np.ndarray) -> np.ndarray:
        """Apply sequential refinement with progressively tighter tolerances."""
        current_points = points.copy()
        
        for refinement_step in range(self.max_refinement_stages):
            # Check time limit
            if time.time() - time.time() > self.max_time_limit:
                break
                
            try:
                # Progressive refinement with tightening tolerances
                tolerance = self.refined_tolerance * (0.1 ** refinement_step)
                maxiter = 500 // (refinement_step + 1)
                
                initial_flat = current_points.flatten()
                bounds = [(0, 1) for _ in range(self.n_points * self.dim)]
                
                # Try different methods for better convergence
                method = 'L-BFGS-B' if refinement_step < 2 else 'SLSQP'
                
                result = minimize(
                    self.objective_function,
                    initial_flat,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': tolerance, 'gtol': tolerance}
                )
                
                if result.success:
                    refined_points = result.x.reshape((self.n_points, self.dim))
                    refined_points = np.clip(refined_points, 0, 1)
                    
                    # Validate improvement by checking distance statistics
                    _, old_min, old_max = self.calculate_ratio_stats(current_points)
                    _, new_min, new_max = self.calculate_ratio_stats(refined_points)
                    
                    if new_min > old_min and (new_max <= old_max or abs(new_max - old_max) < 1e-6):
                        current_points = refined_points
                
            except Exception:
                # Continue with current points if refinement fails
                continue
                
        return current_points
    
    def validate_solution(self, points: np.ndarray) -> np.ndarray:
        """Validate and correct solution if necessary."""
        try:
            # Check for degenerate cases
            distances = pdist(points)
            distances = distances[distances > 1e-12]
            
            if len(distances) > 0:
                min_distance = np.min(distances)
                if min_distance < 1e-12:
                    # Return fallback to initial configuration
                    base_points = self.fibonacci_sphere(self.n_points)
                    return self.normalize_to_cube(base_points)
            else:
                # No valid distances, return fallback
                base_points = self.fibonacci_sphere(self.n_points)
                return self.normalize_to_cube(base_points)
                
        except Exception:
            # Fall back to base points on any validation error
            base_points = self.fibonacci_sphere(self.n_points)
            return self.normalize_to_cube(base_points)
            
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    
    # Initialize optimizer
    optimizer = PointDispersionOptimizer(n_points=14, dim=3, seed=42)
    
    # Main optimization pipeline
    try:
        # Step 1: Adaptive multi-start optimization
        best_points = optimizer.adaptive_multi_start_strategy(start_time)
        
        # Check if we've exceeded time limits
        if time.time() - start_time > optimizer.max_time_limit:
            return best_points
        
        # Step 2: Sequential refinement
        refined_points = optimizer.sequential_refinement(best_points)
        
        # Check if we've exceeded time limits
        if time.time() - start_time > optimizer.max_time_limit:
            return refined_points
        
        # Step 3: Final validation and correction
        final_points = optimizer.validate_solution(refined_points)
        
        return final_points
        
    except Exception as e:
        # Fallback to simple initialization if anything fails
        warnings.warn(f"Fallback due to error: {str(e)}")
        base_points = optimizer.fibonacci_sphere(14)
        return optimizer.normalize_to_cube(base_points)

# EVOLVE-BLOCK-END