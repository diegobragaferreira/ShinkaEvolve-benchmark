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
    
    def sobol_like_init(self, samples: int = 14) -> np.ndarray:
        """Generate points using Sobol-like distribution for better space-filling properties."""
        points = []
        for i in range(samples):
            # Use golden ratio and other irrational numbers for better distribution
            x = (i * 0.618033988749895) % 1.0   # golden ratio
            y = (i * 0.414213562373095) % 1.0   # sqrt(2) - 1
            z = (i * 0.732050807568877) % 1.0   # 2*sqrt(3) - 2

            # Convert to spherical coordinates with proper spacing
            r = np.cbrt(x)  # cubic root for better distribution
            theta = y * 2 * np.pi
            phi_sph = np.arccos(2*z - 1)  # map z to spherical phi

            px = r * np.sin(phi_sph) * np.cos(theta)
            py = r * np.sin(phi_sph) * np.sin(theta)
            pz = r * np.cos(phi_sph)

            points.append([px, py, pz])
        return np.array(points)
    
    def enhanced_multi_strategy_init(self, samples: int = 14) -> np.ndarray:
        """Enhanced initialization combining multiple strategies."""
        points = []
        
        # Strategy 1: Sobol-like distribution for good space filling
        for i in range(samples // 3):
            phi = np.pi * (3. - np.sqrt(5.))
            y = 1 - (i / float(samples // 3 - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i + np.sin(i * 0.3) * 0.15 + np.cos(i * 0.7) * 0.1
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])

        # Strategy 2: Fibonacci spiral for good angular distribution
        phi = np.pi * (3. - np.sqrt(5.))
        for i in range(samples // 3, 2 * samples // 3):
            y = 1 - (i / float(samples // 3 - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i + np.random.uniform(-0.1, 0.1)
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])

        # Strategy 3: Random points for diversity
        for i in range(2 * samples // 3, samples):
            r = np.random.random()
            theta = np.random.uniform(0, 2*np.pi)
            phi_ang = np.arccos(2*r - 1)
            x = np.sin(phi_ang) * np.cos(theta)
            y = np.sin(phi_ang) * np.sin(theta)
            z = np.cos(phi_ang)
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
        """Objective function to maximize min/max distance ratio with variance regularization."""
        points = x_flat.reshape((self.n_points, self.dim))
        distances = pdist(points)
        # Filter out near-zero distances
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        d_mean = np.mean(distances)
        d_var = np.var(distances)

        if d_max == 0:
            return -np.inf
            
        # Regularized objective: maximize ratio while minimizing distance variance
        ratio = d_min / d_max
        # Add penalty for high variance to promote uniform distribution
        variance_penalty = 0.05 * d_var / (d_mean * d_mean + 1e-12)
        regularized_ratio = ratio - variance_penalty
        
        # Return negative because we're minimizing
        return -regularized_ratio
    
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
            options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
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
        
        # Multiple initialization strategies
        init_strategies = [
            self.sobol_like_init,
            self.fibonacci_sphere,
            self.enhanced_multi_strategy_init,
            lambda s: np.random.rand(s, 3),  # Random fallback
        ]
        
        # Adaptive perturbation strategy
        base_perturbation = 0.03
        perturbation_decay = 0.92
        min_perturbation = 0.0005
        max_starts = 25
        
        # Multi-start optimization loop
        for start_iteration in range(max_starts):
            # Adaptive perturbation scaling
            current_perturbation = max(base_perturbation * (perturbation_decay ** start_iteration), 
                                     min_perturbation)
            
            # Select initialization strategy
            strategy_idx = start_iteration % len(init_strategies)
            init_func = init_strategies[strategy_idx]
            
            # Generate initial points
            base_points = init_func(self.n_points)
            base_points = self.normalize_to_cube(base_points)
            
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
                
            # Early termination condition
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
            
            # Try L-BFGS-B first for speed
            try:
                result = minimize(
                    self.objective_function,
                    initial_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12}
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
                
            # Try SLSQP for better constraint handling if needed
            try:
                result = minimize(
                    self.objective_function,
                    initial_flat,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
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
    
    # Apply refinement with multiple optimization methods
    refined_points = optimizer.refine_solution(best_points, max_refinements=3)
    
    # Final validation
    final_points = optimizer.validate_and_correct(refined_points)
    
    return final_points

# EVOLVE-BLOCK-END