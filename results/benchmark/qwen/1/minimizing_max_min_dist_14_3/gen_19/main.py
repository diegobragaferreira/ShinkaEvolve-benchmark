# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple, Optional, List
import warnings

class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""
    
    def __init__(self, n_points: int = 14, dimension: int = 3):
        self.n_points = n_points
        self.dimension = dimension
        self.best_ratio = -np.inf
        self.best_points = None
        self.target_ratio = 1.0 / math.sqrt(4.165849767)  # Benchmark value
        
    def _distance_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance."""
        if points.shape[0] < 2:
            return 0.0
            
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0.0
            
        return min_dist / max_dist
    
    def _objective_function(self, points_flat: np.ndarray) -> float:
        """Minimize negative of distance ratio (since we want to maximize)."""
        points = points_flat.reshape(-1, self.dimension)
        return -self._distance_ratio(points)
    
    def _fibonacci_spiral_points(self) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral."""
        points = []
        phi = math.acos(-1 + 2 * (0 / (self.n_points - 1)))
        theta = math.sqrt(self.n_points * math.pi) * phi
        
        for i in range(1, self.n_points + 1):
            phi = math.acos(-1 + 2 * (i - 1) / (self.n_points - 1))
            theta = math.sqrt(self.n_points * math.pi) * phi
            
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _get_initial_points(self) -> np.ndarray:
        """Get good initial point configuration using multiple strategies."""
        # Strategy 1: Fibonacci spiral points
        fib_points = self._fibonacci_spiral_points()
        
        # Strategy 2: Random perturbation of Fibonacci points
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (self.n_points, self.dimension))
        perturbed_points = fib_points + noise
        
        # Strategy 3: Normalize to unit sphere
        norms = np.linalg.norm(perturbed_points, axis=1)
        normalized_points = perturbed_points / norms[:, np.newaxis]
        
        # Return flattened array
        return normalized_points.flatten()
    
    def _create_spherical_constraints(self) -> List[dict]:
        """Create spherical constraints for unit sphere boundary."""
        constraints = []
        
        def constraint_func(x):
            points = x.reshape(-1, self.dimension)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0  # Should equal 0 for unit sphere
        
        # Add constraint for each point to lie on unit sphere
        for i in range(self.n_points):
            constraints.append({
                'type': 'eq', 
                'fun': lambda x, i=i: constraint_func(x)[i]
            })
        
        return constraints
    
    def _optimize_with_constraints(self, x0: np.ndarray, maxiter: int = 500) -> Tuple[np.ndarray, float]:
        """Optimize with constraints using L-BFGS-B."""
        constraints = self._create_spherical_constraints()
        bounds = [(-1.5, 1.5)] * len(x0)
        
        options = {'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-8}
        
        try:
            result = minimize(
                self._objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options=options,
                tol=1e-8
            )
            
            if result.success:
                optimized_points = result.x
                ratio = self._distance_ratio(optimized_points.reshape(-1, self.dimension))
                return optimized_points, ratio
            else:
                return x0, self._distance_ratio(x0.reshape(-1, self.dimension))
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            return x0, self._distance_ratio(x0.reshape(-1, self.dimension))
    
    def optimize(self, max_restarts: int = 5, maxiter: int = 500) -> np.ndarray:
        """Main optimization loop with multi-start approach."""
        self.best_ratio = -np.inf
        self.best_points = None
        
        # Multi-start optimization with different initializations
        for restart in range(max_restarts):
            # Set seed for reproducibility
            np.random.seed(42 + restart)
            
            # Get initial points
            x0 = self._get_initial_points()
            
            # Optimize
            try:
                optimized_points, ratio = self._optimize_with_constraints(x0, maxiter)
                
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = optimized_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Restart {restart} failed: {str(e)}")
                continue
        
        # Fallback to initial configuration if no improvement found
        if self.best_points is None:
            self.best_points = self._get_initial_points()
        
        # Convert back to n_points x dimension array
        final_points = self.best_points.reshape(self.n_points, self.dimension)
        
        return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Initialize the optimizer
    optimizer = PointDispersionOptimizer(n_points=14, dimension=3)
    
    # Perform optimization
    points = optimizer.optimize(max_restarts=5, maxiter=500)
    
    return points

# EVOLVE-BLOCK-END
