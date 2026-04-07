# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time

class PointOptimizer:
    """Structured optimizer for maximizing min/max distance ratio of 16 points in 2D."""
    
    def __init__(self, n_points=16, max_time=180):
        self.n_points = n_points
        self.max_time = max_time
        self.best_solution = None
        self.best_ratio = -np.inf
        
    def compute_distance_ratio(self, points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Use squareform for numerical stability
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0 or np.isinf(min_dist):
            return 0.0
            
        return min_dist / max_dist
    
    def initialize_points(self):
        """Create structured initialization for better starting configuration."""
        np.random.seed(42)
        
        # Create a 4x4 grid pattern as base
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        
        # Create grid points
        X, Y = np.meshgrid(grid_x, grid_y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Add small random perturbations to break symmetry
        noise_magnitude = 0.01
        noise = np.random.normal(0, noise_magnitude, points.shape)
        points += noise
        
        # Clip to ensure bounds
        points = np.clip(points, 0, 1)
        
        # Ensure exactly n_points
        if len(points) > self.n_points:
            points = points[:self.n_points]
        elif len(points) < self.n_points:
            additional = np.random.rand(self.n_points - len(points), 2)
            points = np.vstack([points, additional])
            
        return points
    
    def objective_function(self, x):
        """Objective function for optimization (minimize negative ratio)."""
        points = x.reshape(-1, 2)
        ratio = self.compute_distance_ratio(points)
        return -ratio
    
    def global_optimization_step(self, x0):
        """Perform global optimization using differential evolution."""
        bounds = [(0, 1)] * (self.n_points * 2)
        
        try:
            result = differential_evolution(
                self.objective_function,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                atol=1e-12,
                rtol=1e-12,
                mutation=(0.7, 1.0),
                recombination=0.7
            )
            return result.success, result.x if result.success else x0
        except Exception:
            return False, x0
    
    def local_refinement_step(self, x0, method='L-BFGS-B'):
        """Perform local refinement optimization."""
        bounds = [(0, 1)] * (self.n_points * 2)
        
        try:
            result = minimize(
                self.objective_function,
                x0,
                method=method,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            return result.success, result.x if result.success else x0
        except Exception:
            return False, x0
    
    def validate_and_update_best(self, points):
        """Validate solution and update best if better."""
        ratio = self.compute_distance_ratio(points)
        if ratio > self.best_ratio:
            self.best_ratio = ratio
            self.best_solution = points.copy()
    
    def optimize(self):
        """Main optimization loop with multiple strategies."""
        # Strategy 1: Structured initialization + global + local refinement
        try:
            # Initialize with structured approach
            initial_points = self.initialize_points()
            x0 = initial_points.flatten()
            
            # Global optimization
            global_success, x_global = self.global_optimization_step(x0)
            
            # Local refinement
            if global_success:
                local_success, x_local = self.local_refinement_step(x_global, 'L-BFGS-B')
                if local_success:
                    final_points = x_local.reshape(-1, 2)
                    self.validate_and_update_best(final_points)
            
            # Additional refinement with SLSQP
            if global_success and self.best_solution is None:
                slsqp_success, x_slsqp = self.local_refinement_step(x_global, 'SLSQP')
                if slsqp_success:
                    final_points = x_slsqp.reshape(-1, 2)
                    self.validate_and_update_best(final_points)
                    
        except Exception:
            pass
        
        # Strategy 2: Direct optimization from random initialization
        if self.best_solution is None:
            try:
                # Random initialization
                np.random.seed(42)
                x0_random = np.random.uniform(0, 1, self.n_points * 2)
                
                # Global optimization
                global_success, x_global = self.global_optimization_step(x0_random)
                
                # Local refinement
                if global_success:
                    local_success, x_local = self.local_refinement_step(x_global)
                    if local_success:
                        final_points = x_local.reshape(-1, 2)
                        self.validate_and_update_best(final_points)
                        
            except Exception:
                pass
        
        # Fallback to initialization if nothing worked
        if self.best_solution is None:
            self.best_solution = self.initialize_points()
            
        return self.best_solution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointOptimizer(n_points=16, max_time=180)
    return optimizer.optimize()

# EVOLVE-BLOCK-END