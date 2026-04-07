# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution

class PointDispersionOptimizer:
    def __init__(self, n_points=16, dimension=2):
        self.n_points = n_points
        self.dimension = dimension
        self.total_vars = n_points * dimension
        
    def initialize_points(self) -> np.ndarray:
        """Initialize points using improved hexagonal arrangement with better spacing."""
        np.random.seed(42)
        
        # Create a hexagonal-like grid with proper spacing and controlled perturbations
        points = []
        rows = 4
        cols = 4
        
        for i in range(rows):
            for j in range(cols):
                # Hexagonal grid with proper spacing
                x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                
                # Add more substantial but controlled random perturbation
                x += (np.random.rand() - 0.5) * 0.15
                y += (np.random.rand() - 0.5) * 0.15
                
                # Ensure points stay within boundaries with epsilon padding
                x = np.clip(x, 0.02, 0.98)
                y = np.clip(y, 0.02, 0.98)
                
                points.append([x, y])
        
        return np.array(points[:self.n_points])
    
    def calculate_objective(self, x_flat: np.ndarray) -> float:
        """Calculate the negative min/max distance ratio using squareform for stability."""
        points = x_flat.reshape(-1, self.dimension)
        
        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing)
        # Handle edge case to avoid division by zero or invalid distances
        if d_max <= 0:
            return -1.0
        return -d_min / d_max
    
    def bounds_constraint(self, x_flat: np.ndarray) -> np.ndarray:
        """Constraint function ensuring all points stay within [0,1] bounds."""
        points = x_flat.reshape(-1, self.dimension)
        # Lower bounds (negative values for inequality constraints)
        lower = -points.flatten()
        # Upper bounds (values that should be <= 0 for inequality constraints)
        upper = points.flatten() - 1.0
        return np.concatenate([lower, upper])
    
    def adaptive_minimize(self, obj_func, x0, bounds, maxiter, ftol, gtol):
        """Minimize with adaptive stopping criteria."""
        previous_obj_val = float('inf')
        consecutive_no_improvement = 0
        max_no_improvement = 20

        # Use a callback to track progress
        def callback(xk):
            nonlocal previous_obj_val, consecutive_no_improvement
            obj_val = obj_func(xk)
            if abs(previous_obj_val - obj_val) < 1e-12:
                consecutive_no_improvement += 1
            else:
                consecutive_no_improvement = 0
            previous_obj_val = obj_val

        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
                callback=callback if consecutive_no_improvement < max_no_improvement else None
            )
            return result
        except Exception:
            return None
    
    def optimize_with_lbfgs_adaptive(self, x0: np.ndarray) -> np.ndarray:
        """Optimize using adaptive L-BFGS-B method."""
        bounds = [(1e-6, 1-1e-6) for _ in range(self.total_vars)]
        
        try:
            result = self.adaptive_minimize(
                self.calculate_objective,
                x0,
                bounds,
                maxiter=1000,
                ftol=1e-10,
                gtol=1e-10
            )
            return result.x if result and result.success else x0
        except Exception:
            return x0
    
    def optimize_with_de(self, x0: np.ndarray) -> np.ndarray:
        """Optimize using Differential Evolution method."""
        bounds = [(1e-6, 1-1e-6) for _ in range(self.total_vars)]
        
        try:
            result = differential_evolution(
                self.calculate_objective,
                bounds,
                maxiter=150,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            return result.x
        except Exception:
            return x0
    
    def validate_solution(self, points_flat: np.ndarray) -> np.ndarray:
        """Ensure the solution respects constraints and is valid."""
        points = points_flat.reshape(-1, self.dimension)
        
        # Clamp points to [1e-6, 1-1e-6] range
        points = np.clip(points, 1e-6, 1 - 1e-6)
        
        return points.flatten()
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine with enhanced phase management."""
        # Phase 1: Initialize points with improved hexagonal arrangement
        initial_points = self.initialize_points()
        x0 = initial_points.flatten()
        
        # Phase 2: Adaptive L-BFGS-B optimization with early stopping
        x_lbfgs = self.optimize_with_lbfgs_adaptive(x0)
        
        # Phase 3: Differential Evolution for global search
        x_de = self.optimize_with_de(x0)
        
        # Phase 4: Compare and select the best solution
        obj_lbfgs = self.calculate_objective(x_lbfgs)
        obj_de = self.calculate_objective(x_de)
        
        # Choose better solution (smaller negative value means better ratio)
        if obj_lbfgs < obj_de:
            final_solution = x_lbfgs
        else:
            final_solution = x_de
        
        # Phase 5: Final validation and cleanup
        validated_solution = self.validate_solution(final_solution)
        
        # Convert to final point array format
        final_points = validated_solution.reshape(-1, self.dimension)
        
        return final_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointDispersionOptimizer(n_points=16, dimension=2)
    return optimizer.optimize()

# EVOLVE-BLOCK-END