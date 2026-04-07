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
        """Initialize points using an enhanced hexagonal arrangement with strategic perturbations."""
        np.random.seed(42)
        
        # Create a more sophisticated hexagonal grid pattern
        points = []
        rows = 4
        cols = 4
        
        # Generate hexagonal grid points with alternating row offsets
        for i in range(rows):
            for j in range(cols):
                # Hexagonal grid with proper spacing
                x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                
                # Add more substantial but controlled random perturbation
                x += (np.random.rand() - 0.5) * 0.15
                y += (np.random.rand() - 0.5) * 0.15
                
                # Ensure points stay within safe boundaries with padding
                x = np.clip(x, 0.02, 0.98)
                y = np.clip(y, 0.02, 0.98)
                
                points.append([x, y])
        
        return np.array(points[:self.n_points])
    
    def calculate_objective(self, x_flat: np.ndarray) -> float:
        """Calculate the negative min/max distance ratio with robust error handling."""
        points = x_flat.reshape(-1, self.dimension)
        
        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)
        
        # Handle edge case where there are no distances (shouldn't happen with 16 points)
        if len(distances[distances != np.inf]) == 0:
            return -1.0
            
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
        
        # Create constraint vectors: lower bounds (negative for inequality) and upper bounds
        lower_bounds = -points.flatten()  # For points to be >= 0  
        upper_bounds = points.flatten() - 1.0  # For points to be <= 1
        
        return np.concatenate([lower_bounds, upper_bounds])
    
    def optimize_with_lbfgs(self, x0: np.ndarray) -> np.ndarray:
        """Optimize using L-BFGS-B method with careful error handling."""
        bounds = [(1e-6, 1-1e-6) for _ in range(self.total_vars)]
        cons = {'type': 'ineq', 'fun': self.bounds_constraint}
        
        try:
            result = minimize(
                self.calculate_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            return result.x if result.success else x0
        except Exception as e:
            # If L-BFGS fails, fall back to initial points
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
    
    def validate_and_finalize(self, points_flat: np.ndarray) -> np.ndarray:
        """Ensure the solution respects constraints and is numerically stable."""
        points = points_flat.reshape(-1, self.dimension)
        
        # Clamp points to [0,1] range with small epsilon to avoid boundary issues
        points = np.clip(points, 1e-6, 1 - 1e-6)
        
        return points.flatten()
    
    def optimize(self) -> np.ndarray:
        """Main optimization routine with enhanced phase management."""
        # Phase 1: Initialize points with improved hexagonal arrangement
        initial_points = self.initialize_points()
        x0 = initial_points.flatten()
        
        # Phase 2: Multi-stage optimization approach
        # Stage 1: Local optimization with L-BFGS-B
        x_lbfgs = self.optimize_with_lbfgs(x0)
        
        # Stage 2: Global optimization with Differential Evolution
        x_de = self.optimize_with_de(x0)
        
        # Stage 3: Compare and select the best solution
        obj_lbfgs = self.calculate_objective(x_lbfgs)
        obj_de = self.calculate_objective(x_de)
        
        # Choose better solution (smaller negative value means better ratio)
        if obj_lbfgs < obj_de:
            final_solution = x_lbfgs
        else:
            final_solution = x_de
        
        # Phase 3: Final validation and cleanup
        validated_solution = self.validate_and_finalize(final_solution)
        
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
