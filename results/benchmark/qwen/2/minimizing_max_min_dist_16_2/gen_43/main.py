# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_ratio(points):
        """Compute the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def objective(x):
        """Objective function to maximize the min/max distance ratio."""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return -np.inf
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return -np.inf
        return -d_min / d_max

    def create_initial_grid():
        """Create a structured 4x4 grid with small perturbations."""
        # Create a 4x4 grid in [0.1, 0.9] range
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
        
        # Add structured perturbations to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.03, (16, 2))
        initial_points = np.clip(grid_points + noise, 0, 1)
        return initial_points

    def optimize_with_refinement(x0):
        """Perform sequential optimization with refinement stages."""
        # Stage 1: Fast optimization with L-BFGS-B
        bounds = [(0, 1) for _ in range(32)]
        result1 = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if not result1.success:
            return None
            
        # Stage 2: Precise optimization with SLSQP
        result2 = minimize(
            objective,
            result1.x,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result2.success:
            return result2.x
        return None

    # Main optimization routine with single initialization strategy
    best_ratio = -np.inf
    best_points = None
    
    # Single high-quality initialization using structured grid
    initial_points = create_initial_grid()
    
    # Optimize using our refined two-stage approach
    optimized_x = optimize_with_refinement(initial_points.flatten())
    
    if optimized_x is not None:
        optimized_points = optimized_x.reshape(-1, 2)
        final_ratio = compute_ratio(optimized_points)
        
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = optimized_points.copy()
    
    # Fallback to initial configuration if nothing worked
    if best_points is None:
        best_points = initial_points
        
    return best_points

# EVOLVE-BLOCK-END