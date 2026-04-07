# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x):
        # Reshape into points
        points = x.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return -1e10
            
        # Return negative ratio to minimize (we want to maximize ratio)
        return -(d_min / d_max)
    
    def constraint_func(x):
        # Ensure points are within [0,1] bounds
        points = x.reshape(-1, 2)
        return np.concatenate([points.flatten(), (1-points.flatten())])
    
    # Set up bounds (0 to 1 for each coordinate)
    bounds = [(0, 1)] * 32
    
    # Initial guess - try multiple random starting points
    best_result = None
    best_ratio = -np.inf
    
    # Try several random initializations
    for _ in range(5):
        # Start with random points
        x0 = np.random.uniform(0, 1, 32)
        
        # Use global optimization first
        try:
            result = differential_evolution(
                objective, 
                bounds, 
                seed=42,
                maxiter=200,
                popsize=15,
                atol=1e-6,
                rtol=1e-6
            )
            
            if result.success:
                # Refine with local optimization
                refined = minimize(
                    objective,
                    result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                
                if refined.success:
                    final_points = refined.x.reshape(-1, 2)
                    distances = pdist(final_points)
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_result = refined.x.copy()
                            
        except Exception:
            continue
    
    # If we found a good solution, use it
    if best_result is not None:
        points = best_result.reshape(-1, 2)
    else:
        # Fallback to simple approach if optimization fails
        np.random.seed(42)
        points = np.random.rand(16, 2)
    
    # Ensure points are within valid bounds
    points = np.clip(points, 0, 1)
    
    return points

# EVOLVE-BLOCK-END
