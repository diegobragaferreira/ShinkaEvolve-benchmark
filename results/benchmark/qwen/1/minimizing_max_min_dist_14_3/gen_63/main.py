# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sobol_seq import i4_sobol_generate

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.
    """
    
    n = 14
    
    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        
        # Calculate pairwise distances
        distances = squareform(pdist(points))
        
        # Set diagonal to large value to avoid self-distance issues
        np.fill_diagonal(distances, np.inf)
        
        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Add penalty for distance imbalance
        if max_dist > 0:
            ratio = min_dist / max_dist
            # Penalty term to discourage extreme distance variations
            imbalance_penalty = 0.1 * (max_dist - min_dist) / max_dist
            return -(ratio - imbalance_penalty)
        else:
            return 0
    
    def constraint_func(x):
        # Ensure points are on unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Generate initial points using Sobol sequence for better distribution
    np.random.seed(42)
    sobol_points = i4_sobol_generate(3, n)
    
    # Scale and normalize Sobol points to unit sphere
    points = sobol_points * 2 - 1  # Scale to [-1,1]^3
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    points = points / norms
    
    # Normalize first point to ensure consistent scaling
    if np.linalg.norm(points[0]) > 0:
        points = points / np.linalg.norm(points[0])
    
    initial_points = points.copy()
    
    # Multiple restarts to find better solution
    best_ratio = -np.inf
    best_points = initial_points.copy()
    
    # Try multiple random perturbations of the initial solution
    for restart in range(15):  # Increased from 10 to 15 restarts
        # Add small noise to break symmetry
        np.random.seed(restart)
        noisy_points = initial_points + np.random.normal(0, 0.01, initial_points.shape)
        
        # Normalize again
        norms = np.linalg.norm(noisy_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        noisy_points = noisy_points / norms
        
        # Flatten for optimization
        x0 = noisy_points.flatten()
        
        # Define constraints
        cons = {'type': 'eq', 'fun': constraint_func}
        
        # Adaptive constraint tightening approach
        # Start with looser bounds and tighten over iterations
        try:
            # First optimization using SLSQP
            result = minimize(objective, x0, method='SLSQP', constraints=cons, 
                            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 500})
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                
                # Secondary refinement with L-BFGS-B
                # Create refined version using L-BFGS-B 
                refined_result = minimize(objective, optimized_points.flatten(), 
                                       method='L-BFGS-B', constraints=cons,
                                       options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 300})
                
                if refined_result.success:
                    final_points = refined_result.x.reshape(-1, 3)
                else:
                    final_points = optimized_points
                
                # Calculate final ratio
                distances = squareform(pdist(final_points))
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
        except Exception:
            continue
    
    return best_points

# EVOLVE-BLOCK-END