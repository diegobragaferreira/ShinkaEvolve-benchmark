# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return -np.inf
        
        # Return negative ratio (since we want to maximize)
        return -d_min / d_max
    
    def constraint(x):
        # Ensure points stay within [0,1] x [0,1]
        points = x.reshape(-1, 2)
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])
    
    # Multi-start optimization with different initializations
    best_ratio = -np.inf
    best_points = None
    
    # Try multiple random initializations
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        
        # Generate grid-based initial points
        # Create a 4x4 grid pattern and add some randomness
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
        
        # Add small random perturbations
        perturbation_magnitude = 0.05
        noise = np.random.uniform(-perturbation_magnitude, perturbation_magnitude, (16, 2))
        initial_points = np.clip(grid_points + noise, 0, 1)
        
        # Flatten for optimization
        x0 = initial_points.flatten()
        
        # Define bounds (points must stay in [0,1] x [0,1])
        bounds = [(0, 1) for _ in range(32)]
        
        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint}
        
        try:
            # Optimize using SLSQP
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                # Extract points and calculate final ratio
                optimized_points = result.x.reshape(-1, 2)
                distances = pdist(optimized_points)
                
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 0:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
        except Exception:
            continue
            
    # If no successful optimization, return the grid-based points
    if best_points is None:
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        best_points = np.array([[x, y] for x in grid_x for y in grid_y])
        
    return best_points

# EVOLVE-BLOCK-END
