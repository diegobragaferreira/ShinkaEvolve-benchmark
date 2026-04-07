# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing)
        if d_max == 0:
            return -1.0  # Avoid division by zero
        return -d_min / d_max
    
    def constraint_bounds(x_flat):
        # Ensure all points are within [0,1] x [0,1]
        points = x_flat.reshape(-1, 2)
        # Return constraints: lower bound (-points) and upper bound (points - 1)
        lower_bound = -points.flatten()
        upper_bound = points.flatten() - 1
        return np.concatenate([lower_bound, upper_bound])
    
    # Create initial configuration using a hexagonal-like arrangement
    # This provides a good starting point that's already somewhat dispersed
    np.random.seed(42)
    
    # Generate points in a hexagonal pattern with some randomness
    points = []
    rows = 4
    cols = 4
    
    # Create a grid with slight perturbations
    for i in range(rows):
        for j in range(cols):
            # Add some jitter to create a more spread-out configuration
            x = (j + 0.5 * (i % 2)) / cols
            y = i / (rows - 1)
            
            # Add small random perturbation to avoid perfect grid
            x += (np.random.rand() - 0.5) * 0.1
            y += (np.random.rand() - 0.5) * 0.1
            
            # Ensure points stay within boundaries
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            points.append([x, y])
    
    points = np.array(points[:16])  # Ensure exactly 16 points
    
    # Flatten the points for optimization
    x0 = points.flatten()
    
    # Define bounds for each coordinate [0, 1]
    bounds = [(0, 1) for _ in range(32)]
    
    # First, try optimization with L-BFGS-B for fine-tuning
    try:
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        # If successful, check if we can further improve with differential evolution
        if result.success:
            # Run differential evolution for global optimization
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            # Compare results and return the better solution
            if -objective(result.x) < -objective(de_result.x):
                final_points = de_result.x.reshape(-1, 2)
            else:
                final_points = result.x.reshape(-1, 2)
        else:
            final_points = x0.reshape(-1, 2)
            
    except Exception as e:
        # Fallback to simple optimization if complex one fails
        try:
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            final_points = result.x.reshape(-1, 2)
        except:
            # Final fallback - return the initial configuration
            final_points = points
    
    return final_points

# EVOLVE-BLOCK-END