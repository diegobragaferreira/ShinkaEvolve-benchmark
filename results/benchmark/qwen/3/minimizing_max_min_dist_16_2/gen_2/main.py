# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    np.random.seed(42)
    
    # Create initial configuration using hexagonal grid pattern
    def create_hexagonal_grid(n_points):
        # For 16 points, we can arrange them in a 4x4 grid with alternating rows offset
        rows = 4
        cols = 4
        points = []
        
        for i in range(rows):
            for j in range(cols):
                x = j + (i % 2) * 0.5  # Offset every other row
                y = i * np.sqrt(3)/2
                points.append([x, y])
        
        # Normalize to unit square [0,1] x [0,1]
        points = np.array(points)
        # Scale to fit within [0,1] x [0,1]
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Ensure they fit in [0,1] x [0,1]
        points[:, 0] = points[:, 0] * 0.9 + 0.05  # Add padding
        points[:, 1] = points[:, 1] * 0.9 + 0.05
        
        return points
    
    # Initialize points
    points = create_hexagonal_grid(16)
    
    # Objective function to maximize ratio of min/max distances
    def objective(params):
        # Reshape params back to points
        points_current = params.reshape(-1, 2)
        
        # Calculate pairwise distances
        distances = pdist(points_current)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio since we want to maximize it
        if max_dist == 0:
            return -1e9  # Avoid division by zero
        return -min_dist / max_dist
    
    # Constraints to keep points within unit square
    def constraint_bounds(x):
        points_current = x.reshape(-1, 2)
        # Ensure all points are within [0,1] x [0,1]
        return np.concatenate([
            points_current[:, 0],  # x coordinates >= 0
            1 - points_current[:, 0],  # x coordinates <= 1
            points_current[:, 1],  # y coordinates >= 0
            1 - points_current[:, 1]   # y coordinates <= 1
        ])
    
    # Define bounds for each coordinate
    bounds = [(0, 1)] * 32  # 16 points * 2 coordinates each
    
    # Optimize using L-BFGS-B method
    result = minimize(
        objective,
        points.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9},
        callback=None
    )
    
    optimized_points = result.x.reshape(-1, 2)
    
    # Final normalization to ensure points are in [0,1] x [0,1]
    # Make sure points stay within bounds
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

# EVOLVE-BLOCK-END
