# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    n = 14
    d = 3
    
    # Generate initial points using Fibonacci spiral on sphere for good distribution
    def fibonacci_sphere(samples=14):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    # Initialize with fibonacci points on sphere
    initial_points = fibonacci_sphere(n)
    
    # Normalize to unit cube [0,1]^3
    # First center around origin and scale appropriately
    initial_points = initial_points - np.mean(initial_points, axis=0)
    max_coord = np.max(np.abs(initial_points))
    if max_coord > 0:
        initial_points = initial_points / max_coord * 0.5
    # Then shift to [0,1]^3
    initial_points = initial_points + 0.5
    
    # Flatten for optimization
    initial_flat = initial_points.flatten()
    
    def objective(x_flat):
        # Reshape back to points
        points = x_flat.reshape((n, d))
        
        # Calculate pairwise distances
        distances = pdist(points)
        distances_squared = distances ** 2
        
        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return -np.inf
            
        # Minimize negative of ratio (since we want to maximize ratio)
        ratio = d_min / d_max
        
        # Return negative because we're minimizing
        return -ratio
    
    # Optimization bounds: [0,1] for all coordinates
    bounds = [(0, 1) for _ in range(n * d)]
    
    # Perform optimization
    result = minimize(
        objective,
        initial_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
    )
    
    # Extract optimized points
    optimized_points = result.x.reshape((n, d))
    
    # Ensure all points are within [0,1]^3
    optimized_points = np.clip(optimized_points, 0, 1)
    
    # Final refinement: ensure we don't have any zero distances due to clipping
    final_distances = pdist(optimized_points)
    if len(final_distances) > 0:
        min_distance = np.min(final_distances)
        if min_distance < 1e-12:
            # If we have degenerate points, fall back to initial points
            return initial_points
    
    return optimized_points

# EVOLVE-BLOCK-END
