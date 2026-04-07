# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def fibonacci_spiral_sphere(n_points):
    """Generate points on a sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance."""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    return np.min(distances) / np.max(distances)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Start with Fibonacci spiral initialization
    points = fibonacci_spiral_sphere(n)
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # Define objective function for optimization (negative because we want to maximize)
    def objective(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
        normalized_points = points_reshaped / norms
        return -min_max_dist_ratio(normalized_points)
    
    # Define constraint function for unit sphere
    def constraint_sphere(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        norms = np.linalg.norm(points_reshaped, axis=1)
        return norms - 1.0
    
    # Set up constraints and bounds
    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(n * d)]
    
    # Use L-BFGS-B for faster convergence
    result = minimize(
        objective, 
        x0, 
        method='L-BFGS-B', 
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12},
        tol=1e-12
    )
    
    # Extract optimized points
    optimized_points = result.x.reshape(n, d)
    
    # Normalize to ensure they're on unit sphere
    normalized_points = optimized_points / np.linalg.norm(optimized_points, axis=1, keepdims=True)
    
    return normalized_points

# EVOLVE-BLOCK-END