# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    np.random.seed(42)
    
    def fibonacci_spiral_on_sphere(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def objective_function(points_flat):
        """Objective function to maximize min/max distance ratio"""
        points = points_flat.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return -np.inf
        return min_dist / max_dist
    
    def constraint_sphere(points_flat):
        """Keep points on unit sphere"""
        points = points_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should be zero for unit sphere
    
    # Stage 1: Initialize with Fibonacci spiral on sphere
    initial_points = fibonacci_spiral_on_sphere(14)
    
    # Stage 2: Optimize using differential evolution with constraints
    # Flatten points for optimization
    initial_flat = initial_points.flatten()
    
    # Define bounds for coordinates (-1, 1) for each coordinate
    bounds = [(-1, 1) for _ in range(42)]
    
    # Optimization parameters
    maxiter = 100
    popsize = 15
    
    # Run differential evolution with constraints
    result = differential_evolution(
        lambda x: -objective_function(x),  # Minimize negative to maximize
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        seed=42,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False
    )
    
    # Extract optimized points
    final_points = result.x.reshape(-1, 3)
    
    # Ensure points are normalized to unit sphere
    norms = np.linalg.norm(final_points, axis=1, keepdims=True)
    final_points = final_points / norms
    
    return final_points

# EVOLVE-BLOCK-END
