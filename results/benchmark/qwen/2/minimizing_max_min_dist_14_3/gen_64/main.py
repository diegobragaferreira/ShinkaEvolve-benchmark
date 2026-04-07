# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Generate initial points on a sphere using Fibonacci spiral
    def fibonacci_sphere(samples):
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
    
    # Initialize with Fibonacci sphere points
    points = fibonacci_sphere(n)
    
    # Add controlled perturbation to break symmetries
    points += 0.01 * np.random.randn(n, 3)
    
    # Normalize to unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Define objective function for optimization
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape((n, 3))
        
        # Compute distance matrix
        dist_matrix = cdist(points, points, 'euclidean')
        
        # Zero out diagonal
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find min and max distances
        min_dist = np.min(dist_matrix)
        max_dist = np.max(dist_matrix)
        
        # If max distance is zero, return a large penalty
        if max_dist == 0:
            return -np.inf
            
        # Return negative ratio (since we want to maximize min/max ratio)
        return -min_dist / max_dist
    
    # Gradient-free optimization using Nelder-Mead for initial coarse optimization
    x_init = points.flatten()
    
    # First level: coarse optimization with larger step sizes
    result_coarse = minimize(objective, x_init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    # Refine with L-BFGS-B for finer optimization
    refined_points = result_coarse.x.reshape((n, 3))
    
    # Apply spherical constraint: project all points back to unit sphere
    refined_points = refined_points / np.linalg.norm(refined_points, axis=1, keepdims=True)
    
    # Second level: fine-tuning optimization
    def constrained_objective(x_flat):
        points = x_flat.reshape((n, 3))
        
        # Project back to sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Compute distance matrix
        dist_matrix = cdist(points, points, 'euclidean')
        
        # Zero out diagonal
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find min and max distances
        min_dist = np.min(dist_matrix)
        max_dist = np.max(dist_matrix)
        
        # If max distance is zero, return a large penalty
        if max_dist == 0:
            return -np.inf
            
        # Return negative ratio (since we want to maximize min/max ratio)
        return -min_dist / max_dist
    
    # Final optimization with improved constraints
    result_fine = minimize(constrained_objective, refined_points.flatten(), 
                          method='L-BFGS-B', 
                          options={'maxiter': 2000, 'gtol': 1e-8})
    
    # Extract final points and ensure they're on unit sphere
    final_points = result_fine.x.reshape((n, 3))
    final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
    
    return final_points

# EVOLVE-BLOCK-END