# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from math import sqrt

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        distances = cdist(points, points)
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        dmin = np.min(distances)
        dmax = np.max(distances)
        return dmin / dmax if dmax > 0 else 0
    
    def objective_function(points_flat):
        """Objective function to minimize (negative ratio)"""
        points = points_flat.reshape(-1, 3)
        return -compute_min_max_ratio(points)
    
    def enforce_constraints(points):
        """Ensure points are within reasonable bounds and maintain minimum separation"""
        # Normalize to unit sphere to keep points bounded
        points = points.copy()
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        points = points / norms * 0.9  # Scale down slightly to avoid boundary issues
        
        return points
    
    def local_refinement(points, max_iter=100):
        """Perform local optimization using scipy minimize"""
        # Flatten points for optimization
        points_flat = points.flatten()
        
        # Define bounds for each coordinate (-1, 1)
        bounds = [(-1, 1) for _ in range(len(points_flat))]
        
        # Optimize using L-BFGS-B which handles bounds well
        result = minimize(
            objective_function,
            points_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        
        refined_points = result.x.reshape(-1, 3)
        return enforce_constraints(refined_points)
    
    def generate_initial_config():
        """Generate initial configuration based on icosahedron plus center"""
        # Start with vertices of regular icosahedron scaled to unit sphere
        phi = (1 + sqrt(5)) / 2  # golden ratio
        # Icosahedron vertices
        vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        vertices = np.array(vertices)
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add additional points near faces and edges for better distribution
        # Simple approach: add points along axes and at face centers
        additional_points = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0],
            [0, 0.5, 0.5], [0, 0.5, -0.5], [0, -0.5, 0.5], [0, -0.5, -0.5]
        ]
        
        # Combine and normalize
        points = np.vstack([vertices[:12], additional_points])
        # Normalize all points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1
        points = points / norms * 0.9
        
        return points
    
    # Generate initial configuration
    best_points = generate_initial_config()
    
    # Apply local refinement multiple times to get better local optimum
    best_ratio = compute_min_max_ratio(best_points)
    
    # Run several rounds of local optimization with different random perturbations
    for round_num in range(3):
        # Slightly perturb the configuration
        perturbation = np.random.normal(0, 0.01, best_points.shape)
        perturbed_points = best_points + perturbation
        
        # Normalize back to unit sphere
        norms = np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        norms[norms == 0] = 1
        perturbed_points = perturbed_points / norms * 0.9
        
        # Local refinement
        refined_points = local_refinement(perturbed_points)
        
        # Evaluate new configuration
        new_ratio = compute_min_max_ratio(refined_points)
        
        if new_ratio > best_ratio:
            best_points = refined_points
            best_ratio = new_ratio
    
    # Final optimization step
    final_points = local_refinement(best_points)
    
    return final_points

# EVOLVE-BLOCK-END
