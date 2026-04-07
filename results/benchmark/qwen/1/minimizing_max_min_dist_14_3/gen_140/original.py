# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def energy_potential(points):
        """Calculate total potential energy between points (inverse distance)"""
        # Compute distance matrix
        dist_matrix = cdist(points, points)
        # Avoid division by zero
        np.fill_diagonal(dist_matrix, 1e-10)
        # Energy is sum of inverse distances
        return np.sum(1.0 / dist_matrix)
    
    def objective_energy(x):
        """Objective function that minimizes potential energy"""
        points = x.reshape(-1, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)
        return energy_potential(points)
    
    def objective_ratio(x):
        """Objective function that maximizes min/max distance ratio"""
        points = x.reshape(-1, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)
        
        # Compute distance matrix
        dist_matrix = cdist(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_dist = np.min(dist_matrix)
        max_dist = np.max(dist_matrix)
        
        if max_dist == 0:
            return -1.0
            
        # We want to maximize min_dist / max_dist, so minimize -min_dist / max_dist
        return -min_dist / max_dist
    
    # Start with an initial configuration using Fibonacci spiral on sphere
    n = 14
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    
    points = []
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points.append([x, y, z])
    
    initial_points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
    initial_points = initial_points / np.where(norms > 0, norms, 1)
    
    # Multi-start optimization with different initializations
    best_points = initial_points.copy()
    best_ratio = -np.inf
    
    # Try multiple random perturbations of initial points
    for restart in range(3):
        np.random.seed(42 + restart)
        
        # Create perturbed version of initial points
        perturbed = initial_points + np.random.normal(0, 0.05, initial_points.shape)
        
        # Normalize again
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.where(norms > 0, norms, 1)
        
        # First, optimize using energy minimization to get a good starting configuration
        x0 = perturbed.flatten()
        
        try:
            # Use L-BFGS-B for energy minimization first
            result_energy = minimize(
                objective_energy,
                x0,
                method='L-BFGS-B',
                options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 300}
            )
            
            if result_energy.success:
                points_energy = result_energy.x.reshape(-1, 3)
                # Re-normalize
                norms = np.linalg.norm(points_energy, axis=1, keepdims=True)
                points_energy = points_energy / np.where(norms > 0, norms, 1)
            else:
                points_energy = perturbed
        except Exception:
            points_energy = perturbed
        
        # Then optimize using the actual ratio objective with SLSQP
        try:
            # Use the energy-optimized points as the starting point
            x0_final = points_energy.flatten()
            
            # Define constraint: points must be on unit sphere
            def constraint_sphere(x):
                points = x.reshape(-1, 3)
                norms = np.linalg.norm(points, axis=1)
                # Return values >= 0 for feasible points (norm = 1)
                return norms - 1.0
            
            cons = {'type': 'eq', 'fun': constraint_sphere}
            
            result = minimize(
                objective_ratio,
                x0_final,
                method='SLSQP',
                constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 500}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                # Final normalization to ensure unit sphere
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                optimized_points = optimized_points / np.where(norms > 0, norms, 1)
                
                # Calculate the ratio achieved
                dist_matrix = cdist(optimized_points, optimized_points)
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)
                max_dist = np.max(dist_matrix)
                ratio = min_dist / max_dist
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
            else:
                # Even if optimization fails, keep the better of the two
                dist_matrix = cdist(points_energy, points_energy)
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)
                max_dist = np.max(dist_matrix)
                ratio = min_dist / max_dist
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points_energy.copy()
                    
        except Exception as e:
            # Fallback to energy-optimized points if SLSQP fails
            dist_matrix = cdist(points_energy, points_energy)
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            max_dist = np.max(dist_matrix)
            ratio = min_dist / max_dist
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points_energy.copy()
    
    return best_points

# EVOLVE-BLOCK-END