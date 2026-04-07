# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple
import math
from scipy.spatial import SphericalVoronoi

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        
        # Handle edge cases
        if len(distances) == 0 or np.max(distances) <= 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max

    def spherical_voronoi_initialization(n_points):
        """Initialize points using spherical Voronoi construction on unit sphere."""
        # Generate Fibonacci sphere points (good approximation to uniform distribution on sphere)
        points_sphere = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points_sphere.append([x, y, z])
        
        # Normalize to unit sphere
        points_sphere = np.array(points_sphere)
        norms = np.linalg.norm(points_sphere, axis=1, keepdims=True)
        points_sphere = points_sphere / norms
        
        return points_sphere

    def project_stereographically(points_3d):
        """Project 3D points to 2D using stereographic projection from south pole."""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        
        points_2d = np.array(points_2d)
        
        # Normalize to unit square
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)
        
        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        return points_2d

    def objective_function(points_flat):
        """Objective function to minimize (negative ratio)"""
        points = points_flat.reshape(-1, 2)
        return -calculate_min_max_ratio(points)

    def adaptive_constraint_handling(points):
        """Ensure points maintain reasonable separation and stay within bounds."""
        # Keep points within bounds
        points = np.clip(points, 0, 1)
        
        # Apply minimum distance constraint to prevent clustering
        min_distance = 0.01  # Minimum allowed distance between points
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_distance:
                    # Move points apart slightly
                    direction = points[i] - points[j]
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        points[i] += direction * (min_distance - dist) * 0.5
                        points[j] -= direction * (min_distance - dist) * 0.5
        
        return np.clip(points, 0, 1)

    def global_optimization(initial_points, max_iter=200):
        """Use differential evolution for global optimization."""
        bounds = [(0, 1) for _ in range(len(initial_points.flatten()))]

        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=max_iter,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return initial_points, calculate_min_max_ratio(initial_points)

    def local_refinement(points, max_iter=100):
        """Apply local refinement using L-BFGS-B."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, 2)
            return -calculate_min_max_ratio(points_candidate)
        
        # Add some noise to avoid local minima
        noise = np.random.normal(0, 0.005, points.shape)
        perturbed = points + noise
        perturbed = np.clip(perturbed, 0, 1)
        
        try:
            result = minimize(
                objective, 
                perturbed.flatten(), 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(perturbed.flatten()))],
                options={'maxiter': max_iter},
                tol=1e-6
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return points, calculate_min_max_ratio(points)

    # Main optimization process
    # Step 1: Generate spherical Voronoi-based initialization
    points_3d = spherical_voronoi_initialization(16)
    initial_points = project_stereographically(points_3d)
    
    # Apply constraint handling
    initial_points = adaptive_constraint_handling(initial_points)
    
    # Step 2: Global optimization with DE
    global_points, global_ratio = global_optimization(initial_points, max_iter=150)
    
    # Step 3: Local refinement
    refined_points, refined_ratio = local_refinement(global_points, max_iter=100)
    
    # Step 4: Additional local refinement with different approach
    final_points, final_ratio = local_refinement(refined_points, max_iter=150)
    
    # Return best solution
    if final_ratio > refined_ratio:
        return final_points
    else:
        return refined_points

# EVOLVE-BLOCK-END
