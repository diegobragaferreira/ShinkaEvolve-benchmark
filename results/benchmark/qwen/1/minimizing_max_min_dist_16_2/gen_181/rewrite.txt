# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time
from scipy.spatial import SphericalVoronoi
import math

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

    def generate_spherical_voronoi_initialization(n_points=16):
        """Generate initial points using spherical Voronoi construction on unit sphere."""
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

    def constrain_to_bounds(points, epsilon=1e-8):
        """Constrain points to stay within valid bounds."""
        return np.clip(points, epsilon, 1-epsilon)

    def local_optimization_step(points, max_iter=200):
        """Apply local optimization to improve point distribution."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, 2)
            points_candidate = constrain_to_bounds(points_candidate)
            return -calculate_min_max_ratio(points_candidate)
        
        # Add small noise to avoid local minima
        noise = np.random.normal(0, 0.001, points.shape)
        perturbed = points + noise
        perturbed = constrain_to_bounds(perturbed)
        
        try:
            result = minimize(
                objective, 
                perturbed.flatten(), 
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(perturbed.flatten()))],
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = constrain_to_bounds(optimized_points)
                ratio = calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return points, calculate_min_max_ratio(points)

    def adaptive_refinement(initial_points, max_iterations=5):
        """Apply progressive refinement with decreasing tolerance."""
        current_points = initial_points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        # Multiple refinement stages with increasing precision
        stages = [
            {'max_iter': 150, 'tol': 1e-6},
            {'max_iter': 200, 'tol': 1e-8}, 
            {'max_iter': 250, 'tol': 1e-10}
        ]
        
        for stage in stages:
            if max_iterations <= 0:
                break
                
            try:
                refined_points, refined_ratio = local_optimization_step(
                    current_points, 
                    max_iter=stage['max_iter']
                )
                
                if refined_ratio > current_ratio:
                    current_points = refined_points
                    current_ratio = refined_ratio
                    max_iterations -= 1
                else:
                    # Try one more aggressive optimization
                    aggressive_points, aggressive_ratio = local_optimization_step(
                        current_points,
                        max_iter=stage['max_iter'] * 2
                    )
                    if aggressive_ratio > current_ratio:
                        current_points = aggressive_points
                        current_ratio = aggressive_ratio
                        max_iterations -= 1
                        
            except Exception:
                continue
                
        return current_points, current_ratio

    # Step 1: Generate spherical Voronoi-based initialization
    points_3d = generate_spherical_voronoi_initialization(16)
    initial_points = project_stereographically(points_3d)
    
    # Step 2: Constrain to valid bounds
    initial_points = constrain_to_bounds(initial_points)
    
    # Step 3: Apply adaptive refinement with progressive precision
    refined_points, refined_ratio = adaptive_refinement(initial_points, max_iterations=3)
    
    # Step 4: Final optimization pass with highest precision
    final_points, final_ratio = local_optimization_step(refined_points, max_iter=300)
    
    # Return best configuration found
    if final_ratio > refined_ratio:
        return final_points
    else:
        return refined_points

# EVOLVE-BLOCK-END