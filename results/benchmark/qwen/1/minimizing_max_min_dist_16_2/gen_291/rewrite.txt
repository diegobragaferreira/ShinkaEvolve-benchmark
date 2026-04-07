# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import math
from numba import jit

@jit(nopython=True)
def fast_min_max_ratio(distances):
    """Fast calculation of min/max ratio using numba-compiled function"""
    if len(distances) == 0:
        return 0.0
    
    min_dist = distances[0]
    max_dist = distances[0]
    
    for i in range(len(distances)):
        if distances[i] < min_dist:
            min_dist = distances[i]
        if distances[i] > max_dist:
            max_dist = distances[i]
    
    if max_dist <= 0:
        return 0.0
    
    return min_dist / max_dist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def fibonacci_sphere_points(n):
        """Generate n points on a sphere using Fibonacci distribution"""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def stereographic_projection(points_3d):
        """Project 3D points to 2D using stereographic projection from south pole"""
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
    
    def voronoi_relaxation(points, max_iterations=10):
        """Improve point distribution using iterative Voronoi relaxation"""
        current_points = points.copy()
        
        for _ in range(max_iterations):
            try:
                # Compute pairwise distances
                distances = pdist(current_points)
                distance_matrix = squareform(distances)
                
                # For each point, find neighbors and compute average position
                new_points = []
                for i in range(len(current_points)):
                    # Get distances to all other points
                    point_distances = distance_matrix[i]
                    # Set self-distance to infinity
                    point_distances[i] = np.inf
                    
                    # Find indices of nearest neighbors (excluding self)
                    nearest_indices = np.argsort(point_distances)[:min(8, len(point_distances)-1)]
                    
                    # Average positions of neighbors
                    if len(nearest_indices) > 0:
                        neighbor_positions = current_points[nearest_indices]
                        avg_position = np.mean(neighbor_positions, axis=0)
                        new_points.append(avg_position)
                    else:
                        new_points.append(current_points[i])
                
                # Update points
                current_points = np.array(new_points)
                
                # Keep within bounds
                current_points = np.clip(current_points, 0, 1)
                
            except Exception:
                break
        
        return current_points
    
    def objective_function(points_flat):
        """Objective function to minimize (negative ratio)"""
        points = points_flat.reshape(-1, 2)
        distances = pdist(points)
        return -fast_min_max_ratio(distances)
    
    def local_optimization(points, max_iter=200):
        """Apply local optimization using L-BFGS-B"""
        try:
            bounds = [(0, 1) for _ in range(len(points.flatten()))]
            result = minimize(
                objective_function,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                return np.clip(optimized_points, 0, 1)
        except Exception:
            pass
        return points
    
    # Step 1: Initialize using Fibonacci sphere distribution projected to 2D
    sphere_points = fibonacci_sphere_points(16)
    initial_points = stereographic_projection(sphere_points)
    
    # Step 2: Apply Voronoi relaxation for global improvement
    relaxed_points = voronoi_relaxation(initial_points, max_iterations=5)
    
    # Step 3: Local optimization to refine
    optimized_points = local_optimization(relaxed_points, max_iter=150)
    
    # Step 4: Additional refinement with another Voronoi relaxation cycle
    final_points = voronoi_relaxation(optimized_points, max_iterations=3)
    
    # Step 5: Final local optimization pass
    final_points = local_optimization(final_points, max_iter=100)
    
    return final_points

# EVOLVE-BLOCK-END