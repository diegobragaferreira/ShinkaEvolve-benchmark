# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import Voronoi
import time
from typing import Tuple
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def fibonacci_sphere_sampling(n_points: int) -> np.ndarray:
        """Generate points on a sphere using Fibonacci sampling for even distribution."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def stereographic_projection(points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using stereographic projection from south pole."""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        
        return np.array(points_2d)
    
    def normalize_to_unit_square(points_2d: np.ndarray) -> np.ndarray:
        """Normalize 2D points to fit within [0,1] x [0,1]."""
        if len(points_2d) == 0:
            return points_2d
            
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)
        
        # Avoid division by zero
        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        return points_2d
    
    def calculate_min_max_ratio(points: np.ndarray) -> float:
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
    
    def voronoi_repulsion_force(points: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Calculate repulsion force based on Voronoi diagram."""
        if len(points) < 2:
            return np.zeros_like(points)
            
        try:
            vor = Voronoi(points)
            forces = np.zeros_like(points)
            
            for i in range(len(points)):
                region = vor.regions[vor.point_region[i]]
                if not region or -1 in region:
                    continue
                
                # Get Voronoi cell vertices
                vertices = [vor.vertices[j] for j in region if j >= 0]
                if len(vertices) < 3:
                    continue
                    
                vertices = np.array(vertices)
                
                # Calculate centroid of Voronoi cell
                centroid = np.mean(vertices, axis=0)
                
                # Repulsion force pointing away from centroid
                force = points[i] - centroid
                if np.linalg.norm(force) > 0:
                    forces[i] = force / np.linalg.norm(force) * alpha
                
            return forces
        except:
            return np.zeros_like(points)
    
    def spherical_optimization_step(points: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """Apply spherical optimization step maintaining radial constraints."""
        # Project to sphere surface
        points_sphere = fibonacci_sphere_sampling(len(points))
        
        # Calculate forces from current configuration
        current_distances = cdist(points, points)
        np.fill_diagonal(current_distances, np.inf)
        
        # Calculate attractive forces (based on distance)
        attractive_forces = np.zeros_like(points)
        for i in range(len(points)):
            # Find nearest neighbors (excluding self)
            distances = current_distances[i]
            nearest_idx = np.argmin(distances)
            
            # Attractive force towards nearest neighbor
            if nearest_idx != i and distances[nearest_idx] > 0:
                direction = points[nearest_idx] - points[i]
                force_magnitude = 1.0 / (distances[nearest_idx] + 1e-8)
                attractive_forces[i] = direction / np.linalg.norm(direction) * force_magnitude
        
        # Apply forces with learning rate
        updated_points = points + attractive_forces * learning_rate
        
        # Project back to sphere and normalize
        # Project to surface of unit sphere
        norms = np.linalg.norm(updated_points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        updated_points = updated_points / norms[:, np.newaxis]
        
        return updated_points
    
    # Initialize using spherical method
    np.random.seed(42)
    
    # Generate points on sphere using Fibonacci method
    points_3d = fibonacci_sphere_sampling(16)
    
    # Project to 2D using stereographic projection
    points_2d = stereographic_projection(points_3d)
    
    # Normalize to unit square
    points_final = normalize_to_unit_square(points_2d)
    
    # Optimization loop with alternating strategies
    best_points = points_final.copy()
    best_ratio = calculate_min_max_ratio(best_points)
    
    num_iterations = 500
    for iteration in range(num_iterations):
        # Every 10 iterations, apply Voronoi-based repulsion
        if iteration % 10 == 0:
            voronoi_forces = voronoi_repulsion_force(best_points, alpha=0.05)
            # Apply forces to points
            new_points = best_points + voronoi_forces
            
            # Ensure points remain in bounds
            new_points = np.clip(new_points, 0, 1)
            
            # Test new configuration
            new_ratio = calculate_min_max_ratio(new_points)
            if new_ratio > best_ratio:
                best_points = new_points
                best_ratio = new_ratio
        
        # Apply spherical optimization step
        if iteration % 5 == 0:
            spherical_updated = spherical_optimization_step(best_points, learning_rate=0.01)
            spherical_updated = np.clip(spherical_updated, 0, 1)
            spherical_ratio = calculate_min_max_ratio(spherical_updated)
            if spherical_ratio > best_ratio:
                best_points = spherical_updated
                best_ratio = spherical_ratio
        
        # Occasionally add random perturbations
        if iteration % 20 == 0:
            noise = np.random.normal(0, 0.005, best_points.shape)
            perturbed = best_points + noise
            perturbed = np.clip(perturbed, 0, 1)
            perturbed_ratio = calculate_min_max_ratio(perturbed)
            if perturbed_ratio > best_ratio:
                best_points = perturbed
                best_ratio = perturbed_ratio
    
    return best_points

# EVOLVE-BLOCK-END