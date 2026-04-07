# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
import math
from typing import Tuple

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def fibonacci_sphere(n_points: int) -> np.ndarray:
        """Generate points on a sphere using Fibonacci distribution for uniformity."""
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
    
    def stereographic_project(points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D using stereographic projection from south pole."""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        
        points_2d = np.array(points_2d)
        
        # Normalize to unit square [0,1] x [0,1]
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)
        
        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        return points_2d
    
    def compute_voronoi_forces(points: np.ndarray) -> np.ndarray:
        """Compute repulsive forces based on Voronoi diagram for point movement."""
        # Handle edge case of too few points
        if len(points) < 2:
            return np.zeros_like(points)
        
        try:
            vor = Voronoi(points)
            forces = np.zeros_like(points)
            
            # Compute forces for each point based on Voronoi cells
            for i in range(len(points)):
                # Find the Voronoi region for point i
                region = vor.regions[vor.point_region[i]]
                
                # Compute centroid of the Voronoi cell
                if region and -1 not in region and len(region) > 0:
                    # Get vertices of the Voronoi cell
                    vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(vertices) > 0:
                        centroid = np.mean(vertices, axis=0)
                        
                        # Repulsion force away from centroid
                        direction = points[i] - centroid
                        distance_to_centroid = np.linalg.norm(direction)
                        
                        if distance_to_centroid > 0:
                            force_magnitude = 1.0 / (distance_to_centroid + 1e-8)
                            forces[i] = force_magnitude * direction
                        
            return forces
            
        except Exception:
            # Fallback to simple repulsion if Voronoi fails
            forces = np.zeros_like(points)
            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        diff = points[i] - points[j]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            # Repulsion force
                            force_magnitude = 1.0 / (dist * dist + 1e-8)
                            forces[i] -= force_magnitude * diff / dist
                            
            return forces
    
    def compute_energy_gradient(points: np.ndarray) -> np.ndarray:
        """Compute energy gradient for optimization using pairwise distances."""
        n = len(points)
        if n < 2:
            return np.zeros_like(points)
        
        # Initialize forces
        forces = np.zeros_like(points)
        
        # Compute pairwise distances and forces
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                
                if dist > 0:
                    # Repulsive force inversely proportional to distance squared
                    force_magnitude = 1.0 / (dist * dist + 1e-8)
                    force_vector = force_magnitude * diff / dist
                    
                    forces[i] -= force_vector
                    forces[j] += force_vector
        
        return forces
    
    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance."""
        if len(points) < 2:
            return 0.0
            
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        if not distances or max(distances) <= 0:
            return 0.0
            
        d_min = min(distances)
        d_max = max(distances)
        
        return d_min / d_max if d_max > 0 else 0.0
    
    def energy_based_optimization(initial_points: np.ndarray, max_iterations: int = 500) -> Tuple[np.ndarray, float]:
        """Optimize using energy-based force dynamics."""
        points = initial_points.copy()
        current_ratio = compute_min_max_ratio(points)
        
        # Energy-based optimization with adaptive step sizes
        for iteration in range(max_iterations):
            # Compute forces on each point
            forces = compute_energy_gradient(points)
            
            # Adaptively scale forces to prevent large jumps
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes)
            
            if max_force > 0:
                # Adaptive learning rate based on force magnitude
                learning_rate = min(0.1, 1.0 / (max_force + 1e-8))
                points += learning_rate * forces
            
            # Keep points within bounds
            points = np.clip(points, 0, 1)
            
            # Periodically compute ratio to track progress
            if iteration % 50 == 0:
                new_ratio = compute_min_max_ratio(points)
                if new_ratio > current_ratio:
                    current_ratio = new_ratio
                    
            # Early stopping if improvement is minimal
            if iteration > 100 and iteration % 100 == 0:
                # Check if ratio has stabilized
                ratio_history = [compute_min_max_ratio(points)]
                test_points = points.copy()
                for _ in range(10):
                    forces = compute_energy_gradient(test_points)
                    test_points += 0.01 * forces
                    test_points = np.clip(test_points, 0, 1)
                    ratio_history.append(compute_min_max_ratio(test_points))
                
                if len(ratio_history) >= 2 and abs(ratio_history[-1] - ratio_history[0]) < 1e-6:
                    break
                    
        return points, current_ratio
    
    def voronoi_refinement(initial_points: np.ndarray, max_iterations: int = 300) -> Tuple[np.ndarray, float]:
        """Refine using Voronoi-based force dynamics."""
        points = initial_points.copy()
        current_ratio = compute_min_max_ratio(points)
        
        for iteration in range(max_iterations):
            # Compute Voronoi-based forces
            forces = compute_voronoi_forces(points)
            
            # Apply forces with adaptive learning rate
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes)
            
            if max_force > 0:
                learning_rate = min(0.05, 0.5 / (max_force + 1e-8))
                points += learning_rate * forces
            
            # Keep points within bounds
            points = np.clip(points, 0, 1)
            
            # Track improvement
            if iteration % 50 == 0:
                new_ratio = compute_min_max_ratio(points)
                if new_ratio > current_ratio:
                    current_ratio = new_ratio
                    
        return points, current_ratio
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate initial points using spherical approach
    points_3d = fibonacci_sphere(16)
    initial_points = stereographic_project(points_3d)
    
    # Apply energy-based optimization first
    optimized_points, ratio1 = energy_based_optimization(initial_points, max_iterations=500)
    
    # Apply Voronoi refinement
    refined_points, ratio2 = voronoi_refinement(optimized_points, max_iterations=300)
    
    # Final energy optimization
    final_points, ratio3 = energy_based_optimization(refined_points, max_iterations=300)
    
    # Return the best result
    best_points = final_points
    if ratio2 > ratio1 and ratio2 > ratio3:
        best_points = refined_points
    elif ratio3 > ratio1 and ratio3 > ratio2:
        best_points = final_points
    
    return best_points

# EVOLVE-BLOCK-END
