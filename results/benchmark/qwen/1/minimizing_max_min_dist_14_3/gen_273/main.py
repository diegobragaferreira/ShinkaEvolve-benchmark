# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical Voronoi evolution algorithm for optimal point distribution.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    n = 14
    
    # Initialize points using Fibonacci spiral on unit sphere
    def fibonacci_sphere(samples=14):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    # Generate initial points on unit sphere
    points = fibonacci_sphere(n)
    
    # Helper function to compute spherical Voronoi areas
    def compute_voronoi_areas(points):
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points)
        # Return areas of all Voronoi regions
        return sv.calculate_areas()
    
    # Helper function to compute forces from Voronoi imbalance
    def compute_voronoi_forces(points):
        try:
            sv = SphericalVoronoi(points)
            # Get centroids of Voronoi regions
            centroids = sv.vertices
            
            # Normalize centroids to unit sphere
            centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            
            # Compute desired centroid positions (should be normalized points)
            # Force is proportional to difference between current and desired
            forces = np.zeros_like(points)
            
            # For each point, accumulate forces from Voronoi vertices
            for i in range(len(points)):
                if len(sv.regions[i]) > 0:
                    # Get Voronoi region vertices
                    region_vertices = sv.regions[i]
                    if len(region_vertices) > 0:
                        # Compute average of region vertices (approximate centroid)
                        avg_vertex = np.mean(centroids[region_vertices], axis=0)
                        # Normalize to unit sphere
                        avg_vertex = avg_vertex / np.linalg.norm(avg_vertex)
                        # Force = desired_position - current_position
                        forces[i] = avg_vertex - points[i]
            
            # Normalize forces
            norm_forces = np.linalg.norm(forces, axis=1, keepdims=True) + 1e-12
            forces = forces / norm_forces
            
            return forces
        except:
            # Fallback to simple repulsion force
            forces = np.zeros_like(points)
            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        diff = points[i] - points[j]
                        dist = np.linalg.norm(diff)
                        if dist > 1e-12:
                            forces[i] -= diff / (dist * dist)
            return forces / (np.linalg.norm(forces, axis=1, keepdims=True) + 1e-12)
    
    # Evolutionary optimization with Voronoi-based forces
    max_iter = 2000
    learning_rate = 0.02
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    best_ratio = -np.inf
    best_points = points.copy()
    
    # Precompute target area for uniform distribution
    target_area = 4 * np.pi / n
    
    for iteration in range(max_iter):
        # Compute current Voronoi areas
        try:
            current_areas = compute_voronoi_areas(points)
            # Compute the ratio of min to max area
            area_ratio = np.min(current_areas) / np.max(current_areas)
        except:
            area_ratio = 0.0
            
        # Compute Voronoi-based forces
        forces = compute_voronoi_forces(points)
        
        # Update velocities with momentum
        velocity = momentum * velocity + learning_rate * forces
        
        # Update points
        points = points + velocity
        
        # Project points back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms
        
        # Compute distance ratio
        distances = pdist(points)
        if len(distances) > 0:
            distances = distances[distances > 1e-12]
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
        
        # Adaptive learning rate decay
        if iteration % 100 == 0 and iteration > 0:
            learning_rate *= 0.95
    
    # Final refinement using constrained optimization
    def objective(x_flat):
        points = x_flat.reshape((n, 3))
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms
        
        distances = pdist(points)
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return np.inf
            
        return -d_min / d_max  # Negative because we minimize
    
    # Use constrained optimization for final refinement
    bounds = [(None, None) for _ in range(n * 3)]
    initial_flat = best_points.flatten()
    
    try:
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=[(None, None)] * (n * 3),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            refined_points = result.x.reshape((n, 3))
            # Project to sphere again
            norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            refined_points = refined_points / norms
            
            # Validate improvement
            distances = pdist(refined_points)
            distances = distances[distances > 1e-12]
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_points = refined_points
    except:
        pass
    
    # Final validation - ensure points are properly distributed
    final_distances = pdist(best_points)
    final_distances = final_distances[final_distances > 1e-12]
    
    if len(final_distances) == 0:
        # Fallback if something went wrong
        return fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END