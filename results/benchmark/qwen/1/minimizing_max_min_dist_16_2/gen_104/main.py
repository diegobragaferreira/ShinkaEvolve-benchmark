# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Voronoi relaxation approach for better global optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_points = 16
    max_iterations = 1000
    tolerance = 1e-6
    cooling_schedule = [0.99, 0.95, 0.90, 0.85, 0.80]  # Cooling factors
    
    # Initialize points randomly within [0,1] x [0,1]
    points = np.random.rand(n_points, 2)
    
    # Track best solution
    best_ratio = 0.0
    best_points = points.copy()
    
    # Energy-based relaxation process
    for iteration in range(max_iterations):
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        # Calculate new positions as centroids of Voronoi cells
        new_points = np.zeros_like(points)
        converged = True
        
        # Process each point
        for i in range(n_points):
            # Get vertices of Voronoi cell for point i
            region = vor.regions[vor.point_region[i]]
            
            if -1 in region or len(region) < 3:
                # Handle unbounded regions (use current position with slight adjustment)
                new_points[i] = points[i] + np.random.normal(0, 0.001, 2)
                continue
                
            # Extract vertices of the Voronoi cell
            vertices = np.array([vor.vertices[j] for j in region if j >= 0])
            
            if len(vertices) < 3:
                # Not enough vertices, use current position
                new_points[i] = points[i]
                continue
                
            # Compute centroid of polygon (Voronoi cell)
            centroid = np.mean(vertices, axis=0)
            
            # Apply boundary constraints
            centroid = np.clip(centroid, 0, 1)
            
            # Update point position
            new_points[i] = centroid
            
            # Check for convergence
            if np.linalg.norm(new_points[i] - points[i]) > tolerance:
                converged = False
                
        # Apply cooling schedule for better convergence
        if iteration < len(cooling_schedule):
            cooling_factor = cooling_schedule[iteration]
            # Apply softening to avoid too aggressive movement
            points = points + cooling_factor * (new_points - points)
        else:
            points = new_points
            
        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)
        
        # Check if we've found a better solution
        if iteration % 10 == 0:  # Only check every few iterations
            # Calculate current ratio
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
        
        # Early stopping if converged
        if converged:
            break
    
    # Final optimization using local search around the relaxed solution
    # Compute pairwise distances for final evaluation
    distances = cdist(best_points, best_points)
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist > 0:
        final_ratio = min_dist / max_dist
    else:
        final_ratio = 0.0
    
    # Refine with simple gradient descent if needed
    if final_ratio < 0.2:  # If solution is poor, do some local optimization
        # Simple local search with small perturbations
        for _ in range(50):
            # Try small random perturbations
            perturbed_points = best_points + np.random.normal(0, 0.001, best_points.shape)
            perturbed_points = np.clip(perturbed_points, 0, 1)
            
            # Evaluate
            dist_mat = cdist(perturbed_points, perturbed_points)
            np.fill_diagonal(dist_mat, np.inf)
            min_d = np.min(dist_mat)
            max_d = np.max(dist_mat)
            
            if max_d > 0 and min_d / max_d > final_ratio:
                best_points = perturbed_points.copy()
                final_ratio = min_d / max_d
    
    return best_points

# EVOLVE-BLOCK-END