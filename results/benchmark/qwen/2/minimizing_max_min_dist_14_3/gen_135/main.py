# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple, Optional
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        
        try:
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 0.0
            
            return min_dist / max_dist
        except Exception:
            return 0.0
    
    def project_to_unit_sphere(points: np.ndarray) -> np.ndarray:
        """Project points onto unit sphere while maintaining relative positions."""
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms[:, np.newaxis]
    
    def fibonacci_sphere_initialization(n_points: int = 14, seed: int = 42) -> np.ndarray:
        """Initialize points using enhanced Fibonacci sphere method."""
        np.random.seed(seed)
        points = np.zeros((n_points, 3))
        
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(n_points):
            # Improved Fibonacci approach for better distribution
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            # Better phi calculation with improved spacing
            phi = (i * golden_ratio) % n_points * (2 * math.pi / n_points)
            
            # Add subtle correction factors to eliminate clustering
            correction = 0.05 * math.sin(0.3 * i * math.pi / n_points)
            phi += correction
            
            # Convert to Cartesian coordinates
            x = radius * math.cos(phi)
            z = radius * math.sin(phi)
            
            points[i] = [x, y, z]
        
        # Apply small random perturbations to break symmetries
        points += np.random.normal(0, 0.02, points.shape)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1.0, norms)
        points = points / safe_norms[:, np.newaxis]
        
        return points
    
    def compute_voronoi_based_gradients(points: np.ndarray) -> np.ndarray:
        """Compute gradient information based on spherical Voronoi diagram."""
        # Create Spherical Voronoi diagram
        sv = SphericalVoronoi(points)
        
        # For each point, compute Voronoi cell area and centroid
        gradients = np.zeros_like(points)
        
        # Simplified approach: compute gradient based on nearest neighbors
        # This is more computationally efficient than full Voronoi analysis
        distances = squareform(pdist(points))
        
        # For each point, we compute a force towards expanding small distances
        # and contracting large distances
        for i in range(len(points)):
            # Find nearest neighbors and analyze their distances
            distances_from_i = distances[i]
            sorted_indices = np.argsort(distances_from_i)
            
            # Focus on first few closest points and furthest points
            close_indices = sorted_indices[1:4]  # Closest 3 points
            far_indices = sorted_indices[-3:]   # Farthest 3 points
            
            # Compute influence of close points (want to expand these)
            close_force = np.zeros(3)
            for j in close_indices:
                if i != j:
                    diff = points[i] - points[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        close_force += diff / dist * (1/dist - 1)
            
            # Compute influence of far points (want to contract these)
            far_force = np.zeros(3)
            for j in far_indices:
                if i != j:
                    diff = points[i] - points[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-10:
                        far_force += diff / dist * (dist - 1)
            
            # Combine forces
            gradients[i] = 0.5 * close_force + 0.3 * far_force
        
        return gradients
    
    def geometric_refinement_step(points: np.ndarray, learning_rate: float = 0.05) -> np.ndarray:
        """Perform geometric refinement step using gradient information."""
        # Calculate current state
        current_ratio = compute_min_max_ratio(points)
        
        # Compute geometric gradients
        gradients = compute_voronoi_based_gradients(points)
        
        # Apply gradients with spherical projection
        new_points = points.copy()
        for i in range(len(points)):
            # Apply gradient step
            new_points[i] = points[i] + learning_rate * gradients[i]
            
            # Project back to sphere
            norms = np.linalg.norm(new_points[i])
            if norms > 1e-10:
                new_points[i] = new_points[i] / norms
        
        return new_points
    
    def spherical_gradient_descent(initial_points: np.ndarray, 
                                 max_iter: int = 5000,
                                 learning_rate: float = 0.02) -> Tuple[np.ndarray, float]:
        """Optimize using spherical gradient descent with geometric insights."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Adaptive learning rate
        adaptive_lr = learning_rate
        
        for iteration in range(max_iter):
            # Apply geometric refinement
            refined_points = geometric_refinement_step(current_points, adaptive_lr)
            
            # Compute new ratio
            new_ratio = compute_min_max_ratio(refined_points)
            
            # Accept improvement or use probabilistic acceptance
            if new_ratio > best_ratio:
                current_points = refined_points.copy()
                best_ratio = new_ratio
                best_points = refined_points.copy()
                
                # Reset adaptive learning rate when we improve
                adaptive_lr = learning_rate
            elif np.random.random() < math.exp((new_ratio - best_ratio) / 0.1):
                current_points = refined_points.copy()
            
            # Gradually decrease learning rate
            adaptive_lr *= 0.999
            
            # Occasionally do a global search
            if iteration % 100 == 0 and iteration > 0:
                # Add small random jitter to escape local minima
                jitter = np.random.normal(0, 0.005, current_points.shape)
                current_points = project_to_unit_sphere(current_points + jitter)
        
        return best_points, best_ratio
    
    def multi_resolution_optimization() -> np.ndarray:
        """Multi-resolution approach: coarse to fine optimization."""
        # Start with coarse optimization using fewer iterations
        coarse_points = fibonacci_sphere_initialization(14, 42)
        coarse_points, coarse_ratio = spherical_gradient_descent(coarse_points, 1000, 0.01)
        
        # Refine with higher resolution
        fine_points, fine_ratio = spherical_gradient_descent(coarse_points, 2000, 0.03)
        
        # Final refinement with very small steps
        final_points, final_ratio = spherical_gradient_descent(fine_points, 1000, 0.005)
        
        # Return best of all three
        final_ratio = compute_min_max_ratio(final_points)
        coarse_ratio = compute_min_max_ratio(coarse_points)
        fine_ratio = compute_min_max_ratio(fine_points)
        
        best_ratio = max(coarse_ratio, fine_ratio, final_ratio)
        if best_ratio == coarse_ratio:
            return coarse_points
        elif best_ratio == fine_ratio:
            return fine_points
        else:
            return final_points
    
    # Run the optimization
    start_time = time.time()
    
    # Multi-resolution optimization
    optimized_points = multi_resolution_optimization()
    
    # Additional local search if time permits
    if time.time() - start_time < 350:
        # Try another initialization and local search
        alt_points = fibonacci_sphere_initialization(14, 123)
        alt_points, alt_ratio = spherical_gradient_descent(alt_points, 1500, 0.02)
        
        # Compare and keep better result
        current_ratio = compute_min_max_ratio(optimized_points)
        if alt_ratio > current_ratio:
            optimized_points = alt_points
    
    # Final projection to ensure spherical constraint
    optimized_points = project_to_unit_sphere(optimized_points)
    
    return optimized_points

# EVOLVE-BLOCK-END