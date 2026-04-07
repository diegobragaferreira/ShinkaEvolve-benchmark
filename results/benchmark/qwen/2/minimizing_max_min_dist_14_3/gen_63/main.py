# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the ratio of minimum to maximum distances."""
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)
        # Handle edge case where dmax might be zero
        return dmin / dmax if dmax > 0 else 0.0

    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        # Use a more uniform Fibonacci distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            # Distribute points more evenly using Fibonacci-like spacing
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)
            
            # Better angular distribution
            theta = np.arccos(1 - 2 * (i / (n - 1)))  # Polar angle
            phi = (i * golden_ratio) % (2 * np.pi)   # Azimuthal angle
            
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            points.append([x, y, z])
        return np.array(points)

    def project_to_sphere(points):
        """Project points onto unit sphere."""
        points = np.asarray(points)
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms[:, np.newaxis]

    def perturb_point_on_sphere(point, perturbation_magnitude=0.01):
        """Perturb a point on the unit sphere while keeping it on the sphere."""
        # Generate random perturbation
        perturbation = np.random.normal(0, perturbation_magnitude, 3)
        
        # Project perturbation to tangent plane of the sphere at current point
        projection_factor = np.dot(perturbation, point)
        perturbation_tangent = perturbation - projection_factor * point
        
        # Apply perturbation and project back to sphere
        new_point = point + perturbation_tangent
        return project_to_sphere(new_point)[0]

    def simulated_annealing_optimization(initial_points, max_iter=15000):
        """Optimize points using simulated annealing with adaptive cooling."""
        current_points = initial_points.copy()
        current_points = project_to_sphere(current_points)
        
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        best_ratio_history = [best_ratio]
        
        # Adaptive cooling schedule
        temperature = 0.1
        cooling_rate = 0.9995
        min_temp = 1e-6
        stall_count = 0
        max_stall = 500
        
        for iteration in range(max_iter):
            # Store current configuration
            old_points = current_points.copy()
            
            # Select random point to perturb
            idx = np.random.randint(len(current_points))
            
            # Apply perturbation
            current_points[idx] = perturb_point_on_sphere(current_points[idx], 0.005)
            
            # Compute new distance matrix
            new_ratio = compute_min_max_ratio(current_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = current_points.copy()
                best_ratio_history.append(best_ratio)
                stall_count = 0
            elif np.random.random() < np.exp((new_ratio - best_ratio) / temperature):
                # Accept worse solution with probability
                pass  # Keep the new configuration
            else:
                # Revert to previous configuration
                current_points = old_points
            
            # Cool down temperature
            temperature = max(min_temp, temperature * cooling_rate)
            
            # Early stopping if no improvement for a while
            stall_count += 1
            if stall_count > max_stall:
                break
                
        return best_points, best_ratio

    def spherical_voronoi_initialization(n_points):
        """Create initial configuration using spherical Voronoi-like arrangement."""
        # Generate points on sphere using Fibonacci method but with better distribution
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n_points):
            # More even distribution
            z = 1 - (i / (n_points - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)
            
            # Improve angular distribution
            theta = np.arccos(1 - 2 * (i / (n_points - 1)))
            # Using golden angle to distribute points more evenly
            phi = (i * 2.399963229728653) % (2 * np.pi)
            
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            points.append([x, y, z])
            
        return np.array(points)

    # Multi-start optimization with diverse initializations
    best_points = None
    best_ratio = -1.0
    
    # Run multiple optimizations with different starting points
    initial_seeds = [
        fibonacci_sphere(14),
        spherical_voronoi_initialization(14),
        np.random.rand(14, 3) * 2 - 1  # Random points in [-1,1]^3
    ]
    
    # Add more diverse initializations using different seeds
    for seed in range(5):
        np.random.seed(seed * 1000)
        # Create variation of Fibonacci sphere with slight randomness
        fib_points = fibonacci_sphere(14)
        jittered_points = fib_points + np.random.normal(0, 0.01, (14, 3))
        initial_seeds.append(jittered_points)
    
    # Optimize each initialization
    for i, initial_points in enumerate(initial_seeds):
        # Normalize to unit sphere
        initial_points = project_to_sphere(initial_points)
        
        # Optimize this initialization
        optimized_points, ratio = simulated_annealing_optimization(initial_points, 10000)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # Final refinement with longer optimization
    if best_points is not None:
        final_points, final_ratio = simulated_annealing_optimization(best_points, 5000)
        if final_ratio > best_ratio:
            best_points = final_points
    
    # Ensure points are on unit sphere
    best_points = project_to_sphere(best_points)
    
    # Map to unit cube [0,1]^3 for output requirements
    best_points = (best_points + 1) / 2
    
    return best_points

# EVOLVE-BLOCK-END