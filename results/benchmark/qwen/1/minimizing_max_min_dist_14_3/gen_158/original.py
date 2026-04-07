# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    random.seed(42)

    n = 14
    d = 3

    # Generate initial points using Fibonacci spiral on sphere for good distribution
    def fibonacci_sphere(samples=14):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)

    def normalize_to_cube(points):
        """Normalize points to fit in [0,1]^3 cube."""
        # Center around origin and scale appropriately
        centered = points - np.mean(points, axis=0)
        max_coord = np.max(np.abs(centered))
        if max_coord > 0:
            scaled = centered / max_coord * 0.5
        else:
            scaled = centered
        # Shift to [0,1]^3
        normalized = scaled + 0.5
        return normalized

    def calculate_ratio(points):
        """Calculate min/max distance ratio."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        # Filter out near-zero distances to avoid numerical issues
        distances = distances[distances > 1e-12]
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max > 0:
            ratio = d_min / d_max
        else:
            ratio = 0.0
            
        return ratio, d_min, d_max

    def objective(x_flat):
        """Objective function to maximize min/max distance ratio."""
        points = x_flat.reshape((n, d))
        distances = pdist(points)
        # Filter out near-zero distances
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -np.inf
            
        # Return negative because we're minimizing
        return -d_min / d_max

    # Multi-start optimization with adaptive strategy
    best_ratio = -np.inf
    best_points = None
    
    # Base initialization
    base_points = fibonacci_sphere(n)
    base_points = normalize_to_cube(base_points)
    
    # Adaptive perturbation strategy
    base_perturbation = 0.02
    perturbation_decay = 0.95
    min_perturbation = 0.001
    max_starts = 15
    
    for start_iteration in range(max_starts):
        # Adaptive perturbation scaling
        current_perturbation = max(base_perturbation * (perturbation_decay ** start_iteration), 
                                 min_perturbation)
        
        # Generate starting points
        if start_iteration == 0:
            # First start: use base points
            current_points = base_points.copy()
        else:
            # Subsequent starts: perturb base points
            perturbation = np.random.normal(0, current_perturbation, base_points.shape)
            current_points = base_points + perturbation
            current_points = np.clip(current_points, 0, 1)
        
        # First optimization with L-BFGS-B (coarse)
        initial_flat = current_points.flatten()
        bounds = [(0, 1) for _ in range(n * d)]
        
        try:
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )
        except Exception:
            continue
            
        if not result.success:
            continue
            
        optimized_points = result.x.reshape((n, d))
        # Ensure bounds are respected
        optimized_points = np.clip(optimized_points, 0, 1)
        
        # Calculate the actual ratio for this optimization run
        ratio, _, _ = calculate_ratio(optimized_points)
        
        # Update best solution
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
            
        # Early termination if we're getting close to good solutions
        if best_ratio > 0.45 and start_iteration > 5:
            break

    # If we didn't find a good solution, return the base points
    if best_points is None:
        return base_points

    # Apply final refinement with different optimization methods
    refined_points = best_points.copy()
    
    # Try SLSQP for better constraint handling
    try:
        initial_flat = refined_points.flatten()
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result.success:
            candidate_points = result.x.reshape((n, d))
            candidate_points = np.clip(candidate_points, 0, 1)
            
            # Validate improvement
            _, old_min, old_max = calculate_ratio(refined_points)
            _, new_min, new_max = calculate_ratio(candidate_points)
            
            if new_min > old_min and new_max <= old_max:
                refined_points = candidate_points
    except Exception:
        pass

    # Final validation to prevent degenerate cases
    final_distances = pdist(refined_points)
    if len(final_distances) > 0:
        min_distance = np.min(final_distances)
        if min_distance < 1e-12:
            # If we have degenerate points, fall back to base points
            return base_points
    
    return refined_points

# EVOLVE-BLOCK-END