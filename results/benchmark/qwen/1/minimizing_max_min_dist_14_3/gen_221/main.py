# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple, Optional
import warnings

def calculate_min_max_ratio(points: np.ndarray) -> float:
    """Calculate the ratio of minimum to maximum distance."""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist < 1e-12:
        return 0.0
    return min_dist / max_dist

def icosahedron_points(n=14) -> np.ndarray:
    """Generate points using icosahedron-based construction"""
    # Vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # If we need more than 12 points, distribute additional points
    if n <= 12:
        # Just return subset of vertices
        return vertices[:n]
    else:
        # For 14 points, we'll start with icosahedron vertices and add two more
        points = vertices.copy()

        # Add two more points that are well-distributed
        # Add points along major axes
        points = np.vstack([points, [[0, 0, 1], [0, 0, -1]]])

        # Apply slight random perturbation to ensure good distribution
        np.random.seed(42)
        points += np.random.normal(0, 0.05, (points.shape[0], 3))

        # Normalize again to maintain unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / np.maximum(norms[:, np.newaxis], 1e-12)

        return points[:n]

def spherical_pessimization_initialization(n_points: int) -> np.ndarray:
    """
    Initialize points using spherical pessimization - construct points that 
    are deliberately designed to be well-spread on the sphere, with emphasis
    on avoiding worst-case configurations.
    """
    # Start with icosahedron base
    base_points = icosahedron_points(n_points)
    
    # Apply a systematic perturbation to avoid symmetric configurations
    # that might get stuck in poor local minima
    np.random.seed(42)
    
    # Add perturbations that preferentially move points away from
    # directions that make distances too concentrated
    for i in range(n_points):
        # Perturb each point with a direction that tends to spread points
        # Use a combination of random and systematic approaches
        direction = base_points[i] + np.random.normal(0, 0.03, 3)
        direction /= np.linalg.norm(direction) + 1e-12
        
        # Adjust magnitude based on local density effects
        # More perturbation for points that might be too close to others
        distances = np.linalg.norm(base_points - base_points[i], axis=1)
        avg_distance = np.mean(distances[distances > 1e-12])
        
        # Scale perturbation by how crowded this region is
        scale = 0.01 + 0.02 * np.exp(-avg_distance * 2.0)
        base_points[i] += scale * direction
    
    # Ensure all points stay on unit sphere
    norms = np.linalg.norm(base_points, axis=1, keepdims=True)
    base_points = base_points / np.maximum(norms, 1e-12)
    
    return base_points

def spherical_refinement_step(points: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Apply a sophisticated refinement step that iteratively improves the
    min/max ratio by using a combination of geometric reasoning and 
    gradient-based optimization.
    """
    n_points, dim = points.shape
    
    def objective(x_flat):
        points_reshaped = x_flat.reshape(n_points, dim)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
        normalized_points = points_reshaped / np.maximum(norms, 1e-12)
        
        # We want to maximize the ratio of min/max distance
        distances = pdist(normalized_points)
        if len(distances) == 0:
            return 1e10
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist < 1e-12:
            return 1e10
        ratio = min_dist / max_dist
        return -ratio  # Negative because we're minimizing negative ratio
    
    # Simplified constraint that keeps points on unit sphere
    def constraint_unit_sphere(x_flat):
        points_reshaped = x_flat.reshape(n_points, dim)
        norms = np.linalg.norm(points_reshaped, axis=1)
        return norms - 1.0
    
    # Use a hybrid approach to reduce computation cost while improving quality
    # First, try fast method with fewer iterations
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            bounds=[(-2, 2) for _ in range(n_points * dim)],
            options={'maxiter': max_iter // 2, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        
        if result.success:
            optimized_points = result.x.reshape(n_points, dim)
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            optimized_points = optimized_points / np.maximum(norms, 1e-12)
            return optimized_points
    except Exception:
        pass
    
    # If first attempt fails, try with more iterations but different method
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='SLSQP',
            bounds=[(-2, 2) for _ in range(n_points * dim)],
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9},
            tol=1e-9
        )
        
        if result.success:
            optimized_points = result.x.reshape(n_points, dim)
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            optimized_points = optimized_points / np.maximum(norms, 1e-12)
            return optimized_points
    except Exception:
        pass
    
    # If all optimization attempts fail, return original points
    return points

def spherical_pessimization_optimization(initial_points: np.ndarray, 
                                       max_iterations: int = 500) -> np.ndarray:
    """
    Main spherical pessimization optimization loop.
    Applies multiple refinement steps with strategic improvements.
    """
    current_points = initial_points.copy()
    
    # Iterative improvement with early stopping
    for iteration in range(max_iterations):
        # Save previous solution
        prev_ratio = calculate_min_max_ratio(current_points)
        
        # Apply refinement step
        refined_points = spherical_refinement_step(current_points, max_iter=50)
        
        # Check if there was improvement
        new_ratio = calculate_min_max_ratio(refined_points)
        
        # If no significant improvement, reduce step size or stop
        if abs(new_ratio - prev_ratio) < 1e-6:
            # Apply a different type of perturbation to escape local minima
            np.random.seed(iteration)
            noise = np.random.normal(0, 0.001, current_points.shape)
            current_points = current_points + noise
            
            # Keep on unit sphere
            norms = np.linalg.norm(current_points, axis=1, keepdims=True)
            current_points = current_points / np.maximum(norms, 1e-12)
        else:
            current_points = refined_points
            
        # Occasionally apply a global reorganization to prevent stagnation
        if iteration % 20 == 0 and iteration > 0:
            # Reinitialize with better distribution
            current_points = spherical_pessimization_initialization(len(current_points))
    
    return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel spherical pessimization algorithm that constructs point sets with optimal geometric
    distribution properties.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate initial points using spherical pessimization method
    initial_points = spherical_pessimization_initialization(14)
    
    # Apply iterative refinement with strategic optimization
    optimized_points = spherical_pessimization_optimization(initial_points, max_iterations=400)
    
    # Final validation and cleanup
    try:
        # Ensure final points are valid on unit sphere
        norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
        final_points = optimized_points / np.maximum(norms, 1e-12)
        
        # Verify we have a reasonable solution
        ratio = calculate_min_max_ratio(final_points)
        if ratio < 0.01:  # If extremely poor quality, fallback
            warnings.warn("Poor quality solution detected, falling back to standard method")
            # Fallback to simpler initialization + optimization if needed
            fallback_points = icosahedron_points(14)
            final_points = spherical_refinement_step(fallback_points, max_iter=100)
            
    except Exception:
        # Fallback to simple approach if anything goes wrong
        warnings.warn("Exception occurred, falling back to basic method")
        fallback_points = icosahedron_points(14)
        final_points = spherical_refinement_step(fallback_points, max_iter=100)
    
    return final_points

# EVOLVE-BLOCK-END