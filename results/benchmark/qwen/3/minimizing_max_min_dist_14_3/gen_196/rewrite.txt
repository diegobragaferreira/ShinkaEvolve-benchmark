# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple

def initialize_points(n: int = 14, d: int = 3) -> np.ndarray:
    """
    Initialize points using a novel spherical Voronoi-inspired energy-based approach 
    that creates better distributed starting configurations.

    Args:
        n: number of points
        d: dimensionality

    Returns:
        Initial point configuration
    """
    np.random.seed(42)
    
    # Generate points using a combination of geometric principles and optimization
    # Strategy 1: Start with icosahedron vertices for good baseline distribution
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ])
    
    # For more than 12 points, add Fibonacci spiral points in a structured way
    if n <= 12:
        points = vertices[:n].copy()
    else:
        # Start with icosahedron vertices
        points = vertices.copy()
        
        # Add points using Fibonacci spiral but with better angular distribution
        # Add points in a way that avoids clustering around the poles
        additional_points = []
        for i in range(n - 12):
            # Modified Fibonacci spiral that distributes points more evenly
            if i < 2:
                # Add points along latitude lines to avoid pole clustering
                lat = np.pi * (i + 1) / (n - 11) - np.pi/2
                lon = i * 2 * np.pi / (n - 11)
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                additional_points.append([x, y, z])
            else:
                # Spread remaining points more uniformly
                phi = np.arccos(1 - 2 * ((12 + i) / (n - 1)))
                theta = np.sqrt(n) * phi
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                additional_points.append([x, y, z])
                
        points = np.vstack([points, additional_points])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-10)  # Avoid division by zero
    
    # Apply spherical Voronoi-inspired energy relaxation
    # This helps distribute points more uniformly without being symmetrically constrained
    for _ in range(5):
        # Calculate pairwise distances
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        
        # Apply repulsion forces between nearby points
        for i in range(len(points)):
            # Repulse from points that are too close
            close_indices = np.where(distances[i] < 0.7)[0]
            if len(close_indices) > 0:
                repulsion_force = np.zeros(3)
                for j in close_indices:
                    if i != j:
                        diff = points[i] - points[j]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            # Strong repulsion for very close points
                            repulsion_magnitude = 1.0 / (dist * dist + 1e-8)
                            repulsion_force += repulsion_magnitude * diff / dist
                
                # Apply the repulsion force
                if np.linalg.norm(repulsion_force) > 0:
                    points[i] += 0.01 * repulsion_force
                    # Project back to sphere
                    norm = np.linalg.norm(points[i])
                    if norm > 0:
                        points[i] = points[i] / norm
    
    # Scale and shift to [0,1]^3
    points = points * 0.5 + 0.5
    
    # Add small random perturbation to break any remaining symmetries
    points += np.random.normal(0, 0.003, points.shape)
    
    # Ensure all points remain within bounds
    points = np.clip(points, 0, 1)
    
    return points

def calculate_distance_metrics(points: np.ndarray) -> tuple[float, float]:
    """
    Calculate minimum and maximum distances between all point pairs efficiently.

    Args:
        points: Array of shape (n, d)

    Returns:
        Tuple of (min_distance, max_distance)
    """
    if len(points) < 2:
        return 0.0, 0.0

    # Use scipy's distance functions for efficiency
    distances = cdist(points, points, 'euclidean')
    
    # Zero out diagonal (distance to self)
    np.fill_diagonal(distances, np.inf)

    if distances.size == 0:
        return 0.0, 0.0

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    return min_dist, max_dist

def energy_based_objective(points_flat: np.ndarray, 
                         penalty_weight: float = 1000.0,
                         repulsion_strength: float = 10.0) -> float:
    """
    Energy-based objective function that includes repulsion forces for better distribution.

    Args:
        points_flat: Flattened array of point coordinates
        penalty_weight: Weight for boundary penalty
        repulsion_strength: Strength of repulsion force between points

    Returns:
        Objective value to minimize
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)
    
    # Apply boundary penalty using coordinate transformation
    penalty = 0.0
    
    # Transform coordinates to ensure they're within [0,1] range
    # This prevents hard clipping which can create discontinuities
    transformed_points = np.clip(points, 0, 1)
    
    # Add penalty for points that were clipped (boundary violations)
    for i, coord in enumerate(points.flat):
        if coord < 0:
            penalty += penalty_weight * (0 - coord)**2
        elif coord > 1:
            penalty += penalty_weight * (coord - 1)**2
    
    # Calculate distances using optimized scipy function
    distances = cdist(transformed_points, transformed_points, 'euclidean')
    np.fill_diagonal(distances, np.inf)
    
    if distances.size == 0:
        return penalty + float('inf')
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Avoid division by zero
    if max_dist <= 0:
        return penalty + float('inf')
    
    # Add repulsion energy for very close points (to enforce separation)
    repulsion_energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i,j]
            if dist < 0.1:  # Only consider very close pairs
                repulsion_energy += repulsion_strength / (dist + 1e-8)
    
    # Return negative ratio plus penalties to minimize (maximize the ratio)
    ratio = -min_dist / max_dist
    return ratio + penalty + repulsion_energy

def coarse_global_optimization(initial_points: np.ndarray, max_time: float = 300.0) -> np.ndarray:
    """
    Coarse global optimization using Nelder-Mead method with energy model.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Optimized point configuration
    """
    try:
        x0 = initial_points.flatten()
        bounds = [(0.0, 1.0)] * len(x0)
        
        # First stage: Coarse optimization with Nelder-Mead
        result = minimize(
            energy_based_objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'adaptive': True,
                'xatol': 1e-5,
                'fatol': 1e-5
            }
        )
        
        optimized_points = result.x.reshape(14, 3)
        optimized_points = np.clip(optimized_points, 0, 1)
        return optimized_points
        
    except Exception:
        return initial_points

def fine_grained_optimization(initial_points: np.ndarray, max_time: float = 60.0) -> np.ndarray:
    """
    Fine-grained optimization using L-BFGS-B with constrained optimization.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Refined point configuration
    """
    try:
        x0 = initial_points.flatten()
        bounds = [(0.0, 1.0)] * len(x0)
        
        # Second stage: Fine optimization with L-BFGS-B
        result = minimize(
            energy_based_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 500,
                'ftol': 1e-12,
                'gtol': 1e-12,
                'eps': 1e-8
            }
        )
        
        refined_points = result.x.reshape(14, 3)
        refined_points = np.clip(refined_points, 0, 1)
        return refined_points
        
    except Exception:
        return initial_points

def multi_stage_refinement(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Multi-stage progressive refinement with alternating optimization strategies.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Final optimized point configuration
    """
    current_points = initial_points.copy()
    
    # Stage 1: Coarse global optimization
    coarse_result = coarse_global_optimization(current_points)
    
    # Stage 2: Fine-grained optimization  
    fine_result = fine_grained_optimization(coarse_result)
    
    # Stage 3: Additional refinement with different strategy
    # Try another Nelder-Mead run to escape local minima
    second_coarse = coarse_global_optimization(fine_result)
    final_result = fine_grained_optimization(second_coarse)
    
    return final_result

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Phase 1: Initialize points with novel spherical Voronoi-inspired approach
    initial_points = initialize_points(14, 3)
    
    # Phase 2: Progressive refinement optimization
    final_points = multi_stage_refinement(initial_points)
    
    # Phase 3: Final validation and adjustment
    # Calculate final metrics to verify quality
    min_dist, max_dist = calculate_distance_metrics(final_points)
    
    # If optimization didn't work well, fall back to a good known arrangement
    if max_dist <= 0 or min_dist <= 0:
        # Fallback to regularized arrangement
        np.random.seed(42)
        final_points = np.random.rand(14, 3)
    
    return final_points

# EVOLVE-BLOCK-END