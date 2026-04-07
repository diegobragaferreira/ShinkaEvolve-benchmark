# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import time
from typing import Tuple

def initialize_points(n: int = 14, d: int = 3) -> np.ndarray:
    """
    Initialize points using a novel spherical Voronoi-based approach with adaptive distribution.

    Args:
        n: number of points
        d: dimensionality

    Returns:
        Initial point configuration
    """
    np.random.seed(42)
    
    # Strategy 1: Generate points on unit sphere using modified icosahedral method
    # Start with vertices of regular icosahedron (12 vertices)
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
    
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # If we need more than 12 points, add them strategically
    if n <= 12:
        points = vertices[:n].copy()
    else:
        # For 13-14 points, add one or two points using a variant of Fibonacci
        points = vertices.copy()
        
        # Add one more point at a strategic location to maintain symmetry
        # Position it roughly opposite to one of the original vertices
        additional_point = -vertices[0]  # Opposite to first vertex
        points = np.vstack([points, additional_point])
        
        if n == 14:
            # Add a second point to achieve 14 points
            additional_point2 = -vertices[1]  # Opposite to second vertex
            points = np.vstack([points, additional_point2])
    
    # Normalize newly added points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.max(norms)
    
    # Apply small random perturbations to break symmetry
    points += np.random.normal(0, 0.01, points.shape)
    
    # Project back to sphere and normalize again
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Scale and shift to [0,1]^3
    points = points * 0.5 + 0.5
    
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

def sphere_projection(xyz: np.ndarray) -> np.ndarray:
    """Project points onto unit sphere."""
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return xyz / norms

def spherical_energy_objective(points_flat: np.ndarray, 
                             penalty_weight: float = 1000.0,
                             repulsion_strength: float = 10.0) -> float:
    """
    Energy-based objective function with spherical constraints and repulsion forces.

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
    Coarse global optimization using gradient-free method with energy model.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Optimized point configuration
    """
    # Use Nelder-Mead for coarse global optimization
    try:
        x0 = initial_points.flatten()
        bounds = [(0.0, 1.0)] * len(x0)
        
        # First, let's do a simple gradient-free optimization with adaptive tolerance
        result = minimize(
            spherical_energy_objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 1000,
                'adaptive': True,
                'xatol': 1e-6,
                'fatol': 1e-6
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
        
        # Use L-BFGS-B with tight tolerances for fine optimization
        result = minimize(
            spherical_energy_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 1000,
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

def progressive_refinement(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Progressive refinement with alternating coarse and fine optimization stages.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Final optimized point configuration
    """
    current_points = initial_points.copy()
    
    # Stage 1: Coarse optimization
    coarse_result = coarse_global_optimization(current_points)
    
    # Stage 2: Fine optimization
    fine_result = fine_grained_optimization(coarse_result)
    
    # Stage 3: Additional refinement cycle
    final_result = fine_grained_optimization(fine_result)
    
    return final_result

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Phase 1: Initialize points with novel spherical Voronoi approach
    initial_points = initialize_points(14, 3)
    
    # Phase 2: Progressive refinement optimization
    final_points = progressive_refinement(initial_points)
    
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