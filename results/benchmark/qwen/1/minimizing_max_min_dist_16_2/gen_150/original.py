# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
from itertools import combinations
import math

def _fibonacci_sphere_points(n: int) -> np.ndarray:
    """Generate n points on a sphere using Fibonacci method."""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)

def _stereographic_project(points_3d: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using stereographic projection from south pole."""
    points_2d = []
    for x, y, z in points_3d:
        # Stereographic projection from south pole (0,0,-1)
        w = 1 / (1 + z)
        proj_x = x * w
        proj_y = y * w
        points_2d.append([proj_x, proj_y])
    
    return np.array(points_2d)

def _normalize_to_unit_square(points: np.ndarray) -> np.ndarray:
    """Normalize points to fit within [0,1] x [0,1]."""
    if len(points) == 0:
        return points
    
    # Find bounding box
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    
    # Avoid division by zero
    if x_max > x_min and y_max > y_min:
        points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
        points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
    
    return points

def _compute_voronoi_min_ratio(points):
    """Compute the minimum distance ratio using Voronoi properties to guide selection"""
    if len(points) < 2:
        return 0.0
    
    # Compute pairwise distances
    distances = squareform(pdist(points))
    np.fill_diagonal(distances, np.inf)
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist == 0:
        return 0.0
        
    return min_dist / max_dist

def _generate_voronoi_candidates(base_points, num_candidates=20):
    """Generate candidate point sets from Voronoi-based perturbations"""
    candidates = []
    
    # Generate perturbations around base points
    for i in range(num_candidates):
        # Create small random perturbations
        perturbation = np.random.normal(0, 0.005, base_points.shape)
        candidate = base_points + perturbation
        
        # Ensure candidates are within bounds
        candidate = np.clip(candidate, 0, 1)
        candidates.append(candidate)
        
    return candidates

def _create_hexagonal_grid(n_points=16, scale=0.8, offset=(0.1, 0.1)):
    """Create a structured hexagonal grid pattern"""
    points = []
    rows = int(np.ceil(np.sqrt(n_points)))
    cols = int(np.ceil(n_points / rows))
    
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n_points:
                break
            x = j * 0.25 + (i % 2) * 0.125
            y = i * 0.25
            points.append([x, y])
    
    points = np.array(points[:n_points])
    
    # Normalize and scale
    if len(points) > 0:
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale and offset
        points[:, 0] *= scale
        points[:, 1] *= scale
        points[:, 0] += offset[0]
        points[:, 1] += offset[1]
    
    return points

def _create_fibonacci_spiral(n_points=16):
    """Create points following Fibonacci spiral pattern"""
    points = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    for i in range(n_points):
        theta = 2 * np.pi * i / phi
        r = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0.5
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        points.append([x, y])
    
    return np.array(points)

def _create_regular_polygon(n_points=16):
    """Create points in regular polygon pattern"""
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = 0.5 + 0.4 * np.cos(angle)
        y = 0.5 + 0.4 * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _local_optimization_step(points, max_iter=100):
    """Perform local optimization using L-BFGS-B on candidate points"""
    
    def objective(x):
        points_matrix = x.reshape(-1, 2)
        distances = squareform(pdist(points_matrix))
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -1.0
            
        return -min_dist / max_dist  # Negative because we minimize to maximize ratio
    
    # Flatten points for optimization
    x0 = points.flatten()
    
    # Define bounds
    bounds = [(0, 1) for _ in range(len(x0))]
    
    try:
        # Optimize with L-BFGS-B
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            return optimized_points
    except:
        pass
        
    return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a Voronoi-based evolutionary approach combining geometric insights with local optimization.
    """
    
    np.random.seed(42)
    
    # Initialize best solution tracking
    best_points = None
    best_ratio = -np.inf
    
    # Generate multiple candidate point sets using different geometric approaches
    candidate_generators = [
        lambda: _create_hexagonal_grid(16),
        lambda: _create_fibonacci_spiral(16),
        lambda: _create_regular_polygon(16),
        lambda: np.random.rand(16, 2)
    ]
    
    # Sample from different initial configurations
    for gen_idx, generator in enumerate(candidate_generators):
        # Create base candidate
        base_points = generator()
        
        # Add small random noise to break symmetry
        noise = np.random.normal(0, 0.01, base_points.shape)
        base_points = np.clip(base_points + noise, 0, 1)
        
        # Generate multiple variations through Voronoi-based sampling
        candidates = _generate_voronoi_candidates(base_points, num_candidates=10)
        
        for candidate in candidates:
            # Apply local optimization
            optimized_candidate = _local_optimization_step(candidate, max_iter=300)
            
            # Evaluate solution
            ratio = _compute_voronoi_min_ratio(optimized_candidate)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_candidate.copy()
    
    # Perform final refinement on the best solution found
    if best_points is not None:
        final_points = _local_optimization_step(best_points, max_iter=500)
        final_ratio = _compute_voronoi_min_ratio(final_points)
        
        if final_ratio > best_ratio:
            best_points = final_points
    
    # Fallback to hexagonal grid if nothing worked
    if best_points is None:
        best_points = _create_hexagonal_grid(16)
    
    return best_points

# EVOLVE-BLOCK-END