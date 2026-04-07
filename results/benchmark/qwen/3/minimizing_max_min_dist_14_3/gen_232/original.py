# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import warnings
from numba import jit

@jit(nopython=True)
def fast_pdist_matrix(points):
    """Fast computation of pairwise distances using Numba."""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def fibonacci_sphere(n):
    """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def spherical_voronoi_initialization(n_points):
    """Create initial points using spherical Voronoi uniformity principle."""
    # Start with Fibonacci distribution
    points = fibonacci_sphere(n_points)

    # Apply iterative adjustment to improve uniformity
    np.random.seed(42)
    for _ in range(20):
        # Perturb points slightly
        perturbations = np.random.normal(0, 0.01, (n_points, 3))
        points += perturbations

        # Project back to sphere surface
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms

    # Scale to unit cube [0,1]^3
    points = (points + 1) / 2
    return points

def spherical_map(points):
    """Map points from 3D space to unit sphere using normalization."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def spherical_voronoi_quality(sphere_points):
    """Calculate quality based on Voronoi cell areas on sphere."""
    if len(sphere_points) < 2:
        return 0

    # Create spherical Voronoi diagram
    try:
        sv = SphericalVoronoi(sphere_points)
        # Calculate total area of Voronoi cells
        cell_areas = sv.calculate_areas()
        # Quality is inversely related to variance of cell areas
        # More uniform areas indicate better distribution
        if len(cell_areas) > 0:
            mean_area = np.mean(cell_areas)
            if mean_area > 0:
                variance = np.var(cell_areas)
                # Return inverse variance (higher is better)
                return 1.0 / (1.0 + variance / mean_area**2)
    except Exception:
        pass
    return 0

def min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0

    # Calculate pairwise distances
    distances = pdist(points)

    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)

    # Avoid division by zero
    if d_max == 0:
        return 0

    return d_min / d_max

def constraint_penalty(points, penalty_weight=1000.0):
    """Calculate penalty for constraint violations."""
    penalty = 0
    for i in range(len(points)):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2
    return penalty

def objective_with_penalty(x_flat, penalty_weight=1000.0):
    """Objective function combining min/max ratio with penalties."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Calculate penalty for constraints
    penalty = constraint_penalty(points, penalty_weight)

    # Return negative ratio plus penalty (since we minimize)
    return -ratio + penalty

def adaptive_gradient_descent(initial_points, max_iterations=500, tolerance=1e-9):
    """Custom adaptive gradient descent with penalty handling."""
    points = initial_points.copy()
    prev_ratio = min_max_ratio(points)
    improvement_threshold = 1e-8

    for iteration in range(max_iterations):
        # Calculate gradients with finite differences
        grad = np.zeros_like(points)
        epsilon = 1e-6

        current_ratio = min_max_ratio(points)

        for i in range(len(points)):
            for j in range(3):
                # Forward difference
                test_points = points.copy()
                test_points[i, j] += epsilon

                # Clip to valid range
                test_points = np.clip(test_points, 0, 1)

                new_ratio = min_max_ratio(test_points)
                grad[i, j] = (new_ratio - current_ratio) / epsilon

        # Apply gradient descent step
        learning_rate = 0.01 * (1.0 - iteration / max_iterations)  # Adaptive learning rate
        new_points = points - learning_rate * grad

        # Enforce constraints
        new_points = np.clip(new_points, 0, 1)

        # Check for improvement
        new_ratio = min_max_ratio(new_points)
        if new_ratio > current_ratio:
            points = new_points
            prev_ratio = new_ratio
        else:
            # Reduce learning rate on stagnation
            learning_rate *= 0.9

        # Early stopping condition
        if abs(new_ratio - prev_ratio) < tolerance:
            break

        # Very small improvement threshold
        if abs(new_ratio - prev_ratio) < improvement_threshold:
            break

    return points

def multi_scale_optimization(initial_points, max_time_seconds=360):
    """Perform multi-scale optimization for better convergence."""
    # Phase 1: Coarse grained optimization with simple steps
    coarse_points = initial_points.copy()

    # Take a few large steps with high learning rate
    coarse_points = adaptive_gradient_descent(coarse_points, max_iterations=100, tolerance=1e-6)

    # Phase 2: Fine grained optimization with smaller steps
    fine_points = coarse_points.copy()
    fine_points = adaptive_gradient_descent(fine_points, max_iterations=300, tolerance=1e-9)

    return fine_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Create initial points using spherical Voronoi approach for better geometric distribution
    initial_points = spherical_voronoi_initialization(14)

    # Try multiple approaches and select the best
    best_points = initial_points.copy()
    best_ratio = min_max_ratio(best_points)

    # Multi-scale optimization
    optimized_points = multi_scale_optimization(initial_points)
    optimized_ratio = min_max_ratio(optimized_points)

    if optimized_ratio > best_ratio:
        best_points = optimized_points
        best_ratio = optimized_ratio

    # Additional refinement using scipy minimize with L-BFGS-B
    try:
        x0 = best_points.flatten()
        bounds = [(0, 1) for _ in range(14 * 3)]

        # Use scipy minimize for final refinement
        result = minimize(
            objective_with_penalty,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000},
            tol=1e-10
        )

        final_points = result.x.reshape((14, 3))
        final_ratio = min_max_ratio(final_points)

        if final_ratio > best_ratio:
            best_points = final_points
    except Exception:
        # Keep the best points found so far if scipy fails
        pass

    return best_points

# EVOLVE-BLOCK-END