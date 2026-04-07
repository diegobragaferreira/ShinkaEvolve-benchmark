# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
from itertools import combinations

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

def _compute_voronoi_statistics(points):
    """Compute Voronoi-related statistics for quality assessment"""
    if len(points) < 3:
        return {'avg_area': 0, 'min_area': 0, 'max_area': 0, 'compactness': 0}

    try:
        vor = Voronoi(points)
        areas = []

        # Get Voronoi cell areas (if available)
        if hasattr(vor, 'areas'):
            areas = vor.areas

        avg_area = np.mean(areas) if len(areas) > 0 else 0
        min_area = np.min(areas) if len(areas) > 0 else 0
        max_area = np.max(areas) if len(areas) > 0 else 0

        # Compactness measure (ratio of actual area to bounding box area)
        if len(points) > 1:
            bbox_area = (np.max(points[:, 0]) - np.min(points[:, 0])) * \
                       (np.max(points[:, 1]) - np.min(points[:, 1]))
            actual_area = np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) -
                               np.dot(points[:, 1], np.roll(points[:, 0], 1))) / 2
            compactness = actual_area / bbox_area if bbox_area > 0 else 0
        else:
            compactness = 0

        return {
            'avg_area': avg_area,
            'min_area': min_area,
            'max_area': max_area,
            'compactness': compactness
        }
    except:
        return {'avg_area': 0, 'min_area': 0, 'max_area': 0, 'compactness': 0}

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