# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
import math


def _initialize_hexagonal_grid(n_points: int = 16) -> np.ndarray:
    """Initialize points using a hexagonal grid pattern."""
    np.random.seed(42)

    points = np.zeros((n_points, 2))
    rows = 4
    cols = 4
    spacing = 0.25

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < n_points:
                # Offset every other row for hexagonal packing
                x = col * spacing + (row % 2) * spacing * 0.5
                y = row * spacing * math.sqrt(3) / 2
                points[idx] = [x, y]
                idx += 1

    # Normalize to [0.1, 0.9] range
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

    # Add small random perturbation to avoid degenerate cases
    points += np.random.normal(0, 0.01, points.shape)

    return points


def _initialize_spiral_pattern(n_points: int = 16) -> np.ndarray:
    """Initialize points using a spiral pattern."""
    points = np.zeros((n_points, 2))

    # Create spiral pattern
    angles = np.linspace(0, 4*np.pi, n_points)
    radii = np.linspace(0.1, 0.4, n_points)

    for i in range(n_points):
        points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
        points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

    return points


def _initialize_random(n_points: int = 16) -> np.ndarray:
    """Initialize points using random distribution."""
    return np.random.uniform(0.1, 0.9, (n_points, 2))


def _initialize_fibonacci_sphere(n_points: int = 16) -> np.ndarray:
    """Initialize points using Fibonacci sphere distribution for good spreading."""
    points = np.zeros((n_points, 2))
    phi = math.pi * (3 - math.sqrt(5))  # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        # Map to 2D plane
        points[i] = [0.5 + x * 0.4, 0.5 + z * 0.4]

    return points


def _compute_distance_ratio(points: np.ndarray) -> float:
    """Compute the ratio of minimum to maximum distance between all point pairs."""
    if len(points) < 2:
        return 0.0

    # Compute pairwise distances efficiently
    distances = squareform(pdist(points))

    # Mask diagonal elements (distance to self is 0)
    np.fill_diagonal(distances, np.inf)

    # Get min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Handle case where all points might be coincident
    if max_dist == 0:
        return 0.0

    return min_dist / max_dist


def _objective_function(points: np.ndarray) -> float:
    """Objective function to maximize (negative because scipy minimizes)."""
    # Reshape points array to (n, 2) format if needed
    if points.ndim == 1:
        points = points.reshape(-1, 2)

    # Compute negative of distance ratio (since we want to maximize)
    return -_compute_distance_ratio(points)


def _constraint_function(points: np.ndarray) -> float:
    """
    Constraint function to ensure points stay within bounds [0,1] x [0,1].
    Returns positive value when constraint is satisfied.
    """
    if points.ndim == 1:
        points = points.reshape(-1, 2)

    # Check if any point is outside [0,1] bounds
    violations = 0

    # Check x bounds
    violations += np.sum(points[:, 0] < 0)
    violations += np.sum(points[:, 0] > 1)

    # Check y bounds
    violations += np.sum(points[:, 1] < 0)
    violations += np.sum(points[:, 1] > 1)

    # Return negative of violations (positive if all constraints satisfied)
    return -violations


def _optimize_with_multiple_methods(initial_points: np.ndarray) -> np.ndarray:
    """Run optimization with multiple methods and return best result."""
    n = initial_points.shape[0]
    d = initial_points.shape[1]

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    # Define bounds for each coordinate [0, 1]
    bounds = [(0, 1) for _ in range(n * d)]

    # Optimization options
    options = {
        'maxiter': 500,
        'ftol': 1e-8,
        'gtol': 1e-8
    }

    best_points = initial_points.copy()
    best_ratio = _compute_distance_ratio(best_points)

    # Method 1: L-BFGS-B
    try:
        result = minimize(
            fun=_objective_function,
            x0=initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options=options,
            tol=1e-8
        )

        if result.success:
            optimized_points = result.x.reshape(n, d)
            ratio = _compute_distance_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass

    # Method 2: SLSQP as fallback
    try:
        result = minimize(
            fun=_objective_function,
            x0=initial_flat,
            method='SLSQP',
            bounds=bounds,
            options=options,
            tol=1e-8
        )

        if result.success:
            optimized_points = result.x.reshape(n, d)
            ratio = _compute_distance_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass

    # Method 3: Nelder-Mead as last resort
    try:
        result = minimize(
            fun=_objective_function,
            x0=initial_flat,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'adaptive': True}
        )

        if result.success:
            optimized_points = result.x.reshape(n, d)
            ratio = _compute_distance_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass

    return best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2

    # Try multiple initialization strategies
    initializations = [
        _initialize_hexagonal_grid(n),
        _initialize_spiral_pattern(n),
        _initialize_random(n),
        _initialize_fibonacci_sphere(n)
    ]

    best_points = None
    best_ratio = -np.inf

    # Run optimization from each initialization
    for init_points in initializations:
        optimized_points = _optimize_with_multiple_methods(init_points)
        ratio = _compute_distance_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    # If no successful optimization, return the best initialization
    if best_points is None:
        best_points = _initialize_hexagonal_grid(n)

    return best_points


# EVOLVE-BLOCK-END