# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
from itertools import product


def _initialize_points(n_points: int = 16, method='hexagonal') -> np.ndarray:
    """Initialize points with a structured pattern that provides good starting configuration."""
    np.random.seed(42)

    if method == 'hexagonal':
        # Create a roughly hexagonal arrangement with some randomness
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25

                # Add small random perturbation
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)

                points.append([x, y])

        # Ensure we have exactly n_points
        points = np.array(points[:n_points])

        # Normalize to [0,1] bounds
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Scale to fit in [0,1] x [0,1]
        points[:, 0] *= 0.95
        points[:, 1] *= 0.95
        points[:, 0] += 0.025
        points[:, 1] += 0.025

    elif method == 'spiral':
        # Create a spiral arrangement
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            radius = 0.4 * (i / n_points)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            points.append([x, y])
        points = np.array(points)

    elif method == 'random':
        # Purely random initialization
        points = np.random.rand(n_points, 2)

    elif method == 'grid':
        # Create a regular grid arrangement
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = j / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                # Add small random perturbation
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                points.append([x, y])
        points = np.array(points[:n_points])

    elif method == 'fibonacci_spiral':
        # Create Fibonacci spiral arrangement for better point distribution
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n_points):
            theta = 2 * np.pi * i / phi
            r = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0.5
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            points.append([x, y])
        points = np.array(points)

    elif method == 'regular_polygon':
        # Create points arranged in a regular polygon pattern
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            # Add small random perturbation
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            points.append([x, y])
        points = np.array(points)

    else:
        # Default hexagonal
        points = _initialize_points(n_points, 'hexagonal')

    return points


def _compute_distance_ratio(points: np.ndarray) -> float:
    """Compute the ratio of minimum to maximum distance between all point pairs."""
    if len(points) < 2:
        return 0.0

    # Compute pairwise distances using squareform for numerical stability
    distances = squareform(pdist(points))

    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)

    # Get min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Handle case where all points might be coincident
    if max_dist == 0 or min_dist == np.inf:
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


def _optimize_single_start(initial_points: np.ndarray, maxiter: int = 1000) -> tuple:
    """Perform optimization from a single starting point with adaptive strategy."""
    # Flatten points for optimization
    initial_flat = initial_points.flatten()

    # Define bounds for each coordinate [0, 1]
    bounds = [(0, 1) for _ in range(len(initial_flat))]

    # Optimization options
    options = {
        'maxiter': maxiter,
        'ftol': 1e-10,
        'gtol': 1e-10
    }

    try:
        # First try L-BFGS-B for local optimization
        result = minimize(
            fun=_objective_function,
            x0=initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options=options,
            tol=1e-10
        )

        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            objective_value = -result.fun
            return optimized_points, objective_value
        else:
            # If L-BFGS-B fails, try with a different method
            result = minimize(
                fun=_objective_function,
                x0=initial_flat,
                method='TNC',
                bounds=bounds,
                options=options,
                tol=1e-10
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                objective_value = -result.fun
                return optimized_points, objective_value
            else:
                return initial_points, _compute_distance_ratio(initial_points)

    except Exception as e:
        # Fallback to initial points if optimization fails due to any reason
        return initial_points, _compute_distance_ratio(initial_points)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2

    # Test multiple initialization methods for better exploration
    init_methods = ['hexagonal', 'spiral', 'random', 'grid', 'fibonacci_spiral', 'regular_polygon']
    best_points = None
    best_ratio = -np.inf

    # Multi-start optimization with increased restarts
    for method in init_methods:
        # Create multiple random restarts for each method
        for restart in range(5):  # 5 restarts per method for better exploration
            np.random.seed(42 + restart)  # Different seed for each restart
            initial_points = _initialize_points(n, method)

            # Perform optimization from this starting point
            optimized_points, ratio = _optimize_single_start(initial_points, maxiter=500)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

    # Final refinement with more iterations if we found a good solution
    if best_points is not None:
        final_points, final_ratio = _optimize_single_start(best_points, maxiter=1000)
        if final_ratio > best_ratio:
            best_points = final_points

    # Return the best solution found
    if best_points is None:
        # Fallback to default initialization
        best_points = _initialize_points(n, 'hexagonal')

    return best_points


# EVOLVE-BLOCK-END