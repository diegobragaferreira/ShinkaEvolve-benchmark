# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    best_ratio = -np.inf
    best_points = None

    # Multiple restart strategies
    restart_strategies = [
        # Strategy 1: Hexagonal packing initialization (better initial spacing)
        lambda: _hexagonal_packing_init(),

        # Strategy 2: Adaptive perturbed grid initialization
        lambda: _adaptive_perturbed_grid_init(),

        # Strategy 3: Random initialization with better spread
        lambda: _random_spread_init()
    ]

    # Try each initialization strategy multiple times
    for strategy_idx, init_func in enumerate(restart_strategies):
        for restart in range(3):  # 3 restarts per strategy
            np.random.seed(strategy_idx * 1000 + restart)

            # Get initial points
            points = init_func()

            # Apply local optimization with multiple methods
            optimized_points = _local_optimization(points)

            # Calculate ratio for this optimization run
            ratio = _calculate_min_max_ratio(optimized_points)

            # Keep track of best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

    return best_points


def _hexagonal_packing_init():
    """Initialize points using hexagonal packing pattern."""
    # Create a hexagonal lattice pattern
    rows = 4
    cols = 4
    points = []

    for i in range(rows):
        for j in range(cols):
            # offset every other row
            x_offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + x_offset) * 0.25 + 0.125  # Scale and shift to [0.125, 0.875]
            y = i * 0.25 + 0.125
            points.append([x, y])

    return np.array(points)


def _adaptive_perturbed_grid_init():
    """Initialize with grid points plus adaptive random perturbations based on current distribution."""
    # Start with a regular grid
    grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
    points = grid_points.astype(float) / 3.0  # Normalize to [0,1] range

    # Calculate initial distance statistics to determine appropriate perturbation scale
    initial_ratio = _calculate_min_max_ratio(points)

    # Adaptive perturbation scaling - smaller perturbations when configuration is already good
    base_perturbation = 0.03
    if initial_ratio > 0.1:  # If already reasonably balanced
        perturbation_magnitude = base_perturbation * 0.3
    elif initial_ratio > 0.05:  # Moderately unbalanced
        perturbation_magnitude = base_perturbation * 0.7
    else:  # Very unbalanced, allow larger perturbations
        perturbation_magnitude = base_perturbation * 1.5

    # Add random perturbations
    np.random.seed(42)
    points += np.random.uniform(-perturbation_magnitude, perturbation_magnitude, points.shape)

    # Ensure points stay within bounds
    points = np.clip(points, 0, 1)

    return points


def _random_spread_init():
    """Initialize with random points that are intentionally spread out."""
    np.random.seed(42)
    points = np.random.rand(16, 2)

    # Apply some basic spacing to prevent clustering
    for i in range(16):
        # Move points away from center slightly
        center_vec = points[i] - [0.5, 0.5]
        center_distance = np.linalg.norm(center_vec)
        if center_distance > 0:
            points[i] += center_vec * 0.1 / center_distance

    # Clip to ensure within bounds
    points = np.clip(points, 0, 1)

    return points


def _calculate_min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0

    # Efficiently compute all pairwise distances
    distances = cdist(points, points, metric='euclidean')

    # Set diagonal to infinity to ignore self-distances
    np.fill_diagonal(distances, np.inf)

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    if max_dist <= 0:
        return 0

    return min_dist / max_dist


def _local_optimization(initial_points):
    """Apply local optimization to improve the point distribution."""
    n = 16
    d = 2

    # Define objective function: negative ratio (we'll minimize this)
    def objective(x):
        # Reshape flat array back to points
        pts = x.reshape(n, d)

        # Calculate all pairwise distances efficiently
        distances = cdist(pts, pts, metric='euclidean')
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize the ratio
        if max_dist <= 0 or min_dist <= 0:
            return 0
        return -min_dist / max_dist

    # Define bounds (points must be in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(n * d)]

    # Try multiple optimization methods
    best_result = None
    best_value = np.inf

    # Method 1: L-BFGS-B
    try:
        result1 = minimize(objective, initial_points.flatten(), method='L-BFGS-B', bounds=bounds,
                          options={'ftol': 1e-12, 'gtol': 1e-12})
        if result1.fun < best_value and result1.success:
            best_value = result1.fun
            best_result = result1
    except Exception as e:
        pass

    # Method 2: Nelder-Mead as fallback
    try:
        if best_result is None:
            result2 = minimize(objective, initial_points.flatten(), method='Nelder-Mead',
                              options={'fatol': 1e-12, 'xatol': 1e-12})
            if result2.fun < best_value and result2.success:
                best_value = result2.fun
                best_result = result2
    except Exception as e:
        pass

    # If no optimization succeeded, return original points
    if best_result is None:
        points = initial_points
    else:
        # Extract optimized points
        points = best_result.x.reshape(n, d)

    # Ensure points are within bounds
    points = np.clip(points, 0, 1)

    # Apply hill-climbing local search for further refinement
    points = _hill_climbing_refinement(points)

    return points


def _hill_climbing_refinement(points, max_iter=1000, perturbation_scale=0.005):
    """Apply hill-climbing local search to refine the point distribution."""
    n = points.shape[0]

    current_ratio = _calculate_min_max_ratio(points)

    for iteration in range(max_iter):
        # Try perturbing each point individually
        best_improvement = 0
        best_point_idx = -1
        best_new_points = None

        # Try perturbing each point
        for i in range(n):
            # Create a copy of current points
            temp_points = points.copy()

            # Perturb one point
            perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, 2)
            temp_points[i] += perturbation

            # Keep within bounds
            temp_points[i] = np.clip(temp_points[i], 0, 1)

            # Calculate new ratio
            new_ratio = _calculate_min_max_ratio(temp_points)

            # Check if this improves the ratio
            if new_ratio > current_ratio:
                improvement = new_ratio - current_ratio
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_point_idx = i
                    best_new_points = temp_points.copy()

        # If we found an improvement, update
        if best_improvement > 0:
            points = best_new_points
            current_ratio = _calculate_min_max_ratio(points)
        else:
            # No improvement found, reduce perturbation scale for finer search
            perturbation_scale *= 0.99

        # Early stopping if improvement is negligible
        if best_improvement < 1e-15:
            break

    return points


# EVOLVE-BLOCK-END