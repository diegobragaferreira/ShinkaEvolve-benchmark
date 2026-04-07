# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple

def initialize_points(n: int = 14, d: int = 3) -> np.ndarray:
    """
    Initialize points using a spherical geometry approach for better distribution.

    Args:
        n: number of points
        d: dimensionality

    Returns:
        Initial point configuration
    """
    np.random.seed(42)

    # Use a more sophisticated spherical initialization approach
    # Generate points on a unit sphere using Fibonacci-based method but with better spacing
    points = []
    for i in range(n):
        # Improved Fibonacci spiral with better distribution
        phi = np.arccos(1 - 2 * (i / (n - 1)))
        theta = np.sqrt(n) * phi

        # Add small perturbation to avoid perfect symmetry
        theta += np.random.normal(0, 0.1)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])

    points = np.array(points)

    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.max(norms)

    # Scale and shift to [0,1]^3
    points = points * 0.5 + 0.5

    # Add small random perturbation to avoid symmetry issues
    points += np.random.normal(0, 0.005, points.shape)

    # Clip to ensure within bounds
    points = np.clip(points, 0, 1)

    return points

def calculate_distance_metrics(points: np.ndarray) -> tuple[float, float]:
    """
    Calculate minimum and maximum distances between all point pairs.

    Args:
        points: Array of shape (n, d)

    Returns:
        Tuple of (min_distance, max_distance)
    """
    if len(points) < 2:
        return 0.0, 0.0

    distances = pdist(points)

    if len(distances) == 0:
        return 0.0, 0.0

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    return min_dist, max_dist

def objective_function_with_penalty(points_flat: np.ndarray, penalty_weight: float = 1000.0) -> float:
    """
    Objective function with penalty for constraint violations.

    Args:
        points_flat: Flattened array of point coordinates
        penalty_weight: Weight for constraint penalty

    Returns:
        Objective value to minimize
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)

    # Compute penalty for boundary violations
    penalty = 0.0
    for i, coord in enumerate(points.flat):
        if coord < 0:
            penalty += penalty_weight * (0 - coord)**2
        elif coord > 1:
            penalty += penalty_weight * (coord - 1)**2

    # Calculate distances
    distances = pdist(points)

    if len(distances) == 0:
        return penalty + float('inf')

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero
    if max_dist <= 0:
        return penalty + float('inf')

    # Return negative ratio plus penalty to minimize (maximize the ratio)
    ratio = -min_dist / max_dist
    return ratio + penalty

def adaptive_differential_evolution(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Optimize point configuration using adaptive differential evolution.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Optimized point configuration
    """
    start_time = time.time()

    # Flatten initial points for optimization
    initial_flat = initial_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0.0, 1.0)] * len(initial_flat)

    # Initialize adaptive parameters
    current_popsize = 15
    maxiter = 500
    stagnation_counter = 0
    best_value = float('inf')
    stagnation_threshold = 10
    max_stagnation = 20

    # History tracking for convergence
    history = []

    try:
        # Run multiple rounds of DE with adaptive population size
        for round_num in range(3):
            # Check if we've exceeded time limits or stagnated
            if len(history) > 0 and len(history) % 50 == 0:
                # Check for convergence
                if len(history) >= 2:
                    recent_change = abs(history[-1] - history[-2])
                    if recent_change < 1e-8:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0

                # Increase population size if stagnated
                if stagnation_counter >= stagnation_threshold and current_popsize < 30:
                    current_popsize = min(current_popsize + 5, 30)
                    stagnation_counter = 0  # Reset counter after adjustment

                # Stop if too many stagnations
                if stagnation_counter >= max_stagnation:
                    break

            # Run differential evolution with current settings
            result = differential_evolution(
                objective_function_with_penalty,
                bounds,
                maxiter=maxiter // 3,
                popsize=current_popsize,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False
            )

            # Track history
            history.append(result.fun)

            # Update best value
            if result.fun < best_value:
                best_value = result.fun
                stagnation_counter = 0  # Reset stagnation counter on improvement

            # Early stopping based on improvement
            if len(history) >= 2 and abs(history[-1] - history[-2]) < 1e-9:
                break

        # Reshape optimized result
        optimized_points = result.x.reshape(14, 3)

        # Ensure all points are within valid range
        optimized_points = np.clip(optimized_points, 0, 1)

        return optimized_points

    except Exception:
        # Return the initial points if optimization fails
        return initial_points

def local_refinement(initial_points: np.ndarray, max_time: float = 60.0) -> np.ndarray:
    """
    Perform local refinement using L-BFGS-B.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Refined point configuration
    """
    try:
        x0 = initial_points.flatten()
        bounds = [(0.0, 1.0)] * len(x0)

        # More aggressive tolerance for better local optimization
        result = minimize(
            objective_function_with_penalty,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
        )

        refined_points = result.x.reshape(14, 3)
        refined_points = np.clip(refined_points, 0, 1)
        return refined_points

    except Exception:
        return initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Phase 1: Initialize points with better strategy
    initial_points = initialize_points(14, 3)

    # Phase 2: Global optimization with adaptive differential evolution
    global_optimized = adaptive_differential_evolution(initial_points)

    # Phase 3: Local refinement with L-BFGS-B
    local_optimized = local_refinement(global_optimized)

    # Phase 4: Final validation and adjustment
    final_points = local_optimized.copy()

    # Calculate final metrics
    min_dist, max_dist = calculate_distance_metrics(final_points)

    # If optimization didn't work well, fall back to a good known arrangement
    if max_dist <= 0 or min_dist <= 0:
        # Fallback to regularized arrangement
        np.random.seed(42)
        final_points = np.random.rand(14, 3)

    return final_points

# EVOLVE-BLOCK-END