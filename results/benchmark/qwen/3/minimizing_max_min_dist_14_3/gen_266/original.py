# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple

def initialize_points(n: int = 14, d: int = 3) -> np.ndarray:
    """
    Initialize points using an enhanced Fibonacci-based spherical approach with adaptive distribution.

    Args:
        n: number of points
        d: dimensionality

    Returns:
        Initial point configuration
    """
    np.random.seed(42)

    # Strategy 1: Enhanced Fibonacci spiral on sphere for even distribution
    points = []
    for i in range(n):
        # Improved Fibonacci spiral with better spacing and reduced clustering
        phi = np.arccos(1 - 2 * (i / (n - 1)))
        theta = np.sqrt(n) * phi

        # Add structured perturbation to avoid perfect symmetry while maintaining good distribution
        perturbation = np.random.normal(0, 0.05) if i % 3 == 0 else np.random.normal(0, 0.02)
        theta += perturbation

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
    points += np.random.normal(0, 0.01, points.shape)

    # Clip to ensure within bounds
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

    distances = pdist(points)

    if len(distances) == 0:
        return 0.0, 0.0

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    return min_dist, max_dist

def objective_function_with_penalty(points_flat: np.ndarray, penalty_weight: float = 1000.0) -> float:
    """
    Objective function with penalty for constraint violations and improved optimization behavior.

    Args:
        points_flat: Flattened array of point coordinates
        penalty_weight: Weight for constraint penalty

    Returns:
        Objective value to minimize
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)

    # Compute penalty for boundary violations with more nuanced handling
    penalty = 0.0
    for i, coord in enumerate(points.flat):
        if coord < 0:
            penalty += penalty_weight * (0 - coord)**2
        elif coord > 1:
            penalty += penalty_weight * (coord - 1)**2

    # Calculate distances with improved numerical stability
    distances = pdist(points)

    if len(distances) == 0:
        return penalty + float('inf')

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero with minimum threshold
    if max_dist <= 1e-12:
        return penalty + float('inf')

    # Return negative ratio plus penalty to minimize (maximize the ratio)
    # Add small epsilon to prevent numerical instability
    ratio = -min_dist / (max_dist + 1e-12)
    return ratio + penalty

def adaptive_differential_evolution(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Optimize point configuration using adaptive differential evolution with enhanced convergence detection.

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

    # Initialize adaptive parameters with improved defaults
    current_popsize = 20  # Increased population size for better diversity
    maxiter = 500
    stagnation_counter = 0
    best_value = float('inf')
    stagnation_threshold = 15  # More conservative stagnation detection
    max_stagnation = 25
    improvement_threshold = 1e-8

    # History tracking for convergence
    history = []
    improvement_history = []

    try:
        # Run multiple rounds of DE with adaptive population size and better termination conditions
        for round_num in range(5):  # Increased rounds for better exploration
            # Check if we've exceeded time limits or stagnated
            if len(history) > 0 and len(history) % 30 == 0:
                # Check for convergence and improvement
                if len(history) >= 2:
                    recent_change = abs(history[-1] - history[-2])
                    improvement_history.append(recent_change)

                    # Track latest improvements
                    if len(improvement_history) > 10:
                        improvement_history.pop(0)

                    # Adjust based on recent improvement trend
                    avg_improvement = np.mean(improvement_history) if improvement_history else 0
                    if avg_improvement < improvement_threshold:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0

                # Increase population size if stagnated
                if stagnation_counter >= stagnation_threshold and current_popsize < 35:
                    current_popsize = min(current_popsize + 5, 35)
                    stagnation_counter = 0  # Reset counter after adjustment

                # Stop if too many stagnations
                if stagnation_counter >= max_stagnation:
                    break

            # Run differential evolution with improved parameters
            result = differential_evolution(
                objective_function_with_penalty,
                bounds,
                maxiter=maxiter // 5,  # Reduced iterations per round
                popsize=current_popsize,
                tol=1e-7,  # Tighter tolerance
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False,
                strategy='best1bin'  # Use best1bin for better convergence
            )

            # Track history
            history.append(result.fun)

            # Update best value
            if result.fun < best_value:
                best_value = result.fun
                stagnation_counter = 0  # Reset stagnation counter on improvement

            # Early stopping based on significant improvement
            if len(history) >= 2 and abs(history[-1] - history[-2]) < 1e-10:
                break

        # Reshape optimized result
        optimized_points = result.x.reshape(14, 3)

        # Ensure all points are within valid range
        optimized_points = np.clip(optimized_points, 0, 1)

        return optimized_points

    except Exception as e:
        # Return the initial points if optimization fails
        return initial_points

def local_refinement(initial_points: np.ndarray, max_time: float = 60.0) -> np.ndarray:
    """
    Perform local refinement using L-BFGS-B with enhanced parameters.

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
            options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12, 'eps': 1e-8}
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

    # Phase 1: Initialize points with enhanced strategy
    initial_points = initialize_points(14, 3)

    # Phase 2: Global optimization with adaptive differential evolution
    global_optimized = adaptive_differential_evolution(initial_points, max_time=300.0)

    # Phase 3: Local refinement with L-BFGS-B
    local_optimized = local_refinement(global_optimized, max_time=50.0)

    # Phase 4: Final validation and adjustment
    final_points = local_optimized.copy()

    # Calculate final metrics
    min_dist, max_dist = calculate_distance_metrics(final_points)

    # If optimization didn't work well, fall back to a good known arrangement
    if max_dist <= 1e-12 or min_dist <= 1e-12:
        # Fallback to regularized arrangement with better quality initialization
        np.random.seed(42)
        fallback_points = np.random.rand(14, 3)

        # Try a more structured fallback using Fibonacci-like approach
        fib_points = []
        for i in range(14):
            phi = np.arccos(1 - 2 * (i / (14 - 1)))
            theta = np.sqrt(14) * phi
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            fib_points.append([x, y, z])

        fib_points = np.array(fib_points) * 0.5 + 0.5
        fib_points += np.random.normal(0, 0.01, fib_points.shape)
        fib_points = np.clip(fib_points, 0, 1)

        # Compare and use better of fallback approaches
        _, max_dist_fallback = calculate_distance_metrics(fallback_points)
        _, max_dist_fib = calculate_distance_metrics(fib_points)

        final_points = fib_points if max_dist_fib > max_dist_fallback else fallback_points

    return final_points

# EVOLVE-BLOCK-END