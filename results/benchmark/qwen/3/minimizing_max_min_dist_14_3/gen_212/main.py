# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple
from scipy.spatial import SphericalVoronoi

def initialize_points(n: int = 14, d: int = 3) -> np.ndarray:
    """
    Initialize points using an enhanced hybrid approach for superior distribution.

    Args:
        n: number of points
        d: dimensionality

    Returns:
        Initial point configuration
    """
    np.random.seed(42)

    # Strategy 1: Start with vertices of a regular icosahedron for excellent base distribution
    # Vertices of regular icosahedron scaled to unit sphere
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

    # Generate points on unit sphere using Fibonacci spiral
    def fibonacci_sphere_points(n_points):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    # If we need more than 12 points, use Fibonacci spiral for additional points
    if n <= 12:
        points = vertices[:n].copy()
    else:
        # Start with icosahedron vertices and add Fibonacci spiral points for better coverage
        points = vertices.copy()

        # Add remaining points using improved Fibonacci spiral
        extra_points_needed = n - 12
        fib_points = fibonacci_sphere_points(extra_points_needed)

        # Add some randomness to break symmetry while maintaining good distribution
        fib_points = fib_points + np.random.normal(0, 0.01, fib_points.shape)

        # Combine with existing points
        points = np.vstack([points, fib_points])

    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.max(norms)

    # Scale and shift to [0,1]^3
    points = points * 0.5 + 0.5

    # Add additional perturbations for even better distribution
    points += np.random.normal(0, 0.003, points.shape)

    # Ensure we don't go outside bounds due to perturbations
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
    Optimize point configuration using adaptive differential evolution with enhanced strategies.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Optimized point configuration
    """
    start_time = time.time()

    # Try multiple strategies with different parameters to find the best approach
    strategies = [
        # Strategy 1: Conservative approach
        {'popsize': 15, 'maxiter': 200, 'mutation': (0.5, 0.8), 'recombination': 0.7},
        # Strategy 2: Balanced approach
        {'popsize': 20, 'maxiter': 150, 'mutation': (0.6, 0.9), 'recombination': 0.8},
        # Strategy 3: Aggressive approach for better exploration
        {'popsize': 25, 'maxiter': 100, 'mutation': (0.7, 1.0), 'recombination': 0.9}
    ]

    best_points = initial_points.copy()
    best_ratio = -float('inf')
    best_result = None

    for i, strategy in enumerate(strategies):
        try:
            # Flatten initial points for optimization
            initial_flat = initial_points.flatten()

            # Define bounds for each coordinate (0 to 1)
            bounds = [(0.0, 1.0)] * len(initial_flat)

            # Run differential evolution with current strategy
            result = differential_evolution(
                objective_function_with_penalty,
                bounds,
                maxiter=strategy['maxiter'],
                popsize=strategy['popsize'],
                tol=1e-8,
                mutation=strategy['mutation'],
                recombination=strategy['recombination'],
                seed=42 + i,  # Different seed for each strategy
                disp=False
            )

            # Reshape result and validate
            optimized_points = result.x.reshape(14, 3)
            optimized_points = np.clip(optimized_points, 0, 1)

            # Evaluate this result
            min_dist, max_dist = calculate_distance_metrics(optimized_points)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    best_result = result

        except Exception:
            continue

    # If no strategy was successful, return the initial points
    if best_result is None:
        return initial_points

    return best_points

def local_refinement(initial_points: np.ndarray, max_time: float = 60.0) -> np.ndarray:
    """
    Perform enhanced local refinement using multiple strategies.

    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Refined point configuration
    """
    # Try multiple local refinement approaches
    refinement_strategies = [
        # Strategy 1: Standard L-BFGS with tight tolerances
        {'method': 'L-BFGS-B', 'options': {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}},
        # Strategy 2: L-BFGS with moderate tolerances (faster)
        {'method': 'L-BFGS-B', 'options': {'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}},
        # Strategy 3: Nelder-Mead as backup (robust method)
        {'method': 'Nelder-Mead', 'options': {'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}}
    ]

    best_points = initial_points.copy()
    best_ratio = -float('inf')

    for strategy in refinement_strategies:
        try:
            x0 = initial_points.flatten()
            bounds = [(0.0, 1.0)] * len(x0)

            # Apply refinement strategy
            result = minimize(
                objective_function_with_penalty,
                x0,
                method=strategy['method'],
                bounds=bounds if strategy['method'] == 'L-BFGS-B' else None,
                options=strategy['options']
            )

            if result.success:
                refined_points = result.x.reshape(14, 3)
                refined_points = np.clip(refined_points, 0, 1)

                # Evaluate result
                min_dist, max_dist = calculate_distance_metrics(refined_points)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()

        except Exception:
            continue

    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Multiple initialization attempts to find a good starting point
    best_initial = None
    best_initial_ratio = -float('inf')

    for attempt in range(3):
        try:
            initial_points = initialize_points(14, 3)
            min_dist, max_dist = calculate_distance_metrics(initial_points)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_initial_ratio:
                    best_initial_ratio = ratio
                    best_initial = initial_points.copy()
        except Exception:
            continue

    # If no good initialization found, use default
    if best_initial is None:
        np.random.seed(42)
        best_initial = np.random.rand(14, 3)

    # Phase 1: Global optimization with adaptive differential evolution
    global_optimized = adaptive_differential_evolution(best_initial)

    # Phase 2: Local refinement with enhanced strategies
    local_optimized = local_refinement(global_optimized)

    # Phase 3: Extra refinement attempt with different approach
    extra_refinement = local_refinement(local_optimized)

    # Phase 4: Final validation and selection
    final_points = local_optimized.copy()
    final_ratio = -float('inf')

    # Compare all candidates and select the best
    candidates = [global_optimized, local_optimized, extra_refinement]

    for candidate in candidates:
        min_dist, max_dist = calculate_distance_metrics(candidate)
        if max_dist > 0:
            ratio = min_dist / max_dist
            if ratio > final_ratio:
                final_ratio = ratio
                final_points = candidate.copy()

    # Final verification
    min_dist, max_dist = calculate_distance_metrics(final_points)

    # If optimization failed, fall back to our best initialization
    if max_dist <= 0 or min_dist <= 0:
        final_points = best_initial.copy()

    return final_points

# EVOLVE-BLOCK-END