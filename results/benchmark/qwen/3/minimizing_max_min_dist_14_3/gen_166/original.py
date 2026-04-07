# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist, squareform
import time

def initialize_points(n: int = 14, d: int = 3) -> np.ndarray:
    """
    Initialize points using Fibonacci spiral on sphere for better starting configuration.

    Args:
        n: number of points
        d: dimensionality

    Returns:
        Initial point configuration
    """
    # Use Fibonacci spiral for more even distribution on sphere
    points = []
    for i in range(n):
        phi = np.arccos(1 - 2 * (i / (n - 1)))
        theta = np.sqrt(n) * phi
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])

    points = np.array(points)

    # Scale to unit cube [0,1]^3
    # First normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.max(norms)
    # Then scale and shift to [0,1]^3
    points = points * 0.5 + 0.5

    # Add small random perturbation to escape local optima
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)

    return points

def calculate_distance_metrics(points: np.ndarray) -> tuple[float, float]:
    """
    Calculate minimum and maximum distances between all point pairs.

    Args:
        points: Array of shape (n, d)

    Returns:
        Tuple of (min_distance, max_distance)
    """
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return min_dist, max_dist

def objective_function(points_flat: np.ndarray) -> float:
    """
    Objective function to maximize the min/max distance ratio.
    Returns negative ratio since optimizers minimize by default.

    Args:
        points_flat: Flattened array of point coordinates

    Returns:
        Negative min/max ratio (to be minimized)
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)

    # Ensure points are within bounds [0,1]^3
    points = np.clip(points, 0, 1)

    # Calculate distances
    distances = pdist(points)

    if len(distances) == 0:
        return float('inf')

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero
    if max_dist <= 0:
        return float('inf')

    # Return negative ratio to minimize (maximize the ratio)
    return -min_dist / max_dist

def optimize_points(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Optimize point configuration using differential evolution with adaptive population sizing.

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

    # Set base optimization options
    opt_options = {
        'maxiter': 1000,
        'popsize': 15,
        'tol': 1e-6,
        'mutation': (0.5, 1.0),
        'recombination': 0.7
    }

    # Track convergence history for adaptive population sizing
    prev_best = float('inf')
    stagnant_count = 0
    max_stagnant = 10
    population_history = []

    def adaptive_objective(x_flat):
        # Wrapper for objective that tracks population size
        result = objective_function(x_flat)
        population_history.append(result)
        return result

    # Perform optimization with adaptive population sizing
    # Start with base population size
    current_popsize = opt_options['popsize']

    # Custom implementation of DE with adaptive population size
    from scipy.optimize import differential_evolution

    # We'll run DE with increasing population sizes if convergence stalls
    result = None
    for iteration in range(5):  # Limited iterations to prevent excessive runtime
        try:
            # Adjust population size based on convergence
            if len(population_history) >= 2:
                recent_improvement = abs(population_history[-1] - population_history[-2])
                if recent_improvement < 1e-8:
                    stagnant_count += 1
                    if stagnant_count >= max_stagnant and current_popsize < 30:
                        current_popsize = min(current_popsize + 5, 30)
                else:
                    stagnant_count = 0
            else:
                stagnant_count = 0

            # Run differential evolution with current population size
            result = differential_evolution(
                adaptive_objective,
                bounds,
                maxiter=opt_options['maxiter'] // 5,  # Divide iterations among sub-runs
                popsize=current_popsize,
                tol=opt_options['tol'],
                mutation=opt_options['mutation'],
                recombination=opt_options['recombination'],
                seed=42,
                disp=False
            )

            # Update best value
            prev_best = result.fun

        except Exception:
            break

    # If we couldn't get a good result, use standard approach
    if result is None:
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=opt_options['maxiter'],
            popsize=opt_options['popsize'],
            tol=opt_options['tol'],
            mutation=opt_options['mutation'],
            recombination=opt_options['recombination'],
            seed=42,
            disp=False
        )

    # Reshape optimized result
    optimized_points = result.x.reshape(14, 3)

    # Ensure all points are within valid range
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Phase 1: Initialize points
    initial_points = initialize_points(14, 3)

    # Phase 2: Optimize points
    optimized_points = optimize_points(initial_points)

    # Phase 3: Final validation and adjustment
    final_points = optimized_points.copy()

    # Calculate final metrics
    min_dist, max_dist = calculate_distance_metrics(final_points)

    # If optimization didn't work well, fall back to a good known arrangement
    if max_dist <= 0 or min_dist <= 0:
        # Fallback to regularized arrangement
        np.random.seed(42)
        final_points = np.random.rand(14, 3)

    return final_points

# EVOLVE-BLOCK-END