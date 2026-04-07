# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from itertools import combinations

def fibonacci_sphere(n: int) -> np.ndarray:
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

def initialize_points(n: int = 14, d: int = 3, num_starts: int = 3) -> np.ndarray:
    """
    Initialize points using multiple strategies for better starting configuration.

    Args:
        n: number of points
        d: dimensionality
        num_starts: number of different initializations to try

    Returns:
        Best initial point configuration
    """
    best_points = None
    best_ratio = -float('inf')

    for start_idx in range(num_starts):
        # Different initialization methods
        if start_idx == 0:
            # Fibonacci sphere initialization
            points = fibonacci_sphere(n)
            # Scale to unit cube [0,1]^3
            points = (points + 1) / 2  # map from [-1,1] to [0,1]

        elif start_idx == 1:
            # Random initialization with seed
            np.random.seed(42 + start_idx)
            points = np.random.rand(n, d)

        else:
            # Perturbed Fibonacci points
            points = fibonacci_sphere(n)
            points = (points + 1) / 2
            # Add small perturbation
            np.random.seed(42 + start_idx)
            points += np.random.normal(0, 0.005, points.shape)
            # Clip to valid range
            points = np.clip(points, 0, 1)

        # Calculate initial ratio
        distances = pdist(points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()

    # Fallback to random if nothing worked
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.rand(n, d)

    return best_points

def calculate_distance_metrics(points: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate minimum, maximum, and ratio of distances between all point pairs.

    Args:
        points: Array of shape (n, d)

    Returns:
        Tuple of (min_distance, max_distance, ratio)
    """
    distances = pdist(points)

    if len(distances) == 0:
        return 0.0, 0.0, 0.0

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    if max_dist <= 0:
        return 0.0, 0.0, 0.0

    ratio = min_dist / max_dist
    return min_dist, max_dist, ratio

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

def adaptive_optimization(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Perform adaptive optimization with changing population sizes and strategies.

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

    # Track convergence history
    prev_best = float('inf')
    stagnant_count = 0
    max_stagnant = 10
    population_history = []
    current_popsize = 15

    def adaptive_objective(x_flat):
        result = objective_function(x_flat)
        population_history.append(result)
        return result

    # Run multiple optimization phases with adaptive population sizing
    for phase in range(5):
        if time.time() - start_time > max_time * 0.95:  # Leave some time for final processing
            break

        try:
            # Adjust population size based on convergence
            if len(population_history) >= 2:
                recent_improvement = abs(population_history[-1] - population_history[-2])
                if recent_improvement < 1e-8:
                    stagnant_count += 1
                    if stagnant_count >= max_stagnant and current_popsize < 30:
                        current_popsize = min(current_popsize + 5, 30)
                        stagnant_count = 0  # Reset after increase
                else:
                    stagnant_count = 0
            else:
                stagnant_count = 0

            # Run differential evolution with current population size
            result = differential_evolution(
                adaptive_objective,
                bounds,
                maxiter=200,  # Reduced iterations per phase
                popsize=current_popsize,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42 + phase,
                disp=False
            )

            # Update best value
            prev_best = result.fun

        except Exception as e:
            # Continue with next phase if current one fails
            continue

    # Reshape optimized result
    optimized_points = result.x.reshape(14, 3)

    # Ensure all points are within valid range
    optimized_points = np.clip(optimized_points, 0, 1)

    # Additional local refinement using L-BFGS-B
    def local_objective(x_flat):
        points = x_flat.reshape(-1, 3)
        points = np.clip(points, 0, 1)
        distances = pdist(points)
        if len(distances) == 0:
            return float('inf')
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 0:
            return float('inf')
        return -min_dist / max_dist

    try:
        x0 = optimized_points.flatten()
        local_result = minimize(
            local_objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        optimized_points = local_result.x.reshape(14, 3)
        optimized_points = np.clip(optimized_points, 0, 1)
    except:
        pass

    return optimized_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Phase 1: Initialize points with multiple strategies
    initial_points = initialize_points(14, 3, num_starts=5)

    # Phase 2: Optimize points with adaptive strategy
    optimized_points = adaptive_optimization(initial_points)

    # Phase 3: Final validation and adjustment
    final_points = optimized_points.copy()

    # Calculate final metrics
    min_dist, max_dist, ratio = calculate_distance_metrics(final_points)

    # If optimization didn't work well, fall back to a good known arrangement
    if max_dist <= 0 or min_dist <= 0 or ratio < 0.1:
        # Fallback to regularized arrangement
        np.random.seed(42)
        final_points = np.random.rand(14, 3)

        # Try one more optimization pass with better parameters if needed
        try:
            bounds = [(0.0, 1.0)] * (14 * 3)
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=500,
                popsize=25,
                tol=1e-8,
                mutation=(0.8, 1.0),
                recombination=0.9,
                seed=42,
                disp=False
            )
            final_points = result.x.reshape(14, 3)
            final_points = np.clip(final_points, 0, 1)
        except:
            pass

    # Final validation
    _, _, final_ratio = calculate_distance_metrics(final_points)
    if final_ratio < 0.05:  # Very poor result, use another fallback
        np.random.seed(42)
        final_points = np.random.rand(14, 3)

    return final_points

# EVOLVE-BLOCK-END