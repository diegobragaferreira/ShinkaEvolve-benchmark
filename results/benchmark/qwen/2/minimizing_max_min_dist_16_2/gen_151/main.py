# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
import time

def compute_min_max_ratio(points):
    """Compute the minimum to maximum distance ratio for given points."""
    if len(points) < 2:
        return 0.0

    # Compute pairwise distances
    distances = cdist(points, points)

    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)

    # Find min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero
    if max_dist == 0:
        return 0.0

    return min_dist / max_dist

def initialize_points(n_points=16, method='adaptive_grid'):
    """Initialize points using a structured adaptive grid approach."""
    np.random.seed(42)

    if method == 'adaptive_grid':
        # Create a 4x4 grid with adaptive perturbations
        points = []
        rows, cols = 4, 4

        # Generate base grid points
        for i in range(rows):
            for j in range(cols):
                # Base grid positions
                x = j * (1.0 / (cols - 1)) if cols > 1 else 0.5
                y = i * (1.0 / (rows - 1)) if rows > 1 else 0.5

                # Adaptive perturbation - vary based on position
                if (i == 0 or i == rows-1) and (j == 0 or j == cols-1):
                    # Corner points - smallest perturbation
                    perturbation = 0.005
                elif i == 0 or i == rows-1 or j == 0 or j == cols-1:
                    # Edge points - medium perturbation
                    perturbation = 0.01
                else:
                    # Interior points - larger perturbation
                    perturbation = 0.02

                # Apply random perturbation
                x += np.random.normal(0, perturbation)
                y += np.random.normal(0, perturbation)
                points.append([x, y])

        # Ensure points are within bounds [0,1]
        points = np.clip(points, 0, 1)
        return np.array(points[:n_points])

    elif method == 'hexagonal':
        # Create hexagonal-like arrangement
        points = []
        rows, cols = 4, 4

        for i in range(rows):
            for j in range(cols):
                x = j * 0.3 + (i % 2) * 0.15
                y = i * 0.3
                x += np.random.normal(0, 0.015)
                y += np.random.normal(0, 0.015)
                points.append([x, y])

        points = np.clip(points, 0, 1)
        return np.array(points[:n_points])

    else:
        # Default random initialization
        return np.random.rand(n_points, 2)

def hybrid_optimization(initial_points, max_time=150):
    """Hybrid optimization combining global and local search strategies."""
    start_time = time.time()

    # Best solution tracking
    best_points = initial_points.copy()
    best_ratio = compute_min_max_ratio(initial_points)

    # Track performance to adapt strategy
    performance_history = []

    # Try multiple initialization approaches
    init_strategies = [
        ('adaptive_grid', 'adaptive_grid'),
        ('hexagonal', 'hexagonal'),
        ('random', 'random')
    ]

    # First-stage: Global optimization attempts
    for strategy_name, init_method in init_strategies:
        if (time.time() - start_time) >= max_time - 10:
            break

        try:
            # Generate initial points with this strategy
            current_points = initialize_points(16, init_method)

            # Define bounds
            bounds = [(0, 1) for _ in range(32)]

            # Objective function
            def objective(x):
                points = x.reshape(-1, 2)
                return -compute_min_max_ratio(points)

            # Global search using differential evolution
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                seed=42 + hash(strategy_name) % 1000,
                timeout=max_time - (time.time() - start_time)
            )

            # If DE succeeded, use its result; otherwise, use current initialization
            if de_result.success:
                global_points = de_result.x.reshape(-1, 2)
            else:
                global_points = current_points.copy()

            # Local refinement with L-BFGS-B
            try:
                lbfgs_result = minimize(
                    objective,
                    global_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12},
                    timeout=max_time - (time.time() - start_time)
                )

                if lbfgs_result.success:
                    refined_points = lbfgs_result.x.reshape(-1, 2)
                    refined_ratio = compute_min_max_ratio(refined_points)

                    # Apply final bound clipping
                    refined_points = np.clip(refined_points, 0, 1)
                    refined_ratio = compute_min_max_ratio(refined_points)

                    if refined_ratio > best_ratio:
                        best_ratio = refined_ratio
                        best_points = refined_points.copy()

            except Exception:
                # If local optimization fails, keep global result
                global_points = np.clip(global_points, 0, 1)
                global_ratio = compute_min_max_ratio(global_points)
                if global_ratio > best_ratio:
                    best_ratio = global_ratio
                    best_points = global_points.copy()

        except Exception as e:
            continue

        # Early stopping if we achieve good results
        if best_ratio > 0.3:
            break

    # Second-stage: Progressive refinement with adaptive parameters
    if (time.time() - start_time) < max_time - 10:
        # Try to further improve using a more intensive local method
        try:
            # Objective function with better numerical properties
            def objective_improved(x):
                points = x.reshape(-1, 2)
                return -compute_min_max_ratio(points)

            bounds = [(0, 1) for _ in range(32)]

            # Use SLSQP for potentially better constraint handling
            slsqp_result = minimize(
                objective_improved,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15},
                timeout=max_time - (time.time() - start_time)
            )

            if slsqp_result.success:
                final_points = slsqp_result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                final_ratio = compute_min_max_ratio(final_points)

                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()

        except Exception:
            pass

    # Third-stage: Adaptive restart with better initialization
    if (time.time() - start_time) < max_time - 10 and best_ratio < 0.3:
        # Try to restart with a new approach if we haven't achieved good results yet
        try:
            # Create a new better initialized configuration
            # Use a more careful initialization pattern
            better_init = []
            rows, cols = 4, 4

            # Distribute points more evenly with higher variance
            for i in range(rows):
                for j in range(cols):
                    x = j * (1.0 / (cols - 1)) if cols > 1 else 0.5
                    y = i * (1.0 / (rows - 1)) if rows > 1 else 0.5

                    # Higher perturbation for all points
                    perturbation = 0.03
                    x += np.random.normal(0, perturbation)
                    y += np.random.normal(0, perturbation)
                    better_init.append([x, y])

            better_init = np.clip(better_init, 0, 1)
            better_points = np.array(better_init[:16])

            # Refine with optimization
            def objective_refine(x):
                points = x.reshape(-1, 2)
                return -compute_min_max_ratio(points)

            bounds = [(0, 1) for _ in range(32)]

            # Use L-BFGS-B for this final refinement
            refine_result = minimize(
                objective_refine,
                better_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-13},
                timeout=max_time - (time.time() - start_time)
            )

            if refine_result.success:
                refined_points = refine_result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)

                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()

        except Exception:
            pass

    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Initialize with adaptive grid
    initial_points = initialize_points(n_points=16, method='adaptive_grid')

    # Optimize using hybrid approach
    optimized_points = hybrid_optimization(initial_points, max_time=150)

    return optimized_points

# EVOLVE-BLOCK-END