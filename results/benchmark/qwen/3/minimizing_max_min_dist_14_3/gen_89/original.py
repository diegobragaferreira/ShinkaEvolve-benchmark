# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape to points
        points = x.reshape(-1, 3)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Zero out diagonal
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (min/max)
        if d_max == 0:
            return -np.inf
        return -d_min / d_max

    def penalty_objective(x, penalty_weight=1e6):
        """Objective with penalty for constraint violations"""
        points = x.reshape(-1, 3)

        # Apply penalty for points outside unit sphere
        norms = np.linalg.norm(points, axis=1)
        penalty = penalty_weight * np.sum(np.maximum(0, norms - 1.0)**2)

        # Original objective
        obj_val = objective(x)

        return obj_val + penalty

    def normalize_points(points):
        """Normalize points to unit sphere"""
        norms = np.linalg.norm(points, axis=1)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms[:, np.newaxis]

    def generate_initial_configurations():
        """Generate multiple good initial configurations"""
        configs = []
        np.random.seed(42)

        # Method 1: Fibonacci sphere sampling
        n = 14
        points1 = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = np.arccos(y)  # angle from z-axis
            phi = (i * 2 * np.pi) / golden_ratio  # azimuthal angle

            x = radius * np.cos(phi)
            z = radius * np.sin(phi)

            points1.append([x, y, z])

        points1 = np.array(points1)
        points1 = normalize_points(points1)
        configs.append(points1)

        # Method 2: Random points in unit cube, then project to sphere
        points2 = np.random.rand(n, 3) * 2 - 1  # Range [-1, 1]
        points2 = normalize_points(points2)
        configs.append(points2)

        # Method 3: Perturbed Fibonacci points
        points3 = points1.copy()
        points3 += np.random.normal(0, 0.02, points3.shape)
        points3 = normalize_points(points3)
        configs.append(points3)

        # Method 4: Another random configuration with different seed
        np.random.seed(2468)
        points4 = np.random.rand(n, 3) * 2 - 1
        points4 = normalize_points(points4)
        configs.append(points4)

        return configs

    def differential_evolution_optimize(x0, max_iter=500):
        """Enhanced differential evolution optimization with adaptive parameters"""
        # Parameters for differential evolution - more adaptive approach
        popsize = 15  # Starting population size
        mutation_rate = 0.8
        crossover_rate = 0.7

        # Bounds for each dimension - allow slight deviation from unit sphere
        bounds = [(-1.5, 1.5)] * len(x0)

        # Custom callback to track progress
        def callback(xk, convergence):
            pass

        # Run differential evolution with enhanced settings
        result = differential_evolution(
            penalty_objective,
            bounds,
            args=(1e4,),  # penalty weight
            maxiter=max_iter,
            popsize=popsize,
            mutation=(mutation_rate, 0.9),
            recombination=crossover_rate,
            seed=42,
            callback=callback,
            disp=False,
            polish=False  # Skip polishing to save time
        )

        return result

    # Generate initial configurations
    initial_configs = generate_initial_configurations()

    best_result = None
    best_ratio = -np.inf
    best_points = None

    # Try multiple starting points with different initial configurations
    for i, config in enumerate(initial_configs):
        x0 = config.flatten()

        try:
            # First, use differential evolution for global optimization
            de_result = differential_evolution_optimize(x0, max_iter=300)

            if de_result.success:
                # Convert back to points and normalize
                optimized_points = de_result.x.reshape(-1, 3)
                optimized_points = normalize_points(optimized_points)

                # Calculate ratio directly
                distances = squareform(pdist(optimized_points))
                np.fill_diagonal(distances, np.inf)
                d_min = np.min(distances)
                d_max = np.max(distances)

                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = optimized_points.copy()

        except Exception as e:
            continue

    # If we didn't find any good solution, fall back to the best initial config
    if best_result is None:
        best_result = initial_configs[0]

    # Local refinement with L-BFGS on the best result so far
    x0_refine = best_result.flatten()

    try:
        # Refinement with L-BFGS-B for fine-tuning
        local_result = minimize(
            penalty_objective,
            x0_refine,
            method='L-BFGS-B',
            args=(1e3,),  # lower penalty weight for refinement
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if local_result.success:
            refined_points = local_result.x.reshape(-1, 3)
            refined_points = normalize_points(refined_points)

            # Final evaluation of refined solution
            distances = squareform(pdist(refined_points))
            np.fill_diagonal(distances, np.inf)
            d_min = np.min(distances)
            d_max = np.max(distances)

            if d_max > 0:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_result = refined_points

    except Exception as e:
        pass

    return best_result

# EVOLVE-BLOCK-END