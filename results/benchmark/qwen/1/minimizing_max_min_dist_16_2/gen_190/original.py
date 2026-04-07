# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances
        distances = pdist(points)

        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        if len(distances) == 0 or np.allclose(distances, 0):
            return -1.0  # Worst possible case

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -1.0

        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points are within [0,1] x [0,1]
        points = x.reshape(-1, 2)

        # Check that all points are within bounds
        violations = []

        # x coordinates in [0,1]
        violations.append(np.min(points[:, 0]))  # Should be >= 0
        violations.append(1 - np.max(points[:, 0]))  # Should be >= 0

        # y coordinates in [0,1]
        violations.append(np.min(points[:, 1]))  # Should be >= 0
        violations.append(1 - np.max(points[:, 1]))  # Should be >= 0

        return np.array(violations)

    def bounded_objective(x):
        # Simple boundary checking - clamp points to bounds
        points = np.clip(x.reshape(-1, 2), 0, 1).flatten()
        return objective(points)

    # Create a better initial configuration based on hexagonal packing
    # For 16 points, we can try a grid-like arrangement that's close to optimal
    np.random.seed(42)

    # Try a structured initialization first to get closer to good solution
    initial_points = np.zeros((16, 2))

    # Create a semi-structured pattern
    x_vals = np.linspace(0.05, 0.95, 4)
    y_vals = np.linspace(0.05, 0.95, 4)

    idx = 0
    for i in range(4):
        for j in range(4):
            if idx < 16:
                initial_points[idx] = [x_vals[i], y_vals[j]]
                idx += 1

    # Add some randomness to avoid getting stuck in symmetric local optima
    initial_points += np.random.normal(0, 0.01, (16, 2))

    # Clamp to bounds
    initial_points = np.clip(initial_points, 0, 1)

    # Flatten for optimization
    x0 = initial_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(32)]

    # First use differential evolution for global search
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        de_result = differential_evolution(
            bounded_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6
        )

    # If DE didn't work well, fall back to local optimization
    if de_result.success and -de_result.fun > 0.1:  # If we found something decent
        x0 = de_result.x
    else:
        # Try a few different starting configurations
        best_x = x0
        best_value = objective(x0)

        for attempt in range(3):
            # Random perturbation
            perturbed = x0 + np.random.normal(0, 0.05, 32)
            # Clamp to bounds
            perturbed = np.clip(perturbed, 0, 1)

            # Local optimization
            local_result = minimize(
                bounded_objective,
                perturbed,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )

            if local_result.success:
                value = objective(local_result.x)
                if value < best_value:  # Since we're minimizing negative ratio
                    best_value = value
                    best_x = local_result.x

        x0 = best_x

    # Final local optimization with L-BFGS-B for fine-tuning
    final_result = minimize(
        bounded_objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}
    )

    # Extract optimized points
    optimized_points = final_result.x.reshape(-1, 2)

    # Ensure all points are within bounds
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points


# EVOLVE-BLOCK-END