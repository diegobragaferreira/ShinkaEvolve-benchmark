# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

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

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero and penalize out-of-bounds points
        if d_max <= 1e-12:
            return -np.inf

        ratio = d_min / d_max

        # Add penalty for points outside [0,1] bounds
        penalty = 0
        for i in range(16):
            if points[i, 0] < 0 or points[i, 0] > 1 or points[i, 1] < 0 or points[i, 1] > 1:
                penalty += 10000

        # Return negative ratio to minimize (since we want to maximize the ratio)
        return -(ratio - penalty / 100000)

    # Create bounds for each coordinate (0 to 1 for both x and y)
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

    # Initialize with a better starting configuration (improved hexagonal pattern)
    np.random.seed(42)

    # Create a more sophisticated initial configuration inspired by optimal packings
    initial_points = np.zeros((16, 2))

    # Generate points in a pattern that's more evenly distributed
    # Using a modified hexagonal lattice that avoids perfect grid alignment
    idx = 0
    for i in range(4):
        for j in range(4):
            # Create hexagonal offset pattern
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + offset) / 3.0
            y = i / 3.0

            # Add controlled random perturbation to break symmetry
            x += np.random.normal(0, 0.015)
            y += np.random.normal(0, 0.015)

            # Ensure within bounds
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            initial_points[idx] = [x, y]
            idx += 1

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    try:
        # Stage 1: Fast global search with differential evolution
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=30,  # Reduced iterations for speed
            popsize=8,   # Smaller population for faster execution
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            callback=None
        )

        # Stage 2: Local refinement with L-BFGS-B
        refined_result = minimize(
            objective,
            result_de.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-9},
            callback=None
        )

        final_result = refined_result.x

    except Exception as e:
        # Fallback to initial configuration
        print(f"Optimization failed: {e}")
        final_result = initial_flat

    # Convert back to points array
    points = final_result.reshape(-1, 2)

    # Ensure all points are within bounds
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)

    return points

# EVOLVE-BLOCK-END