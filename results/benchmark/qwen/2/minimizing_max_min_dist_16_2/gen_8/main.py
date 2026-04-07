# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape the flat array back to points
        points = x.reshape(-1, 2)

        # Compute pairwise distances
        distances = pdist(points)

        # Calculate minimum and maximum distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to minimize (since we want to maximize the ratio)
        # Add small epsilon to avoid division by zero
        if d_max < 1e-10:
            return -np.inf
        return -(d_min / d_max)

    # Initial configuration - place points in a grid-like pattern
    initial_points = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
        [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3],
        [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5],
        [0.1, 0.7], [0.3, 0.7], [0.5, 0.7], [0.7, 0.7], [0.9, 0.7],
        [0.1, 0.9], [0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9]
    ])

    # Take first 16 points and make sure they're within bounds
    initial_points = initial_points[:16]
    initial_points = np.clip(initial_points, 0, 1)

    # Flatten for optimization
    x0 = initial_points.flatten()

    # Define bounds (points must remain in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]

    # Perform optimization
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        atol=1e-6,
        rtol=1e-6
    )

    # Return optimized points
    return result.x.reshape(-1, 2)


# EVOLVE-BLOCK-END