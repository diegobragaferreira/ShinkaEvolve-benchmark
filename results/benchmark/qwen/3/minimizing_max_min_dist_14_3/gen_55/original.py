# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Set random seed for reproducibility
    np.random.seed(42)

    # Run optimization with reasonable settings
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=1000,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False
    )

    # Extract the best solution
    points = result.x.reshape(-1, 3)

    return points


# EVOLVE-BLOCK-END