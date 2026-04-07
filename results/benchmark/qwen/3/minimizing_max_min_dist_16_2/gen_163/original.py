# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Compute pairwise distances
        distances = pdist(points)

        # Avoid division by zero
        if len(distances) == 0:
            return 0

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since minimize minimizes)
        if d_max == 0:
            return 0
        return -d_min / d_max

    def constraint_bounds(x):
        # Ensure all points are within [0,1] x [0,1]
        points = x.reshape(-1, 2)
        return np.concatenate([points.flatten(), (1-points.flatten())])

    # Start with a more intelligent initial configuration
    # Use a regular grid pattern with slight perturbation
    np.random.seed(42)
    grid_size = int(np.ceil(np.sqrt(16)))
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < 16:
                points.append([i/grid_size + np.random.normal(0, 0.05/grid_size),
                              j/grid_size + np.random.normal(0, 0.05/grid_size)])

    # Clip to ensure points stay in [0,1] x [0,1]
    points = np.clip(points, 0, 1)

    # Flatten initial guess
    x0 = points.flatten()

    # Set up bounds (each coordinate must be between 0 and 1)
    bounds = [(0, 1) for _ in range(32)]

    # Define constraints for bounds
    cons = [{'type': 'ineq', 'fun': lambda x: constraint_bounds(x)}]

    # Optimize
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 1000, 'ftol': 1e-9})
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            return optimized_points
        else:
            # If optimization fails, return the initial points
            return points
    except:
        # If anything goes wrong, return initial points
        return points


# EVOLVE-BLOCK-END