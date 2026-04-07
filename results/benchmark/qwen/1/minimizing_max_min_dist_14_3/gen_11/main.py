# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)

        # Compute pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio (since we want to maximize)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist

    def constraint(x):
        # Ensure all points are within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1.0 - norms  # Positive if inside unit sphere

    # Start with a reasonable initial configuration - vertices of a cube
    # Plus some additional points to make 14 total
    initial_points = np.array([
        # Cube vertices
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        # Additional points on sphere
        [0, 0, 1], [0, 0, -1],
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0]
    ])

    # Normalize to unit sphere
    norms = np.linalg.norm(initial_points, axis=1)
    initial_points = initial_points / np.max(norms) * 0.9

    # Flatten for optimization
    x0 = initial_points.flatten()

    # Define constraints
    cons = {'type': 'ineq', 'fun': constraint}

    # Optimize
    result = minimize(objective, x0, method='SLSQP', constraints=cons,
                      options={'maxiter': 1000, 'ftol': 1e-9})

    # Extract optimized points
    optimized_points = result.x.reshape(-1, 3)

    # Further refine by ensuring all points are within unit sphere
    norms = np.linalg.norm(optimized_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        optimized_points[mask] = optimized_points[mask] / norms[mask][:, np.newaxis]

    return optimized_points


# EVOLVE-BLOCK-END