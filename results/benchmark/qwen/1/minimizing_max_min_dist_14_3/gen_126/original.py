# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute pairwise distances
        distances = cdist(points, points)
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Minimize negative of minimum distance (maximize minimum distance)
        return -np.min(distances)

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_max_distance(x):
        # Ensure maximum distance doesn't exceed some reasonable bound
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, 0)
        max_dist = np.max(distances)
        return 2 - max_dist  # Should be >= 0 (allowing up to diameter 2)

    n = 14
    d = 3

    # Start with an initial guess based on known good configurations
    # Use a icosahedron-based approach for better starting point
    np.random.seed(42)

    # Initialize points on unit sphere using Fibonacci spiral method
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        theta = np.arccos(1 - 2*(i/(n-1)))
        phi = np.arctan2(np.sin(i * 2 * np.pi / golden_ratio), np.cos(i * 2 * np.pi / golden_ratio))
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points.append([x, y, z])

    points = np.array(points)

    # Flatten for optimization
    x0 = points.flatten()

    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_sphere},
        {'type': 'ineq', 'fun': constraint_max_distance}
    ]

    # Optimize
    result = minimize(objective, x0, method='SLSQP', constraints=cons,
                      options={'ftol': 1e-8, 'maxiter': 1000})

    # Return optimized points
    optimized_points = result.x.reshape(-1, 3)

    return optimized_points


# EVOLVE-BLOCK-END