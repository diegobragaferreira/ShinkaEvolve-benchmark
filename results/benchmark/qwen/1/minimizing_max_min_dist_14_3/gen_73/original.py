# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci spiral method"""
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


def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance between all point pairs"""
    distances = pdist(points)
    return np.min(distances) / np.max(distances)


def optimize_points(initial_points, max_iter=1000):
    """Optimize point positions to maximize min/max distance ratio"""

    def objective(x):
        # Reshape flat array back to points
        points = x.reshape(-1, 3)
        # We want to maximize the ratio, so we minimize its negative
        return -min_max_dist_ratio(points)

    # Flatten initial points for optimization
    x0 = initial_points.flatten()

    # Define bounds (points should stay within [-1,1] for sphere)
    bounds = [(-1, 1) for _ in range(len(x0))]

    # Use L-BFGS-B for optimization with bounds
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter},
        tol=1e-8
    )

    # Return optimized points
    return result.x.reshape(-1, 3)


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3

    # Initialize points using Fibonacci spiral on sphere
    np.random.seed(42)
    points = fibonacci_sphere(n)

    # Optimize the configuration
    optimized_points = optimize_points(points)

    return optimized_points


# EVOLVE-BLOCK-END