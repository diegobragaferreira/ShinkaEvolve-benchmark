# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def fibonacci_sphere(n):
    """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
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


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    n = 14

    # Initialize points using Fibonacci sphere distribution for better uniformity
    points = fibonacci_sphere(n)

    # Scale to unit cube [0,1]^3
    points = (points + 1) / 2  # map from [-1,1] to [0,1]

    # Apply local optimization to improve the min/max distance ratio
    # Flatten for optimization
    x0 = points.flatten()

    def objective(x_flat):
        # Reshape back to points
        points_opt = x_flat.reshape((n, 3))

        # Calculate distance matrix
        distances = cdist(points_opt, points_opt)

        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize (since we minimize in scipy)
        # Handle case where max_dist is 0 (should never happen with distinct points)
        if max_dist > 1e-10:
            return -min_dist / max_dist
        else:
            return 0

    def penalty_objective(x_flat):
        """Objective function with penalty for boundary violations"""
        points_opt = x_flat.reshape((n, 3))

        # Calculate base objective
        ratio = -objective(x_flat)
        base_obj = ratio

        # Add penalty for points outside [0,1]^3 bounds
        penalty = 0
        for i in range(n):
            for j in range(3):
                coord = points_opt[i, j]
                if coord < 0:
                    penalty += 1e6 * (0 - coord) ** 2
                elif coord > 1:
                    penalty += 1e6 * (coord - 1) ** 2

        return base_obj + penalty

    # Local optimization with L-BFGS-B
    try:
        result = minimize(
            penalty_objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * (n * 3),
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000},
            tol=1e-12
        )

        # Extract optimized points
        optimized_points = result.x.reshape((n, 3))
        points = np.clip(optimized_points, 0, 1)

    except Exception:
        # If optimization fails, return the Fibonacci points
        pass

    return points


# EVOLVE-BLOCK-END