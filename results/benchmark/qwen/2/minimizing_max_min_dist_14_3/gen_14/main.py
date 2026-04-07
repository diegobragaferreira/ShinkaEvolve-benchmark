# EVOLVE-BLOCK-START
import numpy as np


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


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    n = 14
    d = 3

    # Initialize points using Fibonacci sphere distribution for better spread
    points = fibonacci_sphere(n)

    # Normalize to unit sphere (optional, but helps with scaling)
    norms = np.linalg.norm(points, axis=1)
    points = points / np.max(norms) * 0.9  # Scale down slightly to avoid boundary issues

    return points


# EVOLVE-BLOCK-END