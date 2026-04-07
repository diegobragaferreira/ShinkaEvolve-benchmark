# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    # Use Fibonacci spiral initialization on a sphere for better point distribution
    n = 14

    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    points = np.array(points)

    # Normalize to unit sphere
    points = points / np.linalg.norm(points[0]) if np.linalg.norm(points[0]) > 0 else points

    return points


# EVOLVE-BLOCK-END