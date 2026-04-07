# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """
    n = 14

    # Use Fibonacci spiral initialization on sphere for better point distribution
    points = []
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    for i in range(n):
        # Distribute points along longitude (azimuthal angle)
        theta = 2 * np.pi * i / phi

        # Distribute points along latitude (polar angle) using Fibonacci-like spacing
        # This avoids clustering at poles
        y = 1.0 - (2.0 * i) / (n - 1)
        radius = np.sqrt(1 - y*y)

        x = radius * np.cos(theta)
        z = radius * np.sin(theta)

        points.append([x, y, z])

    points = np.array(points)

    # Normalize to unit sphere (this ensures all points are equidistant from center)
    # But we'll scale appropriately for our optimization later
    return points


# EVOLVE-BLOCK-END