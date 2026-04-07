# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14

    # Generate points using Fibonacci spiral on sphere
    # This provides a good initial distribution for point dispersion
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n):
        # Distribute points along the phi dimension (azimuthal angle)
        phi = np.arccos(1 - 2 * i / (n - 1))
        theta = 2 * np.pi * i / golden_ratio

        # Convert to Cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        points.append([x, y, z])

    points = np.array(points)

    # Normalize to unit sphere
    points = points / np.linalg.norm(points[0]) if np.linalg.norm(points[0]) > 0 else points

    return points


# EVOLVE-BLOCK-END