# EVOLVE-BLOCK-START
import numpy as np


def fibonacci_sphere(n_points: int) -> np.ndarray:
    """
    Generate n_points points distributed approximately uniformly on a sphere
    using the Fibonacci lattice method.
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/(n_points-1))
    theta = np.sqrt(n_points * np.pi) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack([x, y, z])


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3

    # Initialize points using Fibonacci sphere distribution for better initial spread
    points = fibonacci_sphere(n)

    # Normalize to unit cube [0,1]^3
    # First center at origin and scale to fit in [-1,1]^3
    points = points * 0.5 + 0.5

    return points


# EVOLVE-BLOCK-END