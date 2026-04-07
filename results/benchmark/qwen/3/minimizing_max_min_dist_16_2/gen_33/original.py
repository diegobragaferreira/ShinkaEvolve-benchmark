# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2

    # Create hexagonal grid initialization for better starting configuration
    # Hexagonal lattice points arranged in 4 rows of 4 points each
    points = np.zeros((n, d))

    # Parameters for hexagonal arrangement
    row_spacing = 1.0
    col_spacing = np.sqrt(3) / 2.0

    # Generate hexagonal grid points
    idx = 0
    for row in range(4):
        for col in range(4):
            x = col * 1.0 + (row % 2) * 0.5  # Offset every other row
            y = row * col_spacing
            points[idx, 0] = x
            points[idx, 1] = y
            idx += 1

    # Normalize to unit square [0,1] x [0,1]
    # Find bounding box and scale appropriately
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    if x_max > x_min:
        points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
    if y_max > y_min:
        points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

    # Apply small random perturbations to break symmetry
    np.random.seed(42)
    perturbation_magnitude = 0.05
    points += np.random.uniform(-perturbation_magnitude, perturbation_magnitude, points.shape)

    # Keep points within [0,1] bounds
    points = np.clip(points, 0, 1)

    return points


# EVOLVE-BLOCK-END