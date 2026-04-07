# EVOLVE-BLOCK-START
import numpy as np

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32

    # Hexagonal grid initialization
    # Determine grid parameters for 32 circles
    rows = 6
    cols = 6
    # Adjust to fit exactly 32 circles
    while rows * cols < n:
        rows += 1
        cols += 1

    # Create hexagonal grid
    hex_radius = 0.1  # Starting radius estimate
    spacing_x = 2 * hex_radius * 1.05  # Slightly increased spacing for safety
    spacing_y = hex_radius * np.sqrt(3) * 1.05

    # Initialize grid points
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 * (i % 2)) * spacing_x
            y = i * spacing_y
            if x <= 1 and y <= 1:
                grid_points.append((x, y))

    # Trim to exactly 32 points and assign initial radii
    grid_points = grid_points[:n]

    # Create circles array with initial positions and equal radii
    circles = np.zeros((n, 3))
    for i, (x, y) in enumerate(grid_points):
        circles[i] = [x, y, hex_radius]

    return circles


# EVOLVE-BLOCK-END