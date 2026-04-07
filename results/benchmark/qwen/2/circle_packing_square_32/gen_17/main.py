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
    circles = np.zeros((n, 3))

    # Hexagonal grid initialization
    # Calculate grid parameters for 32 circles
    rows = 6
    cols = 6  # 6x6 = 36 positions, more than enough for 32 circles

    # Create hexagonal grid points
    sqrt3 = np.sqrt(3)
    spacing_x = 1.0 / cols
    spacing_y = sqrt3 / 2 * spacing_x

    # Generate base grid positions
    grid_points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            if x <= 1.0 and y <= 1.0:
                grid_points.append([x, y])

    # Take first 32 points (or adjust if fewer)
    if len(grid_points) >= n:
        points = np.array(grid_points[:n])
    else:
        # If we don't have enough points, pad with additional positions
        points = np.array(grid_points)
        while len(points) < n:
            # Add some additional points near the edges
            extra_point = [0.5, 0.5]
            points = np.vstack([points, extra_point])

    # Initial radius estimation - start with small radius
    initial_radius = 0.05

    # Place circles with initial configuration
    for i in range(min(n, len(points))):
        circles[i][0] = points[i][0]  # x coordinate
        circles[i][1] = points[i][1]  # y coordinate
        circles[i][2] = initial_radius  # initial radius

    # Ensure all circles respect boundaries
    for i in range(n):
        # Ensure circles don't exceed boundaries
        min_radius = min(circles[i][0], circles[i][1], 1 - circles[i][0], 1 - circles[i][1])
        if circles[i][2] > min_radius:
            circles[i][2] = min_radius

    return circles


# EVOLVE-BLOCK-END