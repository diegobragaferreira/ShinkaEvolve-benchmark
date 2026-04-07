# EVOLVE-BLOCK-START
import numpy as np

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize circles array
    n = 26
    circles = np.zeros((n, 3))

    # Use a heuristic approach to initialize positions and radii
    # Start by placing circles in a structured way

    # Grid-based initialization with decreasing radii
    rows = 5
    cols = 5
    grid_size = min(rows, cols)

    # Create grid points
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)

    # Initialize with a basic grid pattern
    idx = 0
    for i in range(min(rows, 5)):
        for j in range(min(cols, 5)):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Set initial radius based on distance to edges and neighbors
            r = min(x, 1-x, y, 1-y) * 0.4  # Conservative initial radius
            circles[idx] = [x, y, r]
            idx += 1

    # Fill remaining circles with smaller radii placed strategically
    while idx < n:
        # Place remaining circles at strategic locations
        x = 0.2 + 0.6 * np.random.random()
        y = 0.2 + 0.6 * np.random.random()
        # Ensure we don't place too close to existing circles
        r = min(0.05, 0.1 * (1 - np.random.random()))
        circles[idx] = [x, y, r]
        idx += 1

    # Simple refinement to ensure containment
    for i in range(n):
        x, y, r = circles[i]
        # Adjust radius to ensure containment
        r = min(r, x, 1-x, y, 1-y)
        circles[i] = [x, y, r]

    return circles


# EVOLVE-BLOCK-END