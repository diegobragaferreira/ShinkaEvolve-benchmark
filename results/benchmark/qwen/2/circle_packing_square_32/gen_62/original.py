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
    # For 32 circles, arrange them in roughly 6 rows and 6 columns (but adjust to 32)
    rows = 6
    cols = 6

    # If we have too many cells, reduce to 32
    if rows * cols > n:
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

    # Adjust dimensions to use exactly 32 circles
    actual_rows = min(rows, int(np.ceil(np.sqrt(n))))
    actual_cols = min(cols, int(np.ceil(n / actual_rows)))

    # Ensure we get exactly 32 circles
    actual_count = actual_rows * actual_cols
    if actual_count < n:
        # Add extra rows until we have enough
        while actual_count < n:
            actual_rows += 1
            actual_count = actual_rows * actual_cols

    # Create hexagonal grid
    # Calculate spacing based on desired circle density
    spacing_x = 1.0 / (actual_cols + 1)
    spacing_y = 1.0 / (actual_rows + 1)

    # Hexagonal offset for alternating rows
    hex_offset = spacing_x * 0.5

    idx = 0
    for i in range(actual_rows):
        for j in range(actual_cols):
            if idx >= n:
                break
            # Even rows are aligned, odd rows offset
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y

            if i % 2 == 1:
                x += hex_offset

            # Ensure we're within bounds
            x = max(0.1, min(0.9, x))
            y = max(0.1, min(0.9, y))

            # Set initial radius - this will be optimized later
            # Start with small radius so we satisfy containment constraints
            circles[idx, 0] = x
            circles[idx, 1] = y
            circles[idx, 2] = 0.05  # Start with small radius

            idx += 1
            if idx >= n:
                break

    # Fill remaining circles if needed
    for i in range(idx, n):
        circles[i, 0] = 0.5
        circles[i, 1] = 0.5
        circles[i, 2] = 0.05

    return circles


# EVOLVE-BLOCK-END