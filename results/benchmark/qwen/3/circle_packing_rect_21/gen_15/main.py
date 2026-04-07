# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Using a square for simplicity (1x1) but we could optimize this
    width, height = 1.0, 1.0

    # Hexagonal grid initialization
    n = 21
    circles = np.zeros((n, 3))

    # Create hexagonal grid pattern
    rows = 5
    cols = 5
    spacing_x = width / (cols - 1)
    spacing_y = height / (rows - 1)

    # Hexagonal offset for alternate rows
    hex_offset = spacing_x * 0.5

    # Place circles in a hexagonal pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = j * spacing_x
            y = i * spacing_y

            # Offset every other row
            if i % 2 == 1:
                x += hex_offset

            # Ensure circles stay within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))

            circles[idx] = [x, y, 0.01]  # Initial small radius
            idx += 1
        if idx >= n:
            break

    # Fill remaining slots with small radii if needed
    for i in range(idx, n):
        circles[i] = [0.5, 0.5, 0.01]

    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")