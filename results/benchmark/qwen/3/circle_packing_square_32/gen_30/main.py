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
    # Calculate grid parameters
    rows = 6
    cols = 6
    # Adjust grid to fit 32 circles (we'll use 6x6 grid but only take first 32)

    # Create hexagonal grid with proper spacing
    hex_radius = 0.1  # Initial guess for radius
    hex_spacing_x = hex_radius * 2 * 1.0  # Horizontal spacing
    hex_spacing_y = hex_radius * 2 * 0.866  # Vertical spacing (sqrt(3)/2)

    # Generate hexagonal grid points
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break

            # Offset every other row
            x_offset = hex_spacing_x * j if (i % 2 == 0) else hex_spacing_x * j + hex_spacing_x / 2
            y = hex_spacing_y * i

            # Check if position is within bounds
            if x_offset >= hex_radius and x_offset <= 1 - hex_radius and \
               y >= hex_radius and y <= 1 - hex_radius:

                circles[count] = [x_offset, y, hex_radius]
                count += 1

        if count >= n:
            break

    # Fill remaining circles with small random offsets
    for i in range(count, n):
        circles[i] = [
            np.random.uniform(hex_radius, 1 - hex_radius),
            np.random.uniform(hex_radius, 1 - hex_radius),
            hex_radius
        ]

    return circles


# EVOLVE-BLOCK-END