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
    # Place circles in a hexagonal pattern to get good initial distribution

    # Determine grid parameters
    rows = 6
    cols = 6  # This gives us 36 positions, more than enough for 32 circles

    # Calculate spacing based on desired number of circles
    # Start with equal radius for all circles and adjust spacing accordingly
    base_radius = 0.1
    spacing_x = 2 * base_radius
    spacing_y = 2 * base_radius * np.sqrt(3)/2  # Vertical spacing for hexagonal grid

    # Adjust spacing to fit within unit square with some margin
    max_radius = 0.5  # Maximum possible radius for any circle
    min_spacing_x = 2 * max_radius
    min_spacing_y = 2 * max_radius

    # Create hexagonal grid
    x_positions = []
    y_positions = []

    # Generate positions in hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            # Adjust for hexagonal offset
            if i % 2 == 1:
                x += spacing_x / 2

            # Only include positions that could potentially fit in the unit square
            if x >= max_radius and x <= 1 - max_radius and \
               y >= max_radius and y <= 1 - max_radius:
                x_positions.append(x)
                y_positions.append(y)

    # Take first 32 positions (or fewer if needed)
    actual_positions = min(len(x_positions), n)

    # Set up circles with smaller initial radii to avoid overlaps
    for i in range(actual_positions):
        circles[i, 0] = x_positions[i]  # x coordinate
        circles[i, 1] = y_positions[i]  # y coordinate
        circles[i, 2] = 0.05  # Small initial radius to allow for expansion

    # Fill remaining circles with zeros (will be optimized)
    for i in range(actual_positions, n):
        circles[i, 0] = 0.5  # Default center
        circles[i, 1] = 0.5  # Default center
        circles[i, 2] = 0.0  # Zero radius initially

    return circles


# EVOLVE-BLOCK-END