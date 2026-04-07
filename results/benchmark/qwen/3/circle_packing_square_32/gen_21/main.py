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

    # Hexagonal grid initialization approach
    # Create a hexagonal grid of points
    sqrt_3 = np.sqrt(3)
    # Calculate spacing based on desired number of circles
    # For 32 circles in a unit square, we want to cover the area efficiently
    spacing = 0.15  # Initial spacing guess

    # Generate hexagonal grid points
    grid_points = []
    rows = int(np.ceil(1.0 / spacing))
    cols = int(np.ceil(1.0 / (spacing * sqrt_3)))

    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 * (i % 2)) * spacing
            y = i * spacing * sqrt_3
            if x <= 1.0 and y <= 1.0:
                grid_points.append([x, y])

    # Convert to numpy array
    grid_points = np.array(grid_points)

    if len(grid_points) >= n:
        # Take the n closest points to center
        distances = np.linalg.norm(grid_points - [0.5, 0.5], axis=1)
        closest_indices = np.argsort(distances)[:n]
        selected_points = grid_points[closest_indices]
    else:
        # If not enough points, use all and add some randomness
        selected_points = grid_points
        # Pad with random points if needed
        extra_needed = n - len(selected_points)
        for _ in range(extra_needed):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            selected_points = np.vstack([selected_points, [x, y]])

    # Initialize radii to be equal and feasible
    max_radius = 0.05  # Start with a conservative radius
    for i in range(n):
        circles[i] = [selected_points[i][0], selected_points[i][1], max_radius]

    return circles


# EVOLVE-BLOCK-END