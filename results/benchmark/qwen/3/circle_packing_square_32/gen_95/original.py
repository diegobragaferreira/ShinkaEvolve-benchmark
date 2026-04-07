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

    # Create a more systematic hexagonal grid
    # We want to place circles in a hexagonal pattern within the unit square
    # Let's use 5 rows and 7 columns for 35 total positions (enough for 32)

    # Calculate spacing for hexagonal packing
    # For hexagonal packing, the vertical spacing is sqrt(3)/2 times the horizontal spacing
    # We start with an estimated radius and calculate appropriate spacing

    # Try to place circles in a hexagonal grid pattern
    rows = 5
    cols = 7

    # Initial estimate for the radius that would allow 32 circles in a hexagonal pattern
    # This is approximate - we'll use a more systematic approach
    target_total_radius = 2.5  # Estimate to work with
    avg_radius = target_total_radius / n

    # Hexagonal spacing calculation
    spacing_x = 2.2 * avg_radius  # Slightly more than diameter to allow for optimization
    spacing_y = spacing_x * np.sqrt(3) / 2

    # Calculate actual grid dimensions
    grid_width = (cols - 1) * spacing_x
    grid_height = (rows - 1) * spacing_y + spacing_y / 2

    # Center the grid in the unit square
    offset_x = (1 - grid_width) / 2
    offset_y = (1 - grid_height) / 2

    # Generate positions in hexagonal grid
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = offset_x + j * spacing_x
            y = offset_y + i * spacing_y

            # Apply hexagonal offset for odd rows
            if i % 2 == 1:
                x += spacing_x / 2

            positions.append((x, y))

    # Filter positions to only keep those that can fit a circle with reasonable radius
    valid_positions = []
    for x, y in positions:
        # Check if a circle of radius 0.05 could fit at this location
        if (x >= 0.05 and x <= 0.95 and
            y >= 0.05 and y <= 0.95):
            valid_positions.append((x, y))

    # Use first 32 valid positions
    valid_positions = valid_positions[:n]

    # Initialize circles with small radius
    for i, (x, y) in enumerate(valid_positions):
        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = 0.05  # Small initial radius

    # Simple iterative optimization to expand radii while maintaining constraints
    max_iterations = 100
    tolerance = 1e-6

    for _ in range(max_iterations):
        improved = False
        # Try to increase each circle's radius
        for i in range(n):
            old_radius = circles[i, 2]
            new_radius = old_radius

            # Try to increase radius up to the maximum possible without violating constraints
            max_possible_radius = min(
                circles[i, 0],  # Distance to left edge
                1 - circles[i, 0],  # Distance to right edge
                circles[i, 1],  # Distance to bottom edge
                1 - circles[i, 1]  # Distance to top edge
            )

            # Now check against other circles
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    # Must be at least the sum of radii apart
                    max_radius_with_j = distance - circles[j, 2]
                    max_possible_radius = min(max_possible_radius, max_radius_with_j)

            # Limit the new radius to be at most max_possible_radius
            new_radius = min(new_radius + 0.001, max_possible_radius)

            if new_radius > old_radius + tolerance:
                circles[i, 2] = new_radius
                improved = True

        if not improved:
            break

    return circles


# EVOLVE-BLOCK-END