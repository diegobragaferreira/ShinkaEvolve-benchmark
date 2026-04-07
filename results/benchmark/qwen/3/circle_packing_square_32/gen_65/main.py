# EVOLVE-BLOCK-START
import numpy as np

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Use a more precise hexagonal grid approach
    # Optimal hexagonal packing pattern for 32 circles
    rows = 6
    cols = 6

    # Calculate optimal spacing based on desired total radius
    # Target around 2.9 for benchmark (AlphaEvolve)
    target_sum_radius = 2.8
    avg_radius = target_sum_radius / n

    # For hexagonal packing: spacing = 2 * radius
    spacing = 2.0 * avg_radius
    spacing_y = spacing * np.sqrt(3) / 2

    # Calculate grid dimensions
    grid_width = (cols - 1) * spacing
    grid_height = (rows - 1) * spacing_y + spacing_y / 2

    # Center the grid in unit square
    offset_x = (1 - grid_width) / 2
    offset_y = (1 - grid_height) / 2

    # Generate positions in hexagonal grid
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = offset_x + j * spacing
            y = offset_y + i * spacing_y

            # Apply hexagonal offset for odd rows
            if i % 2 == 1:
                x += spacing / 2

            positions.append((x, y))

    # Use first n positions
    positions = positions[:n]

    # Initialize circles with small but reasonable radius
    for i, (x, y) in enumerate(positions):
        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = 0.04  # Slightly smaller initial radius to allow room for expansion

    # Enhanced iterative optimization with better constraint handling
    max_iterations = 1000
    tolerance = 1e-6
    improved = True

    for iteration in range(max_iterations):
        if not improved:
            break

        improved = False

        # Create a copy to avoid using updated values during this iteration
        new_circles = circles.copy()

        # Try to increase each circle's radius
        for i in range(n):
            old_radius = circles[i, 2]
            new_radius = old_radius

            # Calculate maximum possible radius based on boundary constraints
            max_boundary_radius = min(
                circles[i, 0],           # Distance to left edge
                1 - circles[i, 0],       # Distance to right edge
                circles[i, 1],           # Distance to bottom edge
                1 - circles[i, 1]        # Distance to top edge
            )

            # Calculate maximum possible radius based on overlap constraints
            max_overlap_radius = max_boundary_radius

            # Check distance to all other circles
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)

                    # Maximum radius such that circles don't overlap
                    max_radius_with_j = distance - circles[j, 2]
                    max_overlap_radius = min(max_overlap_radius, max_radius_with_j)

            # The final maximum radius is limited by both boundary and overlap constraints
            max_possible_radius = min(max_boundary_radius, max_overlap_radius)

            # Increase radius slightly (but don't exceed maximum)
            if max_possible_radius > old_radius + tolerance:
                # Use a more conservative increment to prevent overshooting
                increment = min(0.005, (max_possible_radius - old_radius) * 0.5)
                new_radius = min(old_radius + increment, max_possible_radius)

                if new_radius > old_radius + tolerance:
                    new_circles[i, 2] = new_radius
                    improved = True

        # Update circles for next iteration
        circles = new_circles

    # Final validation to ensure constraints are met
    def validate_circles(circs):
        # Check boundary constraints
        for i in range(n):
            x, y, r = circs[i]
            if (x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1):
                return False

        # Check overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circs[i]
                x2, y2, r2 = circs[j]
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < r1 + r2:
                    return False
        return True

    # If validation fails, fallback to original hexagonal pattern
    if not validate_circles(circles):
        # Recreate with simple hexagonal pattern and smaller initial radii
        circles = np.zeros((n, 3))
        for i, (x, y) in enumerate(positions):
            circles[i, 0] = x
            circles[i, 1] = y
            circles[i, 2] = 0.03  # Even smaller initial radius

    return circles


# EVOLVE-BLOCK-END