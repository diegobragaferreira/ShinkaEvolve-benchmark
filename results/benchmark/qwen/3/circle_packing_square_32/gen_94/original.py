# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32

    # Hexagonal grid initialization
    # We'll create a hexagonal lattice and then optimize radii
    def create_hexagonal_grid(n_circles):
        # Create a hexagonal grid
        sqrt3 = np.sqrt(3)

        # Determine grid dimensions
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))

        # Adjust to fit exactly n_circles
        if rows * cols < n_circles:
            cols += 1

        # Calculate spacing
        spacing = 0.15  # Initial spacing
        grid_points = []

        for i in range(rows):
            for j in range(cols):
                if len(grid_points) >= n_circles:
                    break
                # Offset every other row
                x = (j + 0.5 * (i % 2)) * spacing + 0.05
                y = i * spacing * sqrt3 / 2 + 0.05

                # Only include points that fit in the unit square
                if x <= 1 - spacing/2 and y <= 1 - spacing/2:
                    grid_points.append([x, y])

        # Trim to exact number needed
        grid_points = grid_points[:n_circles]

        return np.array(grid_points)

    # Initialize with hexagonal grid
    grid_points = create_hexagonal_grid(n)

    # Initialize circles with small uniform radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [grid_points[i][0], grid_points[i][1], 0.02]

    # Simple optimization: try to increase radii while maintaining constraints
    improvement_threshold = 1e-6
    max_iterations = 500

    for iteration in range(max_iterations):
        improved = False

        # Try to increase each radius
        for i in range(n):
            old_radius = circles[i][2]

            # Calculate maximum possible radius for this circle
            max_radius = min(
                circles[i][0],  # Distance to left boundary
                1 - circles[i][0],  # Distance to right boundary
                circles[i][1],  # Distance to bottom boundary
                1 - circles[i][1]  # Distance to top boundary
            )

            # Find minimum distance to other circles
            min_distance = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt(
                        (circles[i][0] - circles[j][0])**2 +
                        (circles[i][1] - circles[j][1])**2
                    )
                    min_distance = min(min_distance, dist)

            # Maximum radius is limited by distance to neighbors minus current radius
            if min_distance < float('inf'):
                max_radius = min(max_radius, min_distance - old_radius)

            # Try to increase radius up to the limit (but keep some safety margin)
            new_radius = min(max_radius, old_radius + 0.001)

            if new_radius > old_radius + improvement_threshold:
                circles[i][2] = new_radius
                improved = True

        # Early stopping if no significant improvement
        if not improved:
            break

    # Final validation - make sure all constraints are met
    # And ensure circles don't go beyond boundaries or overlap
    def validate_and_correct(circles_array):
        # Ensure all circles respect boundary constraints
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Ensure circle fits in unit square
            r = min(r, x, 1-x, y, 1-y)
            circles_array[i] = [x, y, r]

        # Resolve overlaps between circles
        for _ in range(100):  # Allow some iterations to fix overlaps
            changed = False
            distances = cdist(circles_array[:, :2], circles_array[:, :2])

            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    dist = distances[i, j]
                    r_i, r_j = circles_array[i, 2], circles_array[j, 2]

                    # Check if overlapping
                    if dist < r_i + r_j:
                        # Reduce the radii of both circles proportionally
                        total_reduction = (r_i + r_j - dist) * 0.5
                        if r_i > total_reduction and r_j > total_reduction:
                            circles_array[i, 2] -= total_reduction
                            circles_array[j, 2] -= total_reduction
                            changed = True

            if not changed:
                break

        # Ensure all radii are reasonable
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Ensure circle fits in unit square
            r = min(r, x, 1-x, y, 1-y)
            circles_array[i] = [x, y, r]

        return circles_array

    circles = validate_and_correct(circles)

    return circles


# EVOLVE-BLOCK-END