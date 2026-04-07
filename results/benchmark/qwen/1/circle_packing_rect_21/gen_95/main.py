# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    # Rectangle with perimeter = 4, so width + height = 2
    # Optimal rectangle: 1.5 x 0.5 (width x height) for better packing efficiency
    rect_width = 1.5
    rect_height = 0.5

    # Phase 1: Initial placement using hexagonal lattice with adaptive spacing
    circles = np.zeros((n, 3))

    # Hexagonal packing arrangement adapted for rectangular container
    rows = 4
    cols = 6

    # Calculate spacing based on rectangle dimensions
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)

    # Place circles in hexagonal pattern
    idx = 0
    for i in range(rows):
        offset = spacing_x * (i % 2) * 0.5  # Offset every other row
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x + offset
            y = (i + 1) * spacing_y

            # Ensure position is within bounds
            x = max(0.01, min(rect_width - 0.01, x))
            y = max(0.01, min(rect_height - 0.01, y))

            # Set initial radius to a small value
            circles[idx] = [x, y, 0.05]
            idx += 1

        if idx >= n:
            break

    # Fill remaining circles if needed
    while idx < n:
        x = np.random.uniform(0.01, rect_width - 0.01)
        y = np.random.uniform(0.01, rect_height - 0.01)
        circles[idx] = [x, y, 0.05]
        idx += 1

    # Phase 2: Improved optimization using spatial indexing and targeted expansion
    max_iterations = 300  # Increased iterations for better convergence

    # Precompute distance matrix for faster neighbor lookups (memory trade-off for speed)
    # But use a more efficient approach: spatial hashing for collision detection

    # For very fast collision checking, we'll implement a simple spatial grid approach
    def get_collision_bounds(circle_pos, radius, rect_width, rect_height):
        """Calculate the bounds for a circle's collision checking"""
        return [
            max(0.0, circle_pos[0] - radius),
            min(rect_width, circle_pos[0] + radius),
            max(0.0, circle_pos[1] - radius),
            min(rect_height, circle_pos[1] + radius)
        ]

    def is_valid_solution(circles_array, rect_width, rect_height):
        """Fast validation of solution constraints"""
        n = len(circles_array)
        # Check boundary constraints
        for i in range(n):
            x, y, r = circles_array[i]
            if (x - r <= 0 or x + r >= rect_width or
                y - r <= 0 or y + r >= rect_height):
                return False

        # Check all pairwise collisions (optimized for early termination)
        for i in range(n):
            for j in range(i+1, n):
                dx = circles_array[i, 0] - circles_array[j, 0]
                dy = circles_array[i, 1] - circles_array[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < (circles_array[i, 2] + circles_array[j, 2]):
                    return False
        return True

    # Main optimization loop
    for iteration in range(max_iterations):
        improved = False

        # Randomly shuffle circle order for better exploration
        circle_indices = list(range(n))
        np.random.shuffle(circle_indices)

        # Process circles in shuffled order
        for i in circle_indices:
            # Store original values for potential rollback
            original_x, original_y, original_radius = circles[i]

            # Calculate maximum allowable radius for this circle
            max_radius = min(
                circles[i][0],  # Distance to left edge
                rect_width - circles[i][0],  # Distance to right edge
                circles[i][1],  # Distance to bottom edge
                rect_height - circles[i][1]   # Distance to top edge
            ) - 0.001

            # Check collision constraints with all other circles
            # Use optimized approach: compute distances to nearby circles efficiently
            for j in range(n):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - circles[j][2] - 0.001
                    if collision_radius > 0:
                        max_radius = min(max_radius, collision_radius)

            # If we can expand this circle, do so with adaptive strategy
            if max_radius > circles[i][2] and max_radius > 0.001:
                # Adaptive expansion based on local density estimation
                # Count nearby circles (within 3x radius)
                nearby_count = 0
                radius_threshold = circles[i][2] * 3.0
                for j in range(n):
                    if i != j:
                        dx = circles[i][0] - circles[j][0]
                        dy = circles[i][1] - circles[j][1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance < radius_threshold:
                            nearby_count += 1

                # Adjust expansion factor based on local density
                expansion_factor = 1.0
                if nearby_count > 5:  # Dense region
                    expansion_factor = 0.3
                elif nearby_count > 3:  # Moderately dense
                    expansion_factor = 0.6
                elif nearby_count <= 2:  # Sparse region
                    expansion_factor = 1.2

                # Apply bounded expansion
                delta = min(0.03, max_radius - circles[i][2]) * expansion_factor
                if delta > 0.001:
                    circles[i][2] += delta
                    improved = True

        # Early stopping if no significant improvement
        if not improved:
            break

    # Phase 3: Final refinement with more careful constraint checking
    # Perform a few rounds of focused optimization with explicit validation
    for round_num in range(5):
        improved_round = False
        for i in range(n):
            # Calculate maximum allowable radius with full constraint validation
            max_radius = min(
                circles[i][0],  # Distance to left edge
                rect_width - circles[i][0],  # Distance to right edge
                circles[i][1],  # Distance to bottom edge
                rect_height - circles[i][1]   # Distance to top edge
            ) - 0.001

            # Check all collisions carefully
            for j in range(n):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - circles[j][2] - 0.001
                    if collision_radius > 0:
                        max_radius = min(max_radius, collision_radius)

            # Expand if possible and beneficial
            if max_radius > circles[i][2] and max_radius > 0.001:
                # Smaller incremental steps for final refinement
                delta = min(0.01, max_radius - circles[i][2])
                if delta > 0.0005:
                    circles[i][2] += delta
                    improved_round = True

        if not improved_round:
            break

    # Final validation
    if not is_valid_solution(circles, rect_width, rect_height):
        # If invalid, revert to a safe configuration
        print("Warning: Solution validation failed, reverting to safe configuration")
        # Reset to initial configuration with modified spacing
        circles = np.zeros((n, 3))
        rows = 4
        cols = 6
        spacing_x = rect_width / (cols + 1)
        spacing_y = rect_height / (rows + 1)

        idx = 0
        for i in range(rows):
            offset = spacing_x * (i % 2) * 0.5
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x + offset
                y = (i + 1) * spacing_y
                x = max(0.01, min(rect_width - 0.01, x))
                y = max(0.01, min(rect_height - 0.01, y))
                circles[idx] = [x, y, 0.05]
                idx += 1
            if idx >= n:
                break

        while idx < n:
            x = np.random.uniform(0.01, rect_width - 0.01)
            y = np.random.uniform(0.01, rect_height - 0.01)
            circles[idx] = [x, y, 0.05]
            idx += 1

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")