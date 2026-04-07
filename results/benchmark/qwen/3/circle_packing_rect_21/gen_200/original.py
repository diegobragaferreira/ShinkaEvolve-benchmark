# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.2, 0.8

    # Initialize circles array
    n = 21
    circles = np.zeros((n, 3))

    # Phase 1: Enhanced strategic initialization with corner/edge points and k-means clustering
    # Start with corner points for better boundary coverage
    corner_points = [
        [0.1, 0.1],           # Bottom-left
        [width-0.1, 0.1],     # Bottom-right
        [0.1, height-0.1],    # Top-left
        [width-0.1, height-0.1], # Top-right
    ]

    # Add edge midpoints for better boundary coverage
    edge_points = [
        [width/2, 0.1],       # Bottom-middle
        [width/2, height-0.1], # Top-middle
        [0.1, height/2],      # Left-middle
        [width-0.1, height/2], # Right-middle
    ]

    # Combine corner and edge points
    init_points = corner_points + edge_points

    # Fill remaining slots with k-means clustering for good distribution
    dense_grid_size = 50
    grid_x = np.linspace(0.01, width - 0.01, dense_grid_size)
    grid_y = np.linspace(0.01, height - 0.01, dense_grid_size)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])

    # Use k-means to select initial seed points
    centroids, labels = kmeans2(grid_points, n - len(init_points), minit='points')

    # Combine all initial points
    all_init_points = init_points + centroids.tolist()

    # Initialize with small radii at these positions
    for i in range(n):
        circles[i] = [all_init_points[i][0], all_init_points[i][1], 0.01]

    # Phase 2: Multi-scale aggressive optimization with enhanced diversity and early stopping
    max_iterations = 800
    last_improvement_iter = 0
    improvement_count = 0
    prev_sum_radii = 0

    # Iteration-based adaptive parameters
    for iteration in range(max_iterations):
        improved = False

        # Adaptive step size decreases over time
        if iteration < 200:
            step_size = 0.15
            radius_update_factor = 0.9
        elif iteration < 500:
            step_size = 0.08
            radius_update_factor = 0.7
        else:
            step_size = 0.03
            radius_update_factor = 0.5

        # Try to increase each circle's radius
        for i in range(n):
            # Find maximum possible radius for circle i
            max_radius = calculate_max_radius(circles, i, width, height)

            if max_radius > circles[i][2]:
                circles[i][2] = min(max_radius, circles[i][2] * radius_update_factor)
                improved = True

        # Track improvement for early stopping
        current_sum_radii = np.sum(circles[:, 2])
        if current_sum_radii > prev_sum_radii:
            improvement_count += 1
            last_improvement_iter = iteration
            prev_sum_radii = current_sum_radii
        else:
            improvement_count = 0

        # Early termination if stagnated for too long or no significant improvement
        if improvement_count > 50 or (iteration - last_improvement_iter > 300 and current_sum_radii < prev_sum_radii * 1.001):
            break

    # Phase 3: Advanced local search with adaptive neighborhood and enhanced exploration
    local_search_iterations = 400
    improvement_history = []

    for refinement_iteration in range(local_search_iterations):
        # Progressive step size reduction
        if refinement_iteration < 150:
            step_size = 0.08
        elif refinement_iteration < 300:
            step_size = 0.03
        else:
            step_size = 0.01

        # Try moving each circle slightly to see if we can improve the configuration
        for i in range(n):
            current_x, current_y, current_r = circles[i]

            # Try movements in different directions with enhanced search patterns
            best_pos = [current_x, current_y, current_r]
            best_radius = current_r

            # Use adaptive grid pattern based on iteration stage
            if refinement_iteration < 100:
                # Coarse search in early stages
                search_grid = [-step_size*2, -step_size, 0, step_size, step_size*2]
            elif refinement_iteration < 250:
                # Medium search in middle stages
                search_grid = [-step_size, -step_size/2, 0, step_size/2, step_size]
            else:
                # Fine search in later stages
                search_grid = [-step_size/2, 0, step_size/2]

            # Also add diagonal searches for better exploration
            if refinement_iteration > 200:
                search_grid.extend([-step_size*1.5, step_size*1.5])

            # Examine grid around current position with adaptive step size
            for dx in search_grid:
                for dy in search_grid:
                    new_x, new_y = current_x + dx, current_y + dy

                    # Check if new position is within bounds
                    if 0 <= new_x <= width and 0 <= new_y <= height:
                        # Calculate max radius at new position
                        max_radius = calculate_max_radius_at_position(
                            circles, i, new_x, new_y, width, height
                        )

                        if max_radius > best_radius:
                            best_radius = max_radius
                            best_pos = [new_x, new_y, max_radius]

            # Update if we found a better position
            if best_pos[2] > circles[i][2]:
                circles[i] = best_pos

    # Final validation and cleanup
    for i in range(n):
        # Ensure minimum radius
        circles[i][2] = max(circles[i][2], 0.001)

        # Ensure circles stay within bounds
        circles[i][0] = np.clip(circles[i][0], 0.001, width - 0.001)
        circles[i][1] = np.clip(circles[i][1], 0.001, height - 0.001)

    return circles


def calculate_max_radius(circles, index, width, height):
    """Calculate maximum radius for circle at given index without overlapping others."""
    x, y, current_radius = circles[index]

    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)

    # Maximum radius based on other circles
    max_radius_overlap = float('inf')

    for i, (cx, cy, cr) in enumerate(circles):
        if i != index:
            # Distance to other circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - cr

            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle

    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


def calculate_max_radius_at_position(circles, index, x, y, width, height):
    """Calculate maximum radius for circle at given position without overlapping others."""
    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)

    # Maximum radius based on other circles
    max_radius_overlap = float('inf')

    for i, (cx, cy, cr) in enumerate(circles):
        if i != index:
            # Distance to other circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - cr

            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle

    max_radius = min(max_radius_bound, max_radius_overlap)
    return max(max_radius, 0.001)  # Ensure minimum radius


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")