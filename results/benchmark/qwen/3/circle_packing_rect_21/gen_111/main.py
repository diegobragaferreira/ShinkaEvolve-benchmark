# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Using the proven 1.2x0.8 dimensions from prior tests
    width, height = 1.2, 0.8

    # Initialize circles array
    n = 21
    circles = np.zeros((n, 3))

    # Phase 1: Hybrid initialization combining hexagonal and strategic seeding
    # Start with corner and edge points for boundary coverage
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
    
    # Fill remaining slots with hexagonal grid pattern for interior coverage
    rows = 5
    cols = 5
    estimated_radius = 0.08  # Adjusted for container size
    
    # Generate hexagonal grid points
    hex_points = []
    for i in range(rows):
        for j in range(cols):
            if len(hex_points) >= (n - len(init_points)):
                break
            x = j * 2 * estimated_radius + (i % 2) * estimated_radius  # Offset every other row
            y = i * np.sqrt(3) * estimated_radius
            
            # Add point if it's within bounds
            if 0.01 <= x <= width - 0.01 and 0.01 <= y <= height - 0.01:
                hex_points.append([x, y])
        if len(hex_points) >= (n - len(init_points)):
            break

    # Combine all initial points
    all_init_points = init_points + hex_points[:n-len(init_points)]
    
    # Add any remaining points randomly if needed
    np.random.seed(42)  # Fixed seed for reproducibility
    while len(all_init_points) < n:
        x = np.random.uniform(0.05, width - 0.05)
        y = np.random.uniform(0.05, height - 0.05)
        all_init_points.append([x, y])
    
    # Assign initial positions with small radii
    for i in range(n):
        circles[i] = [all_init_points[i][0], all_init_points[i][1], 0.01]

    # Phase 2: Multi-scale adaptive optimization with momentum
    max_iterations = 1200
    
    for iteration in range(max_iterations):
        improved = False
        
        # Three-phase adaptive optimization
        if iteration < 400:
            # Phase 1: Broad exploration with large steps
            step_size = 0.15
            radius_update_factor = 0.95
        elif iteration < 800:
            # Phase 2: Focused refinement with medium steps
            step_size = 0.08
            radius_update_factor = 0.8
        else:
            # Phase 3: Fine-tuning with small steps
            step_size = 0.03
            radius_update_factor = 0.6

        # Try to increase each circle's radius
        for i in range(n):
            # Find maximum possible radius for circle i
            max_radius = calculate_max_radius(circles, i, width, height)
            
            if max_radius > circles[i][2]:
                # Apply adaptive radius update
                new_radius = min(max_radius, circles[i][2] * radius_update_factor)
                circles[i][2] = new_radius
                improved = True

        # Early termination for stagnation
        if not improved and iteration > 500:
            break

    # Phase 3: Progressive local search with adaptive neighborhood
    for refinement_iteration in range(800):
        # Adaptive step size reduction
        if refinement_iteration < 300:
            step_size = 0.1
        elif refinement_iteration < 600:
            step_size = 0.05
        else:
            step_size = 0.02

        # Try moving each circle slightly to see if we can improve the configuration
        for i in range(n):
            current_x, current_y, current_r = circles[i]

            # Track best improvement for this iteration
            best_pos = [current_x, current_y, current_r]
            best_radius = current_r

            # Examine larger grid around current position in early iterations
            if refinement_iteration < 200:
                search_grid = [-step_size*2, -step_size, 0, step_size, step_size*2]
            elif refinement_iteration < 400:
                search_grid = [-step_size, 0, step_size]
            else:
                search_grid = [-step_size/2, 0, step_size/2]

            # Also add diagonal searches for better exploration in later stages
            if refinement_iteration > 400:
                search_grid.extend([-step_size*1.5, step_size*1.5])

            # Examine grid around current position
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