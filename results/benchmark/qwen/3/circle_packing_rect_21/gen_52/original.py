# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from itertools import combinations

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Using width=1.2, height=0.8 for a reasonable aspect ratio
    rect_width = 1.2
    rect_height = 0.8

    # Set seed for reproducibility
    np.random.seed(42)

    # Initialize with Voronoi-based approach
    num_points = 21

    # Create initial points using a combination of regular grid and random perturbations
    rows, cols = 5, 5
    grid_x = np.linspace(0.1, rect_width - 0.1, cols)
    grid_y = np.linspace(0.1, rect_height - 0.1, rows)

    points = []
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            if len(points) < num_points:
                # Add small random perturbation
                pert_x = np.random.uniform(-0.05, 0.05)
                pert_y = np.random.uniform(-0.05, 0.05)
                points.append([x + pert_x, y + pert_y])

    # Trim to exactly 21 points
    points = points[:num_points]
    points = np.array(points)

    # Iterative optimization using Voronoi-guided refinement
    best_sum = 0
    best_circles = None

    # Use a more sophisticated approach combining Voronoi with greedy filling
    # First, let's try a hybrid initialization with strategic corner placement
    corner_positions = [
        (rect_width * 0.1, rect_height * 0.1),
        (rect_width * 0.9, rect_height * 0.1),
        (rect_width * 0.1, rect_height * 0.9),
        (rect_width * 0.9, rect_height * 0.9),
        (rect_width / 2, rect_height / 2)
    ]

    # Start with corner positions
    current_points = []

    # Place circles at strategic locations first
    placed_count = 0
    for x, y in corner_positions:
        if placed_count < 5:
            current_points.append([x, y])
            placed_count += 1

    # Fill remaining slots with greedy approach based on Voronoi cells
    remaining_slots = 21 - len(current_points)

    # Create a grid of candidate positions
    candidate_grid_x = np.linspace(0.05, rect_width - 0.05, 10)
    candidate_grid_y = np.linspace(0.05, rect_height - 0.05, 10)

    # Initialize with corner-based approach but add more systematic filling
    if remaining_slots > 0:
        # Create a structured approach using Voronoi for the core structure
        vor = Voronoi(np.array(current_points))
        candidate_positions = []

        # Generate additional candidate points
        for x in candidate_grid_x:
            for y in candidate_grid_y:
                # Only consider points within boundaries
                if 0.05 <= x <= rect_width - 0.05 and 0.05 <= y <= rect_height - 0.05:
                    candidate_positions.append([x, y])

        # For remaining positions, use greedy approach based on distance to existing points
        for _ in range(remaining_slots):
            best_pos = None
            best_min_distance = 0

            # Try each candidate position to maximize minimum distance to existing points
            for x, y in candidate_positions:
                min_distance_to_existing = float('inf')
                for px, py in current_points:
                    dist = math.sqrt((x - px)**2 + (y - py)**2)
                    min_distance_to_existing = min(min_distance_to_existing, dist)

                # Prefer positions that are far from existing points (encourage spreading)
                if min_distance_to_existing > best_min_distance:
                    best_min_distance = min_distance_to_existing
                    best_pos = [x, y]

            if best_pos is not None:
                current_points.append(best_pos)

    # Limit to exactly 21 points
    current_points = current_points[:21]
    points = np.array(current_points)

    # Main optimization loop
    max_iterations = 500
    for iteration in range(max_iterations):
        # Construct Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to direct calculation
            pass

        # Calculate circle parameters for each point
        circles = []

        # Process each point
        for i, point in enumerate(points):
            center_x, center_y = point

            # Compute minimum distance to rectangle edges
            dist_to_edges = [
                center_x,  # distance to left edge
                rect_width - center_x,  # distance to right edge
                center_y,  # distance to bottom edge
                rect_height - center_y   # distance to top edge
            ]

            # Compute minimum distance to other circles using cKDTree for efficiency
            if len(points) > 1:
                tree = cdist([point], points)[0]
                tree = np.delete(tree, i)  # Remove self-distance
                min_dist_to_others = np.min(tree) if len(tree) > 0 else float('inf')
            else:
                min_dist_to_others = float('inf')

            # Radius is limited by distance to edges and other circles
            max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
            radius = max(0.001, max_radius)
            circles.append([center_x, center_y, radius])

        # Convert to numpy array
        circles = np.array(circles)

        # Validate configuration - check for overlaps
        valid = True
        total_radius = np.sum(circles[:, 2])

        # Check overlaps efficiently using pairwise distance matrix
        if len(circles) > 1:
            # Create distance matrix
            distances = cdist(circles[:, :2], circles[:, :2])
            # Zero out diagonal
            np.fill_diagonal(distances, float('inf'))

            # Check if any pair is too close
            min_distances = np.min(distances, axis=0)
            for i in range(len(circles)):
                expected_separation = circles[i, 2] + circles[np.argmin(distances[i]), 2]
                if min_distances[i] < expected_separation:
                    valid = False
                    break

        # Accept better configurations
        if valid and total_radius > best_sum:
            best_sum = total_radius
            best_circles = circles.copy()

        # Apply small perturbations to points for evolution
        new_points = points.copy()
        for i in range(len(points)):
            # Small random perturbation
            new_points[i, 0] += np.random.normal(0, 0.02)
            new_points[i, 1] += np.random.normal(0, 0.02)

            # Keep within bounds
            new_points[i, 0] = np.clip(new_points[i, 0], 0.01, rect_width - 0.01)
            new_points[i, 1] = np.clip(new_points[i, 1], 0.01, rect_height - 0.01)

        points = new_points

    # Final refinement step with neighbor-based optimization
    if best_circles is not None:
        refined_circles = best_circles.copy()

        # Local optimization around each circle using a more targeted approach
        for _ in range(200):  # Limited iterations for time constraint
            # Select a random circle to optimize
            idx = np.random.randint(0, len(refined_circles))

            # Get current circle
            cx, cy, cr = refined_circles[idx]

            # Store original values
            original_cx, original_cy, original_cr = cx, cy, cr

            # Try to find a better position using a grid sampling approach
            best_cx, best_cy, best_cr = cx, cy, cr
            best_radius = cr

            # Sample nearby positions in a structured way
            step_size = 0.03
            for dx in [-step_size*2, -step_size, 0, step_size, step_size*2]:
                for dy in [-step_size*2, -step_size, 0, step_size, step_size*2]:
                    ncx = cx + dx
                    ncy = cy + dy

                    # Check bounds
                    if (ncx < 0.01 or ncx > rect_width - 0.01 or
                        ncy < 0.01 or ncy > rect_height - 0.01):
                        continue

                    # Compute max radius at new position
                    max_r = min(ncx, rect_width - ncx, ncy, rect_height - ncy)

                    # Check overlap with others efficiently
                    overlap = False
                    for j in range(len(refined_circles)):
                        if j != idx:
                            dist = math.sqrt((ncx - refined_circles[j, 0])**2 + (ncy - refined_circles[j, 1])**2)
                            if dist < max_r + refined_circles[j, 2]:  # Overlap
                                overlap = True
                                break

                    if not overlap and max_r > best_radius:
                        best_radius = max_r
                        best_cx, best_cy = ncx, ncy

            # Update if improvement found
            if best_radius > refined_circles[idx, 2]:
                refined_circles[idx, 0] = best_cx
                refined_circles[idx, 1] = best_cy
                refined_circles[idx, 2] = best_radius

    # Ensure we return exactly 21 circles
    if best_circles is None:
        # Fallback to a simple initial configuration
        circles = np.zeros((21, 3))
        # Place in a simple grid pattern
        row_size = int(np.ceil(np.sqrt(21)))
        col_size = int(np.ceil(21 / row_size))

        spacing_x = rect_width / (col_size + 1)
        spacing_y = rect_height / (row_size + 1)

        count = 0
        for i in range(row_size):
            for j in range(col_size):
                if count < 21:
                    x = spacing_x * (j + 1)
                    y = spacing_y * (i + 1)
                    # Set radius to be proportional to available space
                    radius = min(x, rect_width - x, y, rect_height - y) * 0.4
                    circles[count] = [x, y, max(radius, 0.001)]
                    count += 1

        return circles

    return refined_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")