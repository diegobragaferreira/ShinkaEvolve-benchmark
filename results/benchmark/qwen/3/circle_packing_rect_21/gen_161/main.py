# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from scipy.spatial.distance import cdist
import random
from scipy.spatial import distance
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - since perimeter = 4, width + height = 2
    # Using width=1.2, height=0.8 for optimal packing
    width, height = 1.2, 0.8

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Multi-stage optimization approach
    # Stage 1: Coarse Voronoi initialization
    circles = initialize_voronoi_layout(width, height, 21)

    # Stage 2: Multi-scale optimization with Voronoi constraints
    circles = optimize_multiscale_voronoi(circles, width, height)

    # Stage 3: Fine-grained local optimization
    circles = optimize_fine_local(circles, width, height)

    # Stage 4: Final boundary and overlap optimization
    circles = optimize_boundary_overlap(circles, width, height)

    return circles

def initialize_voronoi_layout(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circles using improved Voronoi-based strategic seeding."""
    # Create a more structured initial point distribution for better Voronoi cells
    # Use hexagonal lattice for better spatial distribution
    circles = np.zeros((n, 3))

    # Use a hexagonal grid pattern for initial distribution
    rows = 5
    cols = 5

    # Calculate spacing for hexagonal packing
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)

    # Offset for hexagonal pattern
    offset = spacing_x * 0.5

    # Generate hexagonal grid points
    hex_points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 1) * spacing_x
            if i % 2 == 1:
                x += offset
            y = (i + 1) * spacing_y
            if 0.1 <= x <= width - 0.1 and 0.1 <= y <= height - 0.1:
                hex_points.append([x, y])

    # Add strategic corner and edge points for better coverage
    corner_points = [
        [0.1, 0.1], [width-0.1, 0.1], [0.1, height-0.1], [width-0.1, height-0.1],
        [width/2, 0.1], [width/2, height-0.1], [0.1, height/2], [width-0.1, height/2]
    ]

    # Combine initial points
    initial_points = hex_points + corner_points

    # Generate Voronoi diagram from initial points
    if len(initial_points) >= n:
        # Use subset of points for Voronoi generation
        selected_points = initial_points[:n]
        vor = Voronoi(selected_points)

        # Use Voronoi vertices as center candidates with better filtering
        vertex_idx = 0
        for i in range(n):
            if vertex_idx < len(vor.vertices) and i < len(vor.vertices):
                x, y = vor.vertices[vertex_idx]
                # Ensure point is within bounds
                if 0.1 <= x <= width - 0.1 and 0.1 <= y <= height - 0.1:
                    # Compute max radius at this location
                    max_radius = compute_max_radius_voronoi(x, y, width, height, circles[:i])
                    circles[i] = [x, y, max_radius]
                    vertex_idx += 1
                else:
                    # Fall back to hex point or random placement
                    if i < len(hex_points):
                        x, y = hex_points[i]
                    else:
                        x = np.random.uniform(0.1, width - 0.1)
                        y = np.random.uniform(0.1, height - 0.1)
                    max_radius = compute_max_radius_voronoi(x, y, width, height, circles[:i])
                    circles[i] = [x, y, max_radius]
            else:
                # Fall back to random or hex point
                if i < len(hex_points):
                    x, y = hex_points[i]
                else:
                    x = np.random.uniform(0.1, width - 0.1)
                    y = np.random.uniform(0.1, height - 0.1)
                max_radius = compute_max_radius_voronoi(x, y, width, height, circles[:i])
                circles[i] = [x, y, max_radius]
    else:
        # Not enough points, fallback to hexagonal grid
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                if i % 2 == 1:
                    x += offset
                y = (i + 1) * spacing_y
                if 0.1 <= x <= width - 0.1 and 0.1 <= y <= height - 0.1:
                    max_radius = compute_max_radius_voronoi(x, y, width, height, circles[:idx])
                    circles[idx] = [x, y, max_radius]
                    idx += 1

        # Fill remaining circles
        for i in range(idx, n):
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            max_radius = compute_max_radius_voronoi(x, y, width, height, circles[:i])
            circles[i] = [x, y, max_radius]

    return circles

def compute_max_radius_voronoi(x: float, y: float, width: float, height: float, existing_circles: np.ndarray) -> float:
    """Compute maximum radius using improved Voronoi-style constraints."""
    # Boundary constraints
    min_dist_from_edge = min(x, width - x, y, height - y)

    if min_dist_from_edge <= 0:
        return 0

    # Overlap constraints with existing circles
    min_dist_from_others = float('inf')

    # Vectorized computation for better performance
    if len(existing_circles) > 0:
        existing_array = np.array(existing_circles)
        # Extract coordinates and radii
        cx_vals = existing_array[:, 0]
        cy_vals = existing_array[:, 1]
        cr_vals = existing_array[:, 2]

        # Compute distances vectorized
        distances = np.sqrt((cx_vals - x)**2 + (cy_vals - y)**2)

        # Minimum distance to other circles (minus their radii)
        min_dist_from_others = np.min(distances - cr_vals)

    # Take minimum of boundary and overlap constraints
    max_radius = min(min_dist_from_edge, min_dist_from_others)

    return max(0.001, max_radius)

def optimize_multiscale_voronoi(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Multi-scale optimization using Voronoi constraints with different refinement levels."""
    current_circles = circles.copy()
    num_circles = len(current_circles)

    # Coarse optimization - high level adjustment
    for iteration in range(200):
        improved = False
        indices = list(range(num_circles))
        random.shuffle(indices)

        for i in indices:
            # Try to increase radius
            old_r = current_circles[i][2]
            max_radius = compute_max_radius_voronoi(
                current_circles[i][0], current_circles[i][1],
                width, height,
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )

            if max_radius > old_r + 1e-6:
                current_circles[i][2] = max_radius
                improved = True

        if not improved:
            break

    # Medium scale optimization - position refinement
    for iteration in range(300):
        improved = False
        indices = list(range(num_circles))
        random.shuffle(indices)

        for i in indices:
            old_x, old_y, old_r = current_circles[i]

            # Try to improve position with grid search
            best_pos = [old_x, old_y, old_r]
            best_radius = old_r

            step_size = 0.05 if iteration < 150 else 0.02
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    new_x = old_x + dx
                    new_y = old_y + dy

                    # Ensure within bounds
                    if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                        max_radius = compute_max_radius_voronoi(
                            new_x, new_y, width, height,
                            np.vstack([current_circles[:i], current_circles[i+1:]])
                        )

                        if max_radius > best_radius + 1e-6:
                            best_radius = max_radius
                            best_pos = [new_x, new_y, max_radius]

            if best_pos[2] > current_circles[i][2] + 1e-6:
                current_circles[i] = best_pos
                improved = True

        if not improved:
            break

    return current_circles

def optimize_fine_local(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Apply fine-grained local optimization."""
    current_circles = circles.copy()

    # Fine optimization with comprehensive grid search
    for iteration in range(200):
        improved = False
        indices = list(range(len(current_circles)))
        random.shuffle(indices)

        for i in indices:
            old_x, old_y, old_r = current_circles[i]

            # More extensive grid search around current position
            best_x, best_y, best_r = old_x, old_y, old_r
            best_radius = old_r

            # Try positions in a finer grid around current location
            step = 0.01
            for dx in [-step*3, -step*2, -step, 0, step, step*2, step*3]:
                for dy in [-step*3, -step*2, -step, 0, step, step*2, step*3]:
                    new_x = old_x + dx
                    new_y = old_y + dy

                    if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                        # Compute max radius at new position
                        max_radius = compute_max_radius_voronoi(
                            new_x, new_y, width, height,
                            np.vstack([current_circles[:i], current_circles[i+1:]])
                        )

                        if max_radius > best_radius + 1e-6:
                            best_radius = max_radius
                            best_x, best_y = new_x, new_y
                            improved = True

            # Update if improvement found
            if improved:
                current_circles[i] = [best_x, best_y, best_radius]

        if not improved:
            break

    return current_circles

def optimize_boundary_overlap(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Final boundary and overlap optimization with comprehensive validation."""
    current_circles = circles.copy()

    # Additional refinement focusing on boundary issues
    for iteration in range(100):
        improved = False
        indices = list(range(len(current_circles)))
        random.shuffle(indices)

        for i in indices:
            old_x, old_y, old_r = current_circles[i]

            # Try various position adjustments including boundary proximity
            best_x, best_y, best_r = old_x, old_y, old_r
            best_radius = old_r

            # Grid search around current position with varying step sizes
            steps = [0.02, 0.01]
            for step in steps:
                for dx in [-step*2, -step, 0, step, step*2]:
                    for dy in [-step*2, -step, 0, step, step*2]:
                        new_x = old_x + dx
                        new_y = old_y + dy

                        # Ensure within bounds with safety
                        if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                            # Compute max radius at new position
                            max_radius = compute_max_radius_voronoi(
                                new_x, new_y, width, height,
                                np.vstack([current_circles[:i], current_circles[i+1:]])
                            )

                            if max_radius > best_radius + 1e-6:
                                best_radius = max_radius
                                best_x, best_y = new_x, new_y
                                improved = True

            # Update if improvement found
            if improved:
                current_circles[i] = [best_x, best_y, best_radius]

        if not improved:
            break

    # Final boundary adjustment with additional safety margins
    for i in range(len(current_circles)):
        x, y, r = current_circles[i]
        # Ensure circle stays within bounds with generous safety margin
        r = min(r, x - 0.005, width - x - 0.005, y - 0.005, height - y - 0.005)
        r = max(r, 0.001)
        current_circles[i] = [x, y, r]

    return current_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")