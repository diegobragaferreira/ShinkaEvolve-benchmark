# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from math import sqrt

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

    # Initialize with Voronoi-based strategy
    circles = initialize_voronoi_layout(width, height, 21)

    # Apply multi-scale optimization with progressive refinement
    circles = multi_scale_optimization(circles, width, height)

    # Final physics-based refinement
    circles = optimize_with_forces(circles, width, height)

    # Final grid search refinement
    circles = final_grid_refinement(circles, width, height)

    return circles

def initialize_voronoi_layout(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circles using Voronoi-based strategic seeding."""
    # Create initial points using a combination of regular grid and random sampling
    # This creates a good distribution for Voronoi generation

    # Regular grid points for structured initial distribution
    grid_rows, grid_cols = 5, 5
    x_grid = np.linspace(0.1, width - 0.1, grid_cols)
    y_grid = np.linspace(0.1, height - 0.1, grid_rows)

    grid_points = []
    for x in x_grid:
        for y in y_grid:
            grid_points.append([x, y])

    # Add corner points for better coverage
    corner_points = [
        [0.1, 0.1], [width-0.1, 0.1], [0.1, height-0.1], [width-0.1, height-0.1]
    ]

    # Combine and shuffle initial points
    initial_points = grid_points + corner_points
    random.shuffle(initial_points)

    # Generate Voronoi diagram from initial points
    vor = Voronoi(initial_points[:min(n, len(initial_points))])

    # Use Voronoi vertices as circle center candidates
    circles = np.zeros((n, 3))

    # For first few points, use Voronoi vertices
    vertex_idx = 0
    for i in range(n):
        if vertex_idx < len(vor.vertices) and i < n:
            x, y = vor.vertices[vertex_idx]
            # Ensure point is within bounds
            if 0.1 <= x <= width - 0.1 and 0.1 <= y <= height - 0.1:
                # Compute max radius at this location
                max_radius = compute_max_radius(x, y, width, height, circles[:i])
                circles[i] = [x, y, max_radius]
                vertex_idx += 1
            else:
                # Fall back to random placement
                x = np.random.uniform(0.1, width - 0.1)
                y = np.random.uniform(0.1, height - 0.1)
                max_radius = compute_max_radius(x, y, width, height, circles[:i])
                circles[i] = [x, y, max_radius]
        else:
            # Random placement for remaining circles
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            max_radius = compute_max_radius(x, y, width, height, circles[:i])
            circles[i] = [x, y, max_radius]

    return circles

def compute_max_radius(x: float, y: float, width: float, height: float, existing_circles: np.ndarray) -> float:
    """Compute maximum radius for a circle at (x,y) given existing circles and container boundaries."""
    # Boundary constraints
    min_dist_from_edge = min(x, width - x, y, height - y)

    if min_dist_from_edge <= 0:
        return 0

    # Overlap constraints with existing circles
    min_dist_from_others = float('inf')

    for circle in existing_circles:
        if circle[2] > 0:  # Only consider placed circles
            cx, cy, cr = circle
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            min_dist_from_others = min(min_dist_from_others, dist - cr)

    # Take minimum of boundary and overlap constraints
    max_radius = min(min_dist_from_edge, min_dist_from_others)

    return max(0.001, max_radius)

def multi_scale_optimization(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Multi-scale optimization with progressive refinement."""
    current_circles = circles.copy()

    # Scale 1: Coarse global optimization (larger steps)
    for iteration in range(100):
        improved = False
        # Shuffle circle indices for better exploration
        indices = list(range(len(current_circles)))
        random.shuffle(indices)

        for i in indices:
            # Try to increase radius
            old_r = current_circles[i][2]
            max_radius = compute_max_radius(
                current_circles[i][0], current_circles[i][1],
                width, height,
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )

            if max_radius > old_r + 1e-6:
                current_circles[i][2] = max_radius
                improved = True

            # Try moving circle with larger steps
            old_x, old_y = current_circles[i][0], current_circles[i][1]
            step_size = 0.1

            # Try several positions
            best_pos = [old_x, old_y, old_r]
            best_radius = old_r

            # Grid search around current position
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    new_x = old_x + dx
                    new_y = old_y + dy

                    # Ensure within bounds
                    if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                        # Compute max radius at new position
                        max_radius = compute_max_radius(
                            new_x, new_y,
                            width, height,
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

    # Scale 2: Medium local search (medium steps)
    for iteration in range(200):
        improved = False
        # Shuffle circle indices for better exploration
        indices = list(range(len(current_circles)))
        random.shuffle(indices)

        for i in indices:
            # Try to increase radius
            old_r = current_circles[i][2]
            max_radius = compute_max_radius(
                current_circles[i][0], current_circles[i][1],
                width, height,
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )

            if max_radius > old_r + 1e-6:
                current_circles[i][2] = max_radius
                improved = True

            # Try moving circle with medium steps
            old_x, old_y = current_circles[i][0], current_circles[i][1]
            step_size = 0.05

            # Try several positions
            best_pos = [old_x, old_y, old_r]
            best_radius = old_r

            # Grid search around current position
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    new_x = old_x + dx
                    new_y = old_y + dy

                    # Ensure within bounds
                    if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                        # Compute max radius at new position
                        max_radius = compute_max_radius(
                            new_x, new_y,
                            width, height,
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

    # Scale 3: Fine local search (small steps)
    for iteration in range(300):
        improved = False
        # Shuffle circle indices for better exploration
        indices = list(range(len(current_circles)))
        random.shuffle(indices)

        for i in indices:
            # Try to increase radius
            old_r = current_circles[i][2]
            max_radius = compute_max_radius(
                current_circles[i][0], current_circles[i][1],
                width, height,
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )

            if max_radius > old_r + 1e-6:
                current_circles[i][2] = max_radius
                improved = True

            # Try moving circle with fine steps
            old_x, old_y = current_circles[i][0], current_circles[i][1]
            step_size = 0.02

            # Try several positions
            best_pos = [old_x, old_y, old_r]
            best_radius = old_r

            # Grid search around current position
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    new_x = old_x + dx
                    new_y = old_y + dy

                    # Ensure within bounds
                    if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                        # Compute max radius at new position
                        max_radius = compute_max_radius(
                            new_x, new_y,
                            width, height,
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

def optimize_with_forces(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize using force-based physics simulation."""
    current_circles = circles.copy()
    num_circles = len(current_circles)

    # Physics simulation parameters
    max_iterations = 300
    damping = 0.95
    repulsion_constant = 100.0
    max_force = 0.1

    for iteration in range(max_iterations):
        # Compute forces on each circle
        forces = np.zeros((num_circles, 2))

        # Repulsion forces from other circles
        for i in range(num_circles):
            x_i, y_i, r_i = current_circles[i]
            force_x, force_y = 0.0, 0.0

            for j in range(num_circles):
                if i != j:
                    x_j, y_j, r_j = current_circles[j]
                    dx = x_i - x_j
                    dy = y_i - y_j
                    dist = np.sqrt(dx*dx + dy*dy)

                    # If circles are too close, create strong repulsion
                    if dist < (r_i + r_j) * 1.05:  # Slight overlap tolerance
                        # Normalized repulsion force
                        if dist > 1e-6:
                            force_magnitude = repulsion_constant / (dist * dist)
                            force_x += force_magnitude * dx / dist
                            force_y += force_magnitude * dy / dist
                    elif dist < 2 * (r_i + r_j):  # Influence zone
                        # Weak repulsion
                        if dist > 1e-6:
                            force_magnitude = repulsion_constant * 0.1 / (dist * dist)
                            force_x += force_magnitude * dx / dist
                            force_y += force_magnitude * dy / dist

            # Boundary forces
            boundary_force_scale = 50.0
            if x_i < 0.1:
                force_x += boundary_force_scale * (0.1 - x_i)
            elif x_i > width - 0.1:
                force_x += boundary_force_scale * (width - 0.1 - x_i)

            if y_i < 0.1:
                force_y += boundary_force_scale * (0.1 - y_i)
            elif y_i > height - 0.1:
                force_y += boundary_force_scale * (height - 0.1 - y_i)

            # Limit maximum force
            force_magnitude = np.sqrt(force_x*force_x + force_y*force_y)
            if force_magnitude > max_force:
                force_x = max_force * force_x / force_magnitude
                force_y = max_force * force_y / force_magnitude

            forces[i] = [force_x, force_y]

        # Apply forces to update positions
        for i in range(num_circles):
            # Update position with velocity and damping
            force_x, force_y = forces[i]
            current_circles[i][0] += force_x * 0.01
            current_circles[i][1] += force_y * 0.01

            # Enforce bounds
            current_circles[i][0] = np.clip(current_circles[i][0], 0.1, width - 0.1)
            current_circles[i][1] = np.clip(current_circles[i][1], 0.1, height - 0.1)

            # Recompute maximum radius for updated position
            x_new, y_new = current_circles[i][0], current_circles[i][1]
            max_radius = compute_max_radius(x_new, y_new, width, height,
                                          np.vstack([current_circles[:i], current_circles[i+1:]]))
            current_circles[i][2] = max_radius

        # Check for convergence (every 50 iterations)
        if iteration % 50 == 0:
            # Quick convergence check - could add more sophisticated check here
            pass

    return current_circles

def final_grid_refinement(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Apply final refinement with grid search."""
    current_circles = circles.copy()

    # Fine optimization with grid search
    for iteration in range(150):
        improved = False

        # Try to improve each circle
        # Shuffle indices for better exploration
        indices = list(range(len(current_circles)))
        random.shuffle(indices)

        for i in indices:
            old_x, old_y, old_r = current_circles[i]

            # Grid search around current position with fine steps
            best_x, best_y, best_r = old_x, old_y, old_r
            best_radius = old_r

            # Try positions in a tighter grid around current location
            step = 0.01
            for dx in [-step*2, -step, 0, step, step*2]:
                for dy in [-step*2, -step, 0, step, step*2]:
                    new_x = old_x + dx
                    new_y = old_y + dy

                    if 0.1 <= new_x <= width - 0.1 and 0.1 <= new_y <= height - 0.1:
                        # Compute max radius at new position
                        max_radius = compute_max_radius(
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

    # Final safety boundary adjustment
    for i in range(len(current_circles)):
        x, y, r = current_circles[i]
        # Ensure circle stays within bounds with safety margin
        r = min(r, x - 0.01, width - x - 0.01, y - 0.01, height - y - 0.01)
        r = max(r, 0.001)
        current_circles[i] = [x, y, r]

    return current_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")