# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from math import sqrt
import time
from sklearn.cluster import KMeans
from numba import jit

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

    # Phase 1: Enhanced initialization with hybrid approach
    circles = initialize_hybrid_layout(width, height, 21)

    # Phase 2: Multi-scale optimization with adaptive parameters
    circles = multi_scale_optimization(circles, width, height)

    # Phase 3: Physics-based refinement with force simulation
    circles = physics_based_refinement(circles, width, height)

    # Phase 4: Final adaptive refinement
    circles = final_adaptive_refinement(circles, width, height)

    return circles

def initialize_hybrid_layout(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circles using enhanced hybrid strategy combining Voronoi, k-means, and strategic points."""
    circles = np.zeros((n, 3))

    # Strategy 1: Corner and edge points for boundary coverage
    corner_points = [
        [0.1, 0.1], [width-0.1, 0.1], [0.1, height-0.1], [width-0.1, height-0.1],
        [width/2, 0.1], [width/2, height-0.1], [0.1, height/2], [width-0.1, height/2]
    ]

    # Strategy 2: Generate dense grid points for k-means initialization
    grid_density = 64
    grid_points = []
    for i in range(grid_density):
        for j in range(grid_density):
            x = (i + 0.5) / grid_density * width
            y = (j + 0.5) / grid_density * height
            grid_points.append([x, y])

    # Add some random points to avoid regular grid bias
    for _ in range(100):
        x = np.random.uniform(0.05, width - 0.05)
        y = np.random.uniform(0.05, height - 0.05)
        grid_points.append([x, y])

    # Use k-means++ to get initial centroids
    kmeans = KMeans(n_clusters=min(len(grid_points), n), init='k-means++', n_init=10, random_state=42)
    all_array = np.array(grid_points)
    kmeans.fit(all_array)
    centroids = kmeans.cluster_centers_

    # Select best points from grid for each centroid
    selected_positions = []
    for centroid in centroids:
        distances = np.linalg.norm(all_array - centroid, axis=1)
        closest_idx = np.argmin(distances)
        selected_positions.append(tuple(all_array[closest_idx]))

    # Combine corner points and selected positions
    combined_positions = corner_points + selected_positions
    
    # Ensure we have exactly n positions
    if len(combined_positions) < n:
        # Fill with additional points from grid
        remaining_needed = n - len(combined_positions)
        for i in range(min(remaining_needed, len(grid_points))):
            combined_positions.append(grid_points[i])
    elif len(combined_positions) > n:
        combined_positions = combined_positions[:n]

    # Initialize circles with positions and appropriate radii
    for i in range(n):
        x, y = combined_positions[i]
        # Initial radius based on proximity to boundaries
        max_radius = min(x, width - x, y, height - y) * 0.2
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

@jit(nopython=True)
def compute_max_radius_fast(x: float, y: float, positions: np.ndarray, radii: np.ndarray, n: int) -> float:
    """Fast computation of maximum radius for a circle at (x,y) using numba compilation."""
    # Boundary constraints
    min_dist_from_edge = min(x, 1.2 - x, y, 0.8 - y)
    
    if min_dist_from_edge <= 0:
        return 0.0

    # Overlap constraints with existing circles
    min_dist_from_others = 1e10

    for i in range(n):
        dx = x - positions[i, 0]
        dy = y - positions[i, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 1e-8:
            max_radius_for_this_circle = dist - radii[i]
            if max_radius_for_this_circle < min_dist_from_others:
                min_dist_from_others = max_radius_for_this_circle

    # Take minimum of boundary and overlap constraints
    max_radius = min(min_dist_from_edge, min_dist_from_others)
    return max(0.001, max_radius)

def multi_scale_optimization(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Multi-scale optimization with adaptive parameters and better convergence handling."""
    current_circles = circles.copy()
    num_circles = len(current_circles)
    best_sum = np.sum(current_circles[:, 2])
    best_config = current_circles.copy()
    
    # Phase 1: Coarse optimization (larger steps)
    for iteration in range(150):
        improved = False
        indices = list(range(num_circles))
        random.shuffle(indices)

        for i in indices:
            # Try to increase radius
            max_radius = compute_max_radius(
                current_circles[i][0], current_circles[i][1],
                width, height,
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )

            if max_radius > current_circles[i][2] + 1e-6:
                current_circles[i][2] = max_radius
                improved = True
                if max_radius > best_sum:
                    best_sum = max_radius
                    best_config = current_circles.copy()

            # Try moving circle with larger steps
            old_x, old_y = current_circles[i][0], current_circles[i][1]
            step_size = 0.15

            # Try several positions
            best_pos = [old_x, old_y, current_circles[i][2]]
            best_radius = current_circles[i][2]

            directions = [(0, 0)]
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    directions.append((dx, dy))

            for dx, dy in directions:
                new_x = old_x + dx
                new_y = old_y + dy

                # Ensure within bounds
                if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                    max_radius = compute_max_radius(
                        new_x, new_y,
                        width, height,
                        np.vstack([current_circles[:i], current_circles[i+1:]])
                    )

                    if max_radius > best_radius + 1e-6:
                        best_radius = max_radius
                        best_pos = [new_x, new_y, max_radius]
                        improved = True

            if best_pos[2] > current_circles[i][2] + 1e-6:
                current_circles[i] = best_pos

        if not improved:
            break

    # Phase 2: Medium optimization (medium steps) with simulated annealing
    temperature = 1.0
    cooling_rate = 0.97
    
    for iteration in range(200):
        improved = False
        indices = list(range(num_circles))
        random.shuffle(indices)

        temperature *= cooling_rate

        for i in indices:
            # Try to increase radius
            max_radius = compute_max_radius(
                current_circles[i][0], current_circles[i][1],
                width, height,
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )

            if max_radius > current_circles[i][2] + 1e-6:
                current_circles[i][2] = max_radius
                improved = True

            # Try moving circle with medium steps
            old_x, old_y = current_circles[i][0], current_circles[i][1]
            step_size = 0.08

            # Try several positions
            best_pos = [old_x, old_y, current_circles[i][2]]
            best_radius = current_circles[i][2]

            # Grid search with some random perturbations
            grid_points = []
            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    grid_points.append((dx, dy))

            # Add some random perturbations
            for _ in range(2):
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                grid_points.append((dx, dy))

            for dx, dy in grid_points:
                new_x = old_x + dx
                new_y = old_y + dy

                # Ensure within bounds
                if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                    max_radius = compute_max_radius(
                        new_x, new_y,
                        width, height,
                        np.vstack([current_circles[:i], current_circles[i+1:]])
                    )

                    # Accept with probability based on temperature
                    if max_radius > best_radius + 1e-6:
                        best_radius = max_radius
                        best_pos = [new_x, new_y, max_radius]
                        improved = True
                    elif random.random() < np.exp(-(best_radius - max_radius) / (temperature * 0.1)) and max_radius > best_radius:
                        best_radius = max_radius
                        best_pos = [new_x, new_y, max_radius]
                        improved = True

            if best_pos[2] > current_circles[i][2] + 1e-6:
                current_circles[i] = best_pos

        if not improved:
            break

    # Phase 3: Fine optimization (small steps)
    for iteration in range(300):
        improved = False
        indices = list(range(num_circles))
        random.shuffle(indices)

        for i in indices:
            # Try to increase radius
            max_radius = compute_max_radius(
                current_circles[i][0], current_circles[i][1],
                width, height,
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )

            if max_radius > current_circles[i][2] + 1e-6:
                current_circles[i][2] = max_radius
                improved = True

            # Try moving circle with fine steps
            old_x, old_y = current_circles[i][0], current_circles[i][1]
            step_size = 0.03

            # Grid search around current position
            best_pos = [old_x, old_y, current_circles[i][2]]
            best_radius = current_circles[i][2]

            for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                    new_x = old_x + dx
                    new_y = old_y + dy

                    # Ensure within bounds
                    if 0.01 <= new_x <= width - 0.01 and 0.01 <= new_y <= height - 0.01:
                        max_radius = compute_max_radius(
                            new_x, new_y,
                            width, height,
                            np.vstack([current_circles[:i], current_circles[i+1:]])
                        )

                        if max_radius > best_radius + 1e-6:
                            best_radius = max_radius
                            best_pos = [new_x, new_y, max_radius]
                            improved = True

            if best_pos[2] > current_circles[i][2] + 1e-6:
                current_circles[i] = best_pos

        if not improved:
            break

    return current_circles

def physics_based_refinement(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Physics-based refinement using force simulation."""
    current_circles = circles.copy()
    num_circles = len(current_circles)

    # Physics simulation parameters
    max_iterations = 150
    damping = 0.95
    repulsion_constant = 50.0
    max_force = 0.05

    for iteration in range(max_iterations):
        # Compute forces on each circle
        forces = np.zeros((num_circles, 2))

        # Repulsion forces from other circles
        positions = current_circles[:, :2]
        radii = current_circles[:, 2]

        for i in range(num_circles):
            x_i, y_i = positions[i]
            r_i = radii[i]
            force_x, force_y = 0.0, 0.0

            for j in range(num_circles):
                if i != j:
                    x_j, y_j = positions[j]
                    r_j = radii[j]
                    dx = x_i - x_j
                    dy = y_i - y_j
                    dist = np.sqrt(dx*dx + dy*dy)

                    # Strong repulsion for overlapping circles
                    if dist < (r_i + r_j) * 1.05:
                        if dist > 1e-6:
                            force_magnitude = repulsion_constant / (dist * dist)
                            force_x += force_magnitude * dx / dist
                            force_y += force_magnitude * dy / dist
                    elif dist < 2 * (r_i + r_j):
                        # Weak repulsion for nearby circles
                        if dist > 1e-6:
                            force_magnitude = repulsion_constant * 0.1 / (dist * dist)
                            force_x += force_magnitude * dx / dist
                            force_y += force_magnitude * dy / dist

            # Boundary forces
            boundary_force_scale = 30.0
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

    return current_circles

def final_adaptive_refinement(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Final adaptive refinement with precise convergence criteria."""
    current_circles = circles.copy()
    num_circles = len(current_circles)

    # Fine optimization with grid search
    max_iterations = 150
    improvement_threshold = 1e-6

    for iteration in range(max_iterations):
        improved = False
        indices = list(range(num_circles))
        random.shuffle(indices)

        for i in indices:
            old_x, old_y, old_r = current_circles[i]

            # Grid search around current position
            best_x, best_y, best_r = old_x, old_y, old_r
            best_radius = old_r

            step_sizes = [0.005, 0.01, 0.02, 0.03]
            for step in step_sizes:
                for dx in [-step*2, -step, 0, step, step*2]:
                    for dy in [-step*2, -step, 0, step, step*2]:
                        new_x = old_x + dx
                        new_y = old_y + dy

                        if 0.1 <= new_x <= width - 0.1 and 0.1 <= new_y <= height - 0.1:
                            max_radius = compute_max_radius(
                                new_x, new_y, width, height,
                                np.vstack([current_circles[:i], current_circles[i+1:]])
                            )

                            if max_radius > best_radius + improvement_threshold:
                                best_radius = max_radius
                                best_x, best_y = new_x, new_y
                                improved = True

            if improved:
                current_circles[i] = [best_x, best_y, best_radius]

        if not improved:
            break

    # Final safety adjustments
    for i in range(num_circles):
        x, y, r = current_circles[i]
        r = min(r, x - 0.01, width - x - 0.01, y - 0.01, height - y - 0.01)
        r = max(r, 0.001)
        current_circles[i] = [x, y, r]

    return current_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")