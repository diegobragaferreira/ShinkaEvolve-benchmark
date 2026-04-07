# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import differential_evolution
import random
from typing import Tuple, List
import time

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_valid_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Check if all circles are within bounds and non-overlapping using efficient vectorized approach."""
    n = len(circles)

    # Check bounds - vectorized
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]

    if np.any(x_coords - radii < 0) or np.any(x_coords + radii > rect_width) or \
       np.any(y_coords - radii < 0) or np.any(y_coords + radii > rect_height):
        return False

    # Check overlaps - vectorized using broadcasting
    x_diff = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    y_diff = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    dists = np.sqrt(x_diff**2 + y_diff**2)
    sums = radii[:, np.newaxis] + radii[np.newaxis, :]

    # Set diagonal to infinity to avoid self-comparison
    np.fill_diagonal(dists, np.inf)

    # Check if any distances are less than sum of radii
    if np.any(dists < sums):
        return False

    return True

def compute_voronoi_constraints(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Voronoi-based constraint information for each circle.

    Returns:
        Tuple of (constraint_distances, voronoi_areas) where:
        - constraint_distances: minimum distance to nearest constraint (neighbor or boundary)
        - voronoi_areas: Voronoi cell areas for each circle
    """
    # Add boundary points for proper Voronoi calculation
    points = circles[:, :2].copy()

    # Add boundary points to make Voronoi more meaningful
    boundary_points = [
        [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
        [rect_width/2, 0], [rect_width/2, rect_height],
        [0, rect_height/2], [rect_width, rect_height/2]
    ]
    points = np.vstack([points, boundary_points])

    try:
        vor = Voronoi(points)

        # For each original point, compute Voronoi cell area and constraint distances
        areas = []
        constraint_dists = []

        for i in range(len(circles)):
            # Compute Voronoi cell area
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1

            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    # Compute area of polygon using shoelace formula
                    vertices = np.array([vor.vertices[j] for j in region])
                    if len(vertices) >= 3:
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
            else:
                areas.append(1.0)

            # Compute minimum constraint distance (to nearest neighbor or boundary)
            x, y = circles[i, 0], circles[i, 1]
            min_dist = float('inf')

            # Check distance to all other circles
            for j in range(len(circles)):
                if i != j:
                    dist = distance((x, y), (circles[j, 0], circles[j, 1]))
                    min_dist = min(min_dist, dist)

            # Check distance to boundaries
            boundary_dists = [x, y, rect_width - x, rect_height - y]
            min_boundary_dist = min(boundary_dists)
            min_dist = min(min_dist, min_boundary_dist)

            constraint_dists.append(min_dist)

        return np.array(constraint_dists), np.array(areas)
    except:
        # Fallback to uniform distribution if Voronoi fails
        return np.ones(len(circles)), np.ones(len(circles))

def mutate_radius(circles: np.ndarray, idx: int, max_delta: float = 0.01,
                  constraint_distances: np.ndarray = None,
                  voronoi_areas: np.ndarray = None) -> np.ndarray:
    """Mutate radius with adaptive delta based on actual geometric constraints."""
    new_circles = circles.copy()
    old_r = new_circles[idx, 2]

    # Adaptive delta based on constraint distance
    # Circles with smaller constraint distances (tighter constraints) get smaller deltas
    if constraint_distances is not None and len(constraint_distances) > idx:
        # Use constraint distance as proxy for how much we can safely change
        # Constraint distance of 0.05 suggests we're at tight constraint, so use very small delta
        # Constraint distance of 0.2 suggests more room, so larger delta
        constraint_factor = np.clip(constraint_distances[idx] / 0.2, 0.1, 1.0)
        delta = max_delta * constraint_factor
    else:
        delta = max_delta

    # Random small perturbation
    delta_r = np.random.uniform(-delta, delta)
    new_r = old_r + delta_r

    # Ensure positive radius
    new_r = max(0.001, new_r)
    new_circles[idx, 2] = new_r

    return new_circles

def mutate_position(circles: np.ndarray, idx: int, max_delta: float = 0.05) -> np.ndarray:
    """Mutate position with small random perturbation."""
    new_circles = circles.copy()
    old_x, old_y = new_circles[idx, 0], new_circles[idx, 1]

    # Small random perturbation
    delta_x = np.random.uniform(-max_delta, max_delta)
    delta_y = np.random.uniform(-max_delta, max_delta)

    new_x = old_x + delta_x
    new_y = old_y + delta_y

    # Ensure within bounds (with some margin)
    new_x = np.clip(new_x, 0.01, 0.99)
    new_y = np.clip(new_y, 0.01, 0.99)

    new_circles[idx, 0] = new_x
    new_circles[idx, 1] = new_y

    return new_circles

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def generate_hexagonal_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Generate initial hexagonal pattern."""
    circles = np.zeros((n, 3))

    # Hexagonal packing parameters
    rows = int(np.sqrt(n))
    cols = int(n / rows) + 1

    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)

    # Adjust spacing to fit better
    min_radius = 0.02

    for i in range(n):
        row = i // cols
        col = i % cols
        x = (col + 1) * spacing_x
        y = (row + 1) * spacing_y

        # Offset every other row for hexagonal arrangement
        if row % 2 == 1:
            x += spacing_x / 2

        circles[i] = [x, y, min_radius]

    return circles

def generate_triangular_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Generate triangular pattern."""
    circles = np.zeros((n, 3))

    # Arrange in triangular pattern
    sqrt_n = int(np.ceil(np.sqrt(n)))
    spacing_x = rect_width / (sqrt_n + 1)
    spacing_y = rect_height / (sqrt_n + 1)

    idx = 0
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Slight offset for triangular pattern
            if i % 2 == 1:
                x += spacing_x / 2
            circles[idx] = [x, y, 0.02]
            idx += 1
        if idx >= n:
            break

    return circles

def generate_square_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Generate square grid pattern."""
    circles = np.zeros((n, 3))

    sqrt_n = int(np.ceil(np.sqrt(n)))
    spacing_x = rect_width / (sqrt_n + 1)
    spacing_y = rect_height / (sqrt_n + 1)

    idx = 0
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            circles[idx] = [x, y, 0.02]
            idx += 1
        if idx >= n:
            break

    return circles

def generate_random_pattern(n: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Generate random initial pattern."""
    circles = np.zeros((n, 3))

    # Generate random positions and large initial radii
    for i in range(n):
        x = np.random.uniform(0.05, rect_width - 0.05)
        y = np.random.uniform(0.05, rect_height - 0.05)
        r = 0.05  # Start with larger radius
        circles[i] = [x, y, r]

    return circles

def local_optimization(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0,
                       max_iter: int = 50, patience: int = 10) -> np.ndarray:
    """Perform local optimization using enhanced approach with boundary awareness."""
    current_circles = circles.copy()
    best_circles = current_circles.copy()
    best_fitness = evaluate_fitness(current_circles)

    patience_counter = 0

    # Get Voronoi constraints for adaptive mutation
    constraint_distances, voronoi_areas = compute_voronoi_constraints(current_circles, rect_width, rect_height)

    for iteration in range(max_iter):
        # Try several mutations to improve solution
        improved = False

        # First, prioritize boundary circles for special treatment
        boundary_indices = []
        for i in range(len(current_circles)):
            x, y, r = current_circles[i]
            # Check if circle is near boundary (within 0.05 units)
            if (x <= r + 0.05 or x >= rect_width - r - 0.05 or
                y <= r + 0.05 or y >= rect_height - r - 0.05):
                boundary_indices.append(i)

        # Process boundary circles with higher precision
        if boundary_indices:
            for i in boundary_indices:
                # Try boundary-aware mutations with constraint-aware deltas
                mutated_rad = mutate_radius(current_circles, i, 0.01, constraint_distances, voronoi_areas)
                mutated_pos = mutate_position(current_circles, i, 0.01)

                # Test both mutations
                rad_fitness = evaluate_fitness(mutated_rad)
                pos_fitness = evaluate_fitness(mutated_pos)

                # Choose better mutation
                if rad_fitness > pos_fitness:
                    if is_valid_solution(mutated_rad, rect_width, rect_height):
                        current_circles = mutated_rad
                        improved = True
                else:
                    if is_valid_solution(mutated_pos, rect_width, rect_height):
                        current_circles = mutated_pos
                        improved = True

        # Then process all circles normally but with better constraint awareness
        # Sort by constraint distance to focus on most constrained first
        if len(constraint_distances) > 0:
            sorted_indices = np.argsort(constraint_distances)
            indices_to_process = [i for i in sorted_indices if i not in boundary_indices]
        else:
            indices_to_process = list(range(len(current_circles)))

        for i in indices_to_process:
            # Try position mutation with adaptive delta
            mutated_pos = mutate_position(current_circles, i, 0.02)

            # Try radius mutation with constraint-aware delta
            mutated_rad = mutate_radius(current_circles, i, 0.01, constraint_distances, voronoi_areas)

            # Evaluate both mutations
            pos_fitness = evaluate_fitness(mutated_pos)
            rad_fitness = evaluate_fitness(mutated_rad)

            # Choose the better one
            if pos_fitness > rad_fitness:
                if is_valid_solution(mutated_pos, rect_width, rect_height):
                    current_circles = mutated_pos
                    improved = True
            else:
                if is_valid_solution(mutated_rad, rect_width, rect_height):
                    current_circles = mutated_rad
                    improved = True

        # Update best solution
        current_fitness = evaluate_fitness(current_circles)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_circles = current_circles.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping if no improvement
        if patience_counter >= patience:
            break

    return best_circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    rect_width = 1.0  # Since perimeter = 4 and width + height = 2, we can set width = height = 1
    rect_height = 1.0

    # Try multiple initialization strategies
    initial_patterns = [
        generate_hexagonal_pattern(n, rect_width, rect_height),
        generate_triangular_pattern(n, rect_width, rect_height),
        generate_square_pattern(n, rect_width, rect_height),
        generate_random_pattern(n, rect_width, rect_height)
    ]

    best_solution = None
    best_score = -float('inf')

    # Multi-start optimization
    for seed_pattern in initial_patterns:
        # Apply local optimization to get better starting points
        optimized_pattern = local_optimization(seed_pattern, rect_width, rect_height, max_iter=30)

        # Further refine using a few rounds of local search
        final_circles = local_optimization(optimized_pattern, rect_width, rect_height, max_iter=20)

        score = evaluate_fitness(final_circles)
        if score > best_score and is_valid_solution(final_circles, rect_width, rect_height):
            best_score = score
            best_solution = final_circles.copy()

    # Final fine-tuning
    if best_solution is not None:
        # Apply more extensive local optimization
        best_solution = local_optimization(best_solution, rect_width, rect_height, max_iter=50)

    # Ensure final validity
    if best_solution is None:
        # Fallback to simple initialization
        best_solution = generate_random_pattern(n, rect_width, rect_height)
        best_solution = local_optimization(best_solution, rect_width, rect_height, max_iter=100)

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")