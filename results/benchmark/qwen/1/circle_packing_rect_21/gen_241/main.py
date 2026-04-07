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

def compute_voronoi_density(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Compute Voronoi cell areas for each circle to estimate constraint density."""
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

        # For each original point, compute Voronoi cell area
        areas = []
        for i in range(len(circles)):
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1

            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    # Compute area of polygon using shoelace formula
                    vertices = np.array([vor.vertices[i] for i in region])
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

        return np.array(areas)
    except:
        # Fallback to uniform distribution if Voronoi fails
        return np.ones(len(circles))

def mutate_radius(circles: np.ndarray, idx: int, density_scores: np.ndarray = None,
                  boundary_distances: np.ndarray = None, base_delta: float = 0.01) -> np.ndarray:
    """Mutate radius with adaptive delta based on Voronoi density and boundary distance."""
    new_circles = circles.copy()
    old_r = new_circles[idx, 2]

    # Adaptive delta based on both Voronoi density and boundary distance
    delta = base_delta

    # Factor based on Voronoi density - smaller cells (more constrained) get smaller deltas
    if density_scores is not None and len(density_scores) > idx:
        # Normalize and invert - smaller Voronoi areas mean more constraints
        normalized_area = density_scores[idx] / max(1e-8, np.mean(density_scores))
        # Use inverse relationship: smaller areas = smaller delta (more conservative)
        density_factor = 1.0 / (1.0 + normalized_area * 0.5)
        delta *= density_factor

    # Factor based on boundary distance - circles near boundaries get smaller steps
    if boundary_distances is not None and len(boundary_distances) > idx:
        boundary_factor = max(0.2, min(1.0, boundary_distances[idx] / 0.05))
        delta *= boundary_factor

    # Random small perturbation with weighted probability
    # 70% chance of small change, 30% chance of larger change
    if np.random.random() < 0.7:
        delta_r = np.random.uniform(-delta * 0.5, delta * 0.5)
    else:
        delta_r = np.random.uniform(-delta, delta)

    new_r = old_r + delta_r

    # Ensure positive radius
    new_r = max(0.001, new_r)
    new_circles[idx, 2] = new_r

    return new_circles

def mutate_position(circles: np.ndarray, idx: int, density_scores: np.ndarray = None,
                   boundary_distances: np.ndarray = None, base_delta: float = 0.05) -> np.ndarray:
    """Mutate position with adaptive delta based on Voronoi density and boundary distance."""
    new_circles = circles.copy()
    old_x, old_y = new_circles[idx, 0], new_circles[idx, 1]

    # Base delta scaled by boundary proximity
    delta = base_delta

    # Adaptive delta based on Voronoi density and boundary distance
    if density_scores is not None and len(density_scores) > idx:
        # Smaller Voronoi areas mean more constrained regions
        normalized_area = density_scores[idx] / max(1e-8, np.mean(density_scores))
        density_factor = 1.0 / (1.0 + normalized_area * 0.3)  # Less aggressive than radius
        delta *= density_factor

    # Boundary factor - circles near edges get smaller steps
    if boundary_distances is not None and len(boundary_distances) > idx:
        boundary_factor = max(0.2, min(1.0, boundary_distances[idx] / 0.05))
        delta *= boundary_factor

    # Ensure minimum delta to prevent stagnation
    delta = max(0.001, delta)

    # Small random perturbation
    delta_x = np.random.uniform(-delta, delta)
    delta_y = np.random.uniform(-delta, delta)

    new_x = old_x + delta_x
    new_y = old_y + delta_y

    # Ensure within bounds (with some margin)
    margin = 0.01
    x, y, r = new_circles[idx]
    new_x = np.clip(new_x, r + margin, 1.0 - r - margin)
    new_y = np.clip(new_y, r + margin, 1.0 - r - margin)

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
    """Perform local optimization using Voronoi-aware adaptive mutation."""
    current_circles = circles.copy()
    best_circles = current_circles.copy()
    best_fitness = evaluate_fitness(current_circles)

    patience_counter = 0

    # Precompute Voronoi and boundary metrics
    try:
        density_scores, boundary_distances = compute_voronoi_metrics(current_circles, rect_width, rect_height)
    except:
        # Fallback if Voronoi computation fails
        density_scores = compute_voronoi_density(current_circles, rect_width, rect_height)
        boundary_distances = np.ones(len(current_circles))

    # Identify boundary circles for special treatment
    boundary_indices = []
    for i in range(len(current_circles)):
        x, y, r = current_circles[i]
        # Check if circle is near boundary
        if (x <= r + 0.05 or x >= rect_width - r - 0.05 or
            y <= r + 0.05 or y >= rect_height - r - 0.05):
            boundary_indices.append(i)

    for iteration in range(max_iter):
        improved = False

        # Process boundary circles first (they're more constrained)
        if boundary_indices:
            for i in boundary_indices:
                # Special handling for boundary circles
                mutated_rad = mutate_radius(current_circles, i, density_scores, boundary_distances, 0.008)
                mutated_pos = mutate_position(current_circles, i, density_scores, boundary_distances, 0.01)

                # Test both mutations
                rad_fitness = evaluate_fitness(mutated_rad)
                pos_fitness = evaluate_fitness(mutated_pos)

                # Choose better mutation, but prefer position mutation for boundary circles
                if pos_fitness > rad_fitness:
                    if is_valid_solution(mutated_pos, rect_width, rect_height):
                        current_circles = mutated_pos
                        improved = True
                else:
                    if is_valid_solution(mutated_rad, rect_width, rect_height):
                        current_circles = mutated_rad
                        improved = True

        # Process remaining circles in batches
        remaining_indices = [i for i in range(len(current_circles)) if i not in boundary_indices]

        # Shuffle for better exploration
        np.random.shuffle(remaining_indices)

        # Process in groups to balance exploration and exploitation
        batch_size = max(1, len(remaining_indices) // 3)
        for start_idx in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[start_idx:start_idx + batch_size]
            for i in batch_indices:
                # Try position mutation with adaptive delta
                mutated_pos = mutate_position(current_circles, i, density_scores, boundary_distances, 0.02)

                # Try radius mutation with adaptive delta
                mutated_rad = mutate_radius(current_circles, i, density_scores, boundary_distances, 0.01)

                # Evaluate both mutations in order of preference
                pos_fitness = evaluate_fitness(mutated_pos)
                rad_fitness = evaluate_fitness(mutated_rad)

                # Priority to position changes when they offer better fitness
                if pos_fitness >= rad_fitness and is_valid_solution(mutated_pos, rect_width, rect_height):
                    current_circles = mutated_pos
                    improved = True
                elif is_valid_solution(mutated_rad, rect_width, rect_height):
                    current_circles = mutated_rad
                    improved = True

        # Update best solution and recalculate metrics periodically
        current_fitness = evaluate_fitness(current_circles)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_circles = current_circles.copy()
            patience_counter = 0

            # Recompute Voronoi metrics every few iterations for better adaptation
            if iteration % 5 == 0:
                try:
                    density_scores, boundary_distances = compute_voronoi_metrics(current_circles, rect_width, rect_height)
                except:
                    density_scores = compute_voronoi_density(current_circles, rect_width, rect_height)
                    boundary_distances = np.ones(len(current_circles))
        else:
            patience_counter += 1

        # Early stopping if no improvement for too long
        if patience_counter >= patience or iteration > max_iter // 2:
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