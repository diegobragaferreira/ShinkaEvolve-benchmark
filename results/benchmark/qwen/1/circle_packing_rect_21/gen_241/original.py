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

def mutate_radius(circles: np.ndarray, idx: int, max_delta: float = 0.01, density_scores: np.ndarray = None) -> np.ndarray:
    """Mutate radius with adaptive delta based on Voronoi density."""
    new_circles = circles.copy()
    old_r = new_circles[idx, 2]

    # Adaptive delta based on density
    if density_scores is not None and len(density_scores) > idx:
        # High density means more constraints, use smaller deltas
        density_factor = 1.0 / (1.0 + density_scores[idx])  # Normalize
        delta = max_delta * density_factor
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
    """Perform local optimization using simple gradient-like approach."""
    current_circles = circles.copy()
    best_circles = current_circles.copy()
    best_fitness = evaluate_fitness(current_circles)

    patience_counter = 0

    # Get Voronoi densities for adaptive mutation
    density_scores = compute_voronoi_density(current_circles, rect_width, rect_height)

    for _ in range(max_iter):
        # Try several mutations to improve solution
        improved = False

        # Mutate each circle
        for i in range(len(current_circles)):
            # Try position mutation
            mutated_pos = mutate_position(current_circles, i, 0.02)

            # Try radius mutation
            mutated_rad = mutate_radius(current_circles, i, 0.01, density_scores)

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