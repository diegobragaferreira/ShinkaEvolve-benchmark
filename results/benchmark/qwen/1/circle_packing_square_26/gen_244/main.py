# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List

# Global constants for the optimization
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 500

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if a configuration of circles is valid (no overlaps, fully contained)."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r + BOUNDARY_MARGIN or x > 1-r - BOUNDARY_MARGIN or y < r + BOUNDARY_MARGIN or y > 1-r - BOUNDARY_MARGIN:
            return False

    # Check overlap constraints with early termination
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_squared = (x1-x2)**2 + (y1-y2)**2
            min_distance_squared = (r1 + r2)**2
            if distance_squared < min_distance_squared:
                return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii."""
    return np.sum(circles[:, 2])

def create_voronoi_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration using enhanced Voronoi-based spreading with priority-based circle packing."""
    # Create a structured grid pattern with randomness for better distribution
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    grid_points = []

    # Generate hexagonal grid pattern
    for i in range(grid_size):
        for j in range(grid_size):
            if len(grid_points) < n_circles:
                # Hexagonal offset for better distribution
                offset = (j % 2) * 0.5
                x = (j + 0.5 + offset) / grid_size
                y = (i + 0.5) / grid_size
                grid_points.append([x, y])

    points = np.array(grid_points)

    # Add some randomness to avoid perfect grids that might cause issues
    noise_level = 0.02
    points += np.random.uniform(-noise_level, noise_level, points.shape)

    # Clip points to ensure they're within bounds
    points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)

    # Add boundary points for better coverage
    boundary_points = []
    for _ in range(10):
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            boundary_points.append([np.random.rand(), 1.0 - BOUNDARY_MARGIN])
        elif side == 1:  # Bottom
            boundary_points.append([np.random.rand(), BOUNDARY_MARGIN])
        elif side == 2:  # Left
            boundary_points.append([BOUNDARY_MARGIN, np.random.rand()])
        else:  # Right
            boundary_points.append([1.0 - BOUNDARY_MARGIN, np.random.rand()])

    points = np.vstack([points, boundary_points])

    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)
        # Use Voronoi cell centers as initial circle positions
        centroids = vor.points[vor.point_region[:-1]]  # Exclude infinite region

        # Calculate Voronoi cell areas to prioritize larger cells for bigger circles
        cell_areas = []
        for i, (x, y) in enumerate(vor.points):
            if i < len(vor.point_region) and vor.point_region[i] >= 0:
                region = vor.regions[vor.point_region[i]]
                if len(region) > 0 and all(r >= 0 for r in region):
                    vertices = np.array([vor.vertices[r] for r in region])
                    if len(vertices) > 0:
                        # Calculate area using shoelace formula
                        n = len(vertices)
                        if n > 2:
                            area = 0.5 * abs(sum(vertices[i][0] * vertices[(i+1)%n][1] -
                                               vertices[(i+1)%n][0] * vertices[i][1]
                                               for i in range(n)))
                        else:
                            area = 0.0
                    else:
                        area = 0.0
                else:
                    area = 0.0
            else:
                area = 0.0
            cell_areas.append(area)

        # Sort centroids by corresponding cell area (descending) to prioritize larger cells
        sorted_indices = np.argsort(cell_areas)[::-1][:n_circles]
        selected_centroids = centroids[sorted_indices]

        # Create circles with initial radii based on Voronoi cell sizes
        circles = np.zeros((n_circles, 3))

        # Assign radii based on Voronoi cell areas - larger cells get larger radii
        # Also ensure proper distribution by considering distance to neighbors
        for i in range(n_circles):
            x, y = selected_centroids[i]

            # Find nearest neighbors to estimate appropriate radius
            distances = np.sqrt(np.sum((selected_centroids - [x, y])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance
            if len(distances) > 0:
                min_distance = np.min(distances)
                # Base radius on cell area and distance to neighbors
                # Larger Voronoi cells get larger initial radii
                area_factor = cell_areas[sorted_indices[i]] / max(cell_areas) if max(cell_areas) > 0 else 1.0
                base_radius = min(min_distance * 0.4, 0.3) * area_factor
                radius = min(base_radius, 0.25)
            else:
                radius = 0.1

            # Ensure it's within bounds
            radius = min(radius, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                        y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)

            circles[i] = [x, y, max(radius, 0.001)]

        return circles
    except:
        # Fallback to grid-based initialization if Voronoi fails
        return generate_grid_initialization(n_circles)

def generate_grid_initialization(n_circles: int) -> np.ndarray:
    """Generate grid-based initial configuration."""
    circles = np.zeros((n_circles, 3))
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / grid_size
    r = spacing * 0.3

    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n_circles:
                x = (j + 0.5) * spacing
                y = (i + 0.5) * spacing
                # Adjust for boundary constraints
                x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                circles[count] = [x, y, r]
                count += 1

    return circles

def greedy_refinement(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Apply greedy refinement to improve the circle packing solution.
    This is a multi-phase approach that alternates between:
    1. Radius expansion phase
    2. Position optimization phase
    """
    # Clone input to avoid modifying original
    current_solution = circles.copy()

    for iteration in range(max_iterations):
        improved = False

        # Phase 1: Try to expand each circle's radius as much as possible
        for i in range(len(current_solution)):
            x, y, r = current_solution[i]

            # Calculate maximum possible radius at this position
            max_radius = min(x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                           y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)

            if max_radius <= r:
                continue

            # Binary search for maximum safe expansion
            low = 0.0
            high = max_radius - r
            best_expansion = 0.0

            # Binary search iterations for precision
            for _ in range(12):
                test_expansion = (low + high) / 2
                test_radius = r + test_expansion

                # Check if expansion violates any constraints
                valid = True
                for j in range(len(current_solution)):
                    if i != j:
                        pos_j = current_solution[j, :2]
                        r_j = current_solution[j, 2]
                        dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                        if dist < (test_radius + r_j):
                            valid = False
                            break

                if valid:
                    best_expansion = test_expansion
                    low = test_expansion
                else:
                    high = test_expansion

            if best_expansion > 0.0001:  # Only accept meaningful improvements
                current_solution[i, 2] = r + best_expansion
                improved = True

        # Phase 2: Try to improve positions to resolve overlaps and allow further expansion
        for i in range(len(current_solution)):
            x, y, r = current_solution[i]
            original_pos = [x, y]

            # Collect overlapping neighbors
            neighbors_to_move = []
            for j in range(len(current_solution)):
                if i != j:
                    pos_j = current_solution[j, :2]
                    r_j = current_solution[j, 2]
                    dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                    if dist < (r + r_j):
                        neighbors_to_move.append(j)

            if neighbors_to_move:
                # Try different small moves to resolve overlaps
                best_move = [0.0, 0.0]
                best_score = -1000

                # Test several small moves in a grid pattern
                moves = []
                for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                        moves.append((dx, dy))

                # Also try some random moves for exploration
                for _ in range(5):
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    moves.append((dx, dy))

                for dx, dy in moves:
                    test_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + dx))
                    test_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + dy))

                    # Check if move helps
                    score = 0
                    valid = True

                    # Check overlap with others
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                            if dist < (r + r_j):
                                valid = False
                                break
                            # Score based on how much we improve overlap
                            if dist < (r + r_j + 0.01):
                                score -= 10  # Penalty for remaining overlap
                            else:
                                score += 1   # Reward for better separation

                    if valid and score > best_score:
                        best_score = score
                        best_move = [dx, dy]

                # Apply best move if beneficial
                if best_score > -1000 and abs(best_move[0]) > 0.0001 or abs(best_move[1]) > 0.0001:
                    current_solution[i, 0] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + best_move[0]))
                    current_solution[i, 1] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + best_move[1]))
                    improved = True

        # If no improvement was made for a while, try aggressive perturbation
        if not improved and iteration > max_iterations // 2:
            # Perturb a few circles randomly
            for _ in range(3):  # Perturb 3 circles
                i = np.random.randint(0, len(current_solution))
                dx = np.random.uniform(-0.005, 0.005)
                dy = np.random.uniform(-0.005, 0.005)
                x, y, r = current_solution[i]
                test_x = np.clip(x + dx, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                test_y = np.clip(y + dy, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                current_solution[i, 0] = test_x
                current_solution[i, 1] = test_y
                improved = True

        # Early stopping if no improvement
        if not improved:
            break

    return current_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility

    n = 26
    best_solution = None
    best_fitness = -np.inf

    # Generate multiple initial configurations and pick the best one
    initial_candidates = []
    for _ in range(8):  # Generate several candidates
        circles = create_voronoi_initialization(n)
        if is_valid_configuration(circles):
            initial_candidates.append(circles)

    # If no valid candidates, fall back to grid
    if not initial_candidates:
        circles = generate_grid_initialization(n)
        return circles

    # Select the best initial candidate
    initial_fitnesses = [evaluate_fitness(c) for c in initial_candidates]
    best_initial_idx = np.argmax(initial_fitnesses)
    current_solution = initial_candidates[best_initial_idx].copy()

    # Apply greedy refinement to the best initial configuration
    refined_solution = greedy_refinement(current_solution)
    current_fitness = evaluate_fitness(refined_solution)

    # Track best solution so far
    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_solution = refined_solution.copy()

    # Do additional refinement passes
    for _ in range(3):  # Multiple rounds of refinement
        # Apply greedy refinement again
        local_refined = greedy_refinement(refined_solution, 200)
        local_fitness = evaluate_fitness(local_refined)

        if local_fitness > best_fitness:
            best_fitness = local_fitness
            best_solution = local_refined.copy()
            refined_solution = local_refined.copy()
        else:
            # Try some random perturbations to escape local minima
            perturbed = refined_solution.copy()
            for i in range(len(perturbed)):
                # Small random perturbation with probability
                if np.random.random() < 0.3:
                    dx = np.random.uniform(-0.005, 0.005)
                    dy = np.random.uniform(-0.005, 0.005)
                    x, y, r = perturbed[i]
                    test_x = np.clip(x + dx, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    test_y = np.clip(y + dy, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    perturbed[i, 0] = test_x
                    perturbed[i, 1] = test_y

            # Validate and update
            if is_valid_configuration(perturbed):
                perturbed_fitness = evaluate_fitness(perturbed)
                if perturbed_fitness > best_fitness:
                    best_fitness = perturbed_fitness
                    best_solution = perturbed.copy()
                    refined_solution = perturbed.copy()

    return best_solution if best_solution is not None else generate_grid_initialization(n)


# EVOLVE-BLOCK-END