# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 300
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.2
MUTATION_RATE_END = 0.005
CROSSOVER_PROB = 0.9
VALIDITY_THRESHOLD = 1e-6
INITIAL_GRID_SIZE = 20
SPATIAL_INDEXING_THRESHOLD = 50  # Use KDTree for overlap detection when circles > this number

def poisson_disk_sampling(n_points: int, min_distance: float = 0.1) -> List[Tuple[float, float]]:
    """Generate points using Poisson disk sampling for better uniformity."""
    points = []
    active_list = []

    # Start with a random point
    points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
    active_list.append(0)

    while len(points) < n_points:
        if not active_list:
            break

        # Pick a random active point
        idx = random.choice(active_list)
        x, y = points[idx]

        # Try to generate a new point
        found = False
        for _ in range(30):  # Limit attempts
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_distance, 2 * min_distance)

            new_x = x + radius * math.cos(angle)
            new_y = y + radius * math.sin(angle)

            # Check bounds
            if new_x < 0.05 or new_x > 0.95 or new_y < 0.05 or new_y > 0.95:
                continue

            # Check distance to existing points
            too_close = False
            for px, py in points:
                dist = math.sqrt((new_x - px)**2 + (new_y - py)**2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                points.append((new_x, new_y))
                active_list.append(len(points) - 1)
                found = True
                break

        if not found:
            active_list.remove(idx)

    # If we didn't get enough points, fill with random ones
    while len(points) < n_points:
        points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))

    return points[:n_points]

def initialize_population(n: int, population_size: int) -> np.ndarray:
    """Initialize population with improved Voronoi-based distribution using Poisson disk sampling."""
    population = []

    # Generate points using Poisson disk sampling for better distribution
    sample_points = poisson_disk_sampling(n, 0.15)

    # Create multiple populations with variation
    for _ in range(population_size):
        circles = np.zeros((n, 3))

        # Distribute circles using the sample points
        for i in range(min(n, len(sample_points))):
            x_base, y_base = sample_points[i]

            # Add jitter for diversity
            x = max(0.01, min(0.99, x_base + random.uniform(-0.03, 0.03)))
            y = max(0.01, min(0.99, y_base + random.uniform(-0.03, 0.03)))

            # Initial radius - start with moderately large values
            circles[i] = [x, y, 0.06]

        # Fill remaining circles
        for i in range(len(sample_points), n):
            # Place remaining circles more randomly but still with some structure
            if random.random() < 0.4:
                # Near an existing circle
                idx = random.randint(0, min(i-1, len(sample_points)-1))
                x_base, y_base = sample_points[idx]
                x = max(0.01, min(0.99, x_base + random.uniform(-0.08, 0.08)))
                y = max(0.01, min(0.99, y_base + random.uniform(-0.08, 0.08)))
            else:
                # Completely random
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)

            circles[i] = [x, y, 0.025]

        # Ensure circles don't overlap initially
        circles = resolve_initial_overlaps(circles)
        population.append(circles)

    return np.array(population)

def resolve_initial_overlaps(circles: np.ndarray) -> np.ndarray:
    """Resolve overlaps in initial configuration using force-based approach."""
    resolved = circles.copy()

    # Iteratively resolve overlaps
    for _ in range(10):
        changed = False
        if len(resolved) > SPATIAL_INDEXING_THRESHOLD:
            # Use efficient KDTree for large populations
            tree = cKDTree(resolved[:, :2])
            pairs = tree.query_pairs(0.001)

            for i, j in pairs:
                xi, yi, ri = resolved[i]
                xj, yj, rj = resolved[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)

                if dist < (ri + rj - VALIDITY_THRESHOLD):
                    # Move circles apart
                    dx = xj - xi
                    dy = yj - yi
                    distance = max(VALIDITY_THRESHOLD, dist)

                    # Normalize
                    dx /= distance
                    dy /= distance

                    # Move based on inverse radius ratio
                    move_amount = (ri + rj - dist) * 0.5

                    # Apply movement in opposite directions
                    resolved[i, 0] -= dx * move_amount * 0.4
                    resolved[i, 1] -= dy * move_amount * 0.4
                    resolved[j, 0] += dx * move_amount * 0.4
                    resolved[j, 1] += dy * move_amount * 0.4
                    changed = True
        else:
            # Use grid-based approach for small populations
            grid = get_grid_cells(resolved, INITIAL_GRID_SIZE)

            for (gx, gy), indices in grid.items():
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        xi, yi, ri = resolved[idx1]
                        xj, yj, rj = resolved[idx2]
                        dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)

                        if dist < (ri + rj - VALIDITY_THRESHOLD):
                            # Move circles apart
                            dx = xj - xi
                            dy = yj - yi
                            distance = max(VALIDITY_THRESHOLD, dist)

                            # Normalize
                            dx /= distance
                            dy /= distance

                            # Move based on inverse radius ratio
                            move_amount = (ri + rj - dist) * 0.5

                            # Apply movement in opposite directions
                            resolved[idx1, 0] -= dx * move_amount * 0.4
                            resolved[idx1, 1] -= dy * move_amount * 0.4
                            resolved[idx2, 0] += dx * move_amount * 0.4
                            resolved[idx2, 1] += dy * move_amount * 0.4
                            changed = True

        # Ensure bounds
        for i in range(len(resolved)):
            x, y, r = resolved[i]
            # Clamp to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            resolved[i] = [x, y, r]

        if not changed:
            break

    return resolved

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained in the unit square."""
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_grid_cells(circles: np.ndarray, grid_size: int = INITIAL_GRID_SIZE) -> dict:
    """Create a spatial grid for fast neighbor lookups."""
    grid = {}
    cell_size = 1.0 / grid_size

    for i, (x, y, r) in enumerate(circles):
        # Determine which grid cells this circle might occupy
        min_x_cell = max(0, int((x - r) / cell_size))
        max_x_cell = min(grid_size - 1, int((x + r) / cell_size))
        min_y_cell = max(0, int((y - r) / cell_size))
        max_y_cell = min(grid_size - 1, int((y + r) / cell_size))

        for gx in range(min_x_cell, max_x_cell + 1):
            for gy in range(min_y_cell, max_y_cell + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)

    return grid

def check_overlap_efficient(circles: np.ndarray, grid: dict = None) -> bool:
    """Check if any circles overlap using multi-scale spatial indexing."""
    if len(circles) <= 1:
        return False

    # Use efficient KDTree for large populations
    if len(circles) > SPATIAL_INDEXING_THRESHOLD:
        tree = cKDTree(circles[:, :2])
        pairs = tree.query_pairs(0.001)
        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = calculate_distance((x1, y1), (x2, y2))
            if distance < (r1 + r2 - VALIDITY_THRESHOLD):
                return True
    else:
        # For smaller populations, use multi-scale grid-based approach
        # Try with different grid sizes for better performance
        grid_sizes = [INITIAL_GRID_SIZE, INITIAL_GRID_SIZE//2, INITIAL_GRID_SIZE*2]
        for grid_size in grid_sizes:
            if grid is None:
                grid = get_grid_cells(circles, grid_size)

            # For each cell, check pairs of circles
            for (gx, gy), indices in grid.items():
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        x1, y1, r1 = circles[idx1]
                        x2, y2, r2 = circles[idx2]

                        distance = calculate_distance((x1, y1), (x2, y2))
                        if distance < (r1 + r2 - VALIDITY_THRESHOLD):
                            return True

    return False

def compute_penalty(circles: np.ndarray, generation: int = 0, total_generations: int = 100) -> float:
    """Compute penalty based on constraint violations with progressive scaling."""
    penalty = 0.0

    # Dynamic penalty scaling factor
    penalty_scale = 1.0 + (generation / total_generations) * 5.0

    # Check containment violations with scaled penalties
    for x, y, r in circles:
        # Boundary violations
        if x - r < 0:
            penalty += (abs(x - r) ** 2) * 10000 * penalty_scale
        elif x + r > 1:
            penalty += (abs(x + r - 1) ** 2) * 10000 * penalty_scale
        if y - r < 0:
            penalty += (abs(y - r) ** 2) * 10000 * penalty_scale
        elif y + r > 1:
            penalty += (abs(y + r - 1) ** 2) * 10000 * penalty_scale

    # Check overlap violations with scaled penalties
    if check_overlap_efficient(circles):
        penalty += 10000000.0 * penalty_scale

    return penalty

def evaluate_fitness(circles: np.ndarray, generation: int = 0, total_generations: int = 100) -> float:
    """Evaluate fitness of a circle configuration."""
    # If invalid, heavily penalize
    if not check_containment(circles) or check_overlap_efficient(circles):
        penalty = compute_penalty(circles, generation, total_generations)
        return -penalty

    # Otherwise, return total radius
    total_radius = np.sum(circles[:, 2])
    return total_radius

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive rates."""
    mutated = circles.copy()

    # Adaptive mutation rate using more sophisticated decay
    # Start high, decrease smoothly with a quadratic decay
    progress = generation / total_generations
    mutation_rate = MUTATION_RATE_START * (1 - progress)**2 + MUTATION_RATE_END * progress

    # Add generation-specific variation to maintain diversity
    diversity_factor = 1.0 + 0.2 * math.sin(2 * math.pi * generation / total_generations)
    mutation_rate *= diversity_factor

    n = len(mutated)

    # Mutate some circles
    for i in range(n):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate
            choice = random.randint(0, 2)

            if choice == 0:  # X coordinate
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, 0.025)))
            elif choice == 1:  # Y coordinate
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, 0.025)))
            else:  # Radius
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.02)))

    # Ensure valid configuration after mutation
    return enforce_constraints(mutated)

def enforce_constraints(circles: np.ndarray) -> np.ndarray:
    """Enforce constraints on circle positions and radii."""
    result = circles.copy()

    # Adjust positions and radii to satisfy bounds
    for i in range(len(result)):
        x, y, r = result[i]

        # Ensure circle fits in the unit square
        r = min(r, x, 1-x, y, 1-y)
        r = max(0.001, min(0.49, r))

        # Clamp coordinates to valid range
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))

        result[i] = [x, y, r]

    return result

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two parent configurations."""
    if random.random() > CROSSOVER_PROB:
        # Return one of the parents randomly
        return parent1.copy() if random.random() < 0.5 else parent2.copy()

    n = len(parent1)
    child = np.zeros_like(parent1)

    # Single point crossover
    crossover_point = random.randint(1, n-1)

    for i in range(n):
        if i < crossover_point:
            child[i] = parent1[i].copy()
        else:
            child[i] = parent2[i].copy()

    # Apply local refinement to ensure validity
    refined_child = refine_configuration(child)

    return refined_child

def refine_configuration(circles: np.ndarray) -> np.ndarray:
    """Refine configuration to remove overlaps and correct constraints."""
    refined = circles.copy()

    # Force-based refinement with better overlap resolution
    for iteration in range(10):
        resolved = False

        if len(refined) > SPATIAL_INDEXING_THRESHOLD:
            # Use efficient KDTree for large populations
            tree = cKDTree(refined[:, :2])
            pairs = tree.query_pairs(0.001)

            for i, j in pairs:
                xi, yi, ri = refined[i]
                xj, yj, rj = refined[j]
                dist = calculate_distance((xi, yi), (xj, yj))

                if dist < (ri + rj - VALIDITY_THRESHOLD):
                    # Resolve overlap by moving circles apart with force-based approach
                    dx = xj - xi
                    dy = yj - yi
                    distance = max(VALIDITY_THRESHOLD, dist)

                    # Normalize direction vector
                    dx /= distance
                    dy /= distance

                    # Move circles apart based on their relative sizes and distances
                    move_amount = (ri + rj - dist) * 0.5

                    # Scale by inverse radii to balance movement
                    scale_factor = min(1.0, ri / (ri + rj + 0.001))
                    refined[i, 0] -= dx * move_amount * scale_factor * 0.3
                    refined[i, 1] -= dy * move_amount * scale_factor * 0.3
                    refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.3
                    refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.3
                    resolved = True
        else:
            # Use grid-based approach for small populations
            grid = get_grid_cells(refined, INITIAL_GRID_SIZE)

            # Check for overlaps and resolve them
            for (gx, gy), indices in grid.items():
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        xi, yi, ri = refined[idx1]
                        xj, yj, rj = refined[idx2]
                        dist = calculate_distance((xi, yi), (xj, yj))

                        if dist < (ri + rj - VALIDITY_THRESHOLD):
                            # Resolve overlap by moving circles apart with force-based approach
                            dx = xj - xi
                            dy = yj - yi
                            distance = max(VALIDITY_THRESHOLD, dist)

                            # Normalize direction vector
                            dx /= distance
                            dy /= distance

                            # Move circles apart based on their relative sizes and distances
                            move_amount = (ri + rj - dist) * 0.5

                            # Scale by inverse radii to balance movement
                            scale_factor = min(1.0, ri / (ri + rj + 0.001))
                            refined[idx1, 0] -= dx * move_amount * scale_factor * 0.3
                            refined[idx1, 1] -= dy * move_amount * scale_factor * 0.3
                            refined[idx2, 0] += dx * move_amount * (1 - scale_factor) * 0.3
                            refined[idx2, 1] += dy * move_amount * (1 - scale_factor) * 0.3
                            resolved = True

        # Enforce bounds
        for i in range(len(refined)):
            x, y, r = refined[i]
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]

        # Early stopping if no changes made
        if not resolved:
            break

    return refined

def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    """Select parent using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 26
    population = initialize_population(n, POPULATION_SIZE)

    # Evaluate initial population
    fitnesses = [evaluate_fitness(individual) for individual in population]

    # Evolution loop
    for gen in range(GENERATIONS):
        # Selection, crossover, and mutation
        new_population = []

        for _ in range(POPULATION_SIZE):
            # Tournament selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child, gen, GENERATIONS)

            new_population.append(child)

        # Evaluate new population
        population = np.array(new_population)
        fitnesses = [evaluate_fitness(individual, gen, GENERATIONS) for individual in population]

        # Print progress
        best_fitness = max(fitnesses)
        if gen % 25 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness}")

    # Return the best individual
    best_index = np.argmax(fitnesses)
    best_solution = population[best_index]

    return best_solution

# EVOLVE-BLOCK-END