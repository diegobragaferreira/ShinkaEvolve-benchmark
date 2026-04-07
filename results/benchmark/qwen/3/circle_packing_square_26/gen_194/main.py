# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 200
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.2
MUTATION_RATE_END = 0.005
CROSSOVER_PROB = 0.9
VALIDITY_THRESHOLD = 1e-6
INITIAL_GRID_SIZE = 20

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
        grid = get_grid_cells(resolved, INITIAL_GRID_SIZE)

        for i in range(len(resolved)):
            for j in range(i+1, len(resolved)):
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
    """Check if any circles overlap using spatial grid indexing."""
    if len(circles) <= 1:
        return False

    if grid is None:
        grid = get_grid_cells(circles, INITIAL_GRID_SIZE)

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
    grid = get_grid_cells(circles, INITIAL_GRID_SIZE)
    if check_overlap_efficient(circles, grid):
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

    # Adaptive mutation rate using sigmoid decay
    mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * \
                   (1 / (1 + math.exp(-10 * (generation / total_generations - 0.5))))

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


def local_refinement(circles: np.ndarray, max_iterations: int = 20, generation: int = 0) -> np.ndarray:
    """Apply intensive local refinement to improve overlap resolution and constraint satisfaction."""
    refined = circles.copy()

    for iteration in range(max_iterations):
        grid = get_grid_cells(refined, INITIAL_GRID_SIZE)
        resolved = False
        overlap_count = 0

        # Track how many overlaps we're resolving per iteration
        for i in range(len(refined)):
            for j in range(i+1, len(refined)):
                xi, yi, ri = refined[i]
                xj, yj, rj = refined[j]
                dist = calculate_distance((xi, yi), (xj, yj))

                if dist < (ri + rj - VALIDITY_THRESHOLD):
                    overlap_count += 1
                    # Calculate adaptive force parameters based on multiple factors

                    # Compute overlap severity
                    overlap_severity = (ri + rj - dist) / (ri + rj + 0.001)

                    # Adaptive force scaling based on generation and overlap severity
                    if generation < 50:
                        force_multiplier = 0.3 + 0.4 * overlap_severity  # Early stage: conservative
                    elif generation < 100:
                        force_multiplier = 0.5 + 0.5 * overlap_severity  # Mid stage: balanced
                    else:
                        force_multiplier = 0.6 + 0.4 * overlap_severity  # Late stage: aggressive

                    # Additional adjustment based on overlap severity
                    if overlap_severity > 0.5:
                        force_multiplier *= 1.5  # Stronger forces for severe overlaps
                    elif overlap_severity > 0.2:
                        force_multiplier *= 1.2  # Moderate forces for medium overlaps

                    # More sophisticated scaling for later generations for fine-tuning
                    if generation > 150:
                        force_multiplier *= 1.1  # Slight additional push in late stage

                    # More aggressive approach for later generations
                    # but avoid over-shooting boundaries
                    if generation > 170:
                        force_multiplier *= 1.2  # Final refinement boost

                    # More aggressive overlap resolution with better force distribution
                    dx = xj - xi
                    dy = yj - yi
                    distance = max(VALIDITY_THRESHOLD, dist)

                    # Normalize direction vector
                    dx /= distance
                    dy /= distance

                    # Move circles apart with enhanced force scaling
                    move_amount = (ri + rj - dist) * force_multiplier

                    # Enhanced scale by inverse radii to better balance movement
                    # But avoid extreme imbalance for very different sized circles
                    if ri > rj:
                        scale_factor = 0.7 + 0.3 * (rj / (ri + 0.001))  # Prefer larger circle movement
                    elif rj > ri:
                        scale_factor = 0.3 + 0.7 * (ri / (rj + 0.001))  # Prefer larger circle movement
                    else:
                        scale_factor = 0.5  # Equal case

                    # Apply bounds to prevent excessive movement
                    scale_factor = max(0.1, min(0.9, scale_factor))

                    # Apply movement with proper scaling
                    refined[i, 0] -= dx * move_amount * scale_factor * 0.5
                    refined[i, 1] -= dy * move_amount * scale_factor * 0.5
                    refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.5
                    refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.5
                    resolved = True

        # Enforce bounds with more precise constraint handling
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Ensure the circle fits properly in the unit square
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            r = max(0.001, r)

            # Clamp coordinates to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]

        # Early stopping criteria:
        # 1. No overlaps detected
        # 2. Very few overlaps (stabilizing)
        # 3. Many iterations with no changes (converged)
        if not resolved or overlap_count < 2:
            break

    return refined

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
    """Perform crossover between two parent configurations with enhanced recombination."""
    if random.random() > CROSSOVER_PROB:
        # Return one of the parents randomly
        return parent1.copy() if random.random() < 0.5 else parent2.copy()

    n = len(parent1)
    child = np.zeros_like(parent1)

    # Use uniform crossover for better recombination
    for i in range(n):
        # Uniform crossover for each parameter
        if random.random() < 0.5:
            child[i] = parent1[i].copy()
        else:
            child[i] = parent2[i].copy()

        # Add some blending for better exploration
        if random.random() < 0.3:  # 30% chance of blending
            alpha = random.random()
            # Blend positions and radii
            child[i][0] = parent1[i][0] + alpha * (parent2[i][0] - parent1[i][0])
            child[i][1] = parent1[i][1] + alpha * (parent2[i][1] - parent1[i][1])
            child[i][2] = parent1[i][2] + alpha * (parent2[i][2] - parent1[i][2])

    # Ensure offspring stays within bounds
    child = enforce_constraints(child)

    # Apply local refinement to ensure validity
    refined_child = refine_configuration(child)

    return refined_child

def refine_configuration(circles: np.ndarray) -> np.ndarray:
    """Refine configuration to remove overlaps and correct constraints."""
    refined = circles.copy()

    # Force-based refinement with better overlap resolution
    for iteration in range(8):
        grid = get_grid_cells(refined, INITIAL_GRID_SIZE)

        # Check for overlaps and resolve them
        resolved = False
        for i in range(len(refined)):
            for j in range(i+1, len(refined)):
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

            # Apply local refinement in later generations for better exploitation
            if gen >= GENERATIONS * 0.7:  # Start local refinement in later generations
                child = local_refinement(child)

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

    # Final local refinement on the best solution with detailed analysis
    # Perform multiple refinement passes to maximize quality
    best_solution = local_refinement(best_solution, max_iterations=50, generation=GENERATIONS)

    # Additional detailed pass if needed
    grid = get_grid_cells(best_solution, INITIAL_GRID_SIZE)
    if check_overlap_efficient(best_solution, grid):
        best_solution = local_refinement(best_solution, max_iterations=30, generation=GENERATIONS)

    # Analyze and improve the final solution one more time
    final_grid = get_grid_cells(best_solution, INITIAL_GRID_SIZE)
    if check_overlap_efficient(best_solution, final_grid):
        # Last resort: aggressive refinement with very high iteration count
        best_solution = local_refinement(best_solution, max_iterations=100, generation=GENERATIONS)

    return best_solution

# EVOLVE-BLOCK-END