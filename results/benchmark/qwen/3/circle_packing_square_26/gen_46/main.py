# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 200
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.15
MUTATION_RATE_END = 0.01
ELITISM_COUNT = 5
BOUNDARY_MARGIN = 0.01
MAX_EVAL_TIME = 60.0
GRID_CELL_SIZE = 0.1  # Size of grid cells for spatial indexing

def initialize_population(n_circles: int, pop_size: int) -> np.ndarray:
    """Initialize population with well-distributed circle configurations"""
    population = []

    for _ in range(pop_size):
        # Initialize circles with random positions and small radii
        circles = np.zeros((n_circles, 3))

        # Distribute circles using a grid-like pattern with some randomness
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        count = 0

        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_circles:
                    break

                x = (i + 0.5 + np.random.normal(0, 0.2)) / grid_size
                y = (j + 0.5 + np.random.normal(0, 0.2)) / grid_size

                # Clip to valid range considering boundary margin
                x = np.clip(x, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                y = np.clip(y, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)

                # Small initial radius
                r = np.random.uniform(0.01, 0.05)

                circles[count] = [x, y, r]
                count += 1

            if count >= n_circles:
                break

        # Adjust radii to be feasible and distribute more evenly
        # Use a simple heuristic to make initial configuration better
        for i in range(n_circles):
            x, y, r = circles[i]
            # Constrain radius by proximity to boundaries
            max_r = min(x, 1-x, y, 1-y)
            circles[i, 2] = min(r, max_r * 0.8)

        population.append(circles)

    return np.array(population)

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration is valid (no overlaps, fully contained)"""
    n_circles = len(circles)

    # Check boundary containment
    for i in range(n_circles):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlaps using grid-based spatial indexing for efficiency
    return is_valid_with_grid_indexing(circles)

def is_valid_with_grid_indexing(circles: np.ndarray) -> bool:
    """Check overlaps using grid-based spatial indexing"""
    n_circles = len(circles)
    if n_circles <= 1:
        return True

    # Create grid cells
    grid = {}
    cell_size = GRID_CELL_SIZE

    # Place each circle into grid cells it occupies
    for i, (x, y, r) in enumerate(circles):
        # Determine which cells this circle affects
        min_x_cell = int((x - r) / cell_size)
        max_x_cell = int((x + r) / cell_size)
        min_y_cell = int((y - r) / cell_size)
        max_y_cell = int((y + r) / cell_size)

        for cx in range(min_x_cell, max_x_cell + 1):
            for cy in range(min_y_cell, max_y_cell + 1):
                if (cx, cy) not in grid:
                    grid[(cx, cy)] = []
                grid[(cx, cy)].append(i)

    # Check for overlaps within each cell and neighboring cells
    for (cx, cy), circle_indices in grid.items():
        # Check overlaps within same cell
        for i in range(len(circle_indices)):
            for j in range(i + 1, len(circle_indices)):
                idx1, idx2 = circle_indices[i], circle_indices[j]
                x1, y1, r1 = circles[idx1]
                x2, y2, r2 = circles[idx2]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False

        # Check overlaps with neighboring cells (only check if circle radius is large enough)
        # We only need to check 8 neighbors plus the center cell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_cell = (cx + dx, cy + dy)
                if neighbor_cell in grid:
                    for idx1 in circle_indices:
                        for idx2 in grid[neighbor_cell]:
                            x1, y1, r1 = circles[idx1]
                            x2, y2, r2 = circles[idx2]
                            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if distance < r1 + r2:
                                return False

    return True

def calculate_fitness(circles: np.ndarray) -> Tuple[float, float]:
    """
    Calculate fitness with penalty for constraint violations
    Returns (total_radius, penalty_score)
    """
    total_radius = np.sum(circles[:, 2])

    if not is_valid_configuration(circles):
        # Large penalty for constraint violations
        penalty = 10000
        return total_radius - penalty, penalty

    return total_radius, 0.0

def tournament_selection(population: np.ndarray, fitness_scores: np.ndarray,
                        tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Select individual using tournament selection"""
    selected_idx = np.random.randint(0, len(population), tournament_size)
    best_idx = selected_idx[np.argmax(fitness_scores[selected_idx])]
    return population[best_idx]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Uniform crossover between two parents"""
    n_circles = len(parent1)
    child = np.zeros_like(parent1)

    # Randomly choose which parent contributes each circle
    mask = np.random.rand(n_circles) > 0.5

    child[mask] = parent1[mask]
    child[~mask] = parent2[~mask]

    # Apply slight adjustments to fix any constraint violations
    for i in range(n_circles):
        x, y, r = child[i]
        # Ensure circle stays within bounds
        max_r = min(x, 1-x, y, 1-y)
        if r > max_r:
            child[i, 2] = max_r * 0.9  # Scale down radius slightly

    return child

def mutate(individual: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Apply mutation to an individual"""
    n_circles = len(individual)
    mutated = individual.copy()

    # Adaptive mutation rate
    mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * (generation / total_generations)

    # Mutate each circle
    for i in range(n_circles):
        if np.random.rand() < mutation_rate:
            # Mutate position and radius
            mutated[i, 0] += np.random.normal(0, 0.02)  # X position
            mutated[i, 1] += np.random.normal(0, 0.02)  # Y position
            mutated[i, 2] += np.random.normal(0, 0.01)  # Radius

            # Clamp values to valid ranges
            mutated[i, 0] = np.clip(mutated[i, 0], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            mutated[i, 1] = np.clip(mutated[i, 1], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            mutated[i, 2] = max(mutated[i, 2], 0.001)

            # Fix potential boundary constraint violations
            x, y, r = mutated[i]
            max_r = min(x, 1-x, y, 1-y)
            mutated[i, 2] = min(r, max_r * 0.95)

    return mutated

def evolve_circles(n_circles: int = 26, generations: int = GENERATIONS) -> np.ndarray:
    """Main evolutionary algorithm"""
    # Initialize population
    population = initialize_population(n_circles, POPULATION_SIZE)

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = np.array([calculate_fitness(ind)[0] for ind in population])

        # Track best fitness
        best_fitness = np.max(fitness_scores)
        best_fitness_history.append(best_fitness)

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
        elites = population[elite_indices].copy()

        # Create new population
        new_population = []

        # Add elites first
        new_population.extend(elites)

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child, gen, generations)

            new_population.append(child)

        # Trim to exact population size
        population = np.array(new_population[:POPULATION_SIZE])

        # Early stopping when improvement plateaus
        if len(best_fitness_history) > 10:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
            if recent_improvement < 0.001:
                break

    # Return best individual
    final_fitness_scores = np.array([calculate_fitness(ind)[0] for ind in population])
    best_idx = np.argmax(final_fitness_scores)
    return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    try:
        circles = evolve_circles(n_circles=26, generations=GENERATIONS)
        return circles
    except Exception as e:
        # Fallback to simple initialization if evolution fails
        print(f"Evolution failed: {e}")
        circles = np.zeros((26, 3))
        # Simple grid initialization
        grid_size = int(np.ceil(np.sqrt(26)))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                r = 0.02
                circles[count] = [x, y, r]
                count += 1
        return circles

# EVOLVE-BLOCK-END