# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if the configuration satisfies all constraints."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > y or r > 1 - x or r > 1 - y:
            return False

    # Check overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def initialize_population(pop_size: int, n_circles: int) -> list:
    """Initialize population with valid configurations."""
    population = []

    # Generate diverse initial configurations
    for _ in range(pop_size):
        # Start with grid-like placement
        circles = np.zeros((n_circles, 3))

        # Place circles on a grid with some randomness
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing = 1.0 / (grid_size + 1)

        for i in range(n_circles):
            row = i // grid_size
            col = i % grid_size
            x = (col + 1) * spacing
            y = (row + 1) * spacing

            # Add some randomness to avoid perfect grid
            x += np.random.uniform(-spacing/4, spacing/4)
            y += np.random.uniform(-spacing/4, spacing/4)

            # Set initial radius based on proximity to edges
            min_dist_to_edge = min(x, y, 1-x, 1-y)
            r = min(0.05, min_dist_to_edge/2)

            circles[i] = [x, y, r]

        # Apply local optimization to make it valid
        circles = optimize_placement(circles)

        if is_valid_configuration(circles):
            population.append(circles.copy())
        else:
            # If invalid, try a different initialization
            circles = generate_random_valid_configuration(n_circles)
            if is_valid_configuration(circles):
                population.append(circles.copy())

    return population

def generate_random_valid_configuration(n_circles: int) -> np.ndarray:
    """Generate a random valid configuration."""
    circles = np.zeros((n_circles, 3))
    max_attempts = 1000

    for attempt in range(max_attempts):
        valid = True

        # Try to place circles randomly
        for i in range(n_circles):
            attempts = 0
            while attempts < 100:
                # Random placement
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)

                # Initial radius based on distance to edges
                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/2)

                # Check if this circle overlaps with existing ones
                overlap = False
                for j in range(i):
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < r + existing_r:
                        overlap = True
                        break

                if not overlap:
                    circles[i] = [x, y, r]
                    break
                attempts += 1

            if attempts >= 100:
                valid = False
                break

        if valid:
            return circles

    # Fallback to a simpler approach
    return create_simple_initialization(n_circles)

def create_simple_initialization(n_circles: int) -> np.ndarray:
    """Create a simple but valid initial configuration."""
    circles = np.zeros((n_circles, 3))

    # Place in a simple grid pattern
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / (grid_size + 1)

    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            if idx >= n_circles:
                break
            x = (col + 1) * spacing
            y = (row + 1) * spacing
            r = spacing / 4  # Conservative radius
            circles[idx] = [x, y, r]
            idx += 1

    return circles

def optimize_placement(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply local optimization to improve placement."""
    # Simple local optimization: move circles to reduce overlaps
    n = len(circles)
    for _ in range(max_iter):
        improved = False

        # Try to move each circle to a better position
        for i in range(n):
            original_pos = circles[i].copy()
            original_radius = circles[i][2]

            # Try to increase radius while maintaining validity
            max_radius = min(circles[i][0], circles[i][1],
                           1 - circles[i][0], 1 - circles[i][1])

            # Try to increase radius slightly
            new_radius = min(original_radius + 0.001, max_radius)
            if new_radius > original_radius:
                circles[i][2] = new_radius
                if not is_valid_configuration(circles):
                    circles[i][2] = original_radius  # Revert if invalid
                else:
                    improved = True

        if not improved:
            break

    return circles

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    if np.random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Uniform crossover for positions and radii
    for i in range(n):
        if np.random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]

    # Ensure children are valid
    child1 = optimize_placement(child1)
    child2 = optimize_placement(child2)

    return child1, child2

def mutate(circles: np.ndarray, mutation_rate: float = MUTATION_RATE) -> np.ndarray:
    """Apply mutation to a configuration."""
    mutated = circles.copy()
    n = len(mutated)

    for i in range(n):
        if np.random.random() < mutation_rate:
            # Mutate either position or radius
            if np.random.random() < 0.5:
                # Mutate position
                mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, 0.05), 0, 1)
                mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, 0.05), 0, 1)
            else:
                # Mutate radius
                mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.01), 0.01, 0.5)

    # Optimize the mutated configuration
    mutated = optimize_placement(mutated)

    return mutated

def select_tournament(population: list, fitnesses: list, tournament_size: int = TOURNAMENT_SIZE) -> int:
    """Select an individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return winner_index

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    population = initialize_population(POPULATION_SIZE, n)

    best_solution = None
    best_fitness = -1

    for generation in range(GENERATIONS):
        # Evaluate fitness for all individuals
        fitnesses = []
        for circles in population:
            if is_valid_configuration(circles):
                fitness = calculate_sum_radii(circles)
                fitnesses.append(fitness)
            else:
                # Penalize invalid solutions
                fitnesses.append(0)

        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Create new population
        new_population = []

        # Elitism: keep best individual
        new_population.append(best_solution.copy())

        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            parent1_idx = select_tournament(population, fitnesses)
            parent2_idx = select_tournament(population, fitnesses)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)

            # Add children to new population
            new_population.extend([child1, child2])

        # Trim population to exact size
        population = new_population[:POPULATION_SIZE]

    # Return the best solution found
    if best_solution is None:
        # Fallback to a simple configuration if nothing worked
        return create_simple_initialization(n)

    return best_solution


# EVOLVE-BLOCK-END