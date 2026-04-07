# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles: np.ndarray, idx: int) -> bool:
    """Check if circle at index idx is valid (within bounds and not overlapping)."""
    x, y, r = circles[idx]

    # Check containment constraints
    if x < r or x > 1 - r or y < r or y > 1 - r:
        return False

    # Check overlap constraints with existing circles
    for i in range(len(circles)):
        if i == idx:
            continue
        x_i, y_i, r_i = circles[i]
        distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
        if distance < r + r_i:
            return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def create_initial_population(pop_size: int, n_circles: int) -> list:
    """Create initial population of valid circle arrangements."""
    population = []

    # Grid-based initialization with adaptive perturbations
    for _ in range(pop_size):
        # Create circles in a grid pattern first
        circles = np.zeros((n_circles, 3))

        # Determine grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        # Start with conservative radius
        base_radius = min(spacing_x, spacing_y) * 0.3

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y

                # Apply strategic perturbations based on position
                # corners: larger perturbations
                # edges: moderate perturbations
                # center: smaller perturbations
                if (i == 0 or i == grid_size-1) and (j == 0 or j == grid_size-1):
                    perturbation = spacing_x * 0.3
                elif i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1:
                    perturbation = spacing_x * 0.15
                else:
                    perturbation = spacing_x * 0.05

                x += np.random.uniform(-perturbation, perturbation)
                y += np.random.uniform(-perturbation, perturbation)

                # Ensure bounds
                x = np.clip(x, base_radius, 1 - base_radius)
                y = np.clip(y, base_radius, 1 - base_radius)

                circles[count] = [x, y, base_radius]
                count += 1
            if count >= n_circles:
                break

        # Improve radii based on available space
        for i in range(n_circles):
            # Try to maximize radius respecting constraints
            max_radius = base_radius

            # Check neighbors to determine max possible radius
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    # Maximum radius without overlap
                    max_radius = min(max_radius, dist - circles[j, 2] - 0.001)

            # Ensure within bounds
            max_radius = min(max_radius, circles[i, 0] - 0.001, 1 - circles[i, 0] - 0.001,
                           circles[i, 1] - 0.001, 1 - circles[i, 1] - 0.001)

            # Use safe fraction of max possible radius
            if max_radius > 0.001:
                circles[i, 2] = max_radius * 0.9

        # Validate and fix positions
        for i in range(n_circles):
            if not is_valid_placement(circles, i):
                # Fix invalid placement by finding valid nearby position
                x, y, r = circles[i]
                attempts = 0
                while not is_valid_placement(circles, i) and attempts < 50:
                    test_x = np.clip(x + np.random.uniform(-r/2, r/2), r, 1-r)
                    test_y = np.clip(y + np.random.uniform(-r/2, r/2), r, 1-r)
                    circles[i] = [test_x, test_y, r]
                    attempts += 1

        # Ensure all circles are valid
        for i in range(n_circles):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])

        population.append(circles.copy())

    return population

def mutate_individual(circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate either position or radius
            if np.random.random() < 0.5:
                # Mutate position
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + np.random.normal(0, 0.02)))
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + np.random.normal(0, 0.02)))
            else:
                # Mutate radius
                mutated[i, 2] = max(0.001, min(0.2, mutated[i, 2] + np.random.normal(0, 0.01)))

    # Fix any invalid placements
    for i in range(len(mutated)):
        if not is_valid_placement(mutated, i):
            # Try to fix by adjusting position and radius
            attempts = 0
            while not is_valid_placement(mutated, i) and attempts < 100:
                mutated[i, 0] = np.random.uniform(0.01, 0.99)
                mutated[i, 1] = np.random.uniform(0.01, 0.99)
                mutated[i, 2] = np.random.uniform(0.001, 0.1)
                attempts += 1

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    # Simple uniform crossover
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(len(child1)):
        if np.random.random() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def select_parents(population: list, fitnesses: list, tournament_size: int = 3) -> list:
    """Select parents using tournament selection."""
    selected = []

    for _ in range(len(population)):
        # Tournament selection
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx])

    return selected

def optimize_circles() -> np.ndarray:
    """Main optimization function using evolutionary algorithm."""
    n_circles = 26
    pop_size = 50
    generations = 100
    mutation_rate = 0.1

    # Create initial population
    population = create_initial_population(pop_size, n_circles)

    best_fitness = 0
    best_individual = None

    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Select parents
        parents = select_parents(population, fitnesses)

        # Create new population through crossover and mutation
        new_population = []

        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)

        # Generate offspring
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Trim to population size
        population = new_population[:pop_size]

    return best_individual if best_individual is not None else population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = optimize_circles()
        return circles
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to simple heuristic
        circles = np.zeros((26, 3))
        # Place circles in a regular grid pattern
        side_length = 1.0
        spacing = side_length / 5.0
        radius = spacing / 3.0

        count = 0
        for i in range(5):
            for j in range(5):
                if count >= 26:
                    break
                x = spacing * i + spacing / 2.0
                y = spacing * j + spacing / 2.0
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break

        return circles


# EVOLVE-BLOCK-END