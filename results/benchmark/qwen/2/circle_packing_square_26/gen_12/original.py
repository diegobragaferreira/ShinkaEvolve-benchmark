# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_constraints(circles):
    """Check if all circles satisfy containment and non-overlap constraints."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check non-overlap constraints
    positions = circles[:, :2]
    radii = circles[:, 2]

    # Calculate pairwise distances between circle centers
    distances = cdist(positions, positions)

    # Check if any circles overlap
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                return False

    return True

def evaluate_fitness(circles):
    """Evaluate fitness as the sum of all radii."""
    return np.sum(circles[:, 2])

def create_individual():
    """Create a single individual (set of 26 circles)."""
    circles = np.zeros((26, 3))

    # Try to place circles randomly while avoiding overlaps
    max_attempts = 1000
    attempts = 0

    while attempts < max_attempts:
        # Reset circles
        circles = np.zeros((26, 3))

        # Generate random positions and radii
        for i in range(26):
            # Random radius between 0 and 0.1 (arbitrary upper bound)
            # We'll adjust this based on how many circles we've placed successfully
            r = np.random.uniform(0.001, 0.1)

            # Random center position
            x = np.random.uniform(r, 1 - r)
            y = np.random.uniform(r, 1 - r)

            circles[i] = [x, y, r]

        # Check if this configuration satisfies constraints
        if check_constraints(circles):
            return circles

        attempts += 1

    # If failed to generate valid configuration, return minimal valid one
    circles = np.zeros((26, 3))
    return circles

def mutate(individual, mutation_rate=0.1):
    """Mutate an individual with given probability."""
    mutated = individual.copy()

    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Mutate either position or radius
            if np.random.rand() < 0.5:
                # Mutate position (x, y)
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.01), 0, 1)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.01), 0, 1)
            else:
                # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.005), 0.001, 0.5)

    # Ensure constraints are satisfied after mutation
    if not check_constraints(mutated):
        # Revert to original if constraints violated
        pass  # We might want to try to fix it more carefully

    return mutated

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    # Simple uniform crossover
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Swap some circles between parents
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual from population using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Evolutionary algorithm parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.1

    # Initialize population
    population = []
    for _ in range(population_size):
        individual = create_individual()
        population.append(individual)

    best_fitness = 0
    best_individual = None

    # Evolve
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(ind) for ind in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Create new population
        new_population = []

        # Keep best individual (elitism)
        new_population.append(best_individual)

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            # Ensure children meet constraints
            if check_constraints(child1):
                new_population.append(child1)
            if len(new_population) < population_size and check_constraints(child2):
                new_population.append(child2)

        population = new_population[:population_size]

    # Return best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to a reasonable configuration
        circles = np.zeros((26, 3))
        # Place circles in a grid pattern with appropriate spacing
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x / 3  # Small radius to ensure they fit

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= 26:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

        return circles


# EVOLVE-BLOCK-END