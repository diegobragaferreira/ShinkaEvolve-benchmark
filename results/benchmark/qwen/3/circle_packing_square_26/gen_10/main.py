# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def check_validity(circles):
    """Check if all circles are within bounds and non-overlapping"""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints
    positions = circles[:, :2]
    radii = circles[:, 2]

    # Compute pairwise distances
    distances = cdist(positions, positions)

    # Check for overlaps (distance < sum of radii)
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] < radii[i] + radii[j]:
                return False

    return True

def evaluate_fitness(circles):
    """Evaluate fitness as sum of radii, with penalty for overlaps"""
    if not check_validity(circles):
        # Return very low fitness if invalid
        return -1e6

    return np.sum(circles[:, 2])

def create_individual():
    """Create a single random valid individual"""
    while True:
        circles = np.zeros((26, 3))

        # Generate random positions and radii
        for i in range(26):
            # Random radius between 0 and 0.1
            r = random.uniform(0.01, 0.1)

            # Ensure valid position given radius
            max_x = 1 - r
            max_y = 1 - r

            if max_x <= 0 or max_y <= 0:
                continue

            x = random.uniform(r, max_x)
            y = random.uniform(r, max_y)

            circles[i] = [x, y, r]

        # Check validity
        if check_validity(circles):
            return circles

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    # Simple uniform crossover
    child = np.copy(parent1)
    for i in range(26):
        if random.random() < 0.5:
            child[i] = parent2[i]

    return child

def mutate(individual, mutation_rate=0.1):
    """Mutate an individual"""
    mutated = np.copy(individual)

    for i in range(26):
        if random.random() < mutation_rate:
            # Mutate either position or radius
            if random.random() < 0.5:
                # Mutate position
                r = mutated[i, 2]
                mutated[i, 0] = random.uniform(r, 1-r)
                mutated[i, 1] = random.uniform(r, 1-r)
            else:
                # Mutate radius
                mutated[i, 2] = random.uniform(0.01, 0.1)

    return mutated

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Evolution parameters
    population_size = 50
    generations = 100
    elite_size = 5

    # Create initial population
    population = [create_individual() for _ in range(population_size)]

    best_fitness = float('-inf')
    best_individual = None

    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness of population
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]

        # Keep elite
        elite = population[:elite_size]

        # Create new population through selection, crossover, and mutation
        new_population = elite[:]

        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness_scores, 3)
            parent2 = tournament_selection(population, fitness_scores, 3)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    # Return best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to creating a valid individual
        return create_individual()

def tournament_selection(population, fitness_scores, tournament_size):
    """Select an individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index]


# EVOLVE-BLOCK-END