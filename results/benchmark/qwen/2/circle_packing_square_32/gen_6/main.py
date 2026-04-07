# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def create_individual(n=32):
    """Create a single individual (circle configuration)."""
    # Each individual is represented as [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    individual = []
    for _ in range(n):
        # Random position within unit square with margin
        x = random.uniform(0.01, 0.99)
        y = random.uniform(0.01, 0.99)
        # Random radius with reasonable bounds
        r = random.uniform(0.001, 0.1)
        individual.extend([x, y, r])
    return individual

def evaluate_fitness(individual):
    """Evaluate fitness of a circle configuration."""
    n = 32
    circles = np.array(individual).reshape(n, 3)

    # Calculate total radius
    total_radius = np.sum(circles[:, 2])

    # Penalty for constraint violations
    penalty = 0

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 10000  # Large penalty for containment violation

    # Check overlap constraints using vectorized computation for efficiency
    if n > 1:
        # Create coordinate matrices for efficient distance calculation
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        r_coords = circles[:, 2]

        # Compute pairwise distances using broadcasting
        dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
        dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
        distances = np.sqrt(dx**2 + dy**2)

        # Create matrix of minimum distances (excluding diagonal)
        min_distances = np.where(np.eye(n, dtype=bool), np.inf, distances)

        # Compute overlap penalties
        min_required_distance = r_coords[:, np.newaxis] + r_coords[np.newaxis, :]
        overlaps = min_required_distance - distances

        # Only penalize negative overlaps (actual overlaps)
        overlap_penalty = np.sum(np.maximum(0, -overlaps))
        penalty += 1000 * overlap_penalty

    return total_radius - penalty,

def mutate_individual(individual, indpb=0.1, mut_strength=0.05):
    """Mutate an individual by slightly adjusting positions and radii."""
    for i in range(len(individual)):
        if random.random() < indpb:
            if i % 3 == 2:  # radius
                # Mutate radius
                old_radius = individual[i]
                new_radius = old_radius * random.uniform(1 - mut_strength, 1 + mut_strength)
                # Keep radius in valid range
                individual[i] = max(0.001, min(0.5, new_radius))
            else:  # x or y position
                # Mutate position
                old_pos = individual[i]
                new_pos = old_pos + random.uniform(-mut_strength, mut_strength)
                # Keep position in valid range
                individual[i] = max(0.001, min(0.999, new_pos))
    return individual,

def initialize_population(pop_size=50, n=32):
    """Initialize population with diverse configurations."""
    population = []
    for _ in range(pop_size):
        individual = create_individual(n)
        population.append(individual)
    return population

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses evolutionary algorithm for optimization.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up DEAP framework
    random.seed(42)  # For reproducibility
    np.random.seed(42)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", initialize_population)
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Parameters for evolution
    pop_size = 100
    n_generations = 200
    cxpb = 0.8   # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Initialize population
    population = toolbox.population(pop_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    for generation in range(n_generations):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the new one
        population[:] = offspring

    # Get the best individual
    best_individual = tools.selBest(population, 1)[0]

    # Convert best individual to numpy array format
    circles = np.array(best_individual).reshape(32, 3)

    return circles


# EVOLVE-BLOCK-END