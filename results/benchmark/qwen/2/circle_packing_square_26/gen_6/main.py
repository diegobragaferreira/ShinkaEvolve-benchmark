# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
import time

# Global constants
POP_SIZE = 100
NUM_GENERATIONS = 500
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# Problem-specific parameters
N_CIRCLES = 26
UNIT_SQUARE_SIZE = 1.0

def check_containment(circles):
    """Check if all circles are fully contained within the unit square."""
    for x, y, r in circles:
        if x - r < 0 or x + r > UNIT_SQUARE_SIZE or y - r < 0 or y + r > UNIT_SQUARE_SIZE:
            return False
    return True

def check_overlap(circles):
    """Check if any circles overlap with each other."""
    # Calculate pairwise distances between circle centers
    coords = circles[:, :2]  # Extract (x,y) coordinates
    radii = circles[:, 2]    # Extract radii

    # Create distance matrix
    distances = cdist(coords, coords)

    # Check for overlaps
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            center_dist = distances[i, j]
            radius_sum = radii[i] + radii[j]
            if center_dist < radius_sum:
                return False  # Overlap found
    return True

def evaluate_individual(individual):
    """Evaluate fitness of an individual (solution)."""
    # Convert flat list to circles array
    circles = np.array(individual).reshape(-1, 3)

    # Check constraints
    if not check_containment(circles) or not check_overlap(circles):
        return (0,)  # Invalid solution

    # Return negative sum of radii (minimize negative -> maximize positive)
    total_radius = np.sum(circles[:, 2])
    return (-total_radius,)

def create_individual():
    """Create a random valid individual."""
    circles = []
    max_attempts = 1000

    for _ in range(N_CIRCLES):
        attempts = 0
        valid_circle = False

        while not valid_circle and attempts < max_attempts:
            # Randomly generate circle parameters
            r = random.uniform(0.01, 0.15)  # Reasonable initial radius range
            x = random.uniform(r, UNIT_SQUARE_SIZE - r)
            y = random.uniform(r, UNIT_SQUARE_SIZE - r)

            # Create temporary circle
            temp_circle = np.array([[x, y, r]])

            # If this is the first circle, it's automatically valid
            if len(circles) == 0:
                circles.append([x, y, r])
                valid_circle = True
            else:
                # Check if this circle overlaps with existing ones
                temp_circles = np.vstack([circles, [x, y, r]])
                if check_containment(temp_circles) and check_overlap(temp_circles):
                    circles.append([x, y, r])
                    valid_circle = True

            attempts += 1

        if not valid_circle:
            # Fallback: use minimal valid configuration
            break

    # If we couldn't create enough valid circles, fill with minimal values
    while len(circles) < N_CIRCLES:
        circles.append([0.5, 0.5, 0.01])

    return np.array(circles.flatten())

def mutate_individual(individual):
    """Mutate an individual by slightly changing circle positions/radii."""
    mutated = individual.copy()
    n = len(mutated) // 3

    # Pick some circles to mutate
    indices_to_mutate = random.sample(range(n), max(1, int(n * MUTATION_RATE)))

    for i in indices_to_mutate:
        # Mutate either x, y, or r coordinate
        param_type = random.choice(['x', 'y', 'r'])
        idx = i * 3

        if param_type == 'x':
            mutated[idx] = max(0.01, min(UNIT_SQUARE_SIZE - 0.01, mutated[idx] + random.gauss(0, 0.02)))
        elif param_type == 'y':
            mutated[idx+1] = max(0.01, min(UNIT_SQUARE_SIZE - 0.01, mutated[idx+1] + random.gauss(0, 0.02)))
        else:  # r
            mutated[idx+2] = max(0.005, min(0.2, mutated[idx+2] + random.gauss(0, 0.01)))

    return mutated

def crossover_individuals(ind1, ind2):
    """Crossover two individuals."""
    # Simple uniform crossover
    size = len(ind1)
    child1, child2 = ind1[:], ind2[:]

    for i in range(0, size, 3):
        if random.random() < CROSSOVER_RATE:
            # Swap the three parameters of each circle
            child1[i:i+3], child2[i:i+3] = child2[i:i+3], child1[i:i+3]

    return child1, child2

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses evolutionary algorithm to optimize the arrangement.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Create initial population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    best_fitness_history = []
    start_time = time.time()

    for generation in range(NUM_GENERATIONS):
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the new one
        pop[:] = offspring

        # Track best fitness
        best_fitness = max(toolbox.map(toolbox.evaluate, pop))
        best_fitness_history.append(-best_fitness[0] if best_fitness[0] != 0 else 0)

    # Get the best individual
    best_ind = tools.selBest(pop, 1)[0]
    circles = np.array(best_ind).reshape(-1, 3)

    # Ensure circles are properly formatted
    if len(circles) < N_CIRCLES:
        # Fill missing circles with minimal values
        padding = N_CIRCLES - len(circles)
        extra_circles = np.array([[0.5, 0.5, 0.01]] * padding)
        circles = np.vstack([circles, extra_circles])

    return circles


# EVOLVE-BLOCK-END