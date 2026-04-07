# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial.distance import cdist

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Problem parameters
    n_circles = 32
    max_iterations = 1000
    population_size = 100

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define bounds for circle positions and radii
    # Each individual is represented as [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    def create_individual():
        individual = []
        for _ in range(n_circles):
            # x, y in [0,1], r in [0, 0.5] (max possible radius)
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            r = random.uniform(0, 0.5)
            individual.extend([x, y, r])
        return creator.Individual(individual)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_circle_packing(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        total_radius = np.sum(circles[:, 2])

        # Check constraints
        # Containment constraints: radius <= x <= 1-radius, radius <= y <= 1-radius
        containment_ok = True
        for i in range(n_circles):
            x, y, r = circles[i]
            if r > x or r > 1 - x or r > y or r > 1 - y:
                containment_ok = False
                break

        if not containment_ok:
            return 0.0,

        # Non-overlap constraints using distance matrix
        overlap_ok = True
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Compute pairwise distances
        distances = cdist(positions, positions)

        # Check for overlaps
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                distance = distances[i, j]
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    overlap_ok = False
                    break
            if not overlap_ok:
                break

        if not overlap_ok:
            return 0.0,

        # Return total radius as fitness
        return total_radius,

    toolbox.register("evaluate", eval_circle_packing)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    pop = toolbox.population(n=population_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop
    for generation in range(max_iterations):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    # Find the best individual
    best_ind = tools.selBest(pop, 1)[0]
    circles = np.array(best_ind).reshape(-1, 3)

    # Ensure we return exactly 32 circles
    if circles.shape[0] != 32:
        # Fallback - generate a simple configuration if optimization failed
        circles = np.zeros((32, 3))
        # Place circles in a grid-like pattern with small radii
        grid_size = int(np.ceil(np.sqrt(32)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x / 2.0

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= 32:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

    return circles


# EVOLVE-BLOCK-END