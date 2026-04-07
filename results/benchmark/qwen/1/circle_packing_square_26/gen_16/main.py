# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def check_constraints(circles):
    """Check if circles satisfy all constraints"""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > y or r > 1-x or r > 1-y:
            return False

    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2:
                return False

    return True

def evaluate_individual(individual):
    """Evaluate fitness of individual (maximize sum of radii)"""
    # Convert individual to circles array
    circles = np.array(individual).reshape(-1, 3)

    # Check if constraints are satisfied
    if not check_constraints(circles):
        return (0,)  # Invalid solution gets very low fitness

    # Return negative sum of radii (since DEAP minimizes by default)
    total_radius = np.sum(circles[:, 2])
    return (-total_radius,)

def initialize_population(pop_size, n_circles=26):
    """Initialize population with valid configurations"""
    population = []

    for _ in range(pop_size):
        # Generate random valid circles
        circles = []
        max_attempts = 1000

        for i in range(n_circles):
            attempts = 0
            valid = False

            while not valid and attempts < max_attempts:
                # Random position and radius
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                r = np.random.uniform(0.01, min(x, y, 1-x, 1-y))

                # Check if this circle overlaps with existing ones
                valid = True
                for cx, cy, cr in circles:
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < r + cr:
                        valid = False
                        break

                attempts += 1

            if valid:
                circles.append([x, y, r])
            else:
                # If we can't place a valid circle, try greedy approach
                circles.append([0.5, 0.5, 0.01])

        population.append(np.array(circles).flatten())

    return population

def mutate_individual(individual, mu=0.1, sigma=0.05):
    """Mutate an individual"""
    mutated = individual.copy()

    # Randomly decide which circles to mutate
    for i in range(0, len(mutated), 3):
        if random.random() < mu:
            # Mutate position
            mutated[i] += np.random.normal(0, sigma)  # x coordinate
            mutated[i+1] += np.random.normal(0, sigma)  # y coordinate
            mutated[i+2] += np.random.normal(0, sigma)  # radius

            # Ensure bounds
            mutated[i] = max(0.01, min(0.99, mutated[i]))
            mutated[i+1] = max(0.01, min(0.99, mutated[i+1]))
            mutated[i+2] = max(0.001, min(0.49, mutated[i+2]))

    return tuple(mutated)

def crossover_individuals(ind1, ind2):
    """Crossover two individuals"""
    # Simple Uniform crossover
    child1, child2 = list(ind1), list(ind2)

    for i in range(len(child1)):
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]

    return tuple(child1), tuple(child2)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: np.random.rand(78), 1)  # 26 circles * 3 params
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation, selection, crossover and mutation operators
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    pop = initialize_population(50)

    # Run evolutionary algorithm
    alg = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3,
                              ngen=300, verbose=False)

    # Get best individual
    best_individual = tools.selBest(alg[0], 1)[0]
    circles = np.array(best_individual).reshape(-1, 3)

    # Final refinement with local search
    refined_circles = local_refinement(circles)

    return refined_circles

def local_refinement(circles, max_iter=50):
    """Refine solution with local search to improve quality"""
    current_circles = circles.copy()

    for iteration in range(max_iter):
        improved = False

        # Try to increase some radii while maintaining constraints
        for i in range(len(current_circles)):
            x, y, r = current_circles[i]
            original_r = r

            # Try to increase radius
            max_possible_r = min(x, y, 1-x, 1-y)

            # Increase radius slightly if possible
            new_r = min(r * 1.1, max_possible_r)
            if new_r > r + 0.001:
                # Temporarily set new radius
                temp_circles = current_circles.copy()
                temp_circles[i, 2] = new_r

                # Check constraints
                if check_constraints(temp_circles):
                    current_circles[i, 2] = new_r
                    improved = True

        if not improved:
            break

    return current_circles


# EVOLVE-BLOCK-END