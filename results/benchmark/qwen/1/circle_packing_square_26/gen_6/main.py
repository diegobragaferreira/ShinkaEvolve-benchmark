# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial.distance import cdist

# Global constants for the problem
N_CIRCLES = 26
BOUNDARY_MARGIN = 1e-6  # Small margin to ensure circles stay inside unit square

# Define the fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def eval_circle_packing(individual):
    """Evaluate the fitness of an individual (set of circles)."""
    # Convert individual to circles array
    circles = np.array(individual).reshape(-1, 3)

    # Calculate sum of radii
    total_radius = np.sum(circles[:, 2])

    # Check constraints
    penalty = 0

    # Check containment constraints
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 1000  # Heavy penalty for containment violations

    # Check overlap constraints
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]

            # Distance between centers
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # Overlap penalty if circles are too close
            if dist < r1 + r2:
                overlap = (r1 + r2 - dist)
                penalty += overlap * 1000  # Penalty proportional to overlap

    # Return fitness (negative because we minimize penalty and maximize radius sum)
    return (total_radius - penalty,)

def init_individual():
    """Initialize a single individual with valid circle positions."""
    individual = []
    # Start with random positions and radii, ensuring they don't overlap
    circles = []

    # Try placing circles one by one
    for _ in range(N_CIRCLES):
        max_attempts = 1000
        placed = False
        attempts = 0

        while not placed and attempts < max_attempts:
            # Random position and radius
            x = random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            y = random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            r = random.uniform(0.001, 0.1)  # Reasonable range for initial radii

            # Check containment constraint
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                attempts += 1
                continue

            # Check overlap with existing circles
            overlap = False
            for cx, cy, cr in circles:
                if np.sqrt((x - cx)**2 + (y - cy)**2) < r + cr:
                    overlap = True
                    break

            if not overlap:
                circles.append((x, y, r))
                placed = True
            else:
                attempts += 1

        if not placed:
            # If we couldn't place a circle, just make something up
            x = random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            y = random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            r = random.uniform(0.001, 0.05)
            circles.append((x, y, r))

    # Flatten into individual list
    for x, y, r in circles:
        individual.extend([x, y, r])

    return individual

def mutate_individual(individual, indpb=0.1):
    """Mutate an individual by slightly modifying circle properties."""
    mutated = individual[:]

    # Mutate some circles
    for i in range(0, len(mutated), 3):
        if random.random() < indpb:
            # Randomly choose which parameter to mutate
            param = random.randint(0, 2)

            if param == 0:  # x coordinate
                mutated[i] = max(BOUNDARY_MARGIN, min(1 - BOUNDARY_MARGIN, mutated[i] + random.gauss(0, 0.01)))
            elif param == 1:  # y coordinate
                mutated[i+1] = max(BOUNDARY_MARGIN, min(1 - BOUNDARY_MARGIN, mutated[i+1] + random.gauss(0, 0.01)))
            elif param == 2:  # radius
                mutated[i+2] = max(0.001, min(0.5, mutated[i+2] + random.gauss(0, 0.005)))

    return tuple(mutated)

def crossover_individuals(ind1, ind2):
    """Crossover two individuals by combining parts of them."""
    # Simple uniform crossover
    child1 = []
    child2 = []

    for i in range(0, len(ind1), 3):
        if random.random() < 0.5:
            child1.extend([ind1[i], ind1[i+1], ind1[i+2]])
            child2.extend([ind2[i], ind2[i+1], ind2[i+2]])
        else:
            child1.extend([ind2[i], ind2[i+1], ind2[i+2]])
            child2.extend([ind1[i], ind1[i+1], ind1[i+2]])

    return tuple(child1), tuple(child2)

def check_validity(individual):
    """Check if an individual represents a valid configuration."""
    circles = np.array(individual).reshape(-1, 3)

    # Check containment
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlaps
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < r1 + r2:
                return False

    return True

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_circle_packing)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population(n=50)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolutionary algorithm parameters
    CXPB = 0.7  # Crossover probability
    MUTPB = 0.3  # Mutation probability
    NGEN = 100  # Number of generations

    # Run evolution
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                child1, child2 = toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                mutant = toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population with offspring
        pop[:] = offspring

        # Print progress
        if gen % 20 == 0:
            fits = [ind.fitness.values[0] for ind in pop]
            length = len(pop)
            mean = sum(fits) / length
            sum_of_best = max(fits)
            print(f"Generation {gen}: Max fitness = {sum_of_best}, Avg fitness = {mean}")

    # Get the best individual
    best_ind = tools.selBest(pop, 1)[0]

    # Convert back to numpy array
    circles = np.array(best_ind).reshape(-1, 3)

    # Ensure final validity
    if not check_validity(best_ind):
        # If invalid, try to fix by applying a local improvement heuristic
        circles = repair_configuration(circles)

    return circles

def repair_configuration(circles):
    """Apply simple repair to ensure configuration is valid."""
    # Simple repair: adjust positions to satisfy constraints
    repaired = circles.copy()

    for i in range(len(repaired)):
        x, y, r = repaired[i]

        # Fix containment
        if x - r < 0:
            x = r
        elif x + r > 1:
            x = 1 - r

        if y - r < 0:
            y = r
        elif y + r > 1:
            y = 1 - r

        repaired[i] = [x, y, r]

    # Now resolve overlaps through iterative adjustment
    improved = repaired.copy()
    changed = True
    iterations = 0
    while changed and iterations < 50:
        changed = False
        iterations += 1

        for i in range(len(improved)):
            x1, y1, r1 = improved[i]

            for j in range(len(improved)):
                if i != j:
                    x2, y2, r2 = improved[j]

                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    if dist < r1 + r2:  # Overlap exists
                        # Shrink first circle slightly to resolve overlap
                        overlap = (r1 + r2 - dist) / 2
                        if r1 - overlap > 0.001:
                            improved[i] = [x1, y1, r1 - overlap]
                            changed = True
                        else:
                            # If we can't shrink, try adjusting position
                            # Move circle away from overlapping circle
                            dx = x1 - x2
                            dy = y1 - y2
                            mag = np.sqrt(dx*dx + dy*dy)
                            if mag > 0:
                                scale = overlap / mag
                                improved[i] = [x1 + dx*scale, y1 + dy*scale, r1]
                                changed = True

    return improved


# EVOLVE-BLOCK-END