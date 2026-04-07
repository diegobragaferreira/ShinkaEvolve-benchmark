# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
import math

# Global constants
POP_SIZE = 150
NGEN = 100
MUTPB = 0.8
CXPB = 0.5
BOUND_LOW = 0.0
BOUND_UP = 1.0

# Define the fitness and individual classes for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def check_containment(circles):
    """Check containment constraints efficiently"""
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]
    
    # Check boundaries for all circles at once
    containment_violations = (
        (x_coords - radii < BOUND_LOW) |
        (x_coords + radii > BOUND_UP) |
        (y_coords - radii < BOUND_LOW) |
        (y_coords + radii > BOUND_UP)
    )
    
    return np.sum(containment_violations)

def calculate_overlap_penalty(circles):
    """Calculate overlap penalty using efficient spatial indexing"""
    if len(circles) <= 1:
        return 0.0
    
    # Build KDTree for efficient neighbor search
    tree = cKDTree(circles[:, :2])
    
    penalty = 0.0
    radii = circles[:, 2]
    
    # For each circle, find neighbors within sum of radii
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        
        # Query nearby points (within 2*(r1+r2) distance)
        neighbors = tree.query_ball_point([x1, y1], 2*(r1 + max(radii)))
        
        # Check overlaps with neighbors
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    penalty += 1000 * (r1 + r2 - distance)
    
    return penalty

def eval_circles(individual):
    """Evaluate the fitness of an individual (set of circles)"""
    # Convert individual to circles array
    circles = np.array(individual).reshape(-1, 3)
    
    # Calculate sum of radii
    total_radius = np.sum(circles[:, 2])
    
    # Check constraints
    containment_violations = check_containment(circles)
    overlap_penalty = calculate_overlap_penalty(circles)
    
    # Combine penalties
    total_penalty = 10000 * containment_violations + overlap_penalty
    
    # Return fitness (higher is better)
    return (total_radius - total_penalty,)

def generate_grid_placement():
    """Generate initial circle positions on a grid"""
    n = 26
    # Create roughly a 5x5 grid (since 5*5=25, add one extra)
    rows_cols = int(math.ceil(math.sqrt(n)))
    spacing_x = 1.0 / (rows_cols + 1)
    spacing_y = 1.0 / (rows_cols + 1)
    
    circles = []
    count = 0
    
    for i in range(rows_cols):
        for j in range(rows_cols):
            if count >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Add some randomness to positions
            x += random.uniform(-spacing_x/4, spacing_x/4)
            y += random.uniform(-spacing_y/4, spacing_y/4)
            # Ensure positions stay within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Initial radii based on proximity to edges and other circles
            min_dist_to_bound = min(x, 1-x, y, 1-y)
            # Start with a reasonable radius based on available space
            r = min(0.05, min_dist_to_bound/2)
            # Add some randomness to radius
            r *= random.uniform(0.8, 1.2)
            r = max(0.005, min(0.15, r))
            
            circles.extend([x, y, r])
            count += 1
            
        if count >= n:
            break
    
    # If we have fewer circles, fill remaining with random ones
    while len(circles) < n * 3:
        x = random.uniform(0.01, 0.99)
        y = random.uniform(0.01, 0.99)
        r = random.uniform(0.005, 0.1)
        circles.extend([x, y, r])
    
    return circles[:n*3]

def init_individual():
    """Initialize an individual with structured grid placement + random perturbation"""
    individual = generate_grid_placement()
    
    # Add some random perturbations to improve local search
    for i in range(len(individual)):
        if i % 3 == 2:  # This is a radius
            # Mutate radius with bounded adjustment
            individual[i] = max(0.001, min(0.4, individual[i] * random.uniform(0.9, 1.1)))
        else:  # This is x or y coordinate
            # Mutate position with bounded adjustment
            individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, 0.01)))
    
    return individual

def mutate_individual(individual):
    """Mutate an individual with adaptive mutation rates"""
    # Get current generation (this is a bit hacky but works for this context)
    # In practice, this should be passed in or managed differently
    gen_rate = 0.1  # Base mutation rate
    
    for i in range(len(individual)):
        if random.random() < gen_rate:  # Apply adaptive mutation
            if i % 3 == 2:  # This is a radius
                # Mutate radius with bounded adjustment (larger mutations early on)
                individual[i] = max(0.001, min(0.4, individual[i] * random.uniform(0.85, 1.15)))
            else:  # This is x or y coordinate
                # Mutate position with bounded adjustment
                individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, 0.03)))
    return individual,

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
    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_circles)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop with adaptive parameters
    for gen in range(NGEN):
        # Adjust mutation rate over generations
        current_mutation_rate = MUTPB * max(0.1, 1.0 - gen/NGEN)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < current_mutation_rate:
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

    return circles


# EVOLVE-BLOCK-END