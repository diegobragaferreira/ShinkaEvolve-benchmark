# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
from scipy.spatial import KDTree
import random
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Problem parameters
    n_circles = 32
    max_radius = 0.5  # Maximum possible radius for any circle
    
    # Initialize population
    toolbox = base.Toolbox()
    
    # Define individual representation: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    IND_SIZE = n_circles * 3
    
    def create_individual():
        # Create initial individual with random positions and radii
        individual = []
        for _ in range(n_circles):
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            r = random.uniform(0.01, min(0.5, max_radius))
            individual.extend([x, y, r])
        return individual
    
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_circle_pack(individual):
        """Evaluate fitness of circle packing"""
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check boundary constraints
        valid = True
        for i, (x, y, r) in enumerate(circles):
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                break
        
        if not valid:
            return (-1e6,)  # Penalize invalid solutions heavily
        
        # Check overlap constraints using KDTree for efficiency
        try:
            tree = KDTree(positions)
            pairs = tree.query_pairs(radii.sum() * 2)  # Rough estimate for neighbors
            
            for i, j in pairs:
                if i >= j:
                    continue
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    return (-1e6,)
        except:
            return (-1e6,)
        
        # Return sum of radii as fitness
        return (np.sum(radii),)
    
    toolbox.register("evaluate", eval_circle_pack)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop_size = 50
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution parameters
    CXPB = 0.7
    MUTPB = 0.3
    NGEN = 50
    
    # Main evolution loop
    best_fitness_history = []
    
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
        
        # Track best fitness
        best_fit = max(population, key=lambda x: x.fitness.values[0]).fitness.values[0]
        best_fitness_history.append(best_fit)
    
    # Get final best solution
    best_ind = max(population, key=lambda x: x.fitness.values[0])
    circles = np.array(best_ind).reshape(-1, 3)
    
    # Ensure final adjustment for boundary constraints
    for i in range(n_circles):
        x, y, r = circles[i]
        # Adjust radius if necessary to stay within bounds
        r = min(r, x, 1-x, y, 1-y)
        circles[i] = [x, y, r]
    
    return circles

# EVOLVE-BLOCK-END
