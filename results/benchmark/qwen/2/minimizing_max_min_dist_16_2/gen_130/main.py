# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms
import random
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0

        # Efficiently compute all pairwise distances
        distances = cdist(points, points, metric='euclidean')

        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist <= 0:
            return 0

        return min_dist / max_dist
    
    def chaos_initialization(n_points=16):
        """Initialize points using logistic map for better spread and symmetry breaking."""
        # Logistic map parameters
        r = 3.99  # Chaotic regime
        x = 0.5  # Initial value
        points = []
        
        for i in range(n_points):
            x = r * x * (1 - x)
            # Map to [0,1] x [0,1] 
            points.append([x, (x * 1.234567) % 1.0])
            
        return np.array(points)
    
    def create_individual():
        """Create a single individual (16 points) using chaotic initialization."""
        return chaos_initialization().flatten()
    
    def evaluate_individual(individual):
        """Evaluate fitness of an individual - maximize min/max ratio"""
        points = individual.reshape(-1, 2)
        ratio = calculate_min_max_ratio(points)
        # Return negative since we want to maximize (minimize negative)
        return (-ratio,)
    
    def mutate_individual(individual, indpb=0.1):
        """Mutate an individual by adding Gaussian noise"""
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += np.random.normal(0, 0.02)
                # Keep within bounds
                individual[i] = max(0.0, min(1.0, individual[i]))
        return individual,
    
    def crossover_individual(ind1, ind2):
        """Crossover two individuals"""
        size = len(ind1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cxpoints
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
            
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        return ind1, ind2
    
    # Evolutionary Algorithm Parameters
    POP_SIZE = 20
    GEN_COUNT = 15
    MUTPB = 0.1
    CXPB = 0.5
    
    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individual)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population with diverse chaotic configurations
    pop = toolbox.population(n=POP_SIZE)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop
    for gen in range(GEN_COUNT):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
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
        
        # Replace the old population with the new one
        pop[:] = offspring
    
    # Get the best individual from evolution
    best_individual = tools.selBest(pop, 1)[0]
    best_points = best_individual.reshape(-1, 2)
    
    # Apply local refinement to the best individual
    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)
        
        # Calculate all pairwise distances efficiently
        distances = cdist(points, points, metric='euclidean')
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio to maximize the ratio (negative because we minimize)
        if max_dist <= 0:
            return 0
        return -min_dist / max_dist
    
    bounds = [(0, 1) for _ in range(32)]
    
    # Refine with local optimization
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 100}
        )
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            
            # Check if refinement improved the result
            original_ratio = calculate_min_max_ratio(best_points)
            refined_ratio = calculate_min_max_ratio(refined_points)
            
            if refined_ratio > original_ratio:
                return refined_points
    except:
        pass
    
    return best_points

# EVOLVE-BLOCK-END