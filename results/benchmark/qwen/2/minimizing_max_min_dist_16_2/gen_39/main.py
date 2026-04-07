# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from deap import base, creator, tools, algorithms
import random
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    def golden_spiral_2d(n_points):
        """Generate points on a 2D golden spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        return np.array(points)
    
    def evaluate_individual(individual):
        """Evaluate fitness of an individual (point configuration)"""
        # Convert flat individual to 2D points
        points = np.array(individual).reshape(-1, 2)
        
        # Ensure points are within bounds [0,1]
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Handle edge case
        if len(distances) == 0 or np.max(distances) == 0:
            return (0,)  # Return very poor fitness
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 1e-12:
            return (0,)
            
        # Return the ratio as fitness (higher is better)
        ratio = min_dist / max_dist
        return (ratio,)
    
    def mutate_individual(individual, indpb=0.1):
        """Mutate an individual by adding Gaussian noise to points"""
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < indpb:
                mutated[i] += random.gauss(0, 0.02)
        return tuple(mutated)
    
    def crossover_individuals(ind1, ind2):
        """Crossover two individuals"""
        size = len(ind1)
        cxpoint1 = random.randint(1, size // 2)
        cxpoint2 = random.randint(cxpoint1, size)
        
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        return ind1, ind2
    
    # Create evaluation function
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=32)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Generate initial population
    # Use golden spiral as starting point for better distribution
    spiral_points = golden_spiral_2d(16)
    # Scale to fit in [0,1] range with padding
    spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
        np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
    spiral_points = spiral_points * 0.8 + 0.1  # Scale to [0.1, 0.9]
    
    # Create initial population with spiral as first individual
    initial_population = []
    for i in range(10):  # 10 individuals
        if i == 0:
            # Use spiral as first individual
            individual = list(spiral_points.flatten())
        else:
            # Add slight perturbations to spiral
            individual = list(spiral_points.flatten() + np.random.normal(0, 0.05, 32))
            # Clip to valid range
            individual = [max(0, min(1, val)) for val in individual]
        initial_population.append(individual)
    
    # Use the best of the initial configurations as fallback
    best_initial = max(initial_population, key=lambda x: evaluate_individual(x)[0])
    
    # Run evolutionary algorithm with timeout protection
    start_time = time.time()
    max_time = 170  # Leave 10 seconds buffer
    
    try:
        # Run evolutionary algorithm with limited time
        pop = initial_population
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Limit number of generations based on time
        max_generations = 100
        for gen in range(max_generations):
            if time.time() - start_time > max_time:
                break
                
            # Select the next generation
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.2:  # Mutation probability
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population with offspring
            pop[:] = offspring
            
            # Update hall of fame
            hof.update(pop)
            
        # Return best solution found
        best_solution = hof[0]
        best_points = np.array(best_solution).reshape(-1, 2)
        
        # Ensure points are within bounds
        best_points = np.clip(best_points, 0, 1)
        
        return best_points
        
    except Exception as e:
        # Fallback to initial spiral configuration
        return spiral_points

# EVOLVE-BLOCK-END