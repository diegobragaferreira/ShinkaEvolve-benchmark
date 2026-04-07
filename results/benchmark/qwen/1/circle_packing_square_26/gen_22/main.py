# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
from scipy.spatial import cKDTree
import random
import math

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Problem parameters
    n_circles = 26
    bounds = [0, 1]  # unit square
    
    # Precompute some constants for optimization
    max_radius = 1.0 / 2.0  # Maximum possible radius for any single circle
    
    def validate_circle(circle):
        """Validate that circle is within bounds and doesn't overlap with others"""
        x, y, r = circle
        # Check bounds
        if x - r < bounds[0] or x + r > bounds[1] or y - r < bounds[0] or y + r > bounds[1]:
            return False
        return True
    
    def check_overlap(c1, c2):
        """Check if two circles overlap"""
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
        return dist_sq < (r1 + r2)**2
    
    def evaluate(individual):
        """Evaluate fitness of individual (maximize sum of radii)"""
        circles = np.array(individual).reshape(-1, 3)
        
        # Check bounds
        for i, circle in enumerate(circles):
            if not validate_circle(circle):
                return (0,)  # Invalid solution
        
        # Check overlaps
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if check_overlap(circles[i], circles[j]):
                    return (0,)  # Overlapping circles
        
        # Return negative sum of radii (minimization problem converted to maximization)
        total_radius = sum(circle[2] for circle in circles)
        return (total_radius,)
    
    def create_individual():
        """Create a valid individual with initial circles"""
        # Start with a greedy approach for initialization
        circles = []
        
        # Initialize with a few circles using a greedy approach
        # Place some circles near corners to start
        corner_positions = [
            (0.1, 0.1, 0.05),
            (0.9, 0.1, 0.05),
            (0.1, 0.9, 0.05),
            (0.9, 0.9, 0.05)
        ]
        
        for cx, cy, r in corner_positions:
            circles.append((cx, cy, r))
        
        # Fill remaining positions with random valid circles
        remaining = n_circles - len(circles)
        for _ in range(remaining):
            # Try several times to place a valid circle
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                # Random position and radius
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = np.random.uniform(0.01, max_radius * 0.5)
                
                # Check if it's valid
                candidate = (x, y, r)
                if not validate_circle(candidate):
                    attempts += 1
                    continue
                    
                # Check against existing circles
                overlap = False
                for existing in circles:
                    if check_overlap(candidate, existing):
                        overlap = True
                        break
                        
                if not overlap:
                    circles.append(candidate)
                    placed = True
                else:
                    attempts += 1
            
            # If we couldn't place a circle, use a smaller default
            if not placed:
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                r = 0.02
                circles.append((x, y, r))
        
        return circles
    
    def mutate(individual):
        """Mutate an individual"""
        # Convert to list for easier manipulation
        circles = list(individual)
        mutated = False
        
        # Mutate one circle at random
        idx = random.randint(0, len(circles)-1)
        cx, cy, r = circles[idx]
        
        # Apply small perturbation to position and radius
        delta_x = np.random.normal(0, 0.01)
        delta_y = np.random.normal(0, 0.01)
        delta_r = np.random.normal(0, 0.005)
        
        # New position and radius
        new_x = max(bounds[0] + r, min(bounds[1] - r, cx + delta_x))
        new_y = max(bounds[0] + r, min(bounds[1] - r, cy + delta_y))
        new_r = max(0.001, min(max_radius, r + delta_r))
        
        # If there was a meaningful change
        if abs(new_x - cx) > 1e-6 or abs(new_y - cy) > 1e-6 or abs(new_r - r) > 1e-6:
            circles[idx] = (new_x, new_y, new_r)
            mutated = True
        
        return tuple(circles), 
    
    def crossover(individual1, individual2):
        """Crossover between two individuals"""
        circles1 = list(individual1)
        circles2 = list(individual2)
        
        # Simple uniform crossover - swap some elements
        child1 = []
        child2 = []
        
        for i in range(len(circles1)):
            if random.random() < 0.5:
                child1.append(circles1[i])
                child2.append(circles2[i])
            else:
                child1.append(circles2[i])
                child2.append(circles1[i])
        
        # Try to fix invalid individuals
        valid1 = check_validity(child1)
        valid2 = check_validity(child2)
        
        if not valid1:
            child1 = child1[:len(circles1)]  # truncate to avoid overflow
        if not valid2:
            child2 = child2[:len(circles2)]
            
        return tuple(child1), tuple(child2)
    
    def check_validity(circles):
        """Check if circles form a valid configuration"""
        for i, circle in enumerate(circles):
            if not validate_circle(circle):
                return False
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if check_overlap(circles[i], circles[j]):
                    return False
        return True
    
    # Setup DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop_size = 200
    population = toolbox.population(n=pop_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution parameters
    n_generations = 50
    cxpb = 0.5      # crossover probability
    mutpb = 0.2     # mutation probability
    
    # Main evolution loop
    for gen in range(n_generations):
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
        
        # Replace population
        population[:] = offspring
    
    # Get best individual
    best_individual = tools.selBest(population, k=1)[0]
    
    # Convert back to array format
    result = np.array(best_individual).reshape(-1, 3)
    
    return result

# EVOLVE-BLOCK-END
