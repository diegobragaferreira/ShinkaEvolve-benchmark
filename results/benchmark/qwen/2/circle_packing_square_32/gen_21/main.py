# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools, algorithms
import uuid

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def generate_voronoi_seeds(n, max_attempts=1000):
    """Generate initial seed points using Voronoi-based approach to ensure good distribution."""
    attempts = 0
    while attempts < max_attempts:
        # Generate random points
        points = np.random.rand(n, 2)
        
        # Add boundary padding to avoid edge issues
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * 0.9 + 0.05
        
        try:
            # Create Voronoi diagram
            vor = Voronoi(points)
            
            # Check if all generators are inside unit square (not at infinity)
            valid_points = True
            for i, point in enumerate(points):
                if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
                    valid_points = False
                    break
            
            if valid_points:
                return points
                
        except:
            pass
            
        attempts += 1
    
    # Fallback to regular grid if Voronoi fails
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            points.append([x, y])
    
    return np.array(points[:n])

def initialize_population(pop_size, n_circles=32, seed=None):
    """Initialize population with diverse Voronoi-based configurations."""
    if seed is not None:
        np.random.seed(seed)
        
    population = []
    for _ in range(pop_size):
        # Generate Voronoi seeds
        seeds = generate_voronoi_seeds(n_circles)
        
        # Assign initial radii based on Voronoi cell areas
        radii = []
        for i in range(n_circles):
            # Simple area-based sizing - larger cells get larger radii
            # But also ensure they fit in their Voronoi region
            x, y = seeds[i]
            # Estimate maximum radius based on proximity to boundaries
            min_dist_to_boundary = min(x, 1-x, y, 1-y)
            # Estimate radius based on Voronoi region
            radii.append(min(0.15, min_dist_to_boundary * 0.8))
        
        # Add some randomness to break symmetry
        radii = [r * (0.8 + random.random() * 0.4) for r in radii]
        radii = [max(0.005, min(0.15, r)) for r in radii]
        
        circles = np.column_stack([seeds, radii])
        population.append(circles)
    
    return population

def calculate_fitness(circles):
    """Calculate fitness with proper penalty handling."""
    n = len(circles)
    total_radius = np.sum(circles[:, 2])
    
    # Penalty calculation with early termination for efficiency
    penalty = 0
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return -float('inf')  # Invalid configuration
    
    # Check overlap constraints with early termination
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                # Use softer penalty to maintain gradient information
                overlap_amount = (r1 + r2 - distance)
                penalty += overlap_amount * 100.0
    
    return total_radius - penalty

def mut_circle(individual, indpb=0.1, mut_radius=0.05):
    """Mutate a circle individual"""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < indpb:
            # Mutate position
            mutated[i, 0] += np.random.normal(0, 0.01)  # x
            mutated[i, 1] += np.random.normal(0, 0.01)  # y
            
            # Clamp position to valid range
            mutated[i, 0] = np.clip(mutated[i, 0], 0.001, 0.999)
            mutated[i, 1] = np.clip(mutated[i, 1], 0.001, 0.999)
            
        if random.random() < indpb:
            # Mutate radius
            mutated[i, 2] *= (1 + np.random.normal(0, mut_radius))
            # Clamp radius to valid range
            mutated[i, 2] = np.clip(mutated[i, 2], 0.001, 0.5)
    
    return mutated,

def cx_circles(ind1, ind2, cxpb=0.5):
    """Cross-over two circle individuals"""
    child1 = ind1.copy()
    child2 = ind2.copy()
    
    for i in range(len(child1)):
        if random.random() < cxpb:
            # Swap positions and radii
            child1[i, :2], child2[i, :2] = child2[i, :2], child1[i, :2]
            child1[i, 2], child2[i, 2] = child2[i, 2], child1[i, 2]
    
    return child1, child2

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses Voronoi-guided evolutionary algorithm.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: generate_voronoi_seeds(32)[0].tolist() + [random.uniform(0.01, 0.1)])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Set up evolutionary operators
    toolbox.register("evaluate", lambda ind: calculate_fitness(np.array(ind).reshape(-1, 3)))
    toolbox.register("mate", cx_circles)
    toolbox.register("mutate", mut_circle)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population
    pop = initialize_population(50, 32, seed=42)
    
    # Convert to DEAP format
    deap_pop = []
    for p in pop:
        ind = [item for sublist in p.tolist() for item in sublist]
        deap_pop.append(ind)
    
    # Run evolution
    hof = tools.ParetoFront()
    
    # Simple evolution with fixed parameters
    for generation in range(20):
        # Select
        offspring = toolbox.select(deap_pop, len(deap_pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = (fit,)
            
        # Replace population
        deap_pop[:] = offspring
        
        # Update hall of fame
        for ind in deap_pop:
            hof.update([ind])
    
    # Extract best individual
    best_individual = max(hof, key=lambda x: x.fitness.values[0])
    best_circles = np.array(best_individual).reshape(-1, 3)
    
    # Final optimization step with scipy to refine
    from scipy.optimize import minimize
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -calculate_fitness(circles)
    
    # Flatten for scipy
    initial_params = best_circles.flatten()
    
    # Bounds
    bounds = []
    for i in range(len(initial_params)):
        if i % 3 == 2:  # radius
            bounds.append((0.001, 0.5))
        else:  # x or y
            bounds.append((0.001, 0.999))
    
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 500})
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            return final_circles
    except:
        pass
    
    return best_circles

# EVOLVE-BLOCK-END