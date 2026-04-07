# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial import cKDTree
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    n_circles = 26
    max_radius = 0.5
    
    def evaluate(individual):
        """Evaluate fitness of individual (circles configuration)"""
        # Reshape individual into (x, y, r) for each circle
        circles = np.array(individual).reshape(-1, 3)
        
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check containment constraints
        containment_ok = (
            np.all(radii <= positions[:, 0]) and 
            np.all(radii <= positions[:, 1]) and 
            np.all(positions[:, 0] <= 1 - radii) and 
            np.all(positions[:, 1] <= 1 - radii)
        )
        
        if not containment_ok:
            return (-1e6,)  # Invalid configuration
        
        # Check overlap constraints using KDTree for efficiency
        try:
            tree = cKDTree(positions)
            # Find pairs within distance (r_i + r_j)
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
            
            # Check if any pair violates the non-overlap constraint
            for i, j in pairs:
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                if dist < (radii[i] + radii[j]):
                    return (-1e6,)  # Violates non-overlap constraint
                    
        except Exception:
            return (-1e6,)  # Error in collision detection
            
        # Objective: maximize sum of radii
        return (np.sum(radii),)
    
    def mutate_individual(individual, indpb=0.1):
        """Custom mutation for circle configurations"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < indpb:
                if i % 3 == 2:  # Radius component
                    mutated[i] = max(0.001, min(0.5, mutated[i] + random.gauss(0, 0.01)))
                else:  # Position components
                    mutated[i] = max(0.001, min(0.999, mutated[i] + random.gauss(0, 0.02)))
        return tuple(mutated)
    
    def create_individual():
        """Create a random valid individual"""
        individual = []
        for _ in range(n_circles):
            # Ensure initial circles are inside the unit square
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            r = random.uniform(0.01, 0.2)  # Reasonable initial radius
            individual.extend([x, y, r])
        return tuple(individual)
    
    # Setup GA framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run genetic algorithm
    population_size = 100
    generations = 50
    
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=0.7, mutpb=0.1, ngen=generations,
            stats=stats, halloffame=hof, verbose=False
        )
        
        # Get best individual
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)
        
    except Exception:
        # Fallback to greedy approach if GA fails
        result = greedy_initialization()
    
    return result

def greedy_initialization():
    """Generate a good initial configuration using greedy approach"""
    n = 26
    circles = np.zeros((n, 3))
    
    # Start with a simple grid-like arrangement plus some randomness
    sqrt_n = int(np.ceil(np.sqrt(n)))
    spacing_x = 1.0 / (sqrt_n + 1)
    spacing_y = 1.0 / (sqrt_n + 1)
    
    idx = 0
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Add small random perturbation
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            # Ensure we're within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            # Set radius to be reasonable
            r = 0.05 + random.uniform(-0.02, 0.02)
            r = max(0.01, min(0.2, r))
            circles[idx] = [x, y, r]
            idx += 1
            
    return circles

# EVOLVE-BLOCK-END
