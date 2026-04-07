# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 32
    max_radius = 0.5  # Maximum possible radius for any circle
    
    def evaluate(individual):
        """Evaluate fitness of individual (chromosome)"""
        # Reshape individual into (x,y,r) coordinates
        coords = individual.reshape(-1, 3)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        radii = coords[:, 2]
        
        # Check containment constraints
        valid_containment = np.all((radii <= x_coords) & 
                                  (x_coords <= 1 - radii) & 
                                  (radii <= y_coords) & 
                                  (y_coords <= 1 - radii))
        
        if not valid_containment:
            return (0,)  # Invalid solution
            
        # Build KDTree for efficient collision checking
        points = np.column_stack([x_coords, y_coords])
        tree = cKDTree(points)
        
        # Check overlap constraints
        collisions = 0
        total_radius = np.sum(radii)
        
        # Only check pairs that could potentially collide based on distance
        for i in range(len(points)):
            # Find neighbors within 2*(max_radius) distance
            neighbors = tree.query_ball_point(points[i], 2 * max_radius)
            for j in neighbors:
                if i != j:
                    dist = np.sqrt((x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2)
                    if dist < (radii[i] + radii[j]):
                        collisions += 1
        
        if collisions > 0:
            # Penalize overlapping solutions heavily
            return (total_radius - 1000 * collisions,)
        
        return (total_radius,)
    
    def mutate_individual(individual, indpb=0.1):
        """Custom mutation operator for circle packing"""
        for i in range(0, len(individual), 3):
            if random.random() < indpb:
                # Mutate x coordinate
                individual[i] = max(0.01, min(0.99, individual[i] + np.random.normal(0, 0.01)))
            if random.random() < indpb:
                # Mutate y coordinate
                individual[i+1] = max(0.01, min(0.99, individual[i+1] + np.random.normal(0, 0.01)))
            if random.random() < indpb:
                # Mutate radius
                individual[i+2] = max(0.001, min(0.49, individual[i+2] + np.random.normal(0, 0.005)))
        return individual,
    
    def crossover_individual(ind1, ind2):
        """Custom crossover operator for circle packing"""
        size = len(ind1)
        cxpoint1 = random.randint(1, size // 3)
        cxpoint2 = random.randint(cxpoint1, size // 3)
        
        # Perform crossover on each parameter group
        for i in range(cxpoint1 * 3, cxpoint2 * 3):
            if i % 3 == 0:  # x coordinate
                ind1[i], ind2[i] = ind2[i], ind1[i]
            elif i % 3 == 1:  # y coordinate
                ind1[i], ind2[i] = ind2[i], ind1[i]
            else:  # radius
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    # Register DEAP components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: random.uniform(0.01, 0.99), n_circles * 3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover_individual)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population with better starting configuration
    def create_initial_population(pop_size):
        population = []
        for _ in range(pop_size):
            # Initial heuristic configuration: hexagonal grid
            individual = []
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = int(np.ceil(n_circles / rows))
            
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)
            
            for i in range(n_circles):
                row = i // cols
                col = i % cols
                
                # Offset odd rows slightly for hexagonal packing
                offset = 0.5 if row % 2 == 1 else 0.0
                x = (col + 1 + offset) * spacing_x
                y = (row + 1) * spacing_y
                
                # Ensure center is within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                # Small random initial radius
                r = np.random.uniform(0.01, 0.1)
                
                individual.extend([x, y, r])
            
            population.append(individual)
        return population
    
    # Initialize population
    pop = create_initial_population(50)
    
    # Run genetic algorithm
    hof = tools.ParetoFront()  # Keep best individuals
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run evolution
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                   ngen=200, stats=stats, halloffame=hof, verbose=False)
    
    # Get best solution
    best_individual = hof[0] if hof else pop[0]
    
    # Convert back to the required format
    result = np.array(best_individual).reshape(-1, 3)
    
    # Ensure final constraint satisfaction
    coords = result[:, :2]
    radii = result[:, 2]
    
    # Re-check and adjust for any slight violations
    for i in range(len(coords)):
        # Apply boundary constraints
        coords[i, 0] = max(radii[i], min(1 - radii[i], coords[i, 0]))
        coords[i, 1] = max(radii[i], min(1 - radii[i], coords[i, 1]))
    
    result[:, :2] = coords
    result[:, 2] = radii
    
    return result

# EVOLVE-BLOCK-END
