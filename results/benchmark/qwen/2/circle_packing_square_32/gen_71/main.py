# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')

def compute_density_weighted_radius(circles, point, k=5):
    """Compute density-aware initial radius based on k-nearest neighbors"""
    if len(circles) <= 1:
        return 0.1
    
    # Calculate distances to all other circles
    distances = np.sqrt(np.sum((circles[:, :2] - point)**2, axis=1))
    
    # Get indices of k nearest neighbors (excluding self)
    sorted_indices = np.argsort(distances)
    nearest_indices = sorted_indices[1:k+1] if len(sorted_indices) > 1 else sorted_indices
    
    # Return inverse of average distance to neighbors as density measure
    if len(nearest_indices) == 0:
        return 0.1
    avg_distance = np.mean(distances[nearest_indices])
    # Higher density (lower avg distance) means smaller initial radius
    return max(0.01, min(0.3, 0.2 / (avg_distance + 1e-8)))

def initialize_density_aware_layout(n_circles, padding=0.05):
    """Initialize circles with density-aware placement strategy"""
    # Start with a basic hexagonal grid
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    spacing_x = (1 - 2*padding) / cols
    spacing_y = (1 - 2*padding) / rows
    hex_spacing_x = spacing_x
    hex_spacing_y = spacing_y * np.sqrt(3)/2
    
    circles = []
    circle_count = 0
    
    # Place circles in hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if circle_count >= n_circles:
                break
                
            # Hexagonal offset
            x_offset = (j if i % 2 == 0 else j + 0.5) * hex_spacing_x + padding
            y_offset = i * hex_spacing_y + padding
            
            # Add slight randomness
            x = max(padding, min(1-padding, x_offset + np.random.normal(0, 0.005*hex_spacing_x)))
            y = max(padding, min(1-padding, y_offset + np.random.normal(0, 0.005*hex_spacing_y)))
            
            circles.append([x, y, 0.0])
            circle_count += 1
            
        if circle_count >= n_circles:
            break
    
    # Convert to numpy array for density calculation
    circle_array = np.array(circles)
    
    # Compute density-weighted radii
    for i in range(len(circle_array)):
        circle_array[i, 2] = compute_density_weighted_radius(circle_array, circle_array[i, :2])
    
    # Ensure we have exactly n_circles
    while len(circle_array) < n_circles:
        x = np.random.uniform(padding, 1-padding)
        y = np.random.uniform(padding, 1-padding)
        radius = min(0.1, 0.5 * min(x, 1-x, y, 1-y))
        circle_array = np.vstack([circle_array, [x, y, radius]])
    
    return circle_array[:n_circles]

def is_valid_placement(circles, threshold=1e-6):
    """Check if circle configuration is valid"""
    n = len(circles)
    if n == 0:
        return False
        
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints using KDTree for efficiency
    if n > 1:
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Query pairs within 2*(max_radius) distance
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
        
        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2 - threshold:
                return False
                
    return True

def evaluate_individual(individual):
    """Evaluate fitness of individual - returns (sum_radii, constraint_violations)"""
    # Convert individual to circles array (x, y, r)
    circles = np.array(individual).reshape(-1, 3)
    
    # Check validity
    if not is_valid_placement(circles):
        # Penalize invalid placements heavily
        return (-1000.0, 1000.0)
    
    # Calculate sum of radii
    sum_radii = np.sum(circles[:, 2])
    
    # Count constraint violations
    violation_count = 0
    n = len(circles)
    
    if n > 1:
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Query pairs within 2*(max_radius) distance
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
        
        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                violation_count += 1
    
    # Apply penalty for violations
    penalty = violation_count * 500.0
    return (sum_radii - penalty, violation_count)

def create_individual():
    """Create a random individual - list of (x, y, r) tuples"""
    # Start with density-aware initialization
    circles = initialize_density_aware_layout(32)
    return circles.flatten().tolist()

def mutate_individual(individual, mutpb=0.1):
    """Mutate an individual by slightly adjusting positions and radii"""
    for i in range(len(individual)):
        if random.random() < mutpb:
            # Randomly adjust each element
            if i % 3 == 0:  # x coordinate
                individual[i] = max(0.05, min(0.95, individual[i] + random.gauss(0, 0.01)))
            elif i % 3 == 1:  # y coordinate
                individual[i] = max(0.05, min(0.95, individual[i] + random.gauss(0, 0.01)))
            else:  # radius
                individual[i] = max(0.01, min(0.4, individual[i] + random.gauss(0, 0.02)))
    return individual,

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Set up DEAP evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population_size = 20
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution parameters
    generations = 50
    cxpb = 0.7  # crossover probability
    mutpb = 0.3  # mutation probability
    
    # Main evolutionary loop
    for gen in range(generations):
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
        
        # Replace population with offspring
        population[:] = offspring
    
    # Get the best individual
    best_individual = tools.selBest(population, 1)[0]
    best_circles = np.array(best_individual).reshape(-1, 3)
    
    # Final validation and refinement
    if not is_valid_placement(best_circles):
        # Fall back to density-aware initialization if invalid
        np.random.seed(42)
        best_circles = initialize_density_aware_layout(32)
    
    # Final constraint enforcement
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        # Constrain positions to valid range
        best_circles[i][0] = max(r, min(1-r, x))
        best_circles[i][1] = max(r, min(1-r, y))
        # Constrain radii to valid range
        best_circles[i][2] = max(0.01, min(0.4, r))
        
    return best_circles

# EVOLVE-BLOCK-END
