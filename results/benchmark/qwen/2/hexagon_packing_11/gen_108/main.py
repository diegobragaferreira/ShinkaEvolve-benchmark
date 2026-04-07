# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from deap import base, creator, tools, algorithms
import random
import time
from functools import partial

def hexagon_vertices(center_x, center_y, rotation_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(rotation_degrees)
    unit_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = unit_vertices @ rotation_matrix.T
    return rotated_vertices * side_length + np.array([center_x, center_y])

def check_containment_single(hex_vertices, outer_polygon):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_collision_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def estimate_min_outer_radius(inner_hex_params):
    """Estimate the minimal outer hexagon radius needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(11):
        x, y, rot = inner_hex_params[3*i], inner_hex_params[3*i+1], inner_hex_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        all_vertices.extend(hex_vertices)
    
    if len(all_vertices) == 0:
        return 100.0
        
    all_vertices = np.array(all_vertices)
    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_y, max_y = all_vertices[:, 1].min(), all_vertices[:, 1].max()
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    return max_dist * 2 / np.sqrt(3) * 1.1

def evaluate_individual(individual):
    """Evaluate a single individual solution for multi-objective optimization."""
    # Extract parameters
    n = 11
    inner_params = individual[:-1]
    outer_radius = individual[-1]
    
    # Check if outer hexagon is large enough
    if outer_radius <= 0:
        return (float('inf'), float('inf'))
    
    # Create outer hexagon vertices once
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = prep(Polygon(outer_vertices))
    
    # Initialize objectives
    objective1 = 1.0 / outer_radius  # We want to maximize this (minimize 1/outer_radius)
    
    # Constraint violations
    constraint_violation = 0
    
    # Check containment of all inner hexagons
    for i in range(n):
        x, y, rot = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            constraint_violation += 1000
    
    # Check collisions between all pairs of inner hexagons
    if constraint_violation == 0:
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, rot1 = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
                x2, y2, rot2 = inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]
                hex1_vertices = hexagon_vertices(x1, y1, rot1, 1)
                hex2_vertices = hexagon_vertices(x2, y2, rot2, 1)
                if check_collision_single(hex1_vertices, hex2_vertices):
                    constraint_violation += 1000
    
    # Apply penalty for constraint violations
    objective2 = constraint_violation
    
    return (objective1, objective2)

def generate_initial_population(pop_size=50):
    """Generate diverse initial population using multiple strategies."""
    initial_solutions = []
    
    # Strategy 1: Hexagonal close-packed arrangement
    hex_pattern = [
        [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
        [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
        [-1.732, 2, 0], [-3.464, 0, 0]
    ]
    
    # Strategy 2: Linear chain with variations
    linear_pattern = [
        [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0], [-1.25, 2.17, 0], [1.25, 2.17, 0],
        [-1.25, -2.17, 0], [1.25, -2.17, 0], [-3.75, 2.17, 0], [3.75, 2.17, 0],
        [-3.75, -2.17, 0], [3.75, -2.17, 0]
    ]
    
    # Strategy 3: Clustered arrangement
    cluster_pattern = [
        [0, 0, 0], [2, 0, 0], [1, 1.732, 0], [-1, 1.732, 0], [-2, 0, 0],
        [-1, -1.732, 0], [1, -1.732, 0], [3, 0, 0], [0, 2.5, 0],
        [0, -2.5, 0], [-3, 0, 0]
    ]
    
    # Strategy 4: Random but constrained arrangement
    random_pattern = []
    for i in range(11):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        rot = np.random.uniform(-180, 180)
        random_pattern.append([x, y, rot])
    
    patterns = [hex_pattern, linear_pattern, cluster_pattern, random_pattern]
    
    # Generate diverse initial individuals from patterns
    for i in range(pop_size):
        pattern_idx = i % len(patterns)
        pattern = patterns[pattern_idx].copy()
        
        # Add some randomness to make diverse solutions
        if i > 0:  # Only add variation to some individuals
            for j in range(len(pattern)):
                pattern[j][0] += np.random.normal(0, 0.5)
                pattern[j][1] += np.random.normal(0, 0.5)
                pattern[j][2] += np.random.normal(0, 15)
        
        # Flatten and add outer radius
        individual = []
        for x, y, rot in pattern:
            individual.extend([x, y, rot])
        
        est_radius = estimate_min_outer_radius(np.array(individual))
        individual.append(max(2.0, est_radius))  # Ensure reasonable size
        
        initial_solutions.append(individual)
    
    return initial_solutions

def mutate_individual(individual, indpb=0.1):
    """Custom mutation function for the individual."""
    mutated = individual.copy()
    
    # Mutate positions and rotations with varying strengths
    for i in range(len(mutated)-1):  # Exclude outer radius
        if random.random() < indpb:
            if i % 3 == 0:  # x coordinate
                mutated[i] += np.random.normal(0, 0.5)
            elif i % 3 == 1:  # y coordinate
                mutated[i] += np.random.normal(0, 0.5)
            else:  # rotation degree
                mutated[i] += np.random.normal(0, 30)
    
    # Mutate outer radius
    if random.random() < indpb:
        mutated[-1] += np.random.normal(0, 0.2)
        mutated[-1] = max(1.0, mutated[-1])
        
    return mutated

def crossover_individuals(ind1, ind2, cxpb=0.5):
    """Custom crossover function."""
    child1 = ind1.copy()
    child2 = ind2.copy()
    
    if random.random() < cxpb:
        # Swap segments of the individuals
        point = random.randint(1, len(ind1)-2)
        for i in range(point, len(ind1)):
            child1[i], child2[i] = child2[i], child1[i]
    
    return child1, child2

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Set up multi-objective optimization
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))  # Maximize first, minimize second
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: random.uniform(-10, 10), n=33)  # 11*3 positions + 1 outer_radius
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selNSGA2)
    
    # Generate initial population
    pop = generate_initial_population(50)
    for i in range(len(pop)):
        pop[i] = creator.Individual(pop[i])
    
    # Parameters for the evolutionary algorithm
    CXPB = 0.7   # Crossover probability
    MUTPB = 0.3  # Mutation probability
    NGEN = 50    # Number of generations
    
    # Run NSGA-II evolution
    hof = tools.ParetoFront()  # Store non-dominated solutions
    
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
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
        
        # Replace the old population with the new one
        pop[:] = offspring
        
        # Collect all individuals (including parents) for pareto front
        all_solutions = pop + hof
        hof.clear()
        for ind in all_solutions:
            hof.add(ind)
    
    # Get best solution from Pareto front
    best_individual = None
    best_fitness = (float('-inf'), float('inf'))
    
    # Find the best solution with lowest constraint violations
    for ind in hof:
        if ind.fitness.values[1] < best_fitness[1]:
            best_fitness = ind.fitness.values
            best_individual = ind
    
    # If no good solution found, fallback to best heuristic
    if best_individual is None:
        # Use simple pattern from previous work
        default_pattern = [
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ]
        
        # Flatten and add outer radius
        flat_pattern = []
        for x, y, rot in default_pattern:
            flat_pattern.extend([x, y, rot])
        est_radius = estimate_min_outer_radius(np.array(flat_pattern))
        flat_pattern.append(est_radius)
        
        best_individual = [val for val in flat_pattern]
    
    # Extract results
    outer_side_length = best_individual[-1]
    
    # Extract inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_individual[3*i], best_individual[3*i+1], best_individual[3*i+2]]
    
    # Outer hexagon data (centered at origin)
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
