# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from deap import base, creator, tools, algorithms
import math
import time

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2
BENCHMARK_RATIO = 0.2544

def create_regular_hexagon(center=(0, 0), side_length=1, rotation_deg=0):
    """Create a regular hexagon as a shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center[0] + side_length * math.cos(angle)
        y = center[1] + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon"""
    return outer_hexagon.contains(hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

def calculate_outer_hex_side_length(inner_hex_data, outer_center=(0, 0)):
    """Calculate the minimum outer hexagon side length that contains all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        
        # Create temporary hexagon to get vertices
        temp_hex = create_regular_hexagon(pos, 1, rot)
        vertices = list(temp_hex.exterior.coords)[:-1]  # Exclude duplicate last point
        
        # Find max distance from center to any vertex
        for vertex in vertices:
            dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add some margin to ensure containment
    return max_dist * 1.1

def get_all_vertices(hexagon_data):
    """Get all vertices from hexagon data for containment calculation"""
    all_vertices = []
    for pos, rot in hexagon_data:
        hexagon = create_regular_hexagon(pos, 1, rot)
        vertices = list(hexagon.exterior.coords)[:-1]
        all_vertices.extend(vertices)
    return all_vertices

def evaluate_constraints(hexagon_data, outer_radius):
    """Evaluate if hexagons satisfy constraints"""
    # Create inner hexagons
    inner_hexagons = []
    for pos, rot in hexagon_data:
        hexagon = create_regular_hexagon(pos, 1, rot)
        inner_hexagons.append(hexagon)

    # Create outer hexagon
    outer_hexagon = create_regular_hexagon((0, 0), outer_radius, 0)
    
    # Check containment
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hexagon):
            return False, False, 0.0  # containment violated
            
    # Check overlaps
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return False, False, 0.0  # overlap violated
    
    return True, True, 1.0 / outer_radius  # valid solution

def construct_spiral_initialization():
    """Construct an initial configuration using spiral arrangement"""
    # Golden spiral pattern for better spatial distribution  
    golden_angle = 2.399963229728653
    spiral_positions = []
    
    # Place first 11 points in spiral
    for i in range(11):
        radius = 0.8 + i * 0.5
        angle = i * golden_angle
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        spiral_positions.append((x, y))
    
    # Create diverse initial configuration
    config = []
    np.random.seed(42)
    for i, (cx, cy) in enumerate(spiral_positions):
        # Add controlled jitter
        jitter_x = np.random.uniform(-0.2, 0.2)
        jitter_y = np.random.uniform(-0.2, 0.2)
        config.append([(cx + jitter_x), (cy + jitter_y), np.random.uniform(0, 360)])
    
    return config

def generate_initial_population(pop_size, n_hexagons=11):
    """Generate initial population with diverse configurations"""
    population = []
    for i in range(pop_size):
        # Alternate between different initialization strategies
        if i % 3 == 0:
            # Golden spiral configuration
            individual = construct_spiral_initialization()
        elif i % 3 == 1:
            # Random configuration with good spread
            individual = []
            for j in range(n_hexagons):
                individual.append([
                    np.random.uniform(-6, 6),
                    np.random.uniform(-6, 6),
                    np.random.uniform(0, 360)
                ])
        else:
            # Grid-based configuration
            individual = []
            positions = [
                (0, 0), (-2, 0), (2, 0), (0, 2), (0, -2), 
                (-2, 2), (2, 2), (-2, -2), (2, -2), 
                (-4, 0), (0, 4)
            ]
            for j, (x, y) in enumerate(positions):
                individual.append([
                    x + np.random.uniform(-0.5, 0.5),
                    y + np.random.uniform(-0.5, 0.5),
                    np.random.uniform(0, 360)
                ])
        
        population.append(individual)
    
    return population

def specialized_crossover(ind1, ind2):
    """Custom crossover operation tailored for hexagon packing"""
    # Perform uniform crossover but preserve hexagon structure
    for i in range(len(ind1)):
        if np.random.random() < 0.5:
            # Swap positions and rotations
            ind1[i][0], ind2[i][0] = ind2[i][0], ind1[i][0]  # x
            ind1[i][1], ind2[i][1] = ind2[i][1], ind1[i][1]  # y
            ind1[i][2], ind2[i][2] = ind2[i][2], ind1[i][2]  # rotation
    
    return ind1, ind2

def specialized_mutation(individual, indpb=0.1):
    """Custom mutation operator for hexagon packing"""
    for i in range(len(individual)):
        if np.random.random() < indpb:
            # Mutate position
            individual[i][0] += np.random.normal(0, 0.3)  # x
            individual[i][1] += np.random.normal(0, 0.3)  # y
            individual[i][2] += np.random.normal(0, 10)   # rotation (clamped later)
            
            # Ensure rotation stays within bounds
            individual[i][2] = individual[i][2] % 360
    
    return individual

def fitness_function(individual, penalty_weight=1000):
    """
    Calculate fitness for individual - maximizes 1/outer_radius while penalizing constraint violations
    """
    # Convert individual to hexagon data format
    hex_data = individual
    
    # Estimate outer radius based on current configuration
    estimated_radius = calculate_outer_hex_side_length(hex_data)
    
    # Evaluate constraints
    containment_ok, overlap_ok, inv_radius = evaluate_constraints(hex_data, estimated_radius)
    
    # Calculate fitness: higher is better
    if not (containment_ok and overlap_ok):
        # Penalize constraint violations heavily
        penalty = penalty_weight * (1 if not containment_ok else 0) + penalty_weight * (1 if not overlap_ok else 0)
        return -penalty  # Very negative fitness for constraint violations
    
    # Return fitness maximizing 1/outer_radius (minimizing outer radius)
    return inv_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid genetic algorithm with specialized operators for hexagon packing.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    
    # Initialize evolutionary algorithm components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Register customized operators
    toolbox.register("individual", construct_spiral_initialization)
    toolbox.register("population", generate_initial_population)
    
    # Custom evaluation function
    def eval_individual(individual):
        # Calculate fitness using our specialized function
        return (fitness_function(individual),)
    
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", specialized_crossover)
    toolbox.register("mutate", specialized_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population = toolbox.population(n=25)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolve with custom parameters
    n_generations = 30
    for gen in range(n_generations):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.6:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.3:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population with offspring
        population[:] = offspring
    
    # Find the best individual
    best_individual = tools.selBest(population, 1)[0]
    
    # Final evaluation to get exact side length
    final_side_length = calculate_outer_hex_side_length(best_individual)
    
    # Refine with local search using a greedy approach
    try:
        # Try a few rounds of local optimization
        best_fitness = fitness_function(best_individual)
        
        for _ in range(10):
            # Create candidate improvements
            mutated_individual = [list(h) for h in best_individual]
            mutated_individual = specialized_mutation(mutated_individual, indpb=0.2)
            
            # Evaluate and accept if better
            new_fitness = fitness_function(mutated_individual)
            if new_fitness > best_fitness:
                best_individual = mutated_individual
                best_fitness = new_fitness
                final_side_length = calculate_outer_hex_side_length(best_individual)
                
    except Exception:
        # If local search fails, keep the best from GA
        pass
    
    # Final constraint verification
    containment_ok, overlap_ok, inv_radius = evaluate_constraints(best_individual, final_side_length)
    
    # If constraint violations exist, fallback to spiral configuration
    if not (containment_ok and overlap_ok):
        # Use the spiral initialization as fallback
        fallback_config = construct_spiral_initialization()
        fallback_side_length = calculate_outer_hex_side_length(fallback_config)
        fallback_containment_ok, fallback_overlap_ok, fallback_inv_radius = evaluate_constraints(fallback_config, fallback_side_length)
        
        if fallback_containment_ok and fallback_overlap_ok:
            best_individual = fallback_config
            final_side_length = fallback_side_length
        else:
            # Ultimate fallback to simple configuration
            best_individual = [
                [0, 0, 0],  # center
                [-2.5, 0, 0],  # left
                [2.5, 0, 0],  # right
                [-1.25, 2.17, 0],  # top-left
                [1.25, 2.17, 0],  # top-right
                [-1.25, -2.17, 0],  # bottom-left
                [1.25, -2.17, 0],  # bottom-right
                [-3.75, 2.17, 0],  # far top-left
                [3.75, 2.17, 0],  # far top-right
                [-3.75, -2.17, 0],  # far bottom-left
                [3.75, -2.17, 0],  # far bottom-right
            ]
            final_side_length = 8
    
    # Prepare return values
    inner_hex_data = np.array(best_individual)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = final_side_length
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END