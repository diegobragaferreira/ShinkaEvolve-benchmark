# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution
import math
from deap import base, creator, tools, algorithms
from joblib import Parallel, delayed
import time

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

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

def evaluate_single_hexagon(hex_data, outer_hexagon):
    """Evaluate a single hexagon for containment"""
    pos = (hex_data[0], hex_data[1])
    rot = hex_data[2]
    hexagon = create_regular_hexagon(pos, 1, rot)
    return check_containment(hexagon, outer_hexagon)

def evaluate_constraint_violations_parallel(inner_hex_data, outer_hexagon, n_jobs=-1):
    """Parallel evaluation of constraint violations"""
    # Check containment for all inner hexagons
    containment_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_hexagon)(hex_data, outer_hexagon) 
        for hex_data in inner_hex_data
    )
    
    containment_violations = sum(1 for result in containment_results if not result)
    
    # Check overlaps between all pairs of inner hexagons
    overlap_violations = 0
    n = len(inner_hex_data)
    
    # Create hexagons once for efficient overlap checking
    hexagons = [create_regular_hexagon((inner_hex_data[i][0], inner_hex_data[i][1]), 1, inner_hex_data[i][2]) 
                for i in range(n)]
    
    # Parallel overlap checking
    def check_pair_overlap(i, j):
        return check_overlap(hexagons[i], hexagons[j])
    
    overlap_results = Parallel(n_jobs=n_jobs)(
        delayed(check_pair_overlap)(i, j) 
        for i in range(n) for j in range(i+1, n)
    )
    
    overlap_violations = sum(1 for result in overlap_results if result)
    
    return containment_violations, overlap_violations

def evaluate_configuration(inner_hex_data, outer_center=(0, 0)):
    """Evaluate a configuration and return penalty and side length"""
    # Create inner hexagons
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        hexagon = create_regular_hexagon(pos, 1, rot)
        inner_hexagons.append(hexagon)

    # Create outer hexagon
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data, outer_center)
    outer_hexagon = create_regular_hexagon(outer_center, outer_side_length, 0)

    # Check containment and overlap constraints in parallel
    containment_violations, overlap_violations = evaluate_constraint_violations_parallel(
        inner_hex_data, outer_hexagon
    )

    # Calculate penalty based on violations
    penalty = 0
    if containment_violations > 0:
        penalty += 1000 * containment_violations
    if overlap_violations > 0:
        penalty += 1000 * overlap_violations

    return outer_side_length, penalty

def construct_voronoi_packing():
    """Construct an initial configuration using Voronoi-based clustering"""
    # Generate points in a Voronoi-like pattern
    base_positions = [
        (0, 0),        # center
        (2.5, 0),      # right
        (-2.5, 0),     # left
        (0, 2.5),      # top
        (0, -2.5),     # bottom
        (2.5, 2.5),    # top-right
        (-2.5, 2.5),   # top-left
        (2.5, -2.5),   # bottom-right
        (-2.5, -2.5),  # bottom-left
        (4.5, 0),      # far right
        (0, 4.5),      # far top
    ]
    
    # Add small random noise to spread out the points
    np.random.seed(42)
    config = []
    for pos in base_positions:
        noise_x = np.random.uniform(-0.5, 0.5)
        noise_y = np.random.uniform(-0.5, 0.5)
        config.append([pos[0] + noise_x, pos[1] + noise_y, np.random.uniform(0, 360)])
    
    return config

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization with Voronoi-based initialization and parallel constraint checking.
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
    
    # Define bounds for each parameter (x, y, rotation for each hexagon)
    # x: [-8, 8], y: [-8, 8], rotation: [0, 360]
    bounds = []
    for i in range(n):
        bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0.0, 360.0)])
    
    # Individual initialization function
    def init_individual():
        # Start with Voronoi-based configuration
        config = construct_voronoi_packing()
        # Add small random perturbations to improve diversity
        individual = []
        for i, hex_data in enumerate(config):
            individual.extend([
                hex_data[0] + np.random.uniform(-0.3, 0.3),
                hex_data[1] + np.random.uniform(-0.3, 0.3),
                hex_data[2] + np.random.uniform(-30, 30)
            ])
        return creator.Individual(individual)
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Evaluation function for the evolutionary algorithm
    def eval_individual(individual):
        # Convert individual to hexagon data format
        hex_data = []
        for i in range(n):
            start_idx = i * 3
            hex_data.append([
                individual[start_idx],
                individual[start_idx + 1],
                individual[start_idx + 2]
            ])
        
        # Evaluate the configuration
        side_length, penalty = evaluate_configuration(hex_data)
        
        # Objective: maximize 1/side_length while penalizing constraint violations
        if penalty > 0:
            # Return fitness that penalizes constraint violations heavily
            return (1.0 / (side_length + penalty)),  # fitness = 1/side_length - penalty
        else:
            # Return fitness maximizing 1/side_length
            return (1.0 / side_length,)
    
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population_size = 30
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolve
    n_generations = 20
    for gen in range(n_generations):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.2:  # Mutation probability
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
    
    # Convert best individual back to hexagon data format
    best_hex_data = []
    for i in range(n):
        start_idx = i * 3
        best_hex_data.append([
            best_individual[start_idx],
            best_individual[start_idx + 1],
            best_individual[start_idx + 2]
        ])
    
    # Final evaluation to get exact side length
    final_side_length, final_penalty = evaluate_configuration(best_hex_data)
    
    # If constraints are violated, fall back to the Voronoi configuration
    if final_penalty > 0:
        # Use Voronoi-based configuration as fallback
        fallback_config = construct_voronoi_packing()
        fallback_side_length, fallback_penalty = evaluate_configuration(fallback_config)
        
        if fallback_penalty == 0:
            best_hex_data = fallback_config
            final_side_length = fallback_side_length
        else:
            # Final fallback to simple configuration
            best_hex_data = [
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
    
    # Local optimization on the best solution
    try:
        # Define objective for local optimization
        def objective_local(params):
            # Reshape parameters
            local_hex_data = []
            for i in range(n):
                local_hex_data.append([params[3*i], params[3*i+1], params[3*i+2]])
            
            # Evaluate current configuration
            side_length, penalty = evaluate_configuration(local_hex_data)
            
            # Return (negative) inverse of side length (to minimize)
            if penalty > 0:
                return 1000 + penalty  # Penalty for constraint violations
            else:
                return -1.0 / side_length  # Want to maximize 1/side_length
        
        # Set up bounds
        bounds_local = []
        for i in range(n):
            # Position bounds
            bounds_local.extend([(-8, 8), (-8, 8)])
            # Rotation bounds
            bounds_local.append((0, 360))
        
        # Start from our best configuration
        initial_params = []
        for hex_data in best_hex_data:
            initial_params.extend(hex_data)
        
        # Use differential evolution for local refinement
        result = differential_evolution(
            objective_local,
            bounds_local,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        
        if result.success:
            # Extract refined parameters
            refined_params = result.x
            refined_config = []
            for i in range(n):
                refined_config.append([refined_params[3*i], refined_params[3*i+1], refined_params[3*i+2]])
            
            # Re-evaluate final refined configuration
            refined_side_length, refined_penalty = evaluate_configuration(refined_config)
            
            if refined_penalty == 0:
                best_hex_data = refined_config
                final_side_length = refined_side_length
                
    except Exception:
        # If local optimization fails, keep the best evolutionary solution
        pass
    
    # Prepare return values
    inner_hex_data = np.array(best_hex_data)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = final_side_length
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END