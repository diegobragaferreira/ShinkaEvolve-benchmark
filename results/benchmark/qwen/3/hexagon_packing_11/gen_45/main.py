# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from deap import base, creator, tools, algorithms
import time
from math import sqrt, cos, sin, pi

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def create_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Create vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = angle_deg * pi / 180
    vertices = []
    for i in range(6):
        theta = angle_rad + i * pi / 3
        x = center_x + side_length * cos(theta)
        y = center_y + side_length * sin(theta)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vx, vy in hex_vertices:
        if not outer_polygon.contains(Point(vx, vy)):
            return False
    return True

def check_hexagon_collision(hex1_vertices, hex2_vertices):
    """Check if two hexagons intersect."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_radius(inner_hex_data, outer_center=(0, 0), outer_angle=0):
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    # Create outer hexagon vertices based on current estimate
    outer_side_length = 10.0  # Initial guess
    
    # Binary search for minimum outer hexagon side length
    def check_containment(side_length):
        outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], outer_angle, side_length)
        outer_polygon = Polygon(outer_vertices)
        
        # Check if all inner hexagon vertices are inside outer hexagon
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle_deg = inner_hex_data[i]
            inner_vertices = create_hexagon_vertices(center_x, center_y, angle_deg, 1.0)
            
            # Check if vertices of inner hex are within outer hexagon
            for vx, vy in inner_vertices:
                if not outer_polygon.contains(Point(vx, vy)):
                    return False
        return True
    
    # Binary search for minimal radius
    low, high = 0.1, 20.0
    while high - low > 1e-6:
        mid = (low + high) / 2
        if check_containment(mid):
            high = mid
        else:
            low = mid
            
    # Final check with refined value
    outer_side_length = high
    
    return outer_side_length

def evaluate_individual(individual):
    """Evaluate the fitness of an individual - returns negative inv_outer_hex_side_length"""
    # Decode individual into hexagon data
    hex_data = []
    idx = 0
    for _ in range(11):  # 11 hexagons
        x = individual[idx]
        y = individual[idx+1]
        angle = individual[idx+2]
        hex_data.append((x, y, angle))
        idx += 3
    
    # Check for collisions between all pairs
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            hex1 = create_hexagon_vertices(hex_data[i][0], hex_data[i][1], hex_data[i][2])
            hex2 = create_hexagon_vertices(hex_data[j][0], hex_data[j][1], hex_data[j][2])
            if check_hexagon_collision(hex1, hex2):
                return (1e6,)  # Penalty for collisions
    
    # Compute outer hexagon radius needed to contain all inner hexes
    outer_side_length = compute_outer_hexagon_radius(np.array(hex_data))
    
    # Return negative value since we want to maximize 1/outer_side_length
    # So we minimize -1/outer_side_length which equals -1/outer_side_length
    if outer_side_length <= 0:
        return (1e6,)
    return (-1.0 / outer_side_length,)

def generate_initial_config():
    """Generate a diverse set of initial configurations."""
    configs = []
    
    # Configuration 1: Grid pattern
    grid_config = [
        (0, 0, 0),
        (-2.5, 0, 0),
        (2.5, 0, 0),
        (-1.25, 2.17, 0),
        (1.25, 2.17, 0),
        (-1.25, -2.17, 0),
        (1.25, -2.17, 0),
        (-3.75, 2.17, 0),
        (3.75, 2.17, 0),
        (-3.75, -2.17, 0),
        (3.75, -2.17, 0),
    ]
    
    # Configuration 2: Spiral pattern
    spiral_config = [
        (0, 0, 0),
        (0, 1.8, 0),
        (0, -1.8, 0),
        (1.5, 0.87, 0),
        (-1.5, 0.87, 0),
        (1.5, -0.87, 0),
        (-1.5, -0.87, 0),
        (3.0, 1.74, 0),
        (-3.0, 1.74, 0),
        (3.0, -1.74, 0),
        (-3.0, -1.74, 0),
    ]
    
    # Configuration 3: Compact cluster
    cluster_config = [
        (0, 0, 0),
        (0, 1.2, 0),
        (0, -1.2, 0),
        (1.2, 0, 0),
        (-1.2, 0, 0),
        (0.6, 1.04, 0),
        (-0.6, 1.04, 0),
        (0.6, -1.04, 0),
        (-0.6, -1.04, 0),
        (1.8, 0.87, 0),
        (-1.8, 0.87, 0),
    ]
    
    configs.extend([grid_config, spiral_config, cluster_config])
    
    # Generate random configurations
    for _ in range(10):
        config = []
        for _ in range(11):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            angle = np.random.uniform(0, 360)
            config.append((x, y, angle))
        configs.append(config)
    
    return configs

def evolutionary_search():
    """Run evolutionary algorithm to optimize hexagon packing."""
    
    # Create toolbox
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Define bounds for parameters
    # Each individual has 33 parameters (11 hexagons × 3 parameters each)
    # For positions: x,y ∈ [-10, 10], angles ∈ [0, 360]
    def generate_individual():
        individual = []
        for _ in range(11):
            individual.extend([
                random.uniform(-5, 5),   # x coordinate
                random.uniform(-5, 5),   # y coordinate
                random.uniform(0, 360)   # angle in degrees
            ])
        return creator.Individual(individual)
    
    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register functions
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Parameters
    pop_size = 50
    n_generations = 100
    crossover_prob = 0.8
    mutate_prob = 0.1
    
    # Generate initial population
    initial_configs = generate_initial_config()
    population = []
    
    for config in initial_configs:
        individual = []
        for x, y, angle in config:
            individual.extend([x, y, angle])
        population.append(creator.Individual(individual))
    
    # Fill remaining population with random individuals
    while len(population) < pop_size:
        population.append(toolbox.individual())
    
    # Evolution loop
    for gen in range(n_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutate_prob)
        
        # Evaluate fitness of offspring
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # Select next generation
        population = toolbox.select(offspring, k=len(population))
        
        # Print progress
        if gen % 20 == 0:
            best_fit = min([ind.fitness.values[0] for ind in population])
            print(f"Generation {gen}: Best fitness = {-best_fit}")
    
    # Get best solution
    best_ind = tools.selBest(population, 1)[0]
    
    # Convert back to hexagon data format
    hex_data = []
    for i in range(11):
        start_idx = i * 3
        x = best_ind[start_idx]
        y = best_ind[start_idx+1]
        angle = best_ind[start_idx+2]
        hex_data.append((x, y, angle))
    
    return np.array(hex_data)

def local_optimization_step(hex_data):
    """Perform local optimization to refine solution."""
    
    def objective(params):
        # Reshape params back to hex_data format
        new_hex_data = []
        for i in range(11):
            start_idx = i * 3
            x = params[start_idx]
            y = params[start_idx+1]
            angle = params[start_idx+2]
            new_hex_data.append((x, y, angle))
        
        # Check collisions
        for i in range(len(new_hex_data)):
            for j in range(i+1, len(new_hex_data)):
                hex1 = create_hexagon_vertices(new_hex_data[i][0], new_hex_data[i][1], new_hex_data[i][2])
                hex2 = create_hexagon_vertices(new_hex_data[j][0], new_hex_data[j][1], new_hex_data[j][2])
                if check_hexagon_collision(hex1, hex2):
                    return 1e6  # Penalty for collisions
        
        # Compute outer radius
        outer_radius = compute_outer_hexagon_radius(np.array(new_hex_data))
        if outer_radius <= 0:
            return 1e6
        return -1.0 / outer_radius  # Negative because we want to maximize 1/r
    
    # Flatten initial hex_data for optimization
    initial_params = []
    for x, y, angle in hex_data:
        initial_params.extend([x, y, angle])
    
    # Optimize using L-BFGS-B
    result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 500})
    
    # Extract optimized hex_data
    optimized_hex_data = []
    for i in range(11):
        start_idx = i * 3
        x = result.x[start_idx]
        y = result.x[start_idx+1]
        angle = result.x[start_idx+2]
        optimized_hex_data.append((x, y, angle))
    
    return np.array(optimized_hex_data)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run evolutionary search
    hex_data = evolutionary_search()
    
    # Apply local optimization
    final_hex_data = local_optimization_step(hex_data)
    
    # Compute outer hexagon radius
    outer_side_length = compute_outer_hexagon_radius(final_hex_data)
    
    # Return results
    inner_hex_data = final_hex_data
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
