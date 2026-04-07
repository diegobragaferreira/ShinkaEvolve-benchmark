# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from collections import defaultdict

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * math.sqrt(3)/2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

def get_hexagon_vertices(center_x, center_y, angle_degrees):
    """Get vertices of a unit regular hexagon given center and rotation"""
    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)

    # Vertices of a unit hexagon centered at origin, pointing up
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi/3
        x = HEX_RADIUS * math.cos(theta)
        y = HEX_RADIUS * math.sin(theta)
        base_vertices.append((x, y))

    # Translate to center
    vertices = [(x + center_x, y + center_y) for x, y in base_vertices]
    return np.array(vertices)

def create_hexagon_polygon(center_x, center_y, angle_degrees):
    """Create Shapely polygon representation of a hexagon"""
    vertices = get_hexagon_vertices(center_x, center_y, angle_degrees)
    return Polygon(vertices)

def check_hexagon_containment(hex_polygon, outer_center_x, outer_center_y, outer_radius):
    """Check if hexagon is contained within outer hexagon using Shapely"""
    outer_polygon = create_hexagon_polygon(outer_center_x, outer_center_y, 0)
    
    # Check if all vertices are within the outer hexagon
    for vertex in hex_polygon.exterior.coords[:-1]:  # Exclude last duplicate point
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def hexagon_collision(hex1_polygon, hex2_polygon):
    """Check if two hexagons collide using Shapely"""
    return hex1_polygon.intersects(hex2_polygon)

def calculate_outer_hex_radius(inner_hex_data, outer_center=(0,0)):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_distance = 0

    for i in range(len(inner_hex_data)):
        center_x = inner_hex_data[i][0]
        center_y = inner_hex_data[i][1]
        angle = inner_hex_data[i][2]

        # Get vertices of this hexagon
        vertices = get_hexagon_vertices(center_x, center_y, angle)

        # Find maximum distance from outer center to any vertex
        for x, y in vertices:
            distance = math.sqrt((x - outer_center[0])**2 + (y - outer_center[1])**2)
            max_distance = max(max_distance, distance)

    # Add buffer to ensure complete containment
    return max_distance + HEX_RADIUS

def evaluate_fitness(individual):
    """
    Evaluate the fitness of a solution configuration
    individual: array of shape (33,) containing [x1,y1,a1,x2,y2,a2,...,x11,y11,a11]
    Returns negative value because we want to maximize 1/R (minimize R)
    """
    # Reshape individual into hexagon data
    inner_hex_data = individual.reshape(-1, 3)

    # Calculate outer hexagon size
    outer_radius = calculate_outer_hex_radius(inner_hex_data)

    # Create polygons for all hexagons
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center_x = inner_hex_data[i][0]
        center_y = inner_hex_data[i][1]
        angle = inner_hex_data[i][2]
        hex_polygons.append(create_hexagon_polygon(center_x, center_y, angle))

    # Check collisions and containment
    num_collisions = 0
    num_out_of_bounds = 0

    # Check containment for each hexagon
    for i, hex_poly in enumerate(hex_polygons):
        if not check_hexagon_containment(hex_poly, 0, 0, outer_radius):
            num_out_of_bounds += 1

    # Check all hexagon pairs for collision
    for i in range(len(hex_polygons)):
        for j in range(i+1, len(hex_polygons)):
            if hexagon_collision(hex_polygons[i], hex_polygons[j]):
                num_collisions += 1

    # Penalty for collisions or out of bounds
    penalty = 1000 * (num_collisions + num_out_of_bounds)

    # If invalid configuration, return poor fitness
    if num_collisions > 0 or num_out_of_bounds > 0:
        return 1000000 + penalty  # Large penalty for invalid solutions

    # Return inverse of outer radius (we want to maximize 1/R)
    return 1.0 / outer_radius

def initialize_population(pop_size, n_hexagons=11):
    """Initialize a diverse population with various geometric arrangements"""
    population = []
    
    # Generate different starting configurations
    for _ in range(pop_size):
        # Create a more strategic initial layout
        individual = np.zeros((n_hexagons, 3))
        
        # Center hexagon
        individual[0] = [0, 0, random.uniform(0, 360)]
        
        # Surrounding hexagons in a pattern
        angles = [0, 60, 120, 180, 240, 300]  # Hexagonal directions
        positions = [(-2.0, 0), (2.0, 0), (0, 2.0), (0, -2.0), (-1.0, 1.0), (1.0, 1.0),
                     (-1.0, -1.0), (1.0, -1.0), (-2.0, 1.0), (2.0, 1.0), (-2.0, -1.0), (2.0, -1.0)]
        
        # Select 10 positions from our list
        selected_positions = random.sample(positions, 10)
        
        # Fill remaining positions
        for i in range(1, 11):
            if i <= len(selected_positions):
                individual[i][0] = selected_positions[i-1][0] + random.uniform(-0.5, 0.5)
                individual[i][1] = selected_positions[i-1][1] + random.uniform(-0.5, 0.5)
                individual[i][2] = random.uniform(0, 360)
            else:
                # Random positions for remaining slots
                individual[i][0] = random.uniform(-4, 4)
                individual[i][1] = random.uniform(-4, 4)
                individual[i][2] = random.uniform(0, 360)
        
        population.append(individual.flatten())
    
    return population

def mutate_individual(individual, mutation_rate=0.1, max_mutation=0.5):
    """Apply mutation to an individual"""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Determine which component to mutate
            if i % 3 == 0:  # x coordinate
                mutated[i] += random.uniform(-max_mutation, max_mutation)
            elif i % 3 == 1:  # y coordinate
                mutated[i] += random.uniform(-max_mutation, max_mutation)
            else:  # angle
                mutated[i] += random.uniform(-30, 30)
                # Ensure angle stays in [0, 360)
                mutated[i] = mutated[i] % 360
    
    return mutated

def crossover_parents(parent1, parent2, crossover_rate=0.8):
    """Crossover two parents to create offspring"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    # Single point crossover
    crossover_point = random.randint(1, len(parent1)-1)
    
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    
    return child1, child2

def evolutionary_search():
    """Perform evolutionary search for optimal hexagon packing"""
    n_generations = 50
    population_size = 20
    elite_size = 4
    
    # Initialize population
    population = initialize_population(population_size)
    best_fitness_history = []
    
    for generation in range(n_generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)
        
        # Track best fitness
        best_fitness = min(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitness_scores)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = sorted_population[:elite_size]
        
        # Create new population
        new_population = elite[:]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.randint(0, population_size//2)
            parent2_idx = random.randint(0, population_size//2)
            
            parent1 = sorted_population[parent1_idx]
            parent2 = sorted_population[parent2_idx]
            
            child1, child2 = crossover_parents(parent1, parent2)
            
            # Apply mutation
            child1 = mutate_individual(child1, mutation_rate=0.15, max_mutation=0.3)
            child2 = mutate_individual(child2, mutation_rate=0.15, max_mutation=0.3)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
        
        # Adaptive mutation rate based on convergence
        if len(best_fitness_history) > 5:
            recent_changes = abs(best_fitness_history[-1] - best_fitness_history[-5])
            if recent_changes < 1e-8:
                # Slowdown convergence, increase mutation
                pass  # Already handled above
    
    # Return best solution
    final_fitness_scores = [evaluate_fitness(ind) for ind in population]
    best_idx = np.argmin(final_fitness_scores)
    return population[best_idx]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    np.random.seed(42)  # For reproducibility
    
    # Try evolutionary approach first
    try:
        best_individual = evolutionary_search()
        inner_hex_data = best_individual.reshape(-1, 3)
        outer_radius = calculate_outer_hex_radius(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_radius
    except Exception as e:
        pass
    
    # Fallback to current optimized approach
    initial_guess = np.array([
        [0, 0, 0],      # center
        [-2.2, 0, 0],   # left
        [2.2, 0, 0],    # right
        [-1.1, 1.9, 0], # top-left
        [1.1, 1.9, 0],  # top-right
        [-1.1, -1.9, 0], # bottom-left
        [1.1, -1.9, 0], # bottom-right
        [-3.3, 1.9, 0], # far top-left
        [3.3, 1.9, 0],  # far top-right
        [-3.3, -1.9, 0], # far bottom-left
        [3.3, -1.9, 0], # far bottom-right
    ]).flatten()

    # Bounds for optimization: positions (-10, 10), rotations (0, 360)
    bounds = []
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle

    # Run optimization with bounds
    try:
        result = differential_evolution(
            func=evaluate_fitness,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            seed=42,
            disp=False,
            tol=1e-6
        )

        if result.success:
            # Get the best solution
            best_individual = result.x
            inner_hex_data = best_individual.reshape(-1, 3)

            # Calculate final outer hexagon radius
            outer_radius = calculate_outer_hex_radius(inner_hex_data)

            # Create outer hexagon data (centered at origin)
            outer_hex_data = np.array([0, 0, 0])

            # Return the best solution found
            return inner_hex_data, outer_hex_data, outer_radius
        else:
            # Fallback to initial guess if optimization fails
            pass
    except Exception:
        # If optimization fails, fall back to initial guess
        pass

    # Fallback to initial configuration if anything goes wrong
    inner_hex_data = np.array([
        [0, 0, 0],      # center
        [-2.2, 0, 0],   # left
        [2.2, 0, 0],    # right
        [-1.1, 1.9, 0], # top-left
        [1.1, 1.9, 0],  # top-right
        [-1.1, -1.9, 0], # bottom-left
        [1.1, -1.9, 0], # bottom-right
        [-3.3, 1.9, 0], # far top-left
        [3.3, 1.9, 0],  # far top-right
        [-3.3, -1.9, 0], # far bottom-left
        [3.3, -1.9, 0], # far bottom-right
    ])

    outer_radius = calculate_outer_hex_radius(inner_hex_data)
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END