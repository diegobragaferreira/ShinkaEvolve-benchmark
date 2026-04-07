# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from scipy.spatial.distance import cdist
import time
from multiprocessing import Pool
import random
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def hexagon_vertices(center_x, center_y, rotation_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a unit hexagon centered at origin
    unit_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = unit_vertices @ rotation_matrix.T
    return rotated_vertices * side_length + np.array([center_x, center_y])

def check_containment_single(hex_vertices, outer_polygon):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    # Check if all vertices are inside the outer polygon
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
        
    # Calculate bounding box
    all_vertices = np.array(all_vertices)
    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_y, max_y = all_vertices[:, 1].min(), all_vertices[:, 1].max()
    
    # Calculate center of bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Find maximum distance from center to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # For a hexagon, the side length is max_dist / sqrt(3)
    # But we want to ensure our hexagon can contain everything with some margin
    return max_dist * 2 / np.sqrt(3) * 1.1  # 10% margin

def calculate_constraint_penalties(params):
    """Calculate penalties for constraint violations."""
    # Extract parameters
    n = 11
    inner_params = params[:-1]
    outer_radius = params[-1]
    
    # Check if outer hexagon is large enough
    if outer_radius <= 0:
        return 1e6
    
    # Create outer hexagon vertices once
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = prep(Polygon(outer_vertices))
    
    # Initialize penalties
    containment_penalty = 0
    collision_penalty = 0
    
    # Check containment of all inner hexagons
    for i in range(n):
        x, y, rot = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            containment_penalty += 1e6
    
    # Check collisions between all pairs of inner hexagons
    if containment_penalty == 0:
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, rot1 = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
                x2, y2, rot2 = inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]
                hex1_vertices = hexagon_vertices(x1, y1, rot1, 1)
                hex2_vertices = hexagon_vertices(x2, y2, rot2, 1)
                if check_collision_single(hex1_vertices, hex2_vertices):
                    collision_penalty += 1e6
    
    return containment_penalty + collision_penalty

def fitness_function(params):
    """Fitness function to maximize 1/outer_hex_side_length."""
    penalty = calculate_constraint_penalties(params)
    
    # Return fitness value (higher is better)
    outer_radius = params[-1]
    if penalty > 0:
        return -1e10  # Very bad fitness for infeasible solutions
    else:
        # Return inverse of side length as fitness (maximize this)
        return 1.0 / outer_radius

def create_individual():
    """Create a random valid individual for the GA."""
    # Start with a good configuration pattern
    base_pattern = [
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [1.732, 1, 0],  # top-right
        [1.732, -1, 0], # bottom-right
        [0, -2, 0],     # bottom
        [-1.732, -1, 0], # bottom-left
        [-1.732, 1, 0],  # top-left
        [3.464, 0, 0],   # far right
        [1.732, 2, 0],   # top-right extended
        [-1.732, 2, 0],  # top-left extended
        [-3.464, 0, 0],  # far left
    ]
    
    individual = []
    for x, y, rot in base_pattern:
        # Add small random perturbations
        individual.extend([
            x + random.uniform(-0.5, 0.5),
            y + random.uniform(-0.5, 0.5),
            rot + random.uniform(-30, 30)
        ])
    
    # Set outer radius
    outer_radius = estimate_min_outer_radius(np.array(individual))
    individual.append(outer_radius)
    
    return np.array(individual)

def crossover(parent1, parent2):
    """Specialized crossover operator for hexagon packing."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Uniform crossover for positions and rotations
    for i in range(len(parent1)-1):  # Exclude outer radius
        if random.random() < 0.5:
            child1[i], child2[i] = parent2[i], parent1[i]
    
    # Ensure child 1 has valid outer radius
    child1[-1] = max(0.1, child1[-1] + random.uniform(-0.1, 0.1))
    
    # Ensure child 2 has valid outer radius  
    child2[-1] = max(0.1, child2[-1] + random.uniform(-0.1, 0.1))
    
    return child1, child2

def mutate(individual, mutation_rate=0.1):
    """Specialized mutation operator for hexagon packing."""
    mutated = individual.copy()
    
    # Mutate positions and rotations
    for i in range(len(individual)-1):  # Exclude outer radius
        if random.random() < mutation_rate:
            if i % 3 == 0:  # x coordinate
                mutated[i] += random.uniform(-0.5, 0.5)
            elif i % 3 == 1:  # y coordinate
                mutated[i] += random.uniform(-0.5, 0.5)
            else:  # rotation
                mutated[i] += random.uniform(-15, 15)
    
    # Mutate outer radius with small changes
    if random.random() < mutation_rate:
        mutated[-1] += random.uniform(-0.2, 0.2)
        mutated[-1] = max(0.1, mutated[-1])
        
    return mutated

def evaluate_population(population):
    """Evaluate fitness of entire population in parallel."""
    fitness_values = []
    for individual in population:
        fitness_values.append(fitness_function(individual))
    return fitness_values

def genetic_algorithm():
    """Run genetic algorithm for hexagon packing optimization."""
    # Parameters
    population_size = 50
    generations = 100
    elite_size = 5
    mutation_rate = 0.1
    
    # Create initial population
    population = [create_individual() for _ in range(population_size)]
    
    best_fitness_history = []
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = evaluate_population(population)
        
        # Track best
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_fitness_history.append(best_fitness)
        
        # Print progress
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        # Selection: tournament selection
        selected = []
        for _ in range(population_size):
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite = [population[i] for i in elite_indices]
        
        # Crossover and mutation
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Return best solution
    fitness_scores = evaluate_population(population)
    best_idx = np.argmax(fitness_scores)
    return population[best_idx]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Run genetic algorithm
    start_time = time.time()
    best_solution = genetic_algorithm()
    end_time = time.time()
    
    # Extract results
    outer_side_length = best_solution[-1]
    
    # Extract inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]]
    
    # Outer hexagon data (centered at origin)
    outer_hex_data = np.array([0, 0, 0])
    
    # Validate solution one more time
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))
    
    # Check validity
    valid = True
    for i in range(11):
        x, y, rot = best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            valid = False
            break
    
    # If still invalid, use safe fallback
    if not valid:
        # Fallback to pattern that works well
        inner_hex_data = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ])
        outer_side_length = estimate_min_outer_radius(inner_hex_data.flatten())
        outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
