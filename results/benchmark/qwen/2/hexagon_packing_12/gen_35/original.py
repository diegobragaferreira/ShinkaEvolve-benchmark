# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time

def generate_hexagon_vertices(center_x, center_y, side_length=1, rotation_deg=0):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    angle_step = np.pi / 3
    rotation_rad = np.radians(rotation_deg)
    vertices = []
    for i in range(6):
        angle = rotation_rad + i * angle_step
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return vertices

def create_hexagon_polygon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a shapely polygon for a hexagon."""
    vertices = generate_hexagon_vertices(center_x, center_y, side_length, rotation_deg)
    return Polygon(vertices)

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon using vertex containment."""
    vertices = list(hexagon_poly.exterior.coords)
    for point in vertices:
        if not outer_hex_poly.contains(Point(point[0], point[1])):
            return False
    return True

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap using shapely intersection."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def calculate_outer_hex_side_length(inner_hex_data, buffer=0.01):
    """Calculate minimum side length of outer hexagon that contains all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, 1, rotation)
        all_vertices.extend(vertices)
    
    # Find bounding box
    if not all_vertices:
        return 1
        
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Calculate distance from center to farthest vertex
    max_dist = 0
    center_x = sum(xs) / len(xs)
    center_y = sum(ys) / len(ys)
    
    for x, y in all_vertices:
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add buffer for numerical stability
    return max_dist + buffer

def evaluate_fitness(individual, outer_hex_side_length):
    """Evaluate fitness of individual configuration."""
    # Convert individual to hex data format
    hex_data = individual.reshape(-1, 3)
    
    # Create all inner hexagon polygons
    inner_hexagons = []
    for i in range(len(hex_data)):
        cx, cy, rot = hex_data[i]
        hex_poly = create_hexagon_polygon(cx, cy, 1, rot)
        inner_hexagons.append(hex_poly)
    
    # Check containment and overlap
    outer_hex = create_hexagon_polygon(0, 0, outer_hex_side_length, 0)
    
    # Count violations
    overlap_count = 0
    containment_count = 0
    
    # Check overlaps between all pairs
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                overlap_count += 1
    
    # Check containment
    for hex_poly in inner_hexagons:
        if not check_containment(hex_poly, outer_hex):
            containment_count += 1
    
    # Penalty for constraint violations
    penalty = overlap_count * 1000 + containment_count * 1000
    
    # Fitness is inverse of outer hex side length minus penalties
    if overlap_count > 0 or containment_count > 0:
        return -penalty  # Very bad fitness if constraints violated
    else:
        return 1.0 / outer_hex_side_length

def mutate_individual(individual, mutation_rate=0.1):
    """Mutate an individual with small changes."""
    mutated = individual.copy()
    
    # Randomly choose which elements to mutate
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            if i % 3 == 0:  # x coordinate
                mutated[i] += random.uniform(-0.5, 0.5)
            elif i % 3 == 1:  # y coordinate
                mutated[i] += random.uniform(-0.5, 0.5)
            else:  # rotation
                mutated[i] = (mutated[i] + random.uniform(-30, 30)) % 360
    
    return mutated

def crossover_individuals(parent1, parent2):
    """Perform uniform crossover between two individuals."""
    child = parent1.copy()
    for i in range(len(child)):
        if random.random() < 0.5:
            child[i] = parent2[i]
    return child

def initialize_population(pop_size, num_hexagons=12):
    """Initialize population with random valid configurations."""
    population = []
    
    # Start with a reasonable heuristic initialization
    # Place some hexagons in a circular pattern with some randomness
    base_positions = []
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 positions around a circle + center
    radii = [0, 1.5, 2.5, 3.5, 2.0, 3.0, 4.0]  # Different distances
    
    for i in range(num_hexagons):
        angle = angles[i % len(angles)]
        radius = radii[i % len(radii)]
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        rotation = random.uniform(0, 360)
        
        base_positions.append([x, y, rotation])
    
    for _ in range(pop_size):
        individual = np.array(base_positions).flatten() + np.random.normal(0, 0.2, 36)
        # Ensure some bounds
        individual[0::3] = np.clip(individual[0::3], -10, 10)  # x coords
        individual[1::3] = np.clip(individual[1::3], -10, 10)  # y coords
        population.append(individual)
    
    return population

def optimize_hexagon_packing():
    """Evolutionary optimization for hexagon packing."""
    start_time = time.time()
    
    pop_size = 100
    num_generations = 500
    mutation_rate = 0.2
    elite_size = 10
    
    # Initialize population
    population = initialize_population(pop_size)
    
    best_fitness = -float('inf')
    best_individual = None
    best_outer_side_length = float('inf')
    
    # Evolution loop
    for generation in range(num_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            # Determine outer hexagon size based on current configuration
            hex_data = individual.reshape(-1, 3)
            outer_side_length = calculate_outer_hex_side_length(hex_data)
            
            # Evaluate fitness with the computed outer size
            fitness = evaluate_fitness(individual, outer_side_length)
            fitness_scores.append(fitness)
            
            # Update best solution
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                best_outer_side_length = outer_side_length
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Elitism: keep top individuals
        elites = population[:elite_size]
        
        # Generate new population
        new_population = elites[:]
        
        # Create offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parent1_idx = random.randint(0, pop_size//4)
            parent2_idx = random.randint(0, pop_size//4)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = crossover_individuals(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
        
        # Adaptive mutation rate
        if generation > 100:
            mutation_rate = max(0.01, mutation_rate * 0.995)
        
        # Early stopping and progress tracking
        if time.time() - start_time > 170:  # Leave 10 seconds for cleanup
            break
            
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, "
                  f"Outer side length = {best_outer_side_length:.6f}")
    
    # Final validation and refinement
    if best_individual is not None:
        hex_data = best_individual.reshape(-1, 3)
        final_outer_side_length = calculate_outer_hex_side_length(hex_data)
        # Re-evaluate final fitness with exact outer size
        final_fitness = evaluate_fitness(best_individual, final_outer_side_length)
    else:
        # Fall back to basic configuration if nothing better found
        hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        final_outer_side_length = 8.0
        final_fitness = 0.125  # Conservative estimate
    
    return hex_data, np.array([0, 0, 0]), final_outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
