# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from deap import base, creator, tools, algorithms
import time

def compute_outer_hexagon_radius(inner_hex_data, outer_center=(0, 0)):
    """Compute minimum radius needed to contain all inner hexagons"""
    # Define a unit hexagon centered at origin
    hex_points = []
    for i in range(6):
        angle = i * np.pi / 3
        x = np.cos(angle)
        y = np.sin(angle)
        hex_points.append((x, y))
    
    # Create polygons for each inner hexagon
    hex_polygons = []
    for x, y, angle in inner_hex_data:
        # Rotate and translate hexagon points
        rotated_points = []
        for px, py in hex_points:
            # Apply rotation
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            # Apply translation
            rotated_points.append((rx + x, ry + y))
        
        # Create polygon
        hex_polygon = Polygon(rotated_points)
        hex_polygons.append(hex_polygon)
    
    # Union all polygons to get the total bounding area
    union_polygon = unary_union(hex_polygons)
    
    # Find the minimum distance from center to any point on boundary
    center_point = Point(outer_center)
    min_distance = float('inf')
    
    # Get boundary points
    if hasattr(union_polygon, 'exterior'):
        boundary_coords = list(union_polygon.exterior.coords)
    else:
        boundary_coords = list(union_polygon.coords)
    
    for coord in boundary_coords:
        distance = np.sqrt((coord[0] - outer_center[0])**2 + (coord[1] - outer_center[1])**2)
        min_distance = min(min_distance, distance)
    
    # Account for hexagon width
    return min_distance * 1.01  # Add small buffer for numerical stability

def check_overlap_hexagons(hex1, hex2):
    """Check if two hexagons overlap using Shapely"""
    # Define hexagon points for first hexagon
    hex_points1 = []
    for i in range(6):
        angle = i * np.pi / 3 + np.radians(hex1[2])
        x = np.cos(angle) + hex1[0]
        y = np.sin(angle) + hex1[1]
        hex_points1.append((x, y))
    
    # Define hexagon points for second hexagon
    hex_points2 = []
    for i in range(6):
        angle = i * np.pi / 3 + np.radians(hex2[2])
        x = np.cos(angle) + hex2[0]
        y = np.sin(angle) + hex2[1]
        hex_points2.append((x, y))
    
    # Create polygons
    poly1 = Polygon(hex_points1)
    poly2 = Polygon(hex_points2)
    
    # Check overlap
    return poly1.intersects(poly2)

def check_containment(hex_data, outer_radius):
    """Check if all hexagons are contained in outer hexagon"""
    # Define outer hexagon points
    outer_points = []
    for i in range(6):
        angle = i * np.pi / 3
        x = outer_radius * np.cos(angle)
        y = outer_radius * np.sin(angle)
        outer_points.append((x, y))
    
    outer_polygon = Polygon(outer_points)
    
    # Check each inner hexagon
    for x, y, angle in hex_data:
        # Define inner hexagon points
        hex_points = []
        for i in range(6):
            angle_i = i * np.pi / 3 + np.radians(angle)
            x_i = np.cos(angle_i) + x
            y_i = np.sin(angle_i) + y
            hex_points.append((x_i, y_i))
        
        inner_polygon = Polygon(hex_points)
        
        # Check if inner polygon is contained in outer polygon
        if not outer_polygon.contains(inner_polygon):
            return False
    
    return True

def evaluate_hexagon_packing(individual):
    """Evaluate fitness of hexagon packing"""
    # Decode individual into hexagon positions and rotations
    hex_positions = individual[:22].reshape(-1, 2)
    hex_rotations = individual[22:]
    
    # Combine into hex_data format
    hex_data = np.column_stack([hex_positions, hex_rotations])
    
    # Compute outer radius
    try:
        outer_radius = compute_outer_hexagon_radius(hex_data)
        # Check constraints
        if not check_containment(hex_data, outer_radius):
            return (1e6,)  # Penalty for containment violation
        
        # Check overlaps
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                if check_overlap_hexagons(hex_data[i], hex_data[j]):
                    return (1e6,)  # Penalty for overlap
        
        # Return inversed radius (we want to minimize radius, so maximize inverse)
        return (1.0 / outer_radius,)
    except Exception:
        return (1e6,)

def create_individual():
    """Create a random individual"""
    # Positions (11 hexagons, 2 coordinates each)
    positions = np.random.uniform(low=-4.0, high=4.0, size=22)
    # Rotations (11 hexagons, single angle each)
    rotations = np.random.uniform(low=0.0, high=360.0, size=11)
    return np.concatenate([positions, rotations])

def main_optimization():
    """Main optimization routine"""
    # Set up evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_hexagon_packing)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population
    population = toolbox.population(n=50)
    
    # Run evolution
    best_individual = None
    best_fitness = -np.inf
    
    start_time = time.time()
    max_time = 150  # seconds
    
    for generation in range(100):  # Limit generations
        if time.time() - start_time > max_time:
            break
            
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Track best individual
        current_best = max(population, key=lambda ind: ind.fitness.values[0])
        if current_best.fitness.values[0] > best_fitness:
            best_fitness = current_best.fitness.values[0]
            best_individual = current_best.copy()
        
        # Crossover and mutation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        population[:] = offspring
    
    # Extract final result
    if best_individual is None:
        # Fallback to heuristic solution
        hex_positions = np.array([
            [0, 0],
            [-1.5, 0],
            [1.5, 0],
            [0, 2.6],
            [0, -2.6],
            [-2.6, 1.3],
            [2.6, 1.3],
            [-2.6, -1.3],
            [2.6, -1.3],
            [-3.9, 2.6],
            [3.9, 2.6]
        ])
        hex_rotations = np.zeros(11)
        best_individual = np.concatenate([hex_positions.flatten(), hex_rotations])
    
    # Decode the best individual
    hex_positions = best_individual[:22].reshape(-1, 2)
    hex_rotations = best_individual[22:]
    hex_data = np.column_stack([hex_positions, hex_rotations])
    
    # Calculate final outer radius
    outer_radius = compute_outer_hexagon_radius(hex_data)
    
    return hex_data, np.array([0, 0, 0]), outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = main_optimization()
    
    # Adjust to proper format if needed
    if len(inner_hex_data.shape) == 1:
        inner_hex_data = inner_hex_data.reshape(-1, 3)
    
    # If we're dealing with positions instead of the expected format, adjust
    if inner_hex_data.shape[1] != 3:
        # Assume we have just positions, add zero rotations
        positions = inner_hex_data
        rotations = np.zeros(positions.shape[0])
        inner_hex_data = np.column_stack([positions, rotations])
    
    # Ensure we return exactly 11 hexagons
    if len(inner_hex_data) < 11:
        # Extend with default values
        extra = 11 - len(inner_hex_data)
        defaults = np.array([[0, 0, 0]] * extra)
        inner_hex_data = np.vstack([inner_hex_data, defaults])
    elif len(inner_hex_data) > 11:
        # Trim to 11
        inner_hex_data = inner_hex_data[:11]
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
