# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
import itertools
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def build_spatial_grid(hex_polygons, cell_size=3.0):
    """Build spatial grid for fast collision detection"""
    grid = defaultdict(list)
    for i, hex_poly in enumerate(hex_polygons):
        bbox = hex_poly.bounds
        min_x, min_y, max_x, max_y = bbox
        for x in range(int(min_x//cell_size), int(max_x//cell_size)+1):
            for y in range(int(min_y//cell_size), int(max_y//cell_size)+1):
                grid[(x,y)].append(i)
    return grid

def get_collision_candidates(grid, hex_index, hex_poly, cell_size=3.0):
    """Get collision candidates efficiently"""
    candidates = []
    bbox = hex_poly.bounds
    min_x, min_y, max_x, max_y = bbox
    for x in range(int(min_x//cell_size)-1, int(max_x//cell_size)+2):
        for y in range(int(min_y//cell_size)-1, int(max_y//cell_size)+2):
            candidates.extend(grid.get((x,y), []))
    return [i for i in candidates if i != hex_index]

def check_containment_and_overlap(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Efficiently check containment and overlaps using spatial indexing"""
    # Create all hexagon polygons
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))
    
    # Check containment first
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment for all hexagons
    for i, hex_poly in enumerate(hex_polygons):
        # Check if any vertex is outside the outer hexagon
        for vertex in hex_poly.exterior.coords[:-1]:
            if not point_in_polygon(vertex, outer_polygon):
                return False, True, None  # containment violated, overlap not checked yet
    
    # Check overlaps using spatial indexing for efficiency
    grid = build_spatial_grid(hex_polygons, cell_size=3.0)
    
    # Check for overlaps
    for i in range(len(hex_polygons)):
        candidates = get_collision_candidates(grid, i, hex_polygons[i], cell_size=3.0)
        for j in candidates:
            if i < j and hex_polygons[i].intersects(hex_polygons[j]):
                return True, False, None  # containment ok, overlap exists
    
    return True, True, hex_polygons  # both ok

def generate_initial_population(size=30):
    """Generate diverse initial population using geometric knowledge"""
    population = []
    
    # Pattern 1: Hexagonal lattice arrangement
    for _ in range(size//6):
        config = []
        # Center hexagon
        config.append([0, 0, 0])
        # First ring (6 hexagons)
        for i in range(6):
            angle = i * np.pi/3
            config.append([2 * np.cos(angle), 2 * np.sin(angle), 0])
        # Second ring (4 hexagons)
        for i in range(4):
            angle = i * np.pi/2 + np.pi/4  # offset for better packing
            config.append([3 * np.cos(angle), 3 * np.sin(angle), 0])
        # Fill remainder with zeros
        while len(config) < NUM_INNER_HEXAGONS:
            config.append([0, 0, 0])
        config = config[:NUM_INNER_HEXAGONS]
        # Add small random noise
        for i in range(len(config)):
            config[i][0] += np.random.normal(0, 0.2)
            config[i][1] += np.random.normal(0, 0.2)
            config[i][2] += np.random.normal(0, 10)
            config[i][2] %= 360
        population.append(np.array(config))
    
    # Pattern 2: Cross pattern
    for _ in range(size//6):
        config = []
        config.append([0, 0, 0])
        config.append([0, 2, 0])
        config.append([0, -2, 0])
        config.append([2, 0, 0])
        config.append([-2, 0, 0])
        # Add more hexagons in cross pattern
        config.append([0, 3, 0])
        config.append([0, -3, 0])
        config.append([3, 0, 0])
        config.append([-3, 0, 0])
        config.append([1, 1, 0])
        config.append([-1, -1, 0])
        # Fill remaining
        while len(config) < NUM_INNER_HEXAGONS:
            config.append([0, 0, 0])
        config = config[:NUM_INNER_HEXAGONS]
        # Add small random noise
        for i in range(len(config)):
            config[i][0] += np.random.normal(0, 0.3)
            config[i][1] += np.random.normal(0, 0.3)
            config[i][2] += np.random.normal(0, 15)
            config[i][2] %= 360
        population.append(np.array(config))
    
    # Pattern 3: Random but constrained
    for _ in range(size//3):
        config = []
        for i in range(NUM_INNER_HEXAGONS):
            config.append([
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
                np.random.uniform(0, 360)
            ])
        population.append(np.array(config))
    
    # Ensure we have exactly the right size
    return population[:size]

def adaptive_neural_evolution():
    """Neural-inspired evolutionary algorithm with self-adaptation"""
    # Initialize population
    population = generate_initial_population(30)
    
    # Track performance statistics
    fitness_history = []
    improvement_history = []
    stagnation_counter = 0
    best_fitness = float('inf') 
    best_individual = None
    
    # Evolution parameters
    generations = 1000
    elite_size = 5
    mutation_rate = 0.8
    max_mutation_rate = 1.0
    min_mutation_rate = 0.1
    cross_rate = 0.8
    max_stagnation = 50
    
    for gen in range(generations):
        if time.time() > MAX_EVAL_TIME:
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = []
        valid_individuals = []
        
        for ind in population:
            # Check if solution is valid and compute fitness
            containment_ok, overlap_ok, hex_polygons = check_containment_and_overlap(ind)
            
            if containment_ok and overlap_ok:
                # Valid solution - calculate objective
                radius = calculate_outer_hexagon_radius(ind)
                fitness = 1.0/radius  # Higher is better
                fitness_scores.append(fitness)
                valid_individuals.append((ind, fitness))
            else:
                # Invalid solution - assign very poor fitness
                fitness_scores.append(-1000000)
                valid_individuals.append((ind, -1000000))
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Update best solution
        if sorted_fitness[0] > best_fitness:
            best_fitness = sorted_fitness[0]
            best_individual = sorted_population[0].copy()
            stagnation_counter = 0
            improvement_history.append(gen)
        else:
            stagnation_counter += 1
        
        if stagnation_counter > max_stagnation:
            # Reset mutation rate if stuck
            mutation_rate = min(max_mutation_rate, mutation_rate * 1.1)
            stagnation_counter = 0
        
        # Adapt mutation rate based on performance
        if len(improvement_history) > 10:
            recent_improvements = len(improvement_history) - len([x for x in improvement_history if x < gen-20])
            if recent_improvements < 2:
                mutation_rate = min(max_mutation_rate, mutation_rate * 1.1)
            else:
                mutation_rate = max(min_mutation_rate, mutation_rate * 0.95)
        
        # Create new population
        new_population = []
        
        # Elitism - keep best individuals
        for i in range(elite_size):
            new_population.append(sorted_population[i].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < len(population):
            # Selection based on fitness (tournament selection)
            tournament_size = 3
            selected_indices = []
            for _ in range(tournament_size):
                idx = np.random.randint(0, len(sorted_population))
                selected_indices.append(idx)
            
            # Pick the best from tournament
            tournament_fitness = [sorted_fitness[i] for i in selected_indices]
            winner_idx = selected_indices[np.argmax(tournament_fitness)]
            parent1 = sorted_population[winner_idx].copy()
            
            # Another parent
            selected_indices = []
            for _ in range(tournament_size):
                idx = np.random.randint(0, len(sorted_population))
                selected_indices.append(idx)
            
            tournament_fitness = [sorted_fitness[i] for i in selected_indices]
            winner_idx = selected_indices[np.argmax(tournament_fitness)]
            parent2 = sorted_population[winner_idx].copy()
            
            # Crossover
            if np.random.random() < cross_rate:
                # Uniform crossover
                child1 = parent1.copy()
                child2 = parent2.copy()
                for i in range(NUM_INNER_HEXAGONS):
                    if np.random.random() < 0.5:
                        child1[i] = parent2[i].copy()
                        child2[i] = parent1[i].copy()
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Mutation with adaptive rate
            for i in range(NUM_INNER_HEXAGONS):
                if np.random.random() < mutation_rate:
                    # Position mutation
                    child1[i][0] += np.random.normal(0, 0.5)
                    child1[i][1] += np.random.normal(0, 0.5)
                    child1[i][2] += np.random.normal(0, 20)
                    child1[i][2] %= 360
                    
                    # Same for child2
                    child2[i][0] += np.random.normal(0, 0.5)
                    child2[i][1] += np.random.normal(0, 0.5)
                    child2[i][2] += np.random.normal(0, 20)
                    child2[i][2] %= 360
            
            new_population.append(child1)
            if len(new_population) < len(population):
                new_population.append(child2)
        
        population = new_population[:len(population)]
    
    return best_individual, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a neural-inspired evolutionary algorithm with adaptive parameters.
    
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run neural evolutionary optimization
    best_individual, best_fitness = adaptive_neural_evolution()
    
    # Get final radius
    final_radius = calculate_outer_hexagon_radius(best_individual)
    
    # Validate final solution
    is_valid, _, _ = check_containment_and_overlap(best_individual)
    if not is_valid:
        # Fallback to well-known good configuration
        best_individual = np.array([
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
        ])
        final_radius = 8.0
    
    # Return result
    inner_hex_data = best_individual
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = final_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END