# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import time
from math import sqrt
from numba import njit
import random

@njit
def create_hexagon_vertices_numba(center_x, center_y, side_length, rotation_degrees):
    """NumPy-accelerated vertex creation for hexagon."""
    angle_rad = np.radians(rotation_degrees)
    angle_step = 2 * np.pi / 6
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_step * i + angle_rad
        vertices[i, 0] = center_x + side_length * np.cos(angle)
        vertices[i, 1] = center_y + side_length * np.sin(angle)
    return vertices

@njit
def get_hexagon_circumradius_numba(side_length):
    """Get the circumradius of a regular hexagon."""
    return side_length

@njit
def fast_check_overlap_pair_numba(hex1_vertices, hex2_vertices):
    """Fast overlap check with approximation."""
    # Quick bounding circle test
    hex1_center = np.mean(hex1_vertices, axis=0)
    hex2_center = np.mean(hex2_vertices, axis=0)
    dist_centers = np.linalg.norm(hex1_center - hex2_center)
    circumradius = get_hexagon_circumradius_numba(1.0)
    
    # If centers are too far apart, no overlap
    if dist_centers > 2 * circumradius:
        return False
    
    # Full polygon intersection test (approximate for speed)
    # Using simple point-in-polygon check of a sample point
    test_point = hex1_vertices[0]
    return point_in_polygon(test_point, hex2_vertices)

@njit
def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon (simple ray casting)."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@njit
def compute_outer_hex_side_from_config_numba(inner_hex_data, center=(0,0)):
    """Compute the minimum required outer hexagon side length from current configuration."""
    if len(inner_hex_data) == 0:
        return 100.0
    
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
        dist_to_edge = dist + get_hexagon_circumradius_numba(1.0)
        max_dist = max(max_dist, dist_to_edge)
    
    return max_dist * 2.0

def check_containment_all_vertices_fast(hex_vertices, outer_hex_center, outer_hex_side_length):
    """Fast containment check."""
    outer_vertices = create_hexagon_vertices_numba(outer_hex_center[0], outer_hex_center[1], outer_hex_side_length, 0)
    outer_polygon = Polygon(outer_vertices)
    
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def evaluate_configuration_fast(inner_hex_data, outer_hex_center=(0,0)):
    """Fast evaluation with optimized geometric checks."""
    if len(inner_hex_data) != 12:
        return 1e-10

    # Precompute all hexagon vertices
    hex_vertices_list = []
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices_numba(cx, cy, 1.0, angle)
        hex_vertices_list.append(vertices)

    # Check containment: all hexagon vertices must be within outer hexagon
    outer_side_length = compute_outer_hex_side_from_config_numba(inner_hex_data, outer_hex_center)
    outer_vertices = create_hexagon_vertices_numba(outer_hex_center[0], outer_hex_center[1], outer_side_length, 0)
    outer_polygon = Polygon(outer_vertices)

    # Check containment for all vertices using fast method
    for vertices in hex_vertices_list:
        for vertex in vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                return 1e-10

    # Check overlaps between all pairs using fast method
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            if fast_check_overlap_pair_numba(hex_vertices_list[i], hex_vertices_list[j]):
                return 1e-10

    # If we reach here, the configuration is valid
    return 1.0 / outer_side_length

def create_symmetric_initial_placement():
    """Generate an initial placement that respects symmetry properties."""
    positions = []
    
    # Central hexagon
    positions.append([0, 0, 0])
    
    # First ring - 6 hexagons around center
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions, excluding duplicate
    radius = 2.0
    
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append([x, y, 0])
    
    # Second ring - 6 hexagons
    angles2 = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions
    radius2 = 3.5
    
    for angle in angles2:
        x = radius2 * np.cos(angle)
        y = radius2 * np.sin(angle)
        positions.append([x, y, 0])
    
    # Adjust to make sure we have exactly 12
    positions = positions[:12]
    
    # Convert to array format
    config = np.array(positions)
    
    # Add slight randomness to break perfect symmetry
    np.random.seed(42)
    config[:, 0] += np.random.normal(0, 0.1, 12)
    config[:, 1] += np.random.normal(0, 0.1, 12)
    
    return config

def generate_random_symmetric_config():
    """Generate a randomized symmetric configuration."""
    base_positions = create_symmetric_initial_placement()
    
    # Introduce variation while keeping symmetric essence
    config = base_positions.copy()
    
    # Randomize positions slightly
    for i in range(12):
        config[i, 0] += np.random.normal(0, 0.2)
        config[i, 1] += np.random.normal(0, 0.2)
        config[i, 2] = np.random.uniform(0, 360)  # Random rotation
    
    return config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Start with a good symmetric configuration
    current_config = create_symmetric_initial_placement()
    best_score = 0
    best_config = None
    best_outer_side = float('inf')
    
    # Evolutionary search with symmetry preservation
    population_size = 20
    generations = 50
    mutation_rate = 0.1
    
    # Initialize population
    population = [generate_random_symmetric_config() for _ in range(population_size)]
    
    for gen in range(generations):
        # Evaluate fitness of current population
        fitness_scores = []
        for individual in population:
            score = evaluate_configuration_fast(individual)
            fitness_scores.append(score)
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_score:
            best_score = fitness_scores[max_fitness_idx]
            best_config = population[max_fitness_idx].copy()
            best_outer_side = 1.0 / best_score
            
        # Early termination if we're approaching SOTA
        if best_score > 0.2535:  # Close to the target
            break
            
        # Selection (tournament selection)
        selected_population = []
        for _ in range(population_size):
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_population.append(population[winner_idx].copy())
        
        # Crossover and mutation to create new population
        new_population = []
        
        # Elitism: keep best individual
        best_individual = selected_population[max_fitness_idx]
        new_population.append(best_individual)
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1 = selected_population[random.randint(0, population_size-1)]
            parent2 = selected_population[random.randint(0, population_size-1)]
            
            # Crossover: blend positions, randomize rotations
            child = parent1.copy()
            for i in range(12):
                if random.random() < 0.5:  # Blend positions
                    child[i, 0] = (parent1[i, 0] + parent2[i, 0]) / 2
                    child[i, 1] = (parent1[i, 1] + parent2[i, 1]) / 2
                # Randomize rotation
                if random.random() < 0.3:
                    child[i, 2] = np.random.uniform(0, 360)
            
            # Mutation
            for i in range(12):
                if random.random() < mutation_rate:
                    child[i, 0] += np.random.normal(0, 0.3)
                    child[i, 1] += np.random.normal(0, 0.3)
                    if random.random() < 0.5:
                        child[i, 2] = np.random.uniform(0, 360)
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    # Local refinement with L-BFGS
    if best_config is not None:
        try:
            # Flatten and optimize with local method
            x0 = best_config.flatten()
            
            def objective(x):
                hex_data = x.reshape(-1, 3)
                score = evaluate_configuration_fast(hex_data)
                return -score  # Negative for maximization
            
            # Bounds for optimization
            bounds = []
            for i in range(24):
                bounds.append((-10, 10))
            for i in range(12):
                bounds.append((0, 360))
                
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            
            refined_config = result.x.reshape(-1, 3)
            refined_score = evaluate_configuration_fast(refined_config)
            
            if refined_score > best_score and refined_score > 1e-5:
                best_score = refined_score
                best_config = refined_config.copy()
                best_outer_side = 1.0 / best_score
                
        except Exception:
            pass
    
    # If we found a good solution, return it
    if best_config is not None and best_score > 1e-5:
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        return best_config, outer_hex_data, best_outer_side
    
    # Fallback to a reasonably good configuration
    inner_hex_data = np.array([
        [0, 0, 0],  # center
        [0, 2, 0],  # top
        [0, -2, 0],  # bottom
        [1.732, 1, 0],  # top right
        [-1.732, 1, 0],  # top left
        [1.732, -1, 0],  # bottom right
        [-1.732, -1, 0],  # bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0],  # far left
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0],  # top far left
        [1.732, -3, 0],  # bottom far right
    ])
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 6.928  # approximated value
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END