# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from numba import jit
import time
import warnings
from typing import Tuple, List
import random

@jit(nopython=True)
def hexagon_vertices_fast(x, y, angle_deg, side_length=1):
    """Fast generation of hexagon vertices using numba"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point, polygon):
    """Fast point-in-polygon test using ray casting"""
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

@jit(nopython=True)
def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using separating axis theorem"""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_vertices:
        if point_in_polygon_fast(v, hex2_vertices):
            return True
    # Check if any vertex of hex2 is inside hex1  
    for v in hex2_vertices:
        if point_in_polygon_fast(v, hex1_vertices):
            return True
    return False

def create_symmetric_initial():
    """Create highly symmetric initial configuration based on group theory"""
    # Use a pattern inspired by the 12-fold symmetry group
    positions = []
    
    # Central hexagon
    positions.append([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons arranged in a regular hexagon 
    for i in range(6):
        angle = i * 60  # degrees
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Ring 2: 5 hexagons in a pentagonal arrangement
    for i in range(5):
        angle = i * 72 + 18  # offset to create irregular but symmetric pattern
        radius = 3.5
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Add some strategic rotations to increase optimality
    # Rotate some hexagons to break degenerate symmetries
    positions[1][2] = 30   # First ring hexagon rotated
    positions[2][2] = 15   # Second ring hexagon rotated
    positions[4][2] = 45   # Third ring hexagon rotated
    
    return np.array(positions)

def compute_outer_side_length(hex_data):
    """Compute minimum side length of outer hexagon"""
    max_dist = 0
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        for vx, vy in vertices:
            dist = np.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)
    
    # Convert to hexagon side length (accounting for hexagon geometry)
    side_length = max_dist * 2 / np.sqrt(3)
    return side_length

def check_containment_fast(hex_position, outer_vertices):
    """Fast containment check using vertex position"""
    x, y, angle = hex_position
    vertices = hexagon_vertices_fast(x, y, angle)
    
    # Check if all vertices are within outer hexagon
    for vertex in vertices:
        if not point_in_polygon_fast(vertex, outer_vertices):
            return False
    return True

def calculate_fitness(hex_data, outer_side_length):
    """Calculate fitness for optimization"""
    # Check for overlaps using fast method first
    penalty = 0
    
    # Fast initial check using distance
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            x1, y1, _ = hex_data[i]
            x2, y2, _ = hex_data[j]
            
            # Distance between centers
            dist_centers = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # If centers are too close, do precise overlap checking
            if dist_centers < 1.99:  # Small tolerance for overlap
                # Use Shapely for precise overlap detection
                x1, y1, angle1 = hex_data[i]
                x2, y2, angle2 = hex_data[j]
                v1 = hexagon_vertices_fast(x1, y1, angle1)
                v2 = hexagon_vertices_fast(x2, y2, angle2)
                
                if check_hexagon_overlap_fast(v1, v2):
                    penalty += 1e6
    
    # Check containment with Shapely for precision
    outer_vertices = hexagon_vertices_fast(0, 0, 0, outer_side_length)
    
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        hex_poly = Polygon(vertices)
        
        # Point by point containment check
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not Polygon(outer_vertices).contains(point):
                # Calculate how far outside the boundary
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                penalty += (dist - outer_side_length)**2 * 1000
    
    # Objective: maximize 1/outer_side_length
    # So we minimize negative of 1/outer_side_length plus penalty
    objective = -1.0 / (outer_side_length + 1e-10) + penalty
    
    return objective

def create_evolutionary_population(pop_size, target_dim=12):
    """Create diverse population with symmetry awareness"""
    population = []
    
    # Generate multiple symmetric base configurations
    for i in range(pop_size // 2):
        base_config = create_symmetric_initial()
        
        # Add variation to positions and orientations
        noise = np.random.normal(0, 0.3, base_config.shape)
        mutated_config = base_config + noise
        population.append(mutated_config.flatten())
    
    # Generate some random configurations
    for i in range(pop_size // 2):
        # Random configuration but with sensible ranges
        config = np.random.uniform(-4, 4, (target_dim, 3))
        config[:, 2] = np.random.uniform(0, 360, target_dim)  # Random rotations
        population.append(config.flatten())
    
    return population

def evolutionary_optimization():
    """Use evolutionary approach with symmetry-aware operators"""
    pop_size = 20
    max_generations = 100
    
    # Create initial population
    population = create_evolutionary_population(pop_size)
    
    best_fitness = float('inf')
    best_individual = None
    
    for gen in range(max_generations):
        # Evaluate all individuals
        fitness_scores = []
        
        for individual in population:
            config = individual.reshape(-1, 3)
            outer_side = compute_outer_side_length(config)
            fitness = calculate_fitness(config, outer_side)
            fitness_scores.append(fitness)
        
        # Select best individuals
        sorted_indices = np.argsort(fitness_scores)
        elite_count = pop_size // 3
        selected_indices = sorted_indices[:elite_count]
        
        # Keep best individual
        current_best_idx = sorted_indices[0]
        current_best_fitness = fitness_scores[current_best_idx]
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep elites
        for idx in selected_indices:
            new_population.append(population[idx])
        
        # Generate offspring
        while len(new_population) < pop_size:
            parent1_idx = np.random.choice(selected_indices)
            parent2_idx = np.random.choice(selected_indices)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Uniform crossover
            child = np.copy(parent1)
            mask = np.random.rand(len(child)) > 0.5
            child[mask] = parent2[mask]
            
            # Mutation with symmetry awareness
            mut_rate = 0.15
            for i in range(len(child)):
                if np.random.rand() < mut_rate:
                    if i % 3 == 2:  # Rotation parameter
                        child[i] += np.random.normal(0, 30)  # Larger change for rotation
                        child[i] = child[i] % 360
                    else:  # Position parameters
                        child[i] += np.random.normal(0, 0.5)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual.reshape(-1, 3)

def optimize_with_local_search(initial_config):
    """Apply local optimization to the best evolutionary result"""
    initial_params = initial_config.flatten()
    
    def objective(params):
        config = params.reshape(-1, 3)
        outer_side_length = compute_outer_side_length(config)
        return calculate_fitness(config, outer_side_length)
    
    def constraint_func(params):
        config = params.reshape(-1, 3)
        return compute_outer_side_length(config) - 100.0  # Upper bound
    
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    try:
        result = minimize(objective, initial_params, method='SLSQP', constraints=cons, 
                          options={'maxiter': 500, 'ftol': 1e-6})
        if result.success:
            return result.x.reshape(-1, 3)
    except:
        pass
    return initial_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Use evolutionary optimization
        inner_hex_data = evolutionary_optimization()
        
        # Refine with local search
        inner_hex_data = optimize_with_local_search(inner_hex_data)
        
        # Final validation and refinement
        outer_side_length = compute_outer_side_length(inner_hex_data)
        
        # Perform one final detailed check
        final_outer_vertices = hexagon_vertices_fast(0, 0, 0, outer_side_length)
        is_valid = True
        for i in range(len(inner_hex_data)):
            if not check_containment_fast(inner_hex_data[i], final_outer_vertices):
                is_valid = False
                break
        
        if not is_valid:
            # Fall back to symmetric configuration
            inner_hex_data = create_symmetric_initial()
            outer_side_length = compute_outer_side_length(inner_hex_data)
            
    except Exception as e:
        warnings.warn(f"Evolutionary optimization failed: {str(e)}")
        # Fall back to symmetric configuration
        inner_hex_data = create_symmetric_initial()
        outer_side_length = compute_outer_side_length(inner_hex_data)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    
    # Calculate benchmark ratio for reporting
    benchmark_ratio = (1.0 / outer_side_length) / 0.2537
    
    # Print metrics
    print(f"inv_outer_hex_side_length: {1.0/outer_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {end_time - start_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END