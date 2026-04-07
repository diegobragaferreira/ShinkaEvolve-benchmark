# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from itertools import combinations
from numba import njit
import random
from collections import defaultdict

# Constants
HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180

@njit
def create_hexagon_vertices(center_x, center_y, rotation):
    """Create vertices of a unit regular hexagon - JIT compiled for speed"""
    vertices = np.empty((6, 2))
    angle_step = np.pi / 3
    for i in range(6):
        angle = rotation + i * angle_step
        x = center_x + HEXAGON_RADIUS * np.cos(angle)
        y = center_y + HEXAGON_RADIUS * np.sin(angle)
        vertices[i] = (x, y)
    return vertices

@njit
def point_in_hexagon(px, py, hex_vertices):
    """Check if point is inside hexagon using ray casting - JIT compiled"""
    n = len(hex_vertices)
    inside = False
    p1x, p1y = hex_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hex_vertices[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@njit
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Distance from point to line segment - JIT compiled"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        # Line segment is actually a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    # Clamp t to [0,1] to stay within line segment
    t = max(0, min(1, t))
    
    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@njit
def calculate_outer_radius_fast(inner_params):
    """Fast calculation of outer radius for all 12 hexagons - JIT compiled"""
    max_dist = 0.0
    for i in range(12):
        center_x, center_y, _ = inner_params[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        # Hexagon diagonal is sqrt(3) * radius
        hex_diag = HEXAGON_RADIUS * np.sqrt(3)
        max_dist = max(max_dist, dist_to_center + hex_diag)
    return max_dist

def hexagon_vertices_array(center_x, center_y, rotation):
    """Create vertices array for a hexagon"""
    vertices = np.empty((6, 2))
    angle_step = np.pi / 3
    for i in range(6):
        angle = rotation + i * angle_step
        x = center_x + HEXAGON_RADIUS * np.cos(angle)
        y = center_y + HEXAGON_RADIUS * np.sin(angle)
        vertices[i] = [x, y]
    return vertices

def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon as a Shapely polygon"""
    angle_step = np.pi / 3
    points = []
    for i in range(6):
        angle = rotation + i * angle_step
        x = center[0] + HEXAGON_RADIUS * np.cos(angle)
        y = center[1] + HEXAGON_RADIUS * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment_shapely(hexagon, outer_hexagon):
    """Check if a hexagon is fully contained within outer hexagon"""
    return outer_hexagon.contains(hexagon)

def compute_packing_density(inner_hex_data):
    """Compute a proxy for how densely packed the hexagons are"""
    if len(inner_hex_data) < 2:
        return 0.0
    
    # Compute average distances between centers
    distances = []
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            cx1, cy1, _ = inner_hex_data[i]
            cx2, cy2, _ = inner_hex_data[j]
            distance = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            distances.append(distance)
    
    if not distances:
        return 0.0
    
    avg_distance = np.mean(distances)
    # Density is inversely related to average distance, normalized by hex diameter
    diameter = 2 * HEXAGON_RADIUS
    density = diameter / avg_distance if avg_distance > 0 else 1000.0
    return density

def calculate_outer_hexagon_radius(inner_hex_data):
    """Calculate minimum outer hexagon radius required to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        # Hexagon diagonal is sqrt(3) * radius
        hex_diag = HEXAGON_RADIUS * np.sqrt(3)
        max_dist = max(max_dist, dist_to_center + hex_diag)
    return max_dist

def evaluate_solution(params):
    """
    Evaluation function with hierarchical filtering and advanced constraint checking
    params: array of shape (12,3) = [x1, y1, theta1, ..., x12, y12, theta12]
    """
    # Reshape into 12 hexagons with (x,y,rotation)
    inner_params = params.reshape(12, 3)
    
    # Phase 1: Quick geometric filtering to reject bad configurations early
    outer_radius = calculate_outer_hexagon_radius(inner_params)
    if outer_radius > 100:  # Too big, reject immediately
        return 1000000.0
    
    # Phase 2: Create hexagons and check main constraints
    hexagons = []
    hexagon_polygons = []
    
    for i in range(12):
        center = (inner_params[i][0], inner_params[i][1])
        angle = np.radians(inner_params[i][2])
        hexagon = create_unit_hexagon(center, angle)
        hexagons.append((center[0], center[1], angle))
        hexagon_polygons.append(hexagon)
    
    # Check containment: use a single large hexagon around all centers
    outer_hex = create_unit_hexagon((0, 0), 0)
    containment_violations = 0
    for hex_poly in hexagon_polygons:
        if not check_containment_shapely(hex_poly, outer_hex):
            containment_violations += 1
    
    # Phase 3: More efficient overlap checking with spatial indexing
    overlap_violations = 0
    
    # Create spatial index for all hexagon centers
    centers = np.array([[h[0], h[1]] for h in hexagons])
    tree = cKDTree(centers)
    
    # Check for nearby pairs first, then precise overlap check
    for i in range(12):
        # Get nearby hexagons within reasonable distance
        indices = tree.query_ball_point(centers[i], 3.0)
        for j in indices:
            if i < j:
                # Quick bounding box check
                h1 = hexagons[i]
                h2 = hexagons[j]
                
                # Simple overlap estimation - check if centers are close enough
                dist = np.sqrt((h1[0]-h2[0])**2 + (h1[1]-h2[1])**2)
                if dist < 2.0:  # Threshold for possible overlap
                    # Precise overlap check
                    if hexagon_polygons[i].intersects(hexagon_polygons[j]):
                        overlap_violations += 1
                        break  # Early termination for efficiency
        if overlap_violations > 0:
            break  # Early termination if any overlap found
    
    # Phase 4: Fitness calculation with penalties
    penalty = 10000 * (containment_violations + overlap_violations)
    
    # If constraints violated, heavily penalize
    if containment_violations > 0 or overlap_violations > 0:
        return 1000000 + penalty
    
    # Primary objective: maximize 1/outer_radius (minimize outer radius)
    # But also consider packing density
    density = compute_packing_density(inner_params)
    fitness = -(1.0 / (outer_radius * 1.05)) + (density * 0.001)
    
    return fitness + penalty

def generate_initial_skeleton():
    """Generate a good initial skeleton configuration using geometric reasoning"""
    # Use a well-known hexagonal packing pattern adapted for 12 elements
    params = []
    
    # Central hexagon
    params.extend([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons arranged in a hexagon pattern
    for i in range(6):
        angle = i * np.pi / 3
        radius = 2.0  # Fixed distance from center
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        params.extend([x, y, 0.0])
    
    # Second ring: 5 more hexagons (central one already placed)
    for i in range(5):
        angle = (i * np.pi / 3) + np.pi/6  # Offset to fill gaps
        radius = 3.5  # Slightly larger radius
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        params.extend([x, y, 0.0])
    
    return np.array(params)

def generate_diverse_initial_population(size=20):
    """Generate diverse initial population with geometric constraints"""
    population = []
    for _ in range(size):
        # Start with skeleton
        individual = generate_initial_skeleton().copy()
        
        # Add some random noise for diversity
        noise_magnitude = 0.5
        for i in range(12):
            individual[i*3 + 0] += (random.random() - 0.5) * noise_magnitude  # x
            individual[i*3 + 1] += (random.random() - 0.5) * noise_magnitude  # y
            individual[i*3 + 2] += (random.random() - 0.5) * 30  # rotation (deg)
        
        population.append(individual)
    
    return population

def evolution_strategy_optimization():
    """Evolutionary optimization strategy with geometric awareness"""
    # Generate initial population
    population_size = 15
    population = generate_diverse_initial_population(population_size)
    
    # Evaluate initial population
    fitness_scores = []
    for individual in population:
        fitness = evaluate_solution(individual)
        fitness_scores.append(fitness)
    
    # Sort by fitness
    sorted_indices = np.argsort(fitness_scores)
    population = [population[i] for i in sorted_indices]
    fitness_scores = [fitness_scores[i] for i in sorted_indices]
    
    # Evolution loop
    generations = 20
    for gen in range(generations):
        # Selection: keep top 50%
        elite_count = population_size // 2
        selected = population[:elite_count]
        selected_fitness = fitness_scores[:elite_count]
        
        # Create new population through crossover and mutation
        new_population = selected.copy()
        
        # Elitism
        new_population.extend(selected)
        
        # Crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Uniform crossover
            child = parent1.copy()
            for i in range(len(child)):
                if random.random() > 0.5:
                    child[i] = parent2[i]
            
            # Mutation
            mutation_rate = 0.2
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    if i % 3 == 0 or i % 3 == 1:  # x or y coordinate
                        child[i] += (random.random() - 0.5) * 1.0
                    else:  # rotation
                        child[i] += (random.random() - 0.5) * 45
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Re-evaluate
        fitness_scores = []
        for individual in population:
            fitness = evaluate_solution(individual)
            fitness_scores.append(fitness)
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
    
    # Return best solution
    return population[0]

def optimize_hexagon_arrangement():
    """
    Multi-stage optimization with evolutionary strategy and local refinement
    """
    # Phase 1: Evolutionary search for good configuration
    try:
        evolution_result = evolution_strategy_optimization()
    except Exception as e:
        print(f"Evolution failed: {e}")
        evolution_result = generate_initial_skeleton()
    
    # Phase 2: Local refinement using scipy optimization
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    
    try:
        # Use L-BFGS-B for fine tuning
        result = minimize(
            evaluate_solution,
            evolution_result,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20}
        )
        
        if result.success:
            optimized_params = result.x
        else:
            optimized_params = evolution_result
    except Exception as e:
        print(f"Local optimization failed: {e}")
        optimized_params = evolution_result
    
    # Convert to standard format
    inner_hex_data = optimized_params.reshape(12, 3)
    
    # Calculate final outer hexagon side length
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data)
    outer_hex_side_length = outer_radius * np.sqrt(3)  # Convert from radius to side length
    
    # Outer hexagon centered at origin with 0 rotation
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use evolutionary optimization approach
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_arrangement()
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Debug output
    inv_outer_side_length = 1.0 / outer_hex_side_length
    benchmark_ratio = inv_outer_side_length / 0.2537
    
    print(f"Eval time: {eval_time:.4f}s")
    print(f"Inv outer side length: {inv_outer_side_length:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END