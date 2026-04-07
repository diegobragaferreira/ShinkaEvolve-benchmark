# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Point, Polygon
import time
import math
from scipy.spatial.distance import cdist
import random

def generate_unit_hexagon_vertices(center=(0, 0), rotation_deg=0):
    """Generate vertices of a unit regular hexagon at given center and rotation."""
    angle = math.radians(rotation_deg)
    radius = 1.0
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def hexagon_intersects(hex1_vertices, hex2_vertices):
    """Check if two hexagons intersect using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_bounds(inner_hex_data):
    """Compute the minimal bounding hexagon that contains all inner hexagons."""
    all_vertices = []
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        vertices = generate_unit_hexagon_vertices(center, rotation)
        all_vertices.extend(vertices)
    
    if not all_vertices:
        return [(0, 0)] * 6
    
    # Find bounding box
    min_x = min(v[0] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)
    
    # Compute approximate hexagon bounds (simplified)
    avg_x = (min_x + max_x) / 2
    avg_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    # Approximate side length based on dimensions
    side_length = max(width, height) / math.sqrt(3) * 2
    
    # Generate final hexagon vertices - centered at average position
    outer_vertices = []
    for i in range(6):
        theta = i * math.pi / 3
        x = avg_x + side_length * math.cos(theta)
        y = avg_y + side_length * math.sin(theta)
        outer_vertices.append((x, y))
        
    return outer_vertices

def evaluate_solution_candidate(inner_hex_data):
    """Evaluate a candidate solution for fitness."""
    # Generate all hexagon vertices
    all_inner_vertices = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        vertices = generate_unit_hexagon_vertices(center, rotation)
        all_inner_vertices.append(vertices)
    
    # Check overlaps between all pairs
    overlap_penalty = 0
    for i in range(len(all_inner_vertices)):
        for j in range(i+1, len(all_inner_vertices)):
            if hexagon_intersects(all_inner_vertices[i], all_inner_vertices[j]):
                overlap_penalty += 1000000  # Heavy penalty for overlaps
    
    # Compute outer hexagon bounds
    outer_vertices = compute_outer_hexagon_bounds(inner_hex_data)
    
    # Check containment
    containment_penalty = 0
    for vertices in all_inner_vertices:
        if not check_hexagon_containment(vertices, outer_vertices):
            containment_penalty += 1000000
    
    # Estimate outer hexagon side length from vertices
    if len(outer_vertices) >= 3:
        # Take maximum distance from center to any vertex as approximation
        center_x = sum(v[0] for v in outer_vertices) / len(outer_vertices)
        center_y = sum(v[1] for v in outer_vertices) / len(outer_vertices)
        distances = [math.sqrt((v[0]-center_x)**2 + (v[1]-center_y)**2) for v in outer_vertices]
        outer_side_length = max(distances) * 2 / math.sqrt(3)  # Approximate side length
    else:
        outer_side_length = 1000  # Large default value
    
    # Fitness: We want to minimize outer hexagon size (maximize 1/outer_side_length)
    # Add penalties for overlaps and containment violations
    if overlap_penalty > 0 or containment_penalty > 0:
        fitness = -1000000  # Invalid solution
    else:
        fitness = 1.0 / outer_side_length
    
    return fitness, outer_side_length

def generate_hexagonal_pattern():
    """Generate a hexagonal arrangement that is likely to be valid."""
    # Start with center hexagon
    positions = [[0, 0]]
    
    # Add first ring of 6 hexagons
    for i in range(6):
        angle = i * math.pi / 3
        x = 2 * math.cos(angle)
        y = 2 * math.sin(angle)
        positions.append([x, y])
    
    # Add second ring of 12 hexagons
    for i in range(12):
        angle = i * math.pi / 6
        x = 3 * math.cos(angle)
        y = 3 * math.sin(angle)
        positions.append([x, y])
    
    # Trim to 11 hexagons
    positions = positions[:11]
    
    # Add random perturbations to create variation
    for i in range(11):
        positions[i][0] += np.random.normal(0, 0.2)
        positions[i][1] += np.random.normal(0, 0.2)
    
    # Return as array with rotations
    result = []
    for pos in positions:
        result.append([pos[0], pos[1], np.random.uniform(0, 360)])
    
    return np.array(result)

def generate_fibonacci_hexagon_placement():
    """Generate a Fibonacci-inspired distribution for better coverage."""
    positions = []
    n = 11
    
    # Fibonacci spiral approach
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n):
        angle = i * 2 * math.pi / golden_ratio
        radius = math.sqrt(i / (n - 1)) * 3  # Scale to fit within reasonable bounds
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append([x, y])
    
    # Convert to array with rotations
    result = []
    for pos in positions:
        result.append([pos[0], pos[1], np.random.uniform(0, 360)])
    
    return np.array(result)

def simulated_annealing_refinement(solution, max_iterations=1000):
    """Refine a solution using simulated annealing."""
    current_solution = solution.copy()
    current_fitness, current_side_length = evaluate_solution_candidate(current_solution)
    
    # Annealing parameters
    temp = 10.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    for iteration in range(max_iterations):
        # Perturb one hexagon
        hex_idx = np.random.randint(0, len(current_solution))
        new_solution = current_solution.copy()
        
        # Small random perturbation
        new_solution[hex_idx][0] += np.random.normal(0, 0.1)
        new_solution[hex_idx][1] += np.random.normal(0, 0.1)
        new_solution[hex_idx][2] += np.random.normal(0, 5)
        
        # Evaluate new solution
        new_fitness, new_side_length = evaluate_solution_candidate(new_solution)
        
        # Accept or reject based on Metropolis criterion
        if new_fitness > current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
        elif np.random.rand() < math.exp((new_fitness - current_fitness) / temp):
            current_solution = new_solution
            current_fitness = new_fitness
        
        # Cool down temperature
        temp *= cooling_rate
        if temp < min_temp:
            break
    
    return current_solution

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    best_fitness = -float('inf')
    best_solution = None
    best_side_length = float('inf')
    
    # Generate multiple initial configurations using different strategies
    initial_configs = []
    
    # Strategy 1: Hexagonal pattern
    initial_configs.append(generate_hexagonal_pattern())
    
    # Strategy 2: Fibonacci spiral pattern  
    initial_configs.append(generate_fibonacci_hexagon_placement())
    
    # Strategy 3: Random with geometric constraints
    np.random.seed(42)
    random_config = np.zeros((11, 3))
    for i in range(11):
        random_config[i][0] = np.random.uniform(-4, 4)
        random_config[i][1] = np.random.uniform(-4, 4)
        random_config[i][2] = np.random.uniform(0, 360)
    initial_configs.append(random_config)
    
    # Strategy 4: Another hexagonal pattern variation
    hex_pattern = generate_hexagonal_pattern()
    # Add noise to make it different
    for i in range(11):
        hex_pattern[i][0] += np.random.normal(0, 0.5)
        hex_pattern[i][1] += np.random.normal(0, 0.5)
    initial_configs.append(hex_pattern)
    
    # Strategy 5: Grid-based pattern
    grid_config = np.zeros((11, 3))
    idx = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            if idx < 11:
                grid_config[idx][0] = x * 2.5
                grid_config[idx][1] = y * 2.5
                grid_config[idx][2] = np.random.uniform(0, 360)
                idx += 1
    initial_configs.append(grid_config)
    
    # Evaluate all initial configurations and refine them
    for config in initial_configs:
        # Local refinement using simulated annealing
        refined_solution = simulated_annealing_refinement(config, max_iterations=500)
        
        # Evaluate refined solution
        fitness, side_length = evaluate_solution_candidate(refined_solution)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = refined_solution
            best_side_length = side_length
    
    # Final refinement of best solution
    if best_solution is not None:
        final_solution = simulated_annealing_refinement(best_solution, max_iterations=500)
        _, best_side_length = evaluate_solution_candidate(final_solution)
        best_solution = final_solution
    
    # Convert to expected format
    if best_solution is None:
        # Fallback to simple configuration if nothing worked
        best_solution = np.array([
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
        ])
        _, best_side_length = evaluate_solution_candidate(best_solution)
    
    # Prepare output data
    inner_hex_data = best_solution
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, best_side_length

# EVOLVE-BLOCK-END