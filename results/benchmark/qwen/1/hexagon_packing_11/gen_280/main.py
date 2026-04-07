# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import math
from collections import defaultdict
import time

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
GRID_CELL_SIZE = 2.8  # Size of grid cells for spatial indexing

def get_hexagon_vertices(center_x, center_y, angle_degrees):
    """Get vertices of a unit regular hexagon given center and rotation"""
    angle_rad = math.radians(angle_degrees)
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi/3
        x = HEX_RADIUS * math.cos(theta)
        y = HEX_RADIUS * math.sin(theta)
        base_vertices.append((x, y))
    vertices = [(x + center_x, y + center_y) for x, y in base_vertices]
    return np.array(vertices)

def check_hexagon_containment(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Check if hexagon vertices are contained within outer hexagon"""
    for vertex in hex_vertices:
        x, y = vertex
        dx = x - outer_center_x
        dy = y - outer_center_y
        distance = math.sqrt(dx*dx + dy*dy)
        if distance >= outer_radius - 1e-10:
            return False
    return True

def hexagon_collision(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Separating Axis Theorem"""
    # Quick bounding box check first
    min1_x = min(v[0] for v in hex1_vertices)
    max1_x = max(v[0] for v in hex1_vertices)
    min1_y = min(v[1] for v in hex1_vertices)
    max1_y = max(v[1] for v in hex1_vertices)

    min2_x = min(v[0] for v in hex2_vertices)
    max2_x = max(v[0] for v in hex2_vertices)
    min2_y = min(v[1] for v in hex2_vertices)
    max2_y = max(v[1] for v in hex2_vertices)

    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False

    # Get all edges of both hexagons
    edges1 = []
    edges2 = []

    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        edge = (p2[0]-p1[0], p2[1]-p1[1])
        edges1.append(edge)

        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i+1)%6]
        edge = (p2[0]-p1[0], p2[1]-p1[1])
        edges2.append(edge)

    # Combine all potential separating axes
    all_axes = edges1 + edges2

    # Normalize axes
    for i, axis in enumerate(all_axes):
        length = math.sqrt(axis[0]**2 + axis[1]**2)
        if length > 0:
            all_axes[i] = (axis[0]/length, axis[1]/length)

    # Check projection overlap on each axis
    for axis in all_axes:
        proj1 = []
        proj2 = []

        for v in hex1_vertices:
            dot = v[0]*axis[0] + v[1]*axis[1]
            proj1.append(dot)

        for v in hex2_vertices:
            dot = v[0]*axis[0] + v[1]*axis[1]
            proj2.append(dot)

        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        if max1 < min2 or max2 < min1:
            return False

    return True

def build_spatial_grid(hex_data):
    """Build a spatial grid for efficient collision detection"""
    grid = defaultdict(list)
    cell_size = GRID_CELL_SIZE

    for i, (cx, cy, angle) in enumerate(hex_data):
        vertices = get_hexagon_vertices(cx, cy, angle)
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        min_col = int((min_x - 10) // cell_size)
        max_col = int((max_x + 10) // cell_size)
        min_row = int((min_y - 10) // cell_size)
        max_row = int((max_y + 10) // cell_size)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                grid[(row, col)].append(i)
    
    return grid

def get_potential_collisions(grid, hex_data, hex_index):
    """Get potential collision partners from spatial grid"""
    cx, cy, angle = hex_data[hex_index]
    vertices = get_hexagon_vertices(cx, cy, angle)
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)

    cell_size = GRID_CELL_SIZE
    min_col = int((min_x - 10) // cell_size)
    max_col = int((max_x + 10) // cell_size)
    min_row = int((min_y - 10) // cell_size)
    max_row = int((max_y + 10) // cell_size)

    candidates = set()
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            if (row, col) in grid:
                candidates.update(grid[(row, col)])
    
    return list(candidates)

def calculate_outer_hex_radius(inner_hex_data, outer_center=(0,0)):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        center_x = inner_hex_data[i][0]
        center_y = inner_hex_data[i][1]
        angle = inner_hex_data[i][2]
        vertices = get_hexagon_vertices(center_x, center_y, angle)
        for x, y in vertices:
            distance = math.sqrt((x - outer_center[0])**2 + (y - outer_center[1])**2)
            max_distance = max(max_distance, distance)
    return max_distance + HEX_RADIUS * 1.001

def evaluate_fitness_adaptive(individual, generation=0, max_generations=200):
    """
    Evaluate the fitness of a solution configuration with adaptive penalties
    """
    inner_hex_data = individual.reshape(-1, 3)
    outer_radius = calculate_outer_hex_radius(inner_hex_data)
    
    num_collisions = 0
    num_out_of_bounds = 0
    
    grid = build_spatial_grid(inner_hex_data)
    
    for i in range(len(inner_hex_data)):
        vertices_i = get_hexagon_vertices(
            inner_hex_data[i][0],
            inner_hex_data[i][1],
            inner_hex_data[i][2]
        )
        
        if not check_hexagon_containment(vertices_i, 0, 0, outer_radius):
            num_out_of_bounds += 1
        
        potential_collisions = get_potential_collisions(grid, inner_hex_data, i)
        for j in potential_collisions:
            if i >= j:
                continue
            vertices_j = get_hexagon_vertices(
                inner_hex_data[j][0],
                inner_hex_data[j][1],
                inner_hex_data[j][2]
            )
            if hexagon_collision(vertices_i, vertices_j):
                num_collisions += 1

    # Adaptive penalty scaling based on generation
    penalty_factor = 1.0 + (generation / max_generations) * 5.0
    penalty = 10000 * penalty_factor * (num_collisions + num_out_of_bounds)
    
    if num_collisions > 0 or num_out_of_bounds > 0:
        return 10000000 + penalty

    return 1.0 / outer_radius

def simulated_annealing_refinement(initial_individual, max_iterations=500):
    """
    Apply simulated annealing refinement to improve a solution
    """
    current_individual = initial_individual.copy()
    current_fitness = evaluate_fitness_adaptive(current_individual)
    
    best_individual = current_individual.copy()
    best_fitness = current_fitness
    
    temperature = 1.0
    cooling_rate = 0.98
    min_temperature = 0.001

    for iteration in range(max_iterations):
        neighbor = current_individual.copy()
        hex_index = np.random.randint(0, 11)
        
        # Perturb position and rotation
        neighbor[hex_index*3] += np.random.normal(0, 0.05)      # x position
        neighbor[hex_index*3 + 1] += np.random.normal(0, 0.05)  # y position
        neighbor[hex_index*3 + 2] += np.random.normal(0, 5)     # rotation
        
        # Keep rotation within [0, 360]
        neighbor[hex_index*3 + 2] = neighbor[hex_index*3 + 2] % 360

        neighbor_fitness = evaluate_fitness_adaptive(neighbor)
        
        if neighbor_fitness > current_fitness:
            current_individual = neighbor
            current_fitness = neighbor_fitness
        else:
            delta = neighbor_fitness - current_fitness
            acceptance_probability = math.exp(delta / temperature)
            if np.random.rand() < acceptance_probability:
                current_individual = neighbor
                current_fitness = neighbor_fitness

        if current_fitness > best_fitness:
            best_individual = current_individual.copy()
            best_fitness = current_fitness

        temperature = max(min_temperature, temperature * cooling_rate)
        if temperature < min_temperature:
            break

    return best_individual, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    np.random.seed(42)  # For reproducibility
    
    # Enhanced initial configurations with better diversity
    initial_configs = []
    
    # Config 1: Honeycomb-like with standard spacing
    config1 = np.array([
        [0, 0, 0],           # center
        [-2.1, 0, 0],        # left
        [2.1, 0, 0],         # right
        [-1.05, 1.8, 0],     # top-left
        [1.05, 1.8, 0],      # top-right
        [-1.05, -1.8, 0],    # bottom-left
        [1.05, -1.8, 0],     # bottom-right
        [-3.15, 1.8, 0],     # far top-left
        [3.15, 1.8, 0],      # far top-right
        [-3.15, -1.8, 0],    # far bottom-left
        [3.15, -1.8, 0],     # far bottom-right
    ]).flatten()
    initial_configs.append(config1)
    
    # Config 2: More spread out with greater separation
    config2 = np.array([
        [0, 0, 0],           # center
        [-2.5, 0, 0],        # left
        [2.5, 0, 0],         # right
        [-1.25, 2.17, 0],    # top-left
        [1.25, 2.17, 0],     # top-right
        [-1.25, -2.17, 0],   # bottom-left
        [1.25, -2.17, 0],    # bottom-right
        [-3.75, 2.17, 0],    # far top-left
        [3.75, 2.17, 0],     # far top-right
        [-3.75, -2.17, 0],   # far bottom-left
        [3.75, -2.17, 0],    # far bottom-right
    ]).flatten()
    initial_configs.append(config2)
    
    # Config 3: Spiral arrangement with more dispersed centers
    config3 = np.array([
        [0, 0, 0],           # center
        [1.2, 0, 0],         # right
        [0.6, 1.039, 0],     # upper right
        [-0.6, 1.039, 0],    # upper left
        [-1.2, 0, 0],        # left
        [-0.6, -1.039, 0],   # lower left
        [0.6, -1.039, 0],    # lower right
        [1.8, 1.559, 0],     # far upper right
        [-1.8, 1.559, 0],    # far upper left
        [-1.8, -1.559, 0],   # far lower left
        [1.8, -1.559, 0],    # far lower right
    ]).flatten()
    initial_configs.append(config3)
    
    # Config 4: Optimized linear arrangement
    config4 = np.array([
        [0, 0, 0],           # center
        [-1.5, 0, 0],        # left
        [1.5, 0, 0],         # right
        [-1.5, 1.0, 0],      # top-left
        [1.5, 1.0, 0],       # top-right
        [-1.5, -1.0, 0],     # bottom-left
        [1.5, -1.0, 0],      # bottom-right
        [-1.5, 2.0, 0],      # far top-left
        [1.5, 2.0, 0],       # far top-right
        [-1.5, -2.0, 0],     # far bottom-left
        [1.5, -2.0, 0],      # far bottom-right
    ]).flatten()
    initial_configs.append(config4)
    
    # Config 5: Rotated hexagon pattern for better utilization
    config5 = np.array([
        [0, 0, 0],           # center
        [-2.0, 0, 30],       # left rotated
        [2.0, 0, 30],        # right rotated
        [-1.0, 1.73, 0],     # top-left
        [1.0, 1.73, 0],      # top-right
        [-1.0, -1.73, 0],    # bottom-left
        [1.0, -1.73, 0],     # bottom-right
        [-3.0, 1.73, 30],    # far top-left rotated
        [3.0, 1.73, 30],     # far top-right rotated
        [-3.0, -1.73, 30],   # far bottom-left rotated
        [3.0, -1.73, 30],    # far bottom-right rotated
    ]).flatten()
    initial_configs.append(config5)

    best_result = None
    best_score = -float('inf')
    
    top_solutions = []
    
    # Run optimization from different initial configurations
    for i, initial_guess in enumerate(initial_configs):
        bounds = []
        for _ in range(11):
            bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle

        try:
            # Enhanced DE parameters with adaptive settings
            def adaptive_fitness(individual):
                return evaluate_fitness_adaptive(individual, generation=0, max_generations=150)

            result = differential_evolution(
                func=adaptive_fitness,
                bounds=bounds,
                maxiter=150,      # Increased iterations
                popsize=35,       # Larger population
                seed=42+i,        # Different seed for each run
                disp=False,
                tol=1e-8,         # Tighter tolerance
                mutation=(0.9, 0.1),  # Better initial mutation
                recombination=0.8   # Higher recombination
            )

            if result.success:
                best_individual = result.x
                inner_hex_data = best_individual.reshape(-1, 3)
                outer_radius = calculate_outer_hex_radius(inner_hex_data)
                score = 1.0 / outer_radius
                
                top_solutions.append((best_individual, score, outer_radius))
                
                if score > best_score:
                    best_score = score
                    best_result = (inner_hex_data, np.array([0, 0, 0]), outer_radius)
                    
        except Exception:
            continue

    # Sort top solutions by score
    top_solutions.sort(key=lambda x: x[1], reverse=True)
    
    # Apply local refinement to top solutions
    refined_solutions = []
    for individual, score, radius in top_solutions[:3]:
        try:
            refined_individual, refined_score = simulated_annealing_refinement(individual, max_iterations=300)
            refined_solutions.append((refined_individual, refined_score, calculate_outer_hex_radius(refined_individual.reshape(-1, 3))))
        except Exception:
            refined_solutions.append((individual, score, radius))

    # Find best among refined solutions
    for individual, score, radius in refined_solutions:
        if score > best_score:
            best_score = score
            inner_hex_data = individual.reshape(-1, 3)
            best_result = (inner_hex_data, np.array([0, 0, 0]), radius)

    # Fallback to best initial configuration if needed
    if best_result is None:
        inner_hex_data = np.array([
            [0, 0, 0],           # center
            [-2.1, 0, 0],        # left
            [2.1, 0, 0],         # right
            [-1.05, 1.8, 0],     # top-left
            [1.05, 1.8, 0],      # top-right
            [-1.05, -1.8, 0],    # bottom-left
            [1.05, -1.8, 0],     # bottom-right
            [-3.15, 1.8, 0],     # far top-left
            [3.15, 1.8, 0],      # far top-right
            [-3.15, -1.8, 0],    # far bottom-left
            [3.15, -1.8, 0],     # far bottom-right
        ])
        outer_radius = calculate_outer_hex_radius(inner_hex_data)
        best_result = (inner_hex_data, np.array([0, 0, 0]), outer_radius)

    return best_result[0], best_result[1], best_result[2]

# EVOLVE-BLOCK-END