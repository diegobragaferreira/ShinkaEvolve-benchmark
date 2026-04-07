# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import math
from collections import defaultdict
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# Constants for hexagon geometry
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APOGEE = HEX_RADIUS * math.sqrt(3)/2  # Distance from center to side midpoint
HEX_HEIGHT = 2 * HEX_APOGEE  # Height of hexagon
HEX_WIDTH = 2 * HEX_RADIUS  # Width of hexagon

# Spatial indexing parameters for efficient collision detection
INITIAL_GRID_CELL_SIZE = 2.5  # Starting grid cell size

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

def build_adaptive_spatial_grid(hex_data, min_x=None, max_x=None, min_y=None, max_y=None):
    """Build an adaptive spatial grid for efficient collision detection"""
    grid = defaultdict(list)

    # Determine grid bounds if not provided
    if min_x is None or max_x is None or min_y is None or max_y is None:
        # Precompute all hexagon vertices for bounds calculation
        hex_vertices_list = []
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for cx, cy, angle in hex_data:
            vertices = get_hexagon_vertices(cx, cy, angle)
            hex_vertices_list.append(vertices)
            for x, y in vertices:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        # Expand bounds slightly to account for edge cases
        margin = HEX_WIDTH * 1.5
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
    else:
        hex_vertices_list = [get_hexagon_vertices(cx, cy, angle) for cx, cy, angle in hex_data]

    # Calculate grid dimensions with adaptive cell size
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    
    # Adaptive grid cell size based on problem dimensions and solution complexity
    avg_cell_size = max(INITIAL_GRID_CELL_SIZE, min(grid_width, grid_height) / 8.0)
    
    num_cols = int(math.ceil(grid_width / avg_cell_size))
    num_rows = int(math.ceil(grid_height / avg_cell_size))

    # For each hexagon, determine which grid cells it occupies
    for i, vertices in enumerate(hex_vertices_list):
        # Get the bounding box of the hexagon
        min_x_h = min(v[0] for v in vertices)
        max_x_h = max(v[0] for v in vertices)
        min_y_h = min(v[1] for v in vertices)
        max_y_h = max(v[1] for v in vertices)

        # Determine grid cells that this hexagon covers
        min_col = int((min_x_h - min_x) // avg_cell_size)
        max_col = int((max_x_h - min_x) // avg_cell_size)
        min_row = int((min_y_h - min_y) // avg_cell_size)
        max_row = int((max_y_h - min_y) // avg_cell_size)

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                grid[(row, col)].append(i)

    return grid, min_x, max_x, min_y, max_y, avg_cell_size

def get_potential_collisions_adaptive(grid, hex_data, hex_index, min_x, max_x, min_y, max_y, grid_cell_size):
    """Get potential collision partners from spatial grid with adaptive expansion"""
    # Get the bounding box of the hexagon we're checking
    cx, cy, angle = hex_data[hex_index]
    vertices = get_hexagon_vertices(cx, cy, angle)
    min_x_h = min(v[0] for v in vertices)
    max_x_h = max(v[0] for v in vertices)
    min_y_h = min(v[1] for v in vertices)
    max_y_h = max(v[1] for v in vertices)

    # Determine grid cells that this hexagon covers
    min_col = int((min_x_h - min_x) // grid_cell_size)
    max_col = int((max_x_h - min_x) // grid_cell_size)
    min_row = int((min_y_h - min_y) // grid_cell_size)
    max_row = int((max_y_h - min_y) // grid_cell_size)

    # Collect potential candidates - expand by 3 cells in each direction for safety
    candidates = set()
    for row in range(min_row - 3, max_row + 4):
        for col in range(min_col - 3, max_col + 4):
            if (row, col) in grid:
                candidates.update(grid[(row, col)])

    return list(candidates)

def check_hexagon_containment(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Check if hexagon vertices are contained within outer hexagon"""
    # For a regular hexagon centered at origin, we can check distance from center
    for vertex in hex_vertices:
        x, y = vertex
        dx = x - outer_center_x
        dy = y - outer_center_y
        distance = math.sqrt(dx*dx + dy*dy)
        if distance >= outer_radius:
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

    # If bounding boxes don't overlap, no collision possible
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

    # Normalize axes and check for zero-length vectors
    normalized_axes = []
    for axis in all_axes:
        length = math.sqrt(axis[0]**2 + axis[1]**2)
        if length > 0:
            normalized_axes.append((axis[0]/length, axis[1]/length))

    if not normalized_axes:
        return False

    # Check projection overlap on each axis
    for axis in normalized_axes:
        # Project both hexagons onto this axis
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

        # If projections don't overlap, then there's separation
        if max1 < min2 or max2 < min1:
            return False

    return True

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

def evaluate_fitness_adaptive(individual):
    """
    Evaluate the fitness of a solution configuration with adaptive optimizations
    individual: array of shape (33,) containing [x1,y1,a1,x2,y2,a2,...,x11,y11,a11]
    Returns negative value because we want to maximize 1/R (minimize R)
    """
    # Reshape individual into hexagon data
    inner_hex_data = individual.reshape(-1, 3)

    # Try different outer hexagon sizes and check feasibility
    # Start with a reasonable estimate
    outer_radius = calculate_outer_hex_radius(inner_hex_data)

    # Check collisions and containment
    num_collisions = 0
    num_out_of_bounds = 0

    # Build spatial grid for efficient collision detection with adaptive sizing
    grid, min_x, max_x, min_y, max_y, grid_cell_size = build_adaptive_spatial_grid(inner_hex_data)

    # Check all hexagon pairs for collision using spatial indexing
    for i in range(len(inner_hex_data)):
        vertices_i = get_hexagon_vertices(
            inner_hex_data[i][0],
            inner_hex_data[i][1],
            inner_hex_data[i][2]
        )

        # Check containment first
        if not check_hexagon_containment(vertices_i, 0, 0, outer_radius):
            num_out_of_bounds += 1
            # Early termination if containment fails
            penalty = 10000 * (num_collisions + num_out_of_bounds)  # Increased penalty
            return 1000000 + penalty  # Large penalty for invalid solutions

        # Efficiently get potential collision partners using spatial indexing
        potential_collisions = get_potential_collisions_adaptive(grid, inner_hex_data, i, min_x, max_x, min_y, max_y, grid_cell_size)

        # Optimize collision checking by prioritizing nearby hexagons and limiting candidates
        # Sort by distance to reduce unnecessary checks
        candidate_distances = [(j, ((inner_hex_data[i][0] - inner_hex_data[j][0])**2 + 
                                  (inner_hex_data[i][1] - inner_hex_data[j][1])**2)) 
                              for j in potential_collisions if i != j]
        candidate_distances.sort(key=lambda x: x[1])  # Sort by distance
        
        # Only check top 30 closest candidates for efficiency
        sorted_candidates = [j for j, _ in candidate_distances[:30]]

        for j in sorted_candidates:
            vertices_j = get_hexagon_vertices(
                inner_hex_data[j][0],
                inner_hex_data[j][1],
                inner_hex_data[j][2]
            )

            if hexagon_collision(vertices_i, vertices_j):
                num_collisions += 1
                # Early termination if collision found
                penalty = 10000 * (num_collisions + num_out_of_bounds)  # Increased penalty
                return 1000000 + penalty  # Large penalty for invalid solutions

    # Penalty for collisions or out of bounds
    penalty = 10000 * (num_collisions + num_out_of_bounds)  # Increased penalty

    # If invalid configuration, return poor fitness
    if num_collisions > 0 or num_out_of_bounds > 0:
        return 1000000 + penalty  # Large penalty for invalid solutions

    # Return inverse of outer radius (we want to maximize 1/R)
    return 1.0 / outer_radius

def generate_voronoi_initial_configurations(n_configs=5, n_hexagons=11):
    """Generate diverse initial configurations using Voronoi-based approach"""
    configurations = []
    
    # Base layout - center hexagon with surrounding hexagons arranged in pattern
    base_layouts = [
        # Layout 1: Hexagonal pattern
        [
            [0, 0, 0],           # center
            [-2.0, 0, 0],        # left
            [2.0, 0, 0],         # right
            [-1.0, 1.732, 0],    # top-left
            [1.0, 1.732, 0],     # top-right
            [-1.0, -1.732, 0],   # bottom-left
            [1.0, -1.732, 0],    # bottom-right
            [-3.0, 1.732, 0],    # far top-left
            [3.0, 1.732, 0],     # far top-right
            [-3.0, -1.732, 0],   # far bottom-left
            [3.0, -1.732, 0],    # far bottom-right
        ],
        # Layout 2: Dense central cluster
        [
            [0, 0, 0],           # center
            [-1.9, 0, 0],        # left
            [1.9, 0, 0],         # right
            [-1.0, 1.732, 0],    # top-left
            [1.0, 1.732, 0],     # top-right
            [-1.0, -1.732, 0],   # bottom-left
            [1.0, -1.732, 0],    # bottom-right
            [-2.9, 1.732, 0],    # far top-left
            [2.9, 1.732, 0],     # far top-right
            [-2.9, -1.732, 0],   # far bottom-left
            [2.9, -1.732, 0],    # far bottom-right
        ],
        # Layout 3: Linear arrangement with clusters
        [
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
        ],
        # Layout 4: Spiral-like arrangement
        [
            [0, 0, 0],           # center
            [0, 2, 0],           # top
            [1.732, 1, 0],       # top-right
            [1.732, -1, 0],      # bottom-right
            [0, -2, 0],          # bottom
            [-1.732, -1, 0],     # bottom-left
            [-1.732, 1, 0],      # top-left
            [0, 3.5, 0],         # far top
            [3.031, 1.75, 0],    # far top-right
            [3.031, -1.75, 0],   # far bottom-right
            [0, -3.5, 0],        # far bottom
        ],
        # Layout 5: Randomized layout
        [
            [0, 0, 0],           # center
            [-2.3, 0, 0],        # left
            [2.3, 0, 0],         # right
            [-1.15, 2.0, 0],     # top-left
            [1.15, 2.0, 0],      # top-right
            [-1.15, -2.0, 0],    # bottom-left
            [1.15, -2.0, 0],     # bottom-right
            [-3.45, 2.0, 0],     # far top-left
            [3.45, 2.0, 0],      # far top-right
            [-3.45, -2.0, 0],    # far bottom-left
            [3.45, -2.0, 0],     # far bottom-right
        ]
    ]
    
    for i, base_layout in enumerate(base_layouts):
        # Add random perturbations to make configurations diverse
        perturbed_layout = []
        for j, (x, y, angle) in enumerate(base_layout):
            # Center hexagon gets minimal perturbation
            if j == 0:
                perturbed_layout.append([
                    x + random.uniform(-0.3, 0.3),
                    y + random.uniform(-0.3, 0.3),
                    angle + random.uniform(-10, 10)
                ])
            else:
                perturbed_layout.append([
                    x + random.uniform(-0.5, 0.5),
                    y + random.uniform(-0.5, 0.5),
                    angle + random.uniform(-15, 15)
                ])
        configurations.append(perturbed_layout)
    
    # Add a few completely randomized configurations
    for i in range(n_configs - len(base_layouts)):
        random_layout = []
        for j in range(n_hexagons):
            # Distribute hexagons with some clustering
            if j == 0:
                # Center hexagon
                random_layout.append([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0, 360)])
            else:
                # Surrounding hexagons
                angle = random.uniform(0, 360)
                distance = random.uniform(1.5, 4.0)
                x = distance * math.cos(math.radians(angle))
                y = distance * math.sin(math.radians(angle))
                random_layout.append([x, y, random.uniform(0, 360)])
        configurations.append(random_layout)
    
    return configurations

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    np.random.seed(42)  # For reproducibility
    
    # Phase 1: Multi-start optimization with Voronoi-based initial configurations
    configurations = generate_voronoi_initial_configurations(8, 11)
    
    best_individual = None
    best_fitness = float('-inf')
    best_radius = float('inf')
    
    # Run multiple optimizations from different starting points
    for i, initial_config in enumerate(configurations):
        try:
            # Flatten initial configuration
            initial_guess = np.array(initial_config).flatten()
            
            # Bounds for optimization: positions (-15, 15), rotations (0, 360)
            bounds = []
            for _ in range(11):
                bounds.extend([(-15, 15), (-15, 15), (0, 360)])  # x, y, angle
            
            # Use more aggressive DE parameters for better exploration
            result = differential_evolution(
                func=evaluate_fitness_adaptive,
                bounds=bounds,
                maxiter=150,  # More iterations
                popsize=20,   # Larger population
                seed=42+i,
                disp=False,
                tol=1e-6,
                strategy='best1bin'  # Slightly different strategy
            )
            
            if result.success:
                # Evaluate final result
                final_fitness = evaluate_fitness_adaptive(result.x)
                if final_fitness > best_fitness:
                    best_fitness = final_fitness
                    best_individual = result.x.copy()
                    # Calculate actual radius for comparison
                    temp_inner_hex_data = best_individual.reshape(-1, 3)
                    temp_radius = calculate_outer_hex_radius(temp_inner_hex_data)
                    best_radius = temp_radius
                    
        except Exception as e:
            continue
    
    # If no good solution found from multiple starts, use the first configuration
    if best_individual is None:
        # Use the first Voronoi configuration as fallback
        initial_config = configurations[0]
        best_individual = np.array(initial_config).flatten()
    
    # Phase 2: Local refinement with simulated annealing approach
    if best_individual is not None:
        # Convert to proper format
        inner_hex_data = best_individual.reshape(-1, 3)
        
        # Apply local refinement with adaptive temperature schedule
        refined_individual = local_refinement_adaptive(inner_hex_data)
        
        # Final verification and calculation
        final_fitness = evaluate_fitness_adaptive(refined_individual)
        if final_fitness > best_fitness:
            best_fitness = final_fitness
            best_individual = refined_individual
    
    # Final processing
    if best_individual is not None:
        inner_hex_data = best_individual.reshape(-1, 3)
        outer_radius = calculate_outer_hex_radius(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_radius
    else:
        # Fallback to a safe configuration
        inner_hex_data = np.array([
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
        ])
        outer_radius = 8.0
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_radius

def local_refinement_adaptive(inner_hex_data):
    """Local refinement using adaptive simulated annealing approach"""
    current_solution = inner_hex_data.copy()
    current_fitness = evaluate_fitness_adaptive(current_solution.flatten())
    
    # Adaptive temperature schedule
    temperature = 5.0
    cooling_rate = 0.995
    min_temperature = 0.001
    max_iterations = 500
    
    for iteration in range(max_iterations):
        if temperature < min_temperature:
            break
            
        # Try random perturbations
        for _ in range(5):  # Multiple tries per iteration
            # Copy current solution
            test_solution = current_solution.copy()
            
            # Select a random hexagon to perturb
            hex_idx = random.randint(0, 10)
            
            # Adaptive perturbation based on temperature
            pos_magnitude = 0.1 * (temperature / 5.0)  # Decrease with temperature
            angle_magnitude = 5.0 * (temperature / 5.0)  # Decrease with temperature
            
            # Perturb position
            test_solution[hex_idx][0] += np.random.normal(0, pos_magnitude)
            test_solution[hex_idx][1] += np.random.normal(0, pos_magnitude)
            
            # Perturb angle
            test_solution[hex_idx][2] += np.random.normal(0, angle_magnitude)
            test_solution[hex_idx][2] %= 360
            
            # Evaluate
            test_fitness = evaluate_fitness_adaptive(test_solution.flatten())
            
            # Accept or reject based on temperature
            if test_fitness > current_fitness:
                current_fitness = test_fitness
                current_solution = test_solution
            else:
                # Accept with probability based on temperature
                delta = test_fitness - current_fitness
                acceptance_prob = np.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_fitness = test_fitness
                    current_solution = test_solution
        
        # Cool down
        temperature *= cooling_rate
    
    return current_solution.flatten()

# EVOLVE-BLOCK-END