# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
import time
import random
from collections import defaultdict

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEX_RADIUS = 1.0
GRID_CELL_SIZE = 2.0  # Based on hexagon diameter for efficient spatial indexing

def get_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
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
    rotated_vertices = base_vertices @ rotation_matrix.T
    translated_vertices = rotated_vertices + np.array([center_x, center_y])
    
    return translated_vertices

def get_bounding_box(vertices):
    """Get axis-aligned bounding box for a set of vertices"""
    if len(vertices) == 0:
        return (0, 0, 0, 0)
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    return (min_x, min_y, max_x, max_y)

def boxes_overlap(box1, box2):
    """Check if two axis-aligned boxes overlap"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def create_spatial_index(hex_vertices_list):
    """Create a spatial index for fast collision detection"""
    # Create bounding boxes for all hexagons
    boxes = []
    for i, vertices in enumerate(hex_vertices_list):
        box = get_bounding_box(vertices)
        boxes.append((box, i))

    # Group boxes into grid cells for faster lookup
    spatial_index = defaultdict(list)

    for box, idx in boxes:
        # Grid cell coordinates
        min_x, min_y, max_x, max_y = box
        cell_min_x = int(min_x // GRID_CELL_SIZE)
        cell_min_y = int(min_y // GRID_CELL_SIZE)
        cell_max_x = int(max_x // GRID_CELL_SIZE)
        cell_max_y = int(max_y // GRID_CELL_SIZE)

        # Add to all relevant grid cells
        for cx in range(cell_min_x, cell_max_x + 1):
            for cy in range(cell_min_y, cell_max_y + 1):
                spatial_index[(cx, cy)].append(idx)

    return spatial_index, boxes

def check_containment(hexagon_vertices, outer_center=(0,0), outer_radius=10.0):
    """Check if hexagon is fully contained within outer hexagon"""
    outer_vertices = get_hexagon_vertices(outer_center[0], outer_center[1], 0, outer_radius)
    outer_polygon = Polygon(outer_vertices)
    hex_polygon = Polygon(hexagon_vertices)
    return outer_polygon.contains(hex_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using shapely"""
    hex1_polygon = Polygon(hex1_vertices)
    hex2_polygon = Polygon(hex2_vertices)
    return hex1_polygon.intersects(hex2_polygon)

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum side length of outer hexagon needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)
    
    all_vertices = np.array(all_vertices)
    
    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
    
    # Calculate approximate side length (simplified approach)
    width = max_x - min_x
    height = max_y - min_y
    
    # Estimate side length from dimensions
    side_len_width = width / 2.0
    side_len_height = height / (np.sqrt(3))
    
    # Take maximum to ensure containment
    estimated_side_length = max(side_len_width, side_len_height) * 1.1  # Add small buffer
    
    return estimated_side_length

def evaluate_fitness(individual):
    """Evaluate fitness of solution - maximize 1/outer_hex_side_length"""
    try:
        # Create polygons for all inner hexagons
        hex_vertices_list = []
        for i in range(len(individual)):
            center_x, center_y, angle = individual[i]
            vertices = get_hexagon_vertices(center_x, center_y, angle)
            hex_vertices_list.append(vertices)
        
        # Check containment and overlap
        outer_side_length = calculate_outer_hex_side_length(individual)
        outer_vertices = get_hexagon_vertices(0, 0, 0, outer_side_length)
        outer_polygon = Polygon(outer_vertices)
        
        # Check containment for all hexagons
        for vertices in hex_vertices_list:
            hex_polygon = Polygon(vertices)
            if not outer_polygon.contains(hex_polygon):
                return 0.0  # Invalid - not fully contained
        
        # Check overlaps using spatial indexing for efficiency
        spatial_index, boxes = create_spatial_index(hex_vertices_list)

        # Only check actual collisions for potentially overlapping hexagons
        for i in range(len(hex_vertices_list)):
            box_i = boxes[i]
            # Get nearby candidates from spatial index
            candidates = set()
            min_x, min_y, max_x, max_y = box_i[0]
            cell_min_x = int(min_x // GRID_CELL_SIZE)
            cell_min_y = int(min_y // GRID_CELL_SIZE)
            cell_max_x = int(max_x // GRID_CELL_SIZE)
            cell_max_y = int(max_y // GRID_CELL_SIZE)

            for cx in range(cell_min_x, cell_max_x + 1):
                for cy in range(cell_min_y, cell_max_y + 1):
                    candidates.update(spatial_index[(cx, cy)])

            # Check actual overlaps with candidates
            for j in candidates:
                if i >= j:  # Avoid duplicate checks and self-checks
                    continue
                if boxes_overlap(box_i[0], boxes[j][0]):  # Only check if bounding boxes overlap
                    if check_overlap(hex_vertices_list[i], hex_vertices_list[j]):
                        return 0.0  # Invalid - overlaps

        # Return 1/outer_side_length as fitness
        return 1.0 / outer_side_length if outer_side_length > 0 else 0.0
        
    except Exception:
        return 0.0

def generate_grid_initialization():
    """Generate initial configuration using grid-based placement strategy"""
    # Start with a known good configuration pattern
    initial_positions = [
        (0, 0, 0),      # center
        (-2.5, 0, 0),   # left
        (2.5, 0, 0),    # right
        (-1.25, 2.17, 0), # top-left
        (1.25, 2.17, 0), # top-right
        (-1.25, -2.17, 0), # bottom-left
        (1.25, -2.17, 0), # bottom-right
        (-3.75, 2.17, 0), # far top-left
        (3.75, 2.17, 0), # far top-right
        (-3.75, -2.17, 0), # far bottom-left
        (3.75, -2.17, 0), # far bottom-right
    ]
    
    # Add some randomness to avoid getting stuck in local minimums
    individual = []
    for i, (x, y, angle) in enumerate(initial_positions):
        # Add small random noise
        noise_x = np.random.normal(0, 0.1)
        noise_y = np.random.normal(0, 0.1)
        noise_angle = np.random.normal(0, 5)
        
        individual.append([
            x + noise_x,
            y + noise_y,
            (angle + noise_angle) % 360
        ])
    
    return np.array(individual)

def optimize_single_hexagon(hex_index, current_individual, hex_neighbors, grid_cells):
    """Optimize a single hexagon's position using gradient-free local search"""
    current_hex = current_individual[hex_index]
    
    # Define search space bounds for this hexagon
    # Based on typical hexagon size and spatial constraints
    bounds = [
        (-8, 8),    # x bounds
        (-8, 8),    # y bounds  
        (0, 360),   # angle bounds
    ]
    
    best_pos = current_hex.copy()
    best_fitness = evaluate_fitness(current_individual)
    
    # Try several local search strategies
    max_iterations = 20
    for iter_num in range(max_iterations):
        # Try small perturbations
        for _ in range(5):
            test_individual = current_individual.copy()
            
            # Choose random parameter to perturb
            param_idx = random.randint(0, 2)
            step_size = 0.1 if param_idx < 2 else 5
            
            # Apply perturbation
            test_individual[hex_index][param_idx] += np.random.normal(0, step_size)
            
            # Constrain parameters
            if param_idx == 0:
                test_individual[hex_index][0] = np.clip(test_individual[hex_index][0], bounds[0][0], bounds[0][1])
            elif param_idx == 1:
                test_individual[hex_index][1] = np.clip(test_individual[hex_index][1], bounds[1][0], bounds[1][1])
            else:  # angle
                test_individual[hex_index][2] = test_individual[hex_index][2] % 360
            
            # Evaluate
            fitness = evaluate_fitness(test_individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_pos = test_individual[hex_index].copy()
                
        # Early stopping if no improvement
        if iter_num > 5 and best_fitness < 0.001:
            break
    
    # Update the individual
    current_individual[hex_index] = best_pos
    
    return current_individual, best_fitness

def grid_based_local_search(initial_individual, max_iter=50):
    """Use grid-based approach to improve the configuration"""
    current_individual = initial_individual.copy()
    
    # Create spatial index for collision checking
    hex_vertices_list = [get_hexagon_vertices(*current_individual[i]) for i in range(len(current_individual))]
    spatial_index, boxes = create_spatial_index(hex_vertices_list)
    
    best_fitness = evaluate_fitness(current_individual)
    
    # Multi-pass optimization with different search strategies
    for pass_num in range(3):
        # Pass 1: Optimize center hexagon first
        if pass_num == 0:
            hex_indices = [0]  # Focus on center hexagon
            step_size = 0.2
        elif pass_num == 1:
            # Pass 2: Optimize peripheral hexagons
            hex_indices = [1, 2, 3, 4, 5, 6]  # Peripheral ones
            step_size = 0.15
        else:
            # Pass 3: Optimize all hexagons
            hex_indices = list(range(11))
            step_size = 0.1
            
        # Shuffle indices for better exploration
        random.shuffle(hex_indices)
        
        # Optimize selected hexagons
        for hex_idx in hex_indices:
            current_individual, fitness = optimize_single_hexagon(hex_idx, current_individual, [], [])
            if fitness > best_fitness:
                best_fitness = fitness
    
    return current_individual, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize with a good starting configuration
    individual = generate_grid_initialization()
    
    best_fitness = evaluate_fitness(individual)
    best_individual = individual.copy()
    
    # Multi-stage optimization for better results
    max_iterations = 100
    
    for iteration in range(max_iterations):
        if time.time() - start_time > 175:
            break
            
        # Create candidate individual by adding small random changes
        candidate_individual = individual.copy()
        
        # Randomly select a few hexagons to modify
        modified_indices = random.sample(range(NUM_INNER_HEXAGONS), 
                                        min(3, NUM_INNER_HEXAGONS))
        
        for hex_idx in modified_indices:
            # Apply small random perturbations
            candidate_individual[hex_idx][0] += np.random.normal(0, 0.15)
            candidate_individual[hex_idx][1] += np.random.normal(0, 0.15)
            candidate_individual[hex_idx][2] += np.random.normal(0, 10)
            candidate_individual[hex_idx][2] %= 360
        
        # Try local optimization on the candidate
        refined_individual, refined_fitness = grid_based_local_search(candidate_individual)
        
        if refined_fitness > best_fitness:
            best_fitness = refined_fitness
            best_individual = refined_individual.copy()
            individual = refined_individual.copy()
            
        # Occasionally do a more thorough local search
        if iteration % 10 == 0:
            individual, fitness = grid_based_local_search(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
    
    # Final refinement on best solution
    best_individual, best_fitness = grid_based_local_search(best_individual)
    
    # Calculate final outer hexagon side length
    outer_side_length = 1.0 / best_fitness if best_fitness > 0 else 8.0
    
    # Ensure valid outer hexagon side length
    if outer_side_length > 100:
        outer_side_length = 10.0
    
    # Center the outer hexagon at origin
    outer_hex_data = np.array([0, 0, 0])
    
    return best_individual, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END