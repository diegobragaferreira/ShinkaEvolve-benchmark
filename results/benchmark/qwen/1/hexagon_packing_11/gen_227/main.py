# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import random
import time
from collections import defaultdict

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def get_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Get vertices of a regular hexagon with given parameters"""
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

def compute_hexagon_bounds(vertices):
    """Compute axis-aligned bounding box for hexagon vertices"""
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

def build_spatial_index(hex_vertices_list):
    """Build spatial index for fast collision detection"""
    # Create bounding boxes for all hexagons
    boxes = []
    for i, vertices in enumerate(hex_vertices_list):
        box = compute_hexagon_bounds(vertices)
        boxes.append((box, i))

    # Group boxes into grid cells for faster lookup
    cell_size = 2.0  # Based on hexagon diameter
    spatial_index = defaultdict(list)

    for box, idx in boxes:
        # Grid cell coordinates
        min_x, min_y, max_x, max_y = box
        cell_min_x = int(min_x // cell_size)
        cell_min_y = int(min_y // cell_size)
        cell_max_x = int(max_x // cell_size)
        cell_max_y = int(max_y // cell_size)

        # Add to all relevant grid cells
        for cx in range(cell_min_x, cell_max_x + 1):
            for cy in range(cell_min_y, cell_max_y + 1):
                spatial_index[(cx, cy)].append(idx)

    return spatial_index, boxes

def check_containment_and_overlap(hex_vertices_list, outer_side_length):
    """Check containment and overlap efficiently"""
    # Create outer hexagon polygon
    outer_vertices = get_hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment and overlap
    for i, vertices in enumerate(hex_vertices_list):
        hex_polygon = Polygon(vertices)
        
        # Check containment
        if not outer_polygon.contains(hex_polygon):
            return False, 0  # Invalid - not fully contained
            
        # Check overlaps with other hexagons using spatial index
        for j in range(i+1, len(hex_vertices_list)):
            vertices_j = hex_vertices_list[j]
            # Quick bounding box check first
            box_i = compute_hexagon_bounds(vertices)
            box_j = compute_hexagon_bounds(vertices_j)
            
            if boxes_overlap(box_i, box_j):
                # Full overlap check
                poly_i = Polygon(vertices)
                poly_j = Polygon(vertices_j)
                if poly_i.intersects(poly_j):
                    return False, 0  # Invalid - overlaps
    
    return True, 1  # Valid configuration

def calculate_outer_hex_side_length(hex_vertices_list):
    """Calculate minimum side length of outer hexagon needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for vertices in hex_vertices_list:
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 1.0
    
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

def compute_objective_and_gradients(params, n_hexagons=11):
    """Compute objective value and gradients for optimization"""
    # Reshape parameters
    positions_angles = params.reshape(n_hexagons, 3)
    
    # Extract positions and angles
    centers = positions_angles[:, :2]
    angles = positions_angles[:, 2]
    
    # Compute hexagon vertices
    hex_vertices_list = []
    for i in range(n_hexagons):
        center_x, center_y = centers[i]
        angle = angles[i]
        vertices = get_hexagon_vertices(center_x, center_y, angle)
        hex_vertices_list.append(vertices)
    
    # Calculate outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(hex_vertices_list)
    
    # Check constraints
    valid, penalty = check_containment_and_overlap(hex_vertices_list, outer_side_length)
    
    # Objective: maximize 1/outer_side_length (minimize outer_side_length)
    objective_value = 1.0 / outer_side_length if valid else -1000000.0
    
    # Return objective and placeholder gradients (optimization will handle actual gradients)
    return objective_value

def optimize_single_config(initial_params, n_hexagons=11):
    """Optimize a single configuration using gradient-based method"""
    # Define bounds for optimization
    bounds = []
    for i in range(n_hexagons):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    
    # Use L-BFGS-B optimization
    try:
        result = minimize(
            lambda x: -compute_objective_and_gradients(x, n_hexagons),  # Minimize negative for maximization
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.success:
            return result.x
    except:
        pass
    
    return initial_params

def generate_symmetric_configurations():
    """Generate known symmetric configurations that are likely to be good"""
    configs = []
    
    # Configuration 1: Central hexagon surrounded by 6 others in hexagonal pattern, plus 4 more
    config1 = [
        [0, 0, 0],  # center
        [-2, 0, 0],  # left
        [2, 0, 0],  # right  
        [0, 2, 0],  # top
        [0, -2, 0],  # bottom
        [-1, 1.732, 0],  # top-left
        [1, 1.732, 0],  # top-right
        [-1, -1.732, 0],  # bottom-left
        [1, -1.732, 0],  # bottom-right
        [-2.5, 0, 0],  # far left
        [2.5, 0, 0],  # far right
    ]
    configs.append(config1)
    
    # Configuration 2: Hexagonal pattern with more spacing
    config2 = [
        [0, 0, 0],
        [-2.5, 0, 0],
        [2.5, 0, 0],
        [0, 2.5, 0],
        [0, -2.5, 0],
        [-1.25, 2.17, 0],
        [1.25, 2.17, 0],
        [-1.25, -2.17, 0],
        [1.25, -2.17, 0],
        [-3.75, 2.17, 0],
        [3.75, 2.17, 0],
    ]
    configs.append(config2)
    
    # Configuration 3: Spiral-like arrangement
    config3 = [
        [0, 0, 0],
        [0, 2, 0],
        [1.732, 1, 0],
        [1.732, -1, 0],
        [0, -2, 0],
        [-1.732, -1, 0],
        [-1.732, 1, 0],
        [0, 3.5, 0],
        [3.031, 1.75, 0],
        [3.031, -1.75, 0],
        [0, -3.5, 0],
    ]
    configs.append(config3)
    
    return configs

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    start_time = time.time()
    max_time = 175  # Leave some margin for cleanup
    
    n = 11
    
    # Initialize best solution
    best_fitness = 0.0
    best_solution = None
    
    # Try multiple symmetric configurations first
    initial_configs = generate_symmetric_configurations()
    
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > max_time:
            break
            
        # Flatten configuration
        initial_params = np.array(config).flatten()
        optimized_params = optimize_single_config(initial_params, n)
        
        # Evaluate the optimized solution
        fitness = compute_objective_and_gradients(optimized_params, n)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = optimized_params.copy()
    
    # Multi-start optimization with random initializations
    if best_fitness < 0.2:  # If still too poor, perform additional optimizations
        num_starts = 10
        for i in range(num_starts):
            if time.time() - start_time > max_time:
                break
                
            # Random initialization
            random_config = []
            for j in range(n):
                center_x = random.uniform(-5, 5)
                center_y = random.uniform(-5, 5)
                angle = random.uniform(0, 360)
                random_config.append([center_x, center_y, angle])
            
            initial_params = np.array(random_config).flatten()
            optimized_params = optimize_single_config(initial_params, n)
            
            # Evaluate the optimized solution
            fitness = compute_objective_and_gradients(optimized_params, n)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = optimized_params.copy()
    
    # Final evaluation of best solution
    if best_solution is None:
        # Fallback to a simple symmetric arrangement
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
        ]).flatten()
    
    # Reshape final solution
    final_solution = best_solution.reshape(n, 3)
    final_fitness = compute_objective_and_gradients(final_solution.flatten(), n)
    
    # Calculate final outer hexagon side length
    hex_vertices_list = []
    for i in range(n):
        center_x, center_y, angle = final_solution[i]
        vertices = get_hexagon_vertices(center_x, center_y, angle)
        hex_vertices_list.append(vertices)
    
    outer_side_length = calculate_outer_hex_side_length(hex_vertices_list)
    
    # Center the outer hexagon at origin
    outer_hex_data = np.array([0, 0, 0])
    
    return final_solution, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END