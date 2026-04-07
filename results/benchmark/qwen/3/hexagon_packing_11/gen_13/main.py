# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import math
import random
import time

def hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon"""
    rotation_rad = math.radians(rotation_degrees)
    vertices = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def hexagon_polygon(center_x, center_y, side_length, rotation_degrees):
    """Create a shapely polygon for a hexagon"""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    return hex1_poly.intersects(hex2_poly)

def calculate_outer_hexagon_radius(inner_hex_data, outer_hex_side_length):
    """Calculate the actual outer hexagon radius needed"""
    # Create a reference outer hexagon (centered at origin)
    outer_hex = hexagon_polygon(0, 0, outer_hex_side_length, 0)
    
    # Check how well the inner hexagons fit
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_hex = hexagon_polygon(center_x, center_y, 1, angle)
        
        # Get the furthest vertex from center
        vertices = hexagon_vertices(center_x, center_y, 1, angle)
        for vx, vy in vertices:
            dist = math.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)
    
    return max_dist

def evaluate_configuration(inner_hex_data, outer_hex_side_length):
    """Evaluate whether configuration meets constraints and return penalty"""
    # Create outer hexagon
    outer_hex = hexagon_polygon(0, 0, outer_hex_side_length, 0)
    
    # Check containment and overlaps for all hexagons
    total_penalty = 0
    
    # Check overlaps between all pairs
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center_x1, center_y1, angle1 = inner_hex_data[i]
            center_x2, center_y2, angle2 = inner_hex_data[j]
            
            hex1 = hexagon_polygon(center_x1, center_y1, 1, angle1)
            hex2 = hexagon_polygon(center_x2, center_y2, 1, angle2)
            
            if check_overlap(hex1, hex2):
                # Overlap penalty - scale with overlap area
                intersection = hex1.intersection(hex2)
                if intersection.geom_type == 'Polygon':
                    overlap_area = intersection.area
                    total_penalty += overlap_area * 10000
                else:
                    total_penalty += 10000
    
    # Check containment
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_hex = hexagon_polygon(center_x, center_y, 1, angle)
        if not check_containment(inner_hex, outer_hex):
            # Penalty for containment violation
            total_penalty += 10000
    
    return total_penalty

def generate_neighbor_config(current_config, step_size=0.5):
    """Generate a neighbor configuration"""
    new_config = current_config.copy()
    
    # Randomly select one hexagon to modify
    hex_idx = random.randint(0, len(new_config)-1)
    
    # Modify either position or rotation
    param_type = random.choice(['position', 'rotation'])
    
    if param_type == 'position':
        # Modify position
        new_config[hex_idx][0] += random.uniform(-step_size, step_size)
        new_config[hex_idx][1] += random.uniform(-step_size, step_size)
    else:
        # Modify rotation
        new_config[hex_idx][2] += random.uniform(-15, 15)
        # Normalize angle to [0, 360)
        new_config[hex_idx][2] %= 360
    
    return new_config

def simulated_annealing():
    """Use simulated annealing to optimize the hexagon arrangement"""
    
    # Initial configuration - attempt to create a good starting point
    initial_config = np.array([
        [0, 0, 0],      # center
        [-2.0, 0, 0],   # left
        [2.0, 0, 0],    # right
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [-1.5, 1.5, 0], # top-left
        [1.5, 1.5, 0],  # top-right
        [-1.5, -1.5, 0], # bottom-left
        [1.5, -1.5, 0], # bottom-right
        [-2.5, 1.0, 0], # offset top
        [2.5, 1.0, 0],  # offset top
    ])
    
    current_config = initial_config.copy()
    current_radius = 10.0  # Start with a reasonable upper bound
    
    # Annealing parameters
    temp = 1000.0
    min_temp = 0.1
    cooling_rate = 0.995
    max_iter = 5000
    
    best_config = current_config.copy()
    best_radius = current_radius
    best_score = float('inf')
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start_time > 170:  # Leave some buffer for finalization
            break
            
        # Generate neighbor configuration
        neighbor_config = generate_neighbor_config(current_config)
        
        # Calculate scores
        current_penalty = evaluate_configuration(current_config, current_radius)
        neighbor_penalty = evaluate_configuration(neighbor_config, current_radius)
        
        # Accept or reject based on simulated annealing criteria
        if neighbor_penalty < current_penalty:
            current_config = neighbor_config
            if neighbor_penalty < best_score:
                best_config = neighbor_config.copy()
                best_score = neighbor_penalty
        else:
            # Metropolis criterion
            delta = neighbor_penalty - current_penalty
            if random.random() < math.exp(-delta / temp):
                current_config = neighbor_config
        
        # Gradually reduce temperature
        temp *= cooling_rate
        
        # Occasionally reduce the outer hexagon radius if valid
        if iteration % 100 == 0 and iteration > 0:
            # Try to decrease outer radius while still satisfying constraints
            test_radius = current_radius - 0.1
            penalty = evaluate_configuration(current_config, test_radius)
            if penalty == 0:  # Feasible solution
                current_radius = test_radius
                
    return best_config, current_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Run optimization
    inner_hex_data, outer_hex_side_length = simulated_annealing()
    
    # Final verification
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
