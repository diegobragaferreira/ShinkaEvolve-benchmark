# EVOLVE-BLOCK-START
import numpy as np
import math
from shapely.geometry import Polygon, Point
from scipy.optimize import minimize
import itertools
import time

def create_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Create vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon with buffer"""
    outer_polygon = Polygon(outer_hex_vertices)
    buffered_outer = outer_polygon.buffer(1e-6)
    for vertex in hex_vertices:
        if not buffered_outer.contains(Point(vertex[0], vertex[1])):
            return False
    return True

def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast bounding box overlap check"""
    x1 = [v[0] for v in hex1_vertices]
    y1 = [v[1] for v in hex1_vertices]
    x2 = [v[0] for v in hex2_vertices]
    y2 = [v[1] for v in hex2_vertices]
    
    min_x1, max_x1 = min(x1), max(x1)
    min_y1, max_y1 = min(y1), max(y1)
    min_x2, max_x2 = min(x2), max(x2)
    min_y2, max_y2 = min(y2), max(y2)
    
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with buffer"""
    if not check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
        return False
        
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    buffered_poly1 = poly1.buffer(1e-6)
    buffered_poly2 = poly2.buffer(1e-6)
    return buffered_poly1.intersects(buffered_poly2)

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length that contains all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1000

    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    if not all_vertices:
        return 1000

    # Calculate tight bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Calculate diagonal of bounding box
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    # For a hexagon, side length should accommodate the diagonal with margin
    diagonal = math.sqrt(bbox_width**2 + bbox_height**2)
    side_length = diagonal / math.sqrt(3)
    side_length *= 1.1  # Add margin
    
    return side_length

def generate_grid_positions():
    """Generate a systematic grid of positions for hexagon centers"""
    # Create a coarse grid around the origin
    positions = []
    
    # Hexagon side length = 1, so the distance between centers of adjacent hexagons is 2
    # We'll sample around a reasonable area including the center
    step = 1.5  # Grid step size
    range_limit = 4.0
    
    for x in np.arange(-range_limit, range_limit + step, step):
        for y in np.arange(-range_limit, range_limit + step, step):
            # Only include positions that are reasonably spaced for 11 hexagons
            if abs(x) <= range_limit and abs(y) <= range_limit:
                positions.append((x, y))
    
    return positions

def generate_symmetric_configurations():
    """Generate various symmetric configurations to serve as starting points"""
    configs = []
    
    # Configuration 1: Central hexagon with ring
    config1 = [
        [0.0, 0.0, 0.0],      # center
        [-2.0, 0.0, 0.0],     # left
        [2.0, 0.0, 0.0],      # right
        [0.0, 2.0, 0.0],      # top
        [0.0, -2.0, 0.0],     # bottom
        [1.73, 1.0, 0.0],     # top-right
        [-1.73, 1.0, 0.0],    # top-left
        [1.73, -1.0, 0.0],    # bottom-right
        [-1.73, -1.0, 0.0],   # bottom-left
        [3.46, 0.0, 0.0],     # far right
        [0.0, 3.46, 0.0],     # far top
    ]
    configs.append(np.array(config1))
    
    # Configuration 2: Star pattern
    config2 = [
        [0.0, 0.0, 0.0],      # center
        [-1.73, 0.0, 0.0],    # left  
        [1.73, 0.0, 0.0],     # right
        [0.0, 1.73, 0.0],     # top
        [0.0, -1.73, 0.0],    # bottom
        [1.5, 1.5, 0.0],      # top-right
        [-1.5, 1.5, 0.0],     # top-left
        [1.5, -1.5, 0.0],     # bottom-right
        [-1.5, -1.5, 0.0],    # bottom-left
        [3.0, 0.0, 0.0],      # far right
        [0.0, 3.0, 0.0],      # far top
    ]
    configs.append(np.array(config2))
    
    # Configuration 3: Spiral pattern
    config3 = [
        [0.0, 0.0, 0.0],      # center
        [-1.5, 0.0, 0.0],     # left
        [1.5, 0.0, 0.0],      # right
        [0.0, 1.5, 0.0],      # top
        [0.0, -1.5, 0.0],     # bottom
        [1.25, 1.25, 0.0],    # top-right
        [-1.25, 1.25, 0.0],   # top-left
        [1.25, -1.25, 0.0],   # bottom-right
        [-1.25, -1.25, 0.0],  # bottom-left
        [2.5, 0.0, 0.0],      # far right
        [0.0, 2.5, 0.0],      # far top
    ]
    configs.append(np.array(config3))
    
    return configs

def optimize_config_local(config):
    """Apply local optimization to a configuration using scipy minimize"""
    def objective(x_flat):
        # Reshape flat array back to hex data
        hex_data = x_flat.reshape(-1, 3)
        
        # Calculate outer hexagon side length
        outer_side_length = calculate_outer_hex_side_length(hex_data)
        
        # Check constraints and apply penalties
        outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)
        total_penalty = 0
        
        # Check containment
        for i in range(len(hex_data)):
            center_x, center_y, angle = hex_data[i]
            inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
            if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                total_penalty += 10000
                
        # Check overlaps
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]
                
                hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)
                
                if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    total_penalty += 10000
                    
        # Return fitness (inverse of outer hex side length + penalties)
        if total_penalty > 0:
            return 1.0 / outer_side_length - total_penalty
        return 1.0 / outer_side_length
    
    # Flatten the configuration
    x0 = config.flatten()
    
    # Define bounds for optimization
    bounds = []
    for i in range(len(x0)):
        if i % 3 == 0:  # x coordinate
            bounds.append((-6.0, 6.0))
        elif i % 3 == 1:  # y coordinate
            bounds.append((-6.0, 6.0))
        else:  # angle
            bounds.append((0.0, 360.0))
    
    try:
        # Apply local refinement
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
        refined_solution = result.x.reshape(-1, 3)
        refined_outer_side_length = calculate_outer_hex_side_length(refined_solution)
        refined_fitness = 1.0 / refined_outer_side_length
        return (refined_solution, refined_outer_side_length, refined_fitness)
    except:
        # If optimization fails, return original
        outer_side_length = calculate_outer_hex_side_length(config)
        current_fitness = 1.0 / outer_side_length
        return (config, outer_side_length, current_fitness)

def grid_search_hexagon_pack():
    """Perform grid-based search for optimal hexagon packing"""
    # Generate reference configurations
    configs = generate_symmetric_configurations()
    
    best_fitness = -float('inf')
    best_config = None
    best_outer_side_length = float('inf')
    
    # Try different symmetric patterns
    for i, config in enumerate(configs):
        try:
            # Local optimization on each configuration
            refined_config, outer_side_length, fitness = optimize_config_local(config)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = refined_config
                best_outer_side_length = outer_side_length
                
        except Exception as e:
            continue
    
    # Additional refinement with grid sampling
    positions = generate_grid_positions()
    
    # Sample combinations of positions for 11 hexagons
    # Try various combinations of 11 positions from our grid
    best_local_fitness = best_fitness
    best_local_config = best_config
    
    # Try different subsets of positions
    for subset_size in [11, 12, 13]:  # Try slightly larger subsets to find better arrangements
        if subset_size > len(positions):
            continue
            
        for combo in itertools.combinations(positions, subset_size):
            # Create configuration with these positions
            temp_config = []
            for j, (x, y) in enumerate(combo):
                if j < 11:  # Take first 11 positions
                    temp_config.append([x, y, 0.0])
                else:
                    break
            
            if len(temp_config) < 11:
                continue
                
            # Make sure we have exactly 11 hexagons
            temp_config = temp_config[:11]
            
            # Try different rotations for some hexagons to find better arrangements
            for rot_combo in itertools.product([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], repeat=3):
                test_config = np.array(temp_config)
                # Apply rotations to some hexagons
                for k in range(3):  # Rotate first 3 hexagons
                    test_config[k][2] = rot_combo[k]
                
                try:
                    # Local optimization
                    refined_config, outer_side_length, fitness = optimize_config_local(test_config)
                    
                    if fitness > best_local_fitness:
                        best_local_fitness = fitness
                        best_local_config = refined_config
                        best_outer_side_length = outer_side_length
                        
                except Exception as e:
                    continue
    
    return best_local_config, best_outer_side_length, best_local_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a grid-based systematic search combined with local optimization.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Perform grid-based search
    best_config, best_outer_side_length, best_fitness = grid_search_hexagon_pack()
    
    # Prepare output in the required format
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    return best_config, outer_hex_data, best_outer_side_length

# EVOLVE-BLOCK-END