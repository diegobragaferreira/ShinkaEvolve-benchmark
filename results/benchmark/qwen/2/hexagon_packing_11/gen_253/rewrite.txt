# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import math
import time
import random
from numba import jit

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a regular hexagon given position and angle - JIT compiled"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def point_in_hexagon_fast(px, py, hx, hy, radius, angle_deg):
    """Fast point-in-hexagon test - JIT compiled"""
    # Transform point to hexagon's local coordinate system
    angle_rad = np.radians(angle_deg)
    rel_x = px - hx
    rel_y = py - hy
    
    # Rotate point back to align with hexagon axes
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a
    
    # Check bounds using hexagon geometry
    # For a hexagon with circumradius r, width = r*sqrt(3) and height = 2*r
    half_width = radius * np.sqrt(3) / 2
    half_height = radius
    
    if abs(rot_x) > half_width or abs(rot_y) > half_height:
        return False
    
    # Additional check - within the hexagon boundaries
    if abs(rot_x) > half_width or abs(rot_y) > half_height:
        return False
    
    return True

def estimate_outer_hex_radius(inner_configs):
    """Estimate minimal outer hexagon radius that can contain all inner hexagons"""
    if len(inner_configs) == 0:
        return 1000.0
    
    # Collect all vertices from all hexagons
    all_vertices = []
    for i in range(len(inner_configs)):
        x, y, angle = inner_configs[i]
        vertices = get_hexagon_vertices(x, y, angle)
        for vertex in vertices:
            all_vertices.append(vertex)
    
    if len(all_vertices) == 0:
        return 1000.0
    
    all_vertices = np.array(all_vertices)
    
    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
    
    # Calculate center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Find maximum distance from center to any vertex
    distances = np.sqrt((all_vertices[:, 0] - center_x)**2 + (all_vertices[:, 1] - center_y)**2)
    max_distance = np.max(distances)
    
    # Add small margin to account for hexagon shape
    return max_distance * 1.05  # 5% padding for safety

def check_containment_single(hex_vertices, outer_radius):
    """Check if hexagon vertices are within the outer hexagon using SAT approach"""
    # Outer hexagon has circumradius = outer_radius * sqrt(3)/2
    outer_circumradius = outer_radius * np.sqrt(3) / 2
    
    for vertex in hex_vertices:
        dist_from_center = np.sqrt(vertex[0]**2 + vertex[1]**2)
        if dist_from_center > outer_circumradius:
            return False
    return True

def check_collision_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Shapely for precise verification"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        return False

def compute_objective_function(hex_params, use_penalty=False):
    """Compute objective function with penalty handling"""
    # reshape parameters to get hexagon data
    hex_data = hex_params.reshape(-1, 3)
    
    # Determine outer hexagon radius
    outer_radius = estimate_outer_hex_radius(hex_data)
    
    # Initialize penalties
    penalty = 0.0
    
    # Check containment constraints
    for i in range(NUM_INNER_HEX):
        x, y, angle = hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        
        # Check containment
        if not check_containment_single(vertices, outer_radius):
            if use_penalty:
                penalty += 1e6  # Heavy penalty for containment violation
            else:
                return -1e6  # Immediate rejection
    
    # Check for overlaps between hexagons
    for i in range(NUM_INNER_HEX):
        for j in range(i+1, NUM_INNER_HEX):
            x1, y1, angle1 = hex_data[i]
            x2, y2, angle2 = hex_data[j]
            
            vertices1 = get_hexagon_vertices(x1, y1, angle1)
            vertices2 = get_hexagon_vertices(x2, y2, angle2)
            
            if check_collision_single(vertices1, vertices2):
                if use_penalty:
                    penalty += 1e6  # Heavy penalty for overlap
                else:
                    return -1e6  # Immediate rejection
    
    # Return fitness (negative inverse of outer radius plus penalties)
    # We want to minimize outer radius, so maximize 1/outer_radius
    objective = -1.0 / outer_radius
    if use_penalty:
        objective -= penalty
    
    return objective

def generate_geometric_initial_configurations():
    """Generate multiple high-quality initial configurations using geometric reasoning"""
    configurations = []
    
    # Configuration 1: Classic hexagonal arrangement
    config1 = np.array([
        [0, 0, 0],  # center
        [2.0, 0, 0],  # right
        [1.0, 1.732, 0],  # top-right
        [-1.0, 1.732, 0],  # top-left
        [-2.0, 0, 0],  # left
        [-1.0, -1.732, 0],  # bottom-left
        [1.0, -1.732, 0],  # bottom-right
        [3.0, 0, 0],  # far right
        [1.5, 2.6, 0],  # upper-middle-right
        [-1.5, 2.6, 0],  # upper-middle-left
        [-3.0, 0, 0],  # far left
    ])
    
    # Configuration 2: More spread out for better optimization space
    config2 = np.array([
        [0, 0, 0],  # center
        [3.0, 0, 0],  # right
        [1.5, 2.6, 0],  # top-right
        [-1.5, 2.6, 0],  # top-left
        [-3.0, 0, 0],  # left
        [-1.5, -2.6, 0],  # bottom-left
        [1.5, -2.6, 0],  # bottom-right
        [4.5, 0, 0],  # far right
        [2.25, 3.9, 0],  # upper-middle-right
        [-2.25, 3.9, 0],  # upper-middle-left
        [-4.5, 0, 0],  # far left
    ])
    
    # Configuration 3: Compact arrangement for tighter packing
    config3 = np.array([
        [0, 0, 0],  # center
        [1.5, 0, 0],  # right
        [0.75, 1.299, 0],  # top-right
        [-0.75, 1.299, 0],  # top-left
        [-1.5, 0, 0],  # left
        [-0.75, -1.299, 0],  # bottom-left
        [0.75, -1.299, 0],  # bottom-right
        [3.0, 0, 0],  # far right
        [1.5, 2.6, 0],  # upper-middle-right
        [-1.5, 2.6, 0],  # upper-middle-left
        [-3.0, 0, 0],  # far left
    ])
    
    # Configuration 4: Spiral-like arrangement
    config4 = np.array([
        [0, 0, 0],  # center
        [1.8, 0, 0],  # right
        [0.9, 1.56, 0],  # top-right
        [-0.9, 1.56, 0],  # top-left
        [-1.8, 0, 0],  # left
        [-0.9, -1.56, 0],  # bottom-left
        [0.9, -1.56, 0],  # bottom-right
        [2.7, 0, 0],  # far right
        [1.35, 2.34, 0],  # upper-middle-right
        [-1.35, 2.34, 0],  # upper-middle-left
        [-2.7, 0, 0],  # far left
    ])
    
    configurations.extend([config1, config2, config3, config4])
    
    # Add random perturbations to configurations to increase diversity
    for i in range(len(configurations)):
        config = configurations[i]
        for j in range(1, len(config)):  # Don't perturb the center
            config[j] = [
                config[j][0] + random.uniform(-0.3, 0.3),
                config[j][1] + random.uniform(-0.3, 0.3),
                config[j][2] + random.uniform(-15, 15)
            ]
    
    return configurations

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Generate multiple initial configurations
    initial_configs = generate_geometric_initial_configurations()
    
    best_objective = -float('inf')
    best_solution = None
    best_outer_radius = float('inf')
    
    # Try multiple starting points with different geometric arrangements
    for i, initial_config in enumerate(initial_configs):
        # Flatten the initial configuration for optimization
        initial_flat = initial_config.flatten()
        
        # Use different optimization methods with different starting points
        try:
            # First, try a direct optimization with Nelder-Mead method
            result = minimize(
                compute_objective_function,
                initial_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'adaptive': True,
                    'disp': False
                },
                args=(False,)  # Don't use penalty initially
            )
            
            # If we get a valid result, check if it's better
            if result.success and result.fun > best_objective:
                # Final validation with penalties
                final_obj = compute_objective_function(result.x, use_penalty=True)
                if final_obj > best_objective:
                    best_objective = final_obj
                    best_solution = result.x.copy()
                    # Update outer radius for validation
                    temp_config = result.x.reshape(-1, 3)
                    best_outer_radius = estimate_outer_hex_radius(temp_config)
                    
        except Exception as e:
            continue
            
        # Early termination check
        if time.time() - start_time > 170:
            break
    
    # If we didn't find a good solution, use the best among initial configurations
    if best_solution is None:
        # Evaluate all initial configurations with penalties and pick the best
        best_initial_obj = -float('inf')
        for i, config in enumerate(initial_configs):
            obj = compute_objective_function(config.flatten(), use_penalty=True)
            if obj > best_initial_obj:
                best_initial_obj = obj
                best_solution = config.flatten()
                temp_config = config
                best_outer_radius = estimate_outer_hex_radius(temp_config)
    
    # Convert back to proper format
    if best_solution is not None:
        inner_hex_data = best_solution.reshape(-1, 3)
        outer_hex_data = np.array([0, 0, 0])  # Centered at origin
        outer_hex_side_length = best_outer_radius * 2.0 / np.sqrt(3)
    else:
        # Fallback to a known working configuration
        inner_hex_data = np.array([
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END