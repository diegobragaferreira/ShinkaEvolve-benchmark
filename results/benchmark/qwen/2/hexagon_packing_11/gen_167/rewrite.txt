# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import time
from numba import jit, prange
import math
import random
from collections import deque

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6
MAX_ITERATIONS = 300

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
def point_in_hexagon_fast(point_x, point_y, hex_center_x, hex_center_y, hex_radius, angle_deg):
    """Fast point-in-hexagon test - JIT compiled"""
    # Transform point to hexagon's local coordinate system
    angle_rad = np.radians(angle_deg)
    rel_x = point_x - hex_center_x
    rel_y = point_y - hex_center_y
    
    # Rotate point back to align with hexagon axes
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a
    
    # Simplified check based on hexagon geometry
    # Width along x-axis: 2 * radius * cos(pi/6) = radius * sqrt(3)
    # Height along y-axis: 2 * radius * sin(pi/3) = radius * sqrt(3)
    width_half = hex_radius * np.sqrt(3) / 2
    height_half = hex_radius * np.sqrt(3) / 2
    
    if abs(rot_x) > width_half or abs(rot_y) > height_half:
        return False
        
    # More precise check 
    if abs(rot_x) > width_half:
        return False
        
    # Further refined check using hexagon boundaries
    # For a hexagon aligned with axes, we can do a more precise check
    # The relationship between x and y in a hexagon is such that:
    # |y| <= hex_radius * 0.5 + (width_half - |x|) * tan(pi/6) = hex_radius * 0.5 + (width_half - |x|) * sqrt(3)/3
    if abs(rot_y) > hex_radius * 0.5 + (width_half - abs(rot_x)) * 0.5773502691896257:  # sqrt(3)/3
        return False
    
    return True

@jit(nopython=True)
def get_edges(vertices):
    """Get edges from vertices"""
    edges = []
    n = len(vertices)
    for i in range(n):
        edges.append(vertices[i] - vertices[(i+1)%n])
    return np.array(edges)

@jit(nopython=True)
def project_polygon_onto_axis(vertices, axis):
    """Project polygon onto axis and return min/max projections"""
    projections = []
    for vertex in vertices:
        proj = vertex[0] * axis[0] + vertex[1] * axis[1]
        projections.append(proj)
    
    min_proj = projections[0]
    max_proj = projections[0]
    for p in projections:
        if p < min_proj:
            min_proj = p
        if p > max_proj:
            max_proj = p
    
    return min_proj, max_proj

@jit(nopython=True)
def sat_collision_check(hex1_vertices, hex2_vertices):
    """SAT-based overlap detection - much faster than Shapely"""
    # Get edges for both polygons
    edges1 = get_edges(hex1_vertices)
    edges2 = get_edges(hex2_vertices)
    
    # Get normals to all edges (perpendicular vectors)
    normals1 = []
    normals2 = []
    
    for edge in edges1:
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm_len = np.sqrt(normal[0]**2 + normal[1]**2)
        if norm_len > 1e-10:
            normal = normal / norm_len
        normals1.append(normal)
    
    for edge in edges2:
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm_len = np.sqrt(normal[0]**2 + normal[1]**2)
        if norm_len > 1e-10:
            normal = normal / norm_len
        normals2.append(normal)
    
    # Test all axes
    all_normals = normals1 + normals2
    
    for axis in all_normals:
        min1, max1 = project_polygon_onto_axis(hex1_vertices, axis)
        min2, max2 = project_polygon_onto_axis(hex2_vertices, axis)
        
        # Check for overlap
        if max1 < min2 or max2 < min1:
            return False  # No overlap along this axis
    
    return True  # Overlap detected

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1000.0
    
    max_distance = 0.0
    center_x, center_y = outer_hex_center
    
    # For each inner hexagon, check all 6 vertices
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        
        # Calculate distance from center to each vertex
        for vertex in vertices:
            distance = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
            max_distance = max(max_distance, distance)
    
    # Account for hexagon radius
    # The outer hexagon needs to be large enough so that any vertex of inner hexagons 
    # lies inside the outer hexagon
    return max_distance * 2.0 / np.sqrt(3)  # Convert circumradius to side length

def check_containment_fast(hex_vertices, outer_center=(0, 0), outer_radius=1000.0):
    """Fast containment check"""
    outer_center_x, outer_center_y = outer_center
    # Check if all vertices are within the outer hexagon
    # Outer hexagon circumscribed circle has radius = outer_radius * sqrt(3)/2
    outer_circumradius = outer_radius * np.sqrt(3) / 2
    
    for vertex in hex_vertices:
        dist_from_center = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
        if dist_from_center > outer_circumradius:
            return False
    return True

def evaluate_fitness_hexagon(inner_hex_data, outer_side_length):
    """Evaluate fitness with improved error handling"""
    try:
        # Calculate required outer hex side length
        computed_outer_side = calculate_outer_hex_side_length(inner_hex_data)
        
        # Initialize penalty
        penalty = 0.0
        
        # Check containment constraints using faster methods
        outer_radius = outer_side_length * np.sqrt(3) / 2  # Circumradius
        
        # Check each hexagon for containment
        for i in range(NUM_INNER_HEX):
            x, y, angle = inner_hex_data[i]
            vertices = get_hexagon_vertices(x, y, angle)
            
            # Fast containment check
            if not check_containment_fast(vertices, (0, 0), outer_radius):
                penalty += 1000000.0  # Heavy penalty
        
        # Check for overlaps between hexagons using SAT
        overlap_pairs = 0
        for i in range(NUM_INNER_HEX):
            for j in range(i+1, NUM_INNER_HEX):
                x1, y1, angle1 = inner_hex_data[i]
                x2, y2, angle2 = inner_hex_data[j]
                
                vertices1 = get_hexagon_vertices(x1, y1, angle1)
                vertices2 = get_hexagon_vertices(x2, y2, angle2)
                
                # Use SAT-based check for efficiency
                if sat_collision_check(vertices1, vertices2):
                    penalty += 1000000.0  # Heavy penalty
                    overlap_pairs += 1
        
        # Fitness is negative inverse of side length plus penalties  
        # We want to minimize side length, so maximize 1/side_length
        fitness = -1.0 / outer_side_length
        if penalty > 0:
            fitness -= penalty  # Add penalty for constraint violations
            
        # Add bonus for valid solutions
        if penalty == 0 and overlap_pairs == 0:
            fitness += 1.0  # Small bonus for valid solutions
            
        return fitness, computed_outer_side
        
    except Exception as e:
        return -1000000.0, 1000.0

def generate_initial_geometric_configs():
    """Generate multiple geometrically motivated initial configurations"""
    configs = []
    
    # Configuration 1: Basic hexagonal arrangement
    base_positions = [
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
    ]
    configs.append(np.array(base_positions))
    
    # Configuration 2: Spiral arrangement
    spiral_positions = []
    for i in range(11):
        angle = i * 0.8
        radius = 0.25 * i
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        spiral_positions.append([x, y, 0])
    configs.append(np.array(spiral_positions))
    
    # Configuration 3: Concentric ring
    ring_positions = [[0, 0, 0]]  # center
    ring_radius = 1.8
    for i in range(1, 11):
        angle = (i - 1) * 2 * np.pi / 10
        x = ring_radius * np.cos(angle)
        y = ring_radius * np.sin(angle)
        ring_positions.append([x, y, 0])
    configs.append(np.array(ring_positions))
    
    # Configuration 4: Grid-like with symmetry
    grid_positions = []
    # Center
    grid_positions.append([0, 0, 0])
    # 4 corners
    for i in range(4):
        angle = i * np.pi / 2
        x = 2.5 * np.cos(angle)
        y = 2.5 * np.sin(angle)
        grid_positions.append([x, y, 0])
    # 4 midpoints
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        x = 1.5 * np.cos(angle)
        y = 1.5 * np.sin(angle)
        grid_positions.append([x, y, 0])
    # 2 additional positions
    grid_positions.append([2.0, 0, 0])
    grid_positions.append([0, 2.0, 0])
    configs.append(np.array(grid_positions))
    
    return configs

def progressive_local_search(initial_config, max_iter=200):
    """Perform progressive local search with increasing resolution"""
    current_config = initial_config.copy()
    best_fitness = float('-inf')
    best_solution = current_config.copy()
    
    # Progressive refinement levels
    levels = [
        {"step_size": 0.5, "max_iter": 50},   # Coarse
        {"step_size": 0.2, "max_iter": 75},   # Medium
        {"step_size": 0.05, "max_iter": 75}   # Fine
    ]
    
    for level in levels:
        step_size = level["step_size"]
        max_iter_level = level["max_iter"]
        
        # Local optimization within this level
        for _ in range(max_iter_level):
            # Generate candidate by perturbing current solution
            candidate = current_config.copy()
            
            # Perturb randomly selected hexagon
            hex_idx = random.randint(0, NUM_INNER_HEX - 1)
            
            # Apply small perturbation to position
            candidate[hex_idx][0] += random.uniform(-step_size, step_size)
            candidate[hex_idx][1] += random.uniform(-step_size, step_size)
            candidate[hex_idx][2] += random.uniform(-5, 5)  # rotation
            candidate[hex_idx][2] = candidate[hex_idx][2] % 360
            
            # Evaluate candidate
            fit, side_length = evaluate_fitness_hexagon(candidate, calculate_outer_hex_side_length(candidate))
            
            if fit > best_fitness:
                best_fitness = fit
                best_solution = candidate.copy()
                current_config = candidate.copy()
            else:
                # Accept with a low probability to escape local minima
                if random.random() < 0.05:
                    current_config = candidate.copy()
    
    return best_solution, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    best_fitness = float('-inf')
    best_config = None
    best_side_length = float('inf')
    
    # Generate diverse starting configurations
    initial_configs = generate_initial_geometric_configs()
    
    # Try multiple initial configurations
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > 170:  # Time budget
            break
            
        # First perform progressive local search on each initial config
        config, fitness = progressive_local_search(config, max_iter=100)
        
        # Calculate the actual outer hexagon side length
        side_length = calculate_outer_hex_side_length(config)
        
        # Evaluate final fitness
        eval_fitness, computed_side = evaluate_fitness_hexagon(config, side_length)
        
        if eval_fitness > best_fitness:
            best_fitness = eval_fitness
            best_config = config.copy()
            best_side_length = computed_side
    
    # If we still haven't found a good solution, fallback to the simpler approach
    if best_config is None or best_fitness < -10000:
        # Use the simplest valid configuration
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
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8.0  # large enough to contain all inner hexagons
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Return the best solution found
    outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin
    return best_config, outer_hex_data, best_side_length

# EVOLVE-BLOCK-END