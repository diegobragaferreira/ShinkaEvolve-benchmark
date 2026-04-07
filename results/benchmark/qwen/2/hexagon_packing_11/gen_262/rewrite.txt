# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from numba import jit, prange
import random
import time
import warnings
warnings.filterwarnings('ignore')

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
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Distance from point to line segment - JIT compiled"""
    line_x = x2 - x1
    line_y = y2 - y1
    point_x = px - x1
    point_y = py - y1
    
    line_len_sq = line_x * line_x + line_y * line_y
    if line_len_sq == 0.0:
        return np.sqrt(point_x * point_x + point_y * point_y)
    
    t = (point_x * line_x + point_y * line_y) / line_len_sq
    t = max(0.0, min(1.0, t))
    
    closest_x = x1 + t * line_x
    closest_y = y1 + t * line_y
    
    dx = px - closest_x
    dy = py - closest_y
    return np.sqrt(dx * dx + dy * dy)

@jit(nopython=True)
def hexagon_distance_fast(hex1_vertices, hex2_vertices):
    """Fast minimum distance between two hexagons - JIT compiled"""
    min_dist = 1e10
    
    # Vertex-to-edge distances
    for i in range(6):
        v1 = hex1_vertices[i]
        v2 = hex1_vertices[(i+1)%6]
        for j in range(6):
            v3 = hex2_vertices[j]
            v4 = hex2_vertices[(j+1)%6]
            dist = distance_point_to_line(v1[0], v1[1], v3[0], v3[1], v4[0], v4[1])
            min_dist = min(min_dist, dist)
    
    # Reverse direction
    for i in range(6):
        v1 = hex2_vertices[i]
        v2 = hex2_vertices[(i+1)%6]
        for j in range(6):
            v3 = hex1_vertices[j]
            v4 = hex1_vertices[(j+1)%6]
            dist = distance_point_to_line(v1[0], v1[1], v3[0], v3[1], v4[0], v4[1])
            min_dist = min(min_dist, dist)
    
    return min_dist

@jit(nopython=True)
def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap detection using distance threshold - JIT compiled"""
    min_dist = hexagon_distance_fast(hex1_vertices, hex2_vertices)
    # Overlap when distance is less than twice the radius (2.0)
    return min_dist < 1.999  # Slight tolerance for numerical errors

@jit(nopython=True)
def point_in_hexagon_fast(px, py, center_x, center_y, angle_deg, radius=1.0):
    """Fast point-in-hexagon test using numba - JIT compiled"""
    angle_rad = np.radians(angle_deg)
    rel_x = px - center_x
    rel_y = py - center_y
    
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a
    
    half_width = radius * np.sqrt(3) / 2
    max_y = radius * np.sqrt(3) / 2
    
    if abs(rot_x) > half_width or abs(rot_y) > max_y:
        return False
    
    return True

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length using geometric insights"""
    if len(inner_hex_data) == 0:
        return 1000.0
    
    # For 11 hexagons, we can estimate the bounding circle by calculating centroids
    # and finding maximum distance + some margin
    all_points = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        for vertex in vertices:
            all_points.append(vertex)
    
    if not all_points:
        return 1000.0
    
    all_points = np.array(all_points)
    # Calculate centroid
    centroid = np.mean(all_points, axis=0)
    
    # Calculate max distance from centroid to any vertex
    distances = np.sqrt(np.sum((all_points - centroid)**2, axis=1))
    max_distance = np.max(distances)
    
    # Convert to side length:
    # For a hexagon, if we know circumradius R, side length s = R
    # But we want to contain all vertices, so we need circumradius = max_distance * 1.05 for safety
    # Then side_length = circumradius
    return max_distance * 1.05

def evaluate_solution(individual):
    """Fast evaluation of hexagon packing solution"""
    # Reshape into (11, 3) hexagon parameters
    hex_data = np.array(individual).reshape(-1, 3)
    
    # Calculate required outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(hex_data)
    
    # Initialize penalty
    penalty = 0.0
    
    # Check containment: All vertices must be within outer hexagon
    # Outer hexagon center is at (0,0) with side length = outer_side_length
    # Outer hexagon circumradius = outer_side_length * sqrt(3) / 2
    outer_circumradius = outer_side_length * np.sqrt(3) / 2
    
    for i in range(NUM_INNER_HEX):
        x, y, angle = hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        
        # Fast containment check: all vertices must be within outer hexagon
        for vertex in vertices:
            dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
            if dist > outer_circumradius - 1e-6:
                penalty += 1000000.0  # Heavy penalty
    
    # Check overlaps between hexagons
    for i in range(NUM_INNER_HEX):
        for j in range(i+1, NUM_INNER_HEX):
            x1, y1, angle1 = hex_data[i]
            x2, y2, angle2 = hex_data[j]
            
            vertices1 = get_hexagon_vertices(x1, y1, angle1)
            vertices2 = get_hexagon_vertices(x2, y2, angle2)
            
            if check_hexagon_overlap_fast(vertices1, vertices2):
                penalty += 1000000.0  # Heavy penalty
    
    # Fitness: negative inverse of side length plus penalties
    fitness = -1.0 / outer_side_length
    if penalty > 0:
        fitness -= penalty
    
    return fitness

def create_diverse_initializations():
    """Create multiple diverse initial configurations"""
    initial_configs = []
    
    # Pattern 1: Hexagonal lattice arrangement
    def hex_lattice():
        positions = [(0, 0), (0, 2), (1.732, 1), (1.732, -1), (0, -2), 
                     (-1.732, -1), (-1.732, 1), (3.464, 0), (1.732, 2), 
                     (-1.732, 2), (-3.464, 0)]
        config = []
        for i, (x, y) in enumerate(positions):
            config.extend([x, y, random.uniform(0, 360)])
        return config
    
    # Pattern 2: Spiral arrangement
    def spiral_arrangement():
        config = []
        for i in range(11):
            angle = i * 0.7
            radius = i * 0.6
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            config.extend([x, y, random.uniform(0, 360)])
        return config
    
    # Pattern 3: Clustered arrangement
    def clustered_arrangement():
        positions = [
            (0.0, 0.0), (-1.8, 0.0), (1.8, 0.0), (0.0, 1.8), (0.0, -1.8),
            (-1.3, 1.3), (1.3, 1.3), (-1.3, -1.3), (1.3, -1.3),
            (-2.2, 0.0), (2.2, 0.0)
        ]
        config = []
        for i, (x, y) in enumerate(positions):
            config.extend([x, y, random.uniform(0, 360)])
        return config
    
    # Pattern 4: Dense grid
    def dense_grid():
        positions = [
            (0.0, 0.0), (-1.5, 0.0), (1.5, 0.0), (0.0, 1.5), (0.0, -1.5),
            (-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
            (-2.0, 0.0), (2.0, 0.0)
        ]
        config = []
        for i, (x, y) in enumerate(positions):
            config.extend([x, y, random.uniform(0, 360)])
        return config
    
    patterns = [hex_lattice, spiral_arrangement, clustered_arrangement, dense_grid]
    
    for pattern_func in patterns:
        # Generate multiple variants per pattern
        for _ in range(8):  # 8 variations per pattern
            config = pattern_func()
            # Add some noise to positions but keep angles random
            for i in range(0, len(config), 3):
                if i < len(config) - 1:  # Skip the last element which is outer radius
                    config[i] += random.uniform(-0.2, 0.2)
                    config[i+1] += random.uniform(-0.2, 0.2)
            # Add outer radius estimate
            config.append(4.0 + random.uniform(0.5, 2.0))
            initial_configs.append(config)
    
    return initial_configs

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses hybrid optimization with fast geometric validation.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Generate diverse initial configurations
    initial_configs = create_diverse_initializations()
    
    # Try to optimize each initial configuration
    best_fitness = float('-inf')
    best_solution = None
    
    # Run optimization on selected top configurations
    for i, config in enumerate(initial_configs[:15]):  # Test first 15 configs
        try:
            # Use DE as global optimizer first
            bounds = [(-8, 8), (-8, 8), (0, 360)] * NUM_INNER_HEX + [(3.0, 8.0)]
            
            result_de = differential_evolution(
                lambda x: -evaluate_solution(x),  # Minimize negative to maximize
                bounds,
                seed=42 + i,
                maxiter=100,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False,
                tol=1e-6
            )
            
            if result_de.success:
                # Local refinement
                refined_result = minimize(
                    lambda x: -evaluate_solution(x),
                    result_de.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-6},
                    tol=1e-6
                )
                
                if refined_result.success:
                    current_fitness = evaluate_solution(refined_result.x)
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_solution = refined_result.x
                        
        except Exception:
            continue
    
    # If we still don't have a valid solution, fall back to a good known pattern
    if best_solution is None:
        # Use a known good starting pattern
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0]
        ])
        outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract final solution
    final_params = best_solution
    inner_hex_data = np.array(final_params[:-1]).reshape(-1, 3)
    outer_hex_side_length = final_params[-1]
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END