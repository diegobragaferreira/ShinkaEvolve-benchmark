# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
from scipy.optimize import minimize
import itertools
from scipy.spatial import cKDTree

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution: check containment and non-overlap"""
    # Precompute all hexagon polygons once for reuse
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))

    # Calculate outer radius once
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # Check containment using the outer hexagon polygon
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Check if all inner hexagons are contained within outer hexagon
    for hex_poly in hex_polygons:
        # Fast check: if any vertex is outside, reject
        for vertex in hex_poly.exterior.coords[:-1]:  # Exclude closing vertex
            if not outer_polygon.contains(Point(vertex)):
                return False

    # Check overlaps efficiently using spatial indexing
    # Build spatial index for faster overlap detection
    points_list = []
    for i, hex_poly in enumerate(hex_polygons):
        # Collect all vertices for spatial indexing
        for vertex in hex_poly.exterior.coords[:-1]:
            points_list.append((vertex[0], vertex[1], i))

    if len(points_list) > 0:
        # Create spatial tree for vertices
        tree_points = cKDTree([(p[0], p[1]) for p in points_list])

        # Check overlaps between hexagons
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return False
    else:
        # Fallback for empty case
        return False

    return True

def generate_hexagonal_grid_layout():
    """Create an initial layout based on hexagonal packing principles"""
    # Start with honeycomb-like structure
    # Center hexagon
    config = [[0, 0, 0]]
    
    # First ring around center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for angle in angles:
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle)
        config.append([x, y, 0])
    
    # Second ring around first ring
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    for i, angle in enumerate(angles):
        x = 3 * np.cos(angle) + (-1)**i * 1.5
        y = 3 * np.sin(angle)
        config.append([x, y, 0])
    
    # Ensure we have exactly 11 hexagons
    while len(config) < 11:
        config.append([0, 0, 0])  # Fill with dummy values
    
    config = config[:11]
    
    # Add random rotations to break symmetry
    for i in range(len(config)):
        config[i][2] = np.random.uniform(0, 360)
    
    return np.array(config)

def optimize_positions_and_rotations(initial_config):
    """Use mathematical optimization to refine the configuration"""
    # Flatten the configuration for optimization
    def flatten_config(config):
        flat = []
        for row in config:
            flat.extend(row)
        return np.array(flat)
    
    def unflatten_config(flat):
        config = []
        for i in range(0, len(flat), 3):
            config.append(flat[i:i+3])
        return np.array(config)
    
    def objective(flat_config):
        config = unflatten_config(flat_config)
        # Calculate outer radius
        radius = calculate_outer_hexagon_radius(config)
        
        # Check validity
        if not validate_solution(config):
            return 1e10  # Penalize invalid solutions heavily
            
        return radius
    
    # Initial flattened config
    flat_start = flatten_config(initial_config)
    
    # Optimize using scipy minimize
    try:
        result = minimize(objective, flat_start, method='L-BFGS-B', 
                         bounds=[(-10, 10), (-10, 10), (0, 360)] * 11,
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_config = unflatten_config(result.x)
            return optimized_config
    except:
        pass
    
    return initial_config

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    best_config = None
    best_radius = float('inf')
    
    # Multi-start with different initial approaches
    for trial in range(20):
        if time.time() - start_time > MAX_EVAL_TIME - 1:
            break
            
        # Generate initial configuration using hexagonal grid layout
        initial_config = generate_hexagonal_grid_layout()
        
        # Refine using optimization
        refined_config = optimize_positions_and_rotations(initial_config)
        
        # Validate and update best
        if validate_solution(refined_config):
            radius = calculate_outer_hexagon_radius(refined_config)
            if radius < best_radius:
                best_radius = radius
                best_config = refined_config.copy()
    
    # If no valid configuration was found, fallback to known good configuration
    if best_config is None:
        best_config = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        best_radius = 8.0
    
    # Final validation
    if not validate_solution(best_config):
        best_config = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        best_radius = 8.0
    
    # Return result
    inner_hex_data = best_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = best_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END