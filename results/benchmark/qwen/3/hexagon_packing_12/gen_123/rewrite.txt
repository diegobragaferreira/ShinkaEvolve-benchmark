# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
import time
from numba import jit, prange
import random

# Numba compiled helper functions for performance
@jit(nopython=True)
def hexagon_vertices_numba(center_x, center_y, angle_deg, scale=1.0):
    """Fast calculation of hexagon vertices using numba"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    cos_a = np.cos(np.radians(angle_deg))
    sin_a = np.sin(np.radians(angle_deg))
    
    vertices = np.empty((6, 2))
    for i in range(6):
        x = scale * np.cos(angles[i])
        y = scale * np.sin(angles[i])
        # Apply rotation matrix
        rotated_x = x * cos_a - y * sin_a
        rotated_y = x * sin_a + y * cos_a
        vertices[i, 0] = rotated_x + center_x
        vertices[i, 1] = rotated_y + center_y
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point, polygon):
    """Fast point-in-polygon test using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def distance_point_to_segment(point, seg_start, seg_end):
    """Calculate distance from point to line segment"""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Vector from seg_start to seg_end
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line segment
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

@jit(nopython=True)
def polygon_distance_fast(poly1, poly2):
    """Fast distance calculation between two polygons"""
    min_dist = np.inf
    
    # Check distances between all edges
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            dist = distance_point_to_segment(poly1[i], poly2[j], poly2[(j+1)%len(poly2)])
            min_dist = min(min_dist, dist)
            
            dist = distance_point_to_segment(poly2[j], poly1[i], poly1[(i+1)%len(poly1)])
            min_dist = min(min_dist, dist)
    
    return min_dist

def get_unit_hexagon_vertices():
    """Get vertices of a unit regular hexagon centered at origin"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles, exclude last to close polygon
    return np.column_stack([np.cos(angles), np.sin(angles)])

# Precomputed vertices for a unit hexagon
UNIT_HEX_VERTICES = get_unit_hexagon_vertices()

def hexagon_vertices(center_x, center_y, angle_deg, scale=1.0):
    """Get vertices of a hexagon with given center, angle, and scale"""
    return hexagon_vertices_numba(center_x, center_y, angle_deg, scale)

def check_containment_shapely(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Check if hexagon vertices are all within the outer hexagon using Shapely"""
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, 0, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_containment_fast(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Fast containment check using precomputed vertices"""
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, 0, outer_radius)
    
    # Check if all hexagon vertices are inside the outer hexagon
    for vertex in hex_vertices:
        if not point_in_polygon_fast(vertex, outer_vertices):
            return False
    return True

def check_overlap_shapely(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap checking using distance-based approximation"""
    # Quick bounding box test
    bbox1_min = np.min(hex1_vertices, axis=0)
    bbox1_max = np.max(hex1_vertices, axis=0)
    bbox2_min = np.min(hex2_vertices, axis=0)
    bbox2_max = np.max(hex2_vertices, axis=0)
    
    if bbox1_max[0] < bbox2_min[0] or bbox2_max[0] < bbox1_min[0]:
        return False
    if bbox1_max[1] < bbox2_min[1] or bbox2_max[1] < bbox1_min[1]:
        return False
        
    # Use distance threshold to avoid expensive calculations
    center1 = np.mean(hex1_vertices, axis=0)
    center2 = np.mean(hex2_vertices, axis=0)
    dist_centers = np.linalg.norm(center1 - center2)
    
    # If centers are far apart, likely no overlap
    if dist_centers > 3.0:  # 3 units is more than 2 hex radii
        return False
    
    # Use exact overlap check for close hexagons
    return check_overlap_shapely(hex1_vertices, hex2_vertices)

def calculate_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y):
    """
    Calculate minimum radius needed for outer hexagon to contain all inner hexagons
    by checking maximum distance from center to any vertex of any hexagon
    """
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_vertices = hexagon_vertices(center_x, center_y, angle)
        for vertex in hex_vertices:
            dist = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
            max_dist = max(max_dist, dist)

    # Add buffer for numerical precision
    return max_dist * 1.01

@jit(nopython=True)
def calculate_outer_radius_fast(inner_hex_data, outer_center_x, outer_center_y):
    """Fast calculation of outer radius using numba"""
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Precomputed hexagon vertices
        angles = np.linspace(0, 2*np.pi, 7)[:-1]
        for j in range(6):
            x = np.cos(angles[j])
            y = np.sin(angles[j])
            # Apply rotation
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            rot_x = x * cos_a - y * sin_a
            rot_y = x * sin_a + y * cos_a
            vertex_x = rot_x + center_x
            vertex_y = rot_y + center_y
            
            dist = np.sqrt((vertex_x - outer_center_x)**2 + (vertex_y - outer_center_y)**2)
            if dist > max_dist:
                max_dist = dist
    return max_dist * 1.01

def prepare_constraints(inner_params, outer_center_x, outer_center_y):
    """Prepare constraints list for fast checking"""
    constraints = []
    for i in range(len(inner_params)):
        center_x, center_y, angle = inner_params[i]
        hex_vertices = hexagon_vertices(center_x, center_y, angle)
        constraints.append((center_x, center_y, angle, hex_vertices))
    return constraints

def evaluate_configuration_fast(inner_params, outer_center_x, outer_center_y, outer_radius):
    """Fast evaluation of configuration validity"""
    # Check containment
    for i in range(len(inner_params)):
        center_x, center_y, angle = inner_params[i]
        hex_vertices = hexagon_vertices(center_x, center_y, angle)
        if not check_containment_fast(hex_vertices, outer_center_x, outer_center_y, outer_radius):
            return False
    
    # Check overlaps using fast method
    for i in range(len(inner_params)):
        for j in range(i+1, len(inner_params)):
            center1_x, center1_y, angle1 = inner_params[i]
            center2_x, center2_y, angle2 = inner_params[j]
            
            hex1_vertices = hexagon_vertices(center1_x, center1_y, angle1)
            hex2_vertices = hexagon_vertices(center2_x, center2_y, angle2)
            
            if check_overlap_fast(hex1_vertices, hex2_vertices):
                return False
    
    return True

def objective_function_fast(params):
    """
    Fast objective function to minimize (negative of 1/outer_radius)
    params: [x1, y1, angle1, ..., x12, y12, angle12, outer_center_x, outer_center_y]
    """
    n = 12
    # Extract inner hexagon parameters
    inner_params = params[:3*n].reshape(n, 3)
    # Extract outer hexagon parameters
    outer_center_x, outer_center_y = params[3*n:3*n+2]

    # Calculate outer radius using fast method
    outer_radius = calculate_outer_radius_fast(inner_params, outer_center_x, outer_center_y)

    # Return negative of inverse radius (we want to maximize 1/R, so minimize -1/R)
    return -1.0 / outer_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 12

    # Multiple initial configurations for multi-start optimization
    initial_configs = [
        # Configuration 1: Symmetric arrangement
        np.array([
            [0, 0, 0],      # center
            [-2.5, 0, 0],   # left
            [2.5, 0, 0],    # right
            [-1.25, 2.17, 0], # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0], # bottom-left
            [1.25, -2.17, 0], # bottom-right
            [-3.75, 2.17, 0], # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0], # far bottom-left
            [3.75, -2.17, 0], # far bottom-right
            [0, -4, 0],     # far bottom-center
        ]),
        
        # Configuration 2: Ring arrangement
        np.array([
            [0, 0, 0],      # center
            [0, 2.0, 0],    # top
            [1.73, 1.0, 0], # top-right
            [1.73, -1.0, 0],# bottom-right
            [0, -2.0, 0],   # bottom
            [-1.73, -1.0, 0],# bottom-left
            [-1.73, 1.0, 0], # top-left
            [0, 3.5, 0],    # top far
            [3.03, 1.75, 0],# top-right far
            [3.03, -1.75, 0],# bottom-right far
            [0, -3.5, 0],   # bottom far
            [-3.03, -1.75, 0],# bottom-left far
        ]),
        
        # Configuration 3: Checkerboard arrangement
        np.array([
            [0, 0, 0],      # center
            [2.0, 0, 0],    # right
            [-2.0, 0, 0],   # left
            [0, 2.0, 0],    # top
            [0, -2.0, 0],   # bottom
            [1.0, 1.0, 0],  # top-right
            [1.0, -1.0, 0], # bottom-right
            [-1.0, 1.0, 0], # top-left
            [-1.0, -1.0, 0],# bottom-left
            [2.0, 2.0, 0],  # far top-right
            [2.0, -2.0, 0], # far bottom-right
            [-2.0, -2.0, 0],# far bottom-left
        ])
    ]

    best_result = None
    best_score = -np.inf
    start_time = time.time()
    
    # Multi-start optimization
    for i, initial_positions in enumerate(initial_configs):
        try:
            # Set initial angles to 0 (no rotation) for simplicity
            initial_angles = np.zeros(n)

            # Start with center at origin
            initial_outer_center = [0.0, 0.0]

            # Flatten parameters
            initial_params = np.hstack([
                initial_positions.flatten(),
                initial_outer_center
            ])

            # Bounds for optimization (limiting positions but keeping flexibility for rotations)
            bounds = []

            # Add bounds for positions (-10, 10)
            for _ in range(n):
                bounds.extend([(-10, 10), (-10, 10)])

            # Add bounds for angles (0 to 360 degrees)
            for _ in range(n):
                bounds.extend([(0, 360)])

            # Add bounds for outer center (limited range)
            bounds.extend([(-20, 20), (-20, 20)])

            # Optimization with fewer iterations for faster convergence
            result = minimize(
                objective_function_fast,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6},
                tol=1e-6
            )

            # Evaluate final result
            if result.success:
                optimized_params = result.x
                optimized_inner_data = optimized_params[:3*n].reshape(n, 3)
                outer_center_x, outer_center_y = optimized_params[3*n:3*n+2]
                
                # Calculate actual outer hexagon radius
                outer_radius = calculate_outer_radius_fast(optimized_inner_data, outer_center_x, outer_center_y)
                
                # Calculate score (inverse of outer radius)
                score = 1.0 / outer_radius
                
                # If this is better than previous best, store it
                if score > best_score:
                    best_score = score
                    best_result = {
                        'inner_data': optimized_inner_data,
                        'outer_center': [outer_center_x, outer_center_y],
                        'outer_radius': outer_radius,
                        'success': True
                    }
                    
        except Exception as e:
            # Continue with other initial configurations if one fails
            continue

    # If we found any valid result, use it; otherwise fallback
    if best_result is not None and best_result['success']:
        optimized_inner_data = best_result['inner_data']
        outer_center_x, outer_center_y = best_result['outer_center']
        outer_hex_side_length = best_result['outer_radius']
    else:
        # Fallback to first initial configuration
        initial_positions = initial_configs[0]
        optimized_inner_data = initial_positions.copy()
        outer_hex_side_length = 8.0

    end_time = time.time()

    # Construct final result according to required format
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Outer hexagon centered at origin

    return optimized_inner_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END