# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.deg2rad(angle_deg)
    # Vertices of a unit hexagon centered at origin
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
    return rotated_vertices + np.array([center_x, center_y])

def check_containment(hex_vertices, outer_center_x, outer_center_y, outer_side_length):
    """Check if hexagon vertices are contained within outer hexagon"""
    # Create outer hexagon vertices
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, 0, outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Check if each vertex of inner hexagon is inside outer hexagon
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_penalty(inner_hex_data, outer_side_length):
    """Calculate penalty based on constraint violations"""
    penalty = 0
    
    # Generate outer hexagon vertices (centered at origin)
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment and overlap for all hexagons
    hex_polygons = []
    for i in range(12):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(center_x, center_y, angle)
        hex_polygons.append(Polygon(vertices))
        
        # Check containment
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                penalty += 1e6
        
        # Check overlap with other hexagons
        for j in range(i+1, 12):
            if hex_polygons[i].intersects(hex_polygons[j]):
                penalty += 1e6
    
    return penalty

def objective_function(params):
    """Objective function to maximize 1/outer_hex_side_length"""
    # Reshape params into 12 hexagons with (x,y,angle) each
    inner_hex_data = params.reshape(12, 3)
    outer_side_length = params[-1]  # last parameter is outer hex side length
    
    # Calculate penalty for constraint violations
    penalty = calculate_penalty(inner_hex_data, outer_side_length)
    
    # Objective: maximize 1/outer_side_length (minimize outer_side_length)
    # Add penalty for constraint violations
    return -1.0 / outer_side_length + penalty

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use a more informed initial guess based on known symmetric arrangements
    initial_positions = [
        [0, 0, 0],      # center
        [-2.0, 0, 0],   # left
        [2.0, 0, 0],    # right
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [-1.0, 1.0, 0], # top-left
        [1.0, 1.0, 0],  # top-right
        [-1.0, -1.0, 0], # bottom-left
        [1.0, -1.0, 0],  # bottom-right
        [-2.0, 1.0, 0],  # far top-left
        [2.0, 1.0, 0],   # far top-right
        [-2.0, -1.0, 0], # far bottom-left
    ]
    
    # Start with a reasonable initial outer hexagon size
    initial_outer_size = 4.0
    
    # Flatten initial parameters (12 hexagons * 3 params + 1 outer size)
    initial_params = np.array(initial_positions).flatten()
    initial_params = np.append(initial_params, initial_outer_size)
    
    # Define bounds: positions (-10, 10), angles (0, 360), outer size (1, 20)
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    bounds.append((1, 20))  # outer hex side length
    
    # Optimization options
    options = {'maxiter': 200, 'disp': False}
    
    # Run optimization
    result = minimize(
        objective_function,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options=options,
        tol=1e-6
    )
    
    # Extract results
    optimized_params = result.x
    inner_hex_data = optimized_params[:-1].reshape(12, 3)
    outer_hex_side_length = optimized_params[-1]
    
    # Center the outer hexagon at origin
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# Import shapely Point if needed (for compatibility)
try:
    from shapely.geometry import Point
except ImportError:
    # Fallback if shapely is not available
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

# EVOLVE-BLOCK-END
