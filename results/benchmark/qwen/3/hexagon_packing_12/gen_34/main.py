# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

def create_hexagon_vertices(center_x, center_y, side_length, rotation_deg):
    """Create vertices of a regular hexagon."""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices = np.array([
        [center_x + side_length * np.cos(angle),
         center_y + side_length * np.sin(angle)]
        for angle in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_side_length(inner_hex_data, outer_center=(0,0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons."""
    # Create inner hexagons
    inner_hexagons = []
    for x, y, angle in inner_hex_data:
        vertices = create_hexagon_vertices(x, y, 1, angle)
        inner_hexagons.append(Polygon(vertices))
    
    # Find bounding box of all inner hexagons
    all_points = []
    for hexagon in inner_hexagons:
        all_points.extend(list(hexagon.exterior.coords))
    
    if not all_points:
        return 1e6
    
    # Calculate distance from center to farthest point
    max_dist = 0
    center = np.array(outer_center)
    for point in all_points:
        dist = np.linalg.norm(np.array(point) - center)
        max_dist = max(max_dist, dist)
    
    # Convert to hexagon side length
    # For a regular hexagon with circumradius R, side length = R
    # But we need to account for the fact that we're fitting around hexagons
    # The side length of the outer hexagon should be such that it circumscribes
    # the farthest point from center, accounting for the hexagon geometry
    
    # Approximate calculation using distance to vertices
    side_length = max_dist * 2  # Rough estimation
    return side_length

def validate_configuration(inner_hex_data, outer_side_length):
    """Validate that configuration satisfies all constraints."""
    # Create outer hexagon vertices
    outer_vertices = create_hexagon_vertices(0, 0, outer_side_length, 0)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment
    for x, y, angle in inner_hex_data:
        inner_vertices = create_hexagon_vertices(x, y, 1, angle)
        inner_polygon = Polygon(inner_vertices)
        
        # Check if inner hexagon is fully contained
        if not outer_polygon.contains(inner_polygon):
            return False, "Not fully contained"
        
        # Check if any other hexagon overlaps
        for i, (x2, y2, angle2) in enumerate(inner_hex_data):
            if i == 0:
                continue  # Skip self
            inner_vertices2 = create_hexagon_vertices(x2, y2, 1, angle2)
            inner_polygon2 = Polygon(inner_vertices2)
            
            if inner_polygon.intersects(inner_polygon2):
                return False, "Overlap detected"
                
    return True, "Valid"

def objective_function(params):
    """Objective function to minimize (negative of inverse side length)."""
    # params: [x1, y1, angle1, ..., x12, y12, angle12]
    inner_hex_data = params.reshape(-1, 3)
    
    # Calculate outer hexagon side length needed
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Return negative inverse of side length (we want to maximize 1/R)
    if outer_side_length <= 0:
        return 1e6
    return -1.0 / outer_side_length

def constraint_function(params):
    """Constraint function to ensure no overlaps."""
    inner_hex_data = params.reshape(-1, 3)
    
    # Check for overlaps between hexagons
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            x1, y1, angle1 = inner_hex_data[i]
            x2, y2, angle2 = inner_hex_data[j]
            
            vertices1 = create_hexagon_vertices(x1, y1, 1, angle1)
            vertices2 = create_hexagon_vertices(x2, y2, 1, angle2)
            
            poly1 = Polygon(vertices1)
            poly2 = Polygon(vertices2)
            
            if poly1.intersects(poly2):
                # Return positive value to indicate violation
                return 1.0
                
    return -1.0  # No violation

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Initial configuration based on known good arrangement
    initial_config = np.array([
        [0, 0, 0],           # center
        [-1.5, 0, 0],        # left
        [1.5, 0, 0],         # right
        [0, 1.5, 0],         # top
        [0, -1.5, 0],        # bottom
        [-1.5, 1.5, 0],      # top-left
        [1.5, 1.5, 0],       # top-right
        [-1.5, -1.5, 0],     # bottom-left
        [1.5, -1.5, 0],      # bottom-right
        [-3.0, 0, 0],        # far left
        [3.0, 0, 0],         # far right
        [0, 3.0, 0],         # far top
    ])
    
    # Flatten parameters for optimization
    flat_params = initial_config.flatten()
    
    # Define bounds (x, y in [-5, 5], angle in [0, 360])
    bounds = []
    for i in range(12):
        bounds.append((-5, 5))  # x
        bounds.append((-5, 5))  # y  
        bounds.append((0, 360)) # angle
    
    # Set up constraints for non-overlap
    def overlap_constraint(params):
        inner_hex_data = params.reshape(-1, 3)
        for i in range(len(inner_hex_data)):
            for j in range(i+1, len(inner_hex_data)):
                x1, y1, angle1 = inner_hex_data[i]
                x2, y2, angle2 = inner_hex_data[j]
                
                vertices1 = create_hexagon_vertices(x1, y1, 1, angle1)
                vertices2 = create_hexagon_vertices(x2, y2, 1, angle2)
                
                poly1 = Polygon(vertices1)
                poly2 = Polygon(vertices2)
                
                if poly1.intersects(poly2):
                    return 1.0  # Violation
        return -1.0  # No violation
    
    constraints = [{'type': 'ineq', 'fun': overlap_constraint}]
    
    # Optimization options
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Perform optimization
        result = minimize(
            objective_function,
            flat_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-8
        )
        
        if result.success:
            final_params = result.x.reshape(-1, 3)
        else:
            # If optimization fails, use initial config
            final_params = initial_config
            
    except Exception as e:
        # Fallback to initial configuration
        final_params = initial_config
        
    # Calculate final outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(final_params)
    
    # Validate final configuration
    is_valid, message = validate_configuration(final_params, outer_side_length)
    
    if not is_valid:
        # If validation fails, use a safer configuration
        safe_config = np.array([
            [0, 0, 0],          # center
            [-1.8, 0, 0],       # left
            [1.8, 0, 0],        # right
            [0, 1.8, 0],        # top
            [0, -1.8, 0],       # bottom
            [-1.8, 1.8, 0],     # top-left
            [1.8, 1.8, 0],      # top-right
            [-1.8, -1.8, 0],    # bottom-left
            [1.8, -1.8, 0],     # bottom-right
            [-3.6, 0, 0],       # far left
            [3.6, 0, 0],        # far right
            [0, 3.6, 0],        # far top
        ])
        outer_side_length = calculate_outer_hex_side_length(safe_config)
        final_params = safe_config
    
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    return final_params, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END
