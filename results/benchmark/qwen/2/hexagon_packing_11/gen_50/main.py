# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

def create_hexagon_vertices(center_x, center_y, angle_degrees, side_length=1):
    """Create vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = np.radians(angle_degrees)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = []
    for angle in angles[:-1]:  # exclude last to close the polygon
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return vertices

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly)

def calculate_outer_hex_radius(inner_hex_data, margin_factor=1.1):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons with margin."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle, 1)
        all_vertices.extend(vertices)
    
    if not all_vertices:
        return 1.0
    
    # Compute bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Calculate centroid
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    # Find maximum distance from centroid to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = max(max_dist, dist)
    
    # Add margin and convert to hexagon radius
    return max_dist * margin_factor

def evaluate_configuration(x):
    """Evaluate fitness of a configuration."""
    # Reshape input to get all hexagon data
    inner_hex_data = x.reshape(-1, 3)
    
    # Check collisions between all pairs of hexagons
    hexagons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle, 1)
        hexagons.append(Polygon(vertices))
    
    # Check for overlaps
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if check_overlap(hexagons[i], hexagons[j]):
                return float('inf')  # Invalid configuration
    
    # Calculate outer hex radius
    outer_radius = calculate_outer_hex_radius(inner_hex_data)
    
    # Return negative inverse of radius (since we want to maximize 1/r)
    return -1.0 / outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Initial configuration (better than baseline)
    initial_config = np.array([
        [0, 0, 0],      # center
        [-2.5, 0, 0],   # left
        [2.5, 0, 0],    # right
        [-1.25, 2.17, 0],  # top-left
        [1.25, 2.17, 0],   # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0],  # bottom-right
        [-3.75, 2.17, 0],  # far top-left
        [3.75, 2.17, 0],   # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0],  # far bottom-right
    ])
    
    # Optimization bounds for center coordinates (-10, 10) and angle (0, 360)
    bounds = []
    for i in range(11):
        # x and y coordinates
        bounds.extend([(-10, 10), (-10, 10)])
        # angle (0 to 360 degrees)
        bounds.append((0, 360))
    
    # Run differential evolution optimization
    start_time = time.time()
    
    # Use differential evolution with our objective function
    result = differential_evolution(evaluate_configuration, bounds, maxiter=1000, 
                                   popsize=15, mutation=(0.5, 1), recombination=0.7,
                                   seed=42, tol=1e-6, atol=1e-6)
    
    end_time = time.time()
    
    # Extract optimized configuration
    optimized_config = result.x.reshape(-1, 3)
    
    # Validate final configuration
    hexagons = []
    for i in range(len(optimized_config)):
        center_x, center_y, angle = optimized_config[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle, 1)
        hexagons.append(Polygon(vertices))
    
    # Final consistency check
    for i in range(len(hexagons)):
        for j in range(i+1, len(hexagons)):
            if check_overlap(hexagons[i], hexagons[j]):
                # If still overlapping, use the initial config
                optimized_config = initial_config.copy()
                break
    
    # Calculate final outer hexagon radius
    outer_radius = calculate_outer_hex_radius(optimized_config)
    
    # Return the best configuration found
    return optimized_config, np.array([0, 0, 0]), outer_radius

# EVOLVE-BLOCK-END
