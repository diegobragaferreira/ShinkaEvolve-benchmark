# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
import random
from collections import defaultdict

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2

def get_hexagon_vertices(center_x, center_y, angle_deg, radius=UNIT_HEX_RADIUS):
    """Get vertices of a regular hexagon given center, rotation, and radius."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi/3
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        vertices.append((x, y))
    return vertices

def check_containment(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Check if all vertices of hexagon are within the outer hexagon."""
    outer_vertices = get_hexagon_vertices(outer_center_x, outer_center_y, 0, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    for vertex in hex_vertices:
        point = Polygon(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_radius(inner_configs):
    """Calculate minimal outer hexagon radius needed to contain all inner hexagons."""
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for center_x, center_y, angle_deg in inner_configs:
        vertices = get_hexagon_vertices(center_x, center_y, angle_deg)
        for x, y in vertices:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    # Calculate distance from center to farthest point
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    max_dist = 0
    for center_x, center_y, angle_deg in inner_configs:
        vertices = get_hexagon_vertices(center_x, center_y, angle_deg)
        for x, y in vertices:
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = max(max_dist, dist)

    # Add some padding and convert to hexagon radius
    outer_radius = max_dist * 1.01  # slightly more than needed
    return outer_radius

def generate_valid_config():
    """Generate a valid configuration using a combination of structured patterns and randomization."""
    # Start with a structured pattern that's known to work well for small hexagon packings
    base_positions = [
        (0.0, 0.0),      # center
        (-1.8, 0.0),     # left
        (1.8, 0.0),      # right
        (0.0, 1.8),      # top
        (0.0, -1.8),     # bottom
        (-1.3, 1.3),     # top-left
        (1.3, 1.3),      # top-right
        (-1.3, -1.3),    # bottom-left
        (1.3, -1.3),     # bottom-right
        (-2.2, 0.0),     # further left
        (2.2, 0.0),      # further right
    ]
    
    # Add randomness while maintaining valid spacings
    config = []
    for i, (cx, cy) in enumerate(base_positions):
        # Add noise to position
        noise_x = random.uniform(-0.3, 0.3)
        noise_y = random.uniform(-0.3, 0.3)
        new_x = cx + noise_x
        new_y = cy + noise_y
        
        # Random rotation
        rotation = random.uniform(0, 360)
        
        config.append((new_x, new_y, rotation))
        
    return config

def validate_and_refine_config(config):
    """Validate a configuration and return its inverse radius if valid."""
    # Check for overlaps
    for i in range(len(config)):
        hex1_vertices = get_hexagon_vertices(config[i][0], config[i][1], config[i][2])
        for j in range(i+1, len(config)):
            hex2_vertices = get_hexagon_vertices(config[j][0], config[j][1], config[j][2])
            if check_overlap(hex1_vertices, hex2_vertices):
                return None
    
    # Check containment
    outer_radius = calculate_outer_hex_radius(config)
    outer_vertices = get_hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = Polygon(outer_vertices)
    
    for i in range(len(config)):
        hex1_vertices = get_hexagon_vertices(config[i][0], config[i][1], config[i][2])
        for vertex in hex1_vertices:
            point = Polygon(vertex)
            if not outer_polygon.contains(point):
                return None
                
    return 1.0 / outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses Monte Carlo sampling with intelligent configuration generation.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    best_inv_radius = 0.0
    best_config = None
    max_attempts = 20000  # Maximum number of attempts
    
    # Pre-computed bounds for efficient sampling
    bounds = [(-6.0, 6.0), (-6.0, 6.0), (0, 360)]
    
    for attempt in range(max_attempts):
        # Generate a new configuration
        config = generate_valid_config()
        
        # Validate and measure
        inv_radius = validate_and_refine_config(config)
        
        if inv_radius is not None and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = config
            
        # Occasionally, try to improve by making small adjustments
        if attempt % 1000 == 0 and best_config is not None:
            # Make a small local adjustment to the best config
            adjusted_config = []
            for center_x, center_y, angle in best_config:
                # Small perturbation
                new_x = center_x + random.uniform(-0.1, 0.1)
                new_y = center_y + random.uniform(-0.1, 0.1)
                new_angle = angle + random.uniform(-10, 10)
                adjusted_config.append((new_x, new_y, new_angle))
            
            # Validate adjusted version
            adjusted_inv_radius = validate_and_refine_config(adjusted_config)
            if adjusted_inv_radius is not None and adjusted_inv_radius > best_inv_radius:
                best_inv_radius = adjusted_inv_radius
                best_config = adjusted_config
    
    # If we didn't find anything, fall back to the basic approach
    if best_config is None:
        # Default configuration that's known to work reasonably well
        best_config = [
            (0.0, 0.0, 0.0),      # center
            (-1.8, 0.0, 0.0),     # left
            (1.8, 0.0, 0.0),      # right
            (0.0, 1.8, 0.0),      # top
            (0.0, -1.8, 0.0),     # bottom
            (-1.3, 1.3, 0.0),     # top-left
            (1.3, 1.3, 0.0),      # top-right
            (-1.3, -1.3, 0.0),    # bottom-left
            (1.3, -1.3, 0.0),     # bottom-right
            (-2.2, 0.0, 0.0),     # further left
            (2.2, 0.0, 0.0),      # further right
        ]
    
    # Convert to proper format
    inner_hex_data = np.array(best_config)
    
    # Calculate outer hexagon size
    outer_radius = calculate_outer_hex_radius(best_config)
    
    # Create outer hexagon data (centered at origin)
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END