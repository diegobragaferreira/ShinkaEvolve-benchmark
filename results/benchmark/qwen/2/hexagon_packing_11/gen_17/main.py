# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
import math


def get_hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Get vertices of a regular hexagon with given center, side length, and rotation."""
    # Convert degrees to radians
    theta = math.radians(rotation_degrees)

    # For a regular hexagon with side length s, the distance from center to vertex is also s
    radius = side_length

    # Generate vertices around the center with rotation
    vertices = []
    for i in range(6):
        angle = theta + i * math.pi / 3
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        vertices.append((x, y))

    return vertices


def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if a hexagon is completely contained within the outer hexagon."""
    inner_polygon = Polygon(hexagon_vertices)
    outer_polygon = Polygon(outer_hexagon_vertices)
    return outer_polygon.contains(inner_polygon)


def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)


def calculate_outer_hexagon_radius(inner_hex_data, margin=0.01):
    """Calculate the minimum radius needed for outer hexagon to contain all inner hexagons."""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Get all vertices of this hexagon
        vertices = get_hexagon_vertices(center_x, center_y, 1, angle)
        # Find maximum distance from center to any vertex
        for vx, vy in vertices:
            dist = math.sqrt((vx - center_x)**2 + (vy - center_y)**2)
            max_dist = max(max_dist, dist)

    # Add small margin to ensure complete containment
    return max_dist + margin


def evaluate_packing(config):
    """Evaluate how well a given configuration works."""
    # Reshape config into 11 hexagons with (x, y, angle)
    hex_data = config.reshape(-1, 3)

    # Check constraints
    # 1. Check for overlaps between any pair of hexagons
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            hex1_vertices = get_hexagon_vertices(hex_data[i][0], hex_data[i][1], 1, hex_data[i][2])
            hex2_vertices = get_hexagon_vertices(hex_data[j][0], hex_data[j][1], 1, hex_data[j][2])
            if check_overlap(hex1_vertices, hex2_vertices):
                # If overlapping, return very poor score
                return 1e10

    # 2. Check containment
    outer_radius = calculate_outer_hexagon_radius(hex_data)
    outer_vertices = get_hexagon_vertices(0, 0, outer_radius, 0)

    for i in range(len(hex_data)):
        hex_vertices = get_hexagon_vertices(hex_data[i][0], hex_data[i][1], 1, hex_data[i][2])
        if not check_containment(hex_vertices, outer_vertices):
            # If not contained, return very poor score
            return 1e10

    # Return the inverse of outer hexagon radius (we want to minimize outer radius)
    return 1.0 / outer_radius


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initial guess: start with a reasonable configuration
    # We'll use a more optimized initial arrangement
    initial_config = np.array([
        [0.0, 0.0, 0.0],      # center
        [-1.5, 0.0, 0.0],     # left
        [1.5, 0.0, 0.0],      # right
        [0.0, 2.6, 0.0],      # top
        [0.0, -2.6, 0.0],     # bottom
        [-1.5, 2.6, 0.0],     # top-left
        [1.5, 2.6, 0.0],      # top-right
        [-1.5, -2.6, 0.0],    # bottom-left
        [1.5, -2.6, 0.0],     # bottom-right
        [-3.0, 0.0, 0.0],     # far left
        [3.0, 0.0, 0.0],      # far right
    ]).flatten()

    # Optimization bounds: (x_min, x_max), (y_min, y_max), (angle_min, angle_max) for each hexagon
    bounds = []
    for _ in range(11):
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (-180.0, 180.0)])

    # Run optimization
    result = differential_evolution(evaluate_packing, bounds, seed=42, maxiter=200, popsize=15)

    # Extract the best configuration
    best_config = result.x.reshape(-1, 3)

    # Calculate outer hexagon size
    outer_radius = calculate_outer_hexagon_radius(best_config)

    # Set up return values
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = outer_radius

    return best_config, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END