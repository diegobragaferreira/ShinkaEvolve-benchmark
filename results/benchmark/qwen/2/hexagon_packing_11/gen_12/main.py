# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    hexagon_vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hexagon_vertices.append((x, y))

    # Scale and translate
    scaled_vertices = [(side_length * vx + center_x, side_length * vy + center_y)
                       for vx, vy in hexagon_vertices]
    return scaled_vertices

def check_hexagon_containment(hexagon_vertices, outer_hex_vertices):
    """Check if a hexagon is fully contained within the outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hex_vertices)
    return outer_poly.contains(inner_poly)

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_bound(inner_hex_data, margin_factor=1.1):
    """
    Compute the minimal bounding hexagon that contains all inner hexagons.
    """
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_vertices = generate_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(hex_vertices)

    if len(all_vertices) == 0:
        return 1.0

    # Find min/max coordinates to estimate hexagon size
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2

    # Calculate distance from center to farthest vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_dist = max(max_dist, dist)

    # Add some margin and convert to side length
    # For a hexagon, if we know the distance from center to vertex,
    # the side length is that distance
    side_length = max_dist * margin_factor
    return side_length

def objective_function(params):
    """Objective function to maximize 1/outer_hex_side_length"""
    # Parse parameters: 11 hexagons with (x, y, angle) each + outer hexagon center and rotation
    # We'll treat the outer hexagon as fixed for now
    hex_params = params[:33].reshape(11, 3)

    # Assume outer hexagon is centered at origin, with rotation 0, and compute side length
    outer_side_length = compute_outer_hexagon_bound(hex_params)

    # Return negative because we want to maximize 1/outer_side_length
    return -1.0 / outer_side_length

def constraint_function(params):
    """Constraint function to ensure no overlaps and full containment"""
    hex_params = params[:33].reshape(11, 3)

    # First check containment constraint
    outer_side_length = compute_outer_hexagon_bound(hex_params, margin_factor=1.05)
    outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_side_length)

    # Check each inner hexagon for containment
    for i in range(len(hex_params)):
        center_x, center_y, angle = hex_params[i]
        inner_vertices = generate_hexagon_vertices(center_x, center_y, angle)

        # Check containment
        if not check_hexagon_containment(inner_vertices, outer_vertices):
            return -1.0  # Violation

    # Check overlaps between all pairs of hexagons
    for i in range(len(hex_params)):
        for j in range(i+1, len(hex_params)):
            center_x1, center_y1, angle1 = hex_params[i]
            center_x2, center_y2, angle2 = hex_params[j]

            inner_vertices1 = generate_hexagon_vertices(center_x1, center_y1, angle1)
            inner_vertices2 = generate_hexagon_vertices(center_x2, center_y2, angle2)

            # Check overlapping
            if check_hexagon_overlap(inner_vertices1, inner_vertices2):
                return -1.0  # Violation

    return 1.0  # No violations

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start with a good initial configuration - this is inspired by known optimal packings
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
    ])

    # Flatten for optimization
    initial_params = initial_config.flatten()

    # Define bounds for optimization: x, y in [-5, 5], angle in [0, 360)
    bounds = []
    for i in range(33):  # 11 hexagons * 3 parameters each
        if i % 3 < 2:   # x, y coordinates
            bounds.append((-5.0, 5.0))
        else:           # angle
            bounds.append((0.0, 360.0))

    # Optimization with bounds
    result = differential_evolution(objective_function, bounds, maxiter=100, popsize=15, seed=42)

    # Extract results
    optimized_params = result.x
    optimized_hex_params = optimized_params[:33].reshape(11, 3)

    # Calculate final outer hexagon side length
    outer_side_length = compute_outer_hexagon_bound(optimized_hex_params)

    # Return data
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Fixed at origin

    return optimized_hex_params, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END