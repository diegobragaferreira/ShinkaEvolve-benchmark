# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import math

def generate_hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon given center, side length, and rotation"""
    rotation_rad = math.radians(rotation_degrees)
    vertices = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def hexagon_area(side_length):
    """Calculate area of regular hexagon"""
    return (3 * math.sqrt(3) / 2) * side_length**2

def check_containment(inner_hex_vertices, outer_hex_vertices):
    """Check if inner hexagon is fully contained within outer hexagon"""
    inner_polygon = Polygon(inner_hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)

    # Check if all vertices of inner hexagon are inside outer hexagon
    for vertex in inner_hex_vertices:
        if not outer_polygon.contains(Point(vertex[0], vertex[1])):
            return False

    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_size(inner_hex_data, margin=0.01):
    """Compute the minimal outer hexagon size needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, 1.0, angle)
        all_vertices.extend(vertices)

    if len(all_vertices) == 0:
        return 1.0

    # Find bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    # Find the minimum hexagon that contains all vertices
    # We'll estimate the required radius by finding the maximum distance from origin
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist)

    # Add some margin for safety
    return max_dist + margin

def objective_function(params):
    """Objective function to minimize the outer hexagon side length"""
    # Extract parameters
    # params = [x1, y1, theta1, x2, y2, theta2, ..., x11, y11, theta11]
    n = 11
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i] = [params[3*i], params[3*i+1], params[3*i+2]]

    # Compute outer hexagon size
    outer_side_length = compute_outer_hexagon_size(inner_hex_data)

    # Add penalty for violations
    penalty = 0

    # Check containment and overlap constraints
    outer_vertices = generate_hexagon_vertices(0, 0, outer_side_length, 0)

    # Check containments
    for i in range(n):
        center_x, center_y, angle = inner_hex_data[i]
        inner_vertices = generate_hexagon_vertices(center_x, center_y, 1.0, angle)
        if not check_containment(inner_vertices, outer_vertices):
            penalty += 1000  # Large penalty for containment violation

    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            center_x1, center_y1, angle1 = inner_hex_data[i]
            center_x2, center_y2, angle2 = inner_hex_data[j]

            inner_vertices1 = generate_hexagon_vertices(center_x1, center_y1, 1.0, angle1)
            inner_vertices2 = generate_hexagon_vertices(center_x2, center_y2, 1.0, angle2)

            if check_overlap(inner_vertices1, inner_vertices2):
                penalty += 1000  # Large penalty for overlap violation

    return outer_side_length + penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimization to find the best configuration.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initial guess for inner hexagons (more carefully arranged)
    initial_guess = [
        0.0, 0.0, 0.0,      # center
        -1.732, 0.0, 0.0,   # left
        1.732, 0.0, 0.0,    # right
        -0.866, 1.5, 0.0,   # top-left
        0.866, 1.5, 0.0,    # top-right
        -0.866, -1.5, 0.0,  # bottom-left
        0.866, -1.5, 0.0,   # bottom-right
        -2.598, 1.5, 0.0,   # far top-left
        2.598, 1.5, 0.0,    # far top-right
        -2.598, -1.5, 0.0,  # far bottom-left
        2.598, -1.5, 0.0,   # far bottom-right
    ]

    # Bounds for optimization (positions and angles)
    bounds = []
    for i in range(11):
        # Positions: [-5, 5] for x and y
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0)])
        # Angles: [0, 360) degrees
        bounds.append((0.0, 360.0))

    # Optimize
    try:
        result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-6})

        if result.success:
            # Extract the optimized configuration
            params = result.x
            inner_hex_data = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data[i] = [params[3*i], params[3*i+1], params[3*i+2]]

            # Compute final outer hexagon size
            outer_side_length = compute_outer_hexagon_size(inner_hex_data)

            # Create outer hexagon data
            outer_hex_data = np.array([0, 0, 0])  # centered at origin

            return inner_hex_data, outer_hex_data, outer_side_length
        else:
            print("Optimization failed:", result.message)
    except Exception as e:
        print("Error during optimization:", str(e))

    # Fallback to original configuration if optimization fails
    n = 11
    inner_hex_data = np.array([
        [0, 0, 0],          # center
        [-1.732, 0, 0],     # left
        [1.732, 0, 0],      # right
        [-0.866, 1.5, 0],   # top-left
        [0.866, 1.5, 0],    # top-right
        [-0.866, -1.5, 0],  # bottom-left
        [0.866, -1.5, 0],   # bottom-right
        [-2.598, 1.5, 0],   # far top-left
        [2.598, 1.5, 0],    # far top-right
        [-2.598, -1.5, 0],  # far bottom-left
        [2.598, -1.5, 0],   # far bottom-right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_side_length = 4.0  # Estimate based on initial placement

    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END