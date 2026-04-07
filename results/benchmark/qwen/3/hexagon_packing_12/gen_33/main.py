# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import math


def hexagon_vertices(center_x, center_y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon"""
    angle_rad = math.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def hexagon_area(side_length):
    """Calculate area of a regular hexagon"""
    return (3 * math.sqrt(3) / 2) * side_length ** 2


def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if a hexagon is fully contained within another hexagon using Shapely"""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hex_vertices)
    return outer_poly.contains(inner_poly)


def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)


def calculate_outer_hex_side_length(inner_hex_data, outer_center=(0, 0)):
    """Calculate the minimum side length of outer hexagon that contains all inner hexagons"""
    # Get all vertices from all inner hexagons
    all_vertices = []

    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    # Calculate bounding box and then minimum enclosing hexagon
    if not all_vertices:
        return 1000

    # Find extreme points
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Estimate based on maximum distance from center
    max_dist = 0
    center = (0, 0)
    for x, y in all_vertices:
        dist = math.sqrt((x - center[0])**2 + (y - center[1])**2)
        max_dist = max(max_dist, dist)

    # The side length needs to accommodate this distance plus some margin
    # For a regular hexagon with side length s, distance from center to vertex is s
    return max_dist + 1.0  # Add margin for safety


def validate_configuration(inner_hex_data, outer_side_length):
    """Validate that the configuration meets all requirements"""
    # Check all pairs for overlap
    for i in range(len(inner_hex_data)):
        for j in range(i + 1, len(inner_hex_data)):
            center1_x, center1_y, angle1 = inner_hex_data[i]
            center2_x, center2_y, angle2 = inner_hex_data[j]

            vertices1 = hexagon_vertices(center1_x, center1_y, angle1)
            vertices2 = hexagon_vertices(center2_x, center2_y, angle2)

            if check_overlap(vertices1, vertices2):
                return False, "Overlap detected"

    # Check containment - create outer hexagon vertices
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)

    # Check if all inner hexagons are contained
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(center_x, center_y, angle)

        if not check_containment(vertices, outer_vertices):
            return False, "Containment violation"

    return True, "Valid configuration"


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Initial configuration based on more efficient hexagonal packing
    # Using a 3x4 grid with hexagonal arrangement
    inner_hex_data = np.array([
        # Center row
        [0, 0, 0],
        [0, 2, 0],
        [0, -2, 0],
        [0, 4, 0],
        [0, -4, 0],

        # First ring
        [1.732, 1, 0],   # 1.732 = sqrt(3)
        [-1.732, 1, 0],
        [1.732, -1, 0],
        [-1.732, -1, 0],

        # Second ring
        [1.732, 3, 0],
        [-1.732, 3, 0],
        [1.732, -3, 0],
        [-1.732, -3, 0],
    ])

    # Calculate the optimal outer hexagon size
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)

    # Validate and refine if necessary
    valid, message = validate_configuration(inner_hex_data, outer_side_length)

    # If not valid, try a simpler but more robust configuration
    if not valid:
        # Use a more conservative approach with known good values
        inner_hex_data = np.array([
            # Center
            [0, 0, 0],

            # Around center
            [0, 2, 0],
            [1.732, 1, 0],
            [1.732, -1, 0],
            [0, -2, 0],
            [-1.732, -1, 0],
            [-1.732, 1, 0],

            # Outer ring
            [0, 4, 0],
            [3.464, 2, 0],
            [3.464, -2, 0],
            [0, -4, 0],
            [-3.464, -2, 0],
            [-3.464, 2, 0],
        ])

        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)

    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END