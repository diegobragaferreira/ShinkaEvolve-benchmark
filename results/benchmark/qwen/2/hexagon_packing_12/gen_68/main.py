# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from math import sqrt, cos, sin, pi


def generate_hexagon_vertices(center_x, center_y, rotation_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length"""
    rotation_rad = np.radians(rotation_degrees)
    vertices = []
    for i in range(6):
        angle = rotation_rad + i * pi / 3
        x = center_x + side_length * cos(angle)
        y = center_y + side_length * sin(angle)
        vertices.append((x, y))
    return vertices


def create_hexagon_polygon(center_x, center_y, rotation_degrees, side_length=1):
    """Create a Shapely polygon representation of a hexagon"""
    vertices = generate_hexagon_vertices(center_x, center_y, rotation_degrees, side_length)
    return Polygon(vertices)


def check_containment(hexagon_polygon, outer_hex_polygon):
    """Check if a hexagon is fully contained within the outer hexagon"""
    return outer_hex_polygon.contains(hexagon_polygon)


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)


def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []

    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        hex_poly = create_hexagon_polygon(center_x, center_y, rotation, 1.0)

        # Add all vertices of this hexagon
        for vertex in hex_poly.exterior.coords[:-1]:  # exclude last point (same as first)
            all_vertices.append(vertex)

    # Find bounding circle radius
    max_distance = 0
    center_point = (0, 0)

    for x, y in all_vertices:
        distance = sqrt((x - center_point[0])**2 + (y - center_point[1])**2)
        max_distance = max(max_distance, distance)

    # For a regular hexagon, the circumradius equals the side length
    # We need to account for the fact that our hexagons might be rotated
    # and we want the minimal enclosing hexagon
    return max_distance * 2 / sqrt(3)  # convert to side length of enclosing hexagon


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Better arrangement based on hexagonal lattice pattern
    # This follows a more efficient packing arrangement

    # Hexagonal lattice arrangement with 3 rings:
    # Ring 1: center hexagon
    # Ring 2: 6 surrounding hexagons at distance 2
    # Ring 3: 6 more hexagons at distance 3 from center

    # Using sqrt(3) factor for proper hexagonal spacing
    sqrt3 = sqrt(3)

    # Define positions in a hexagonal lattice pattern
    inner_hex_data = np.array([
        [0, 0, 0],           # center
        [2.0, 0, 0],         # right
        [-2.0, 0, 0],        # left
        [1.0, sqrt3, 0],     # top-right
        [-1.0, sqrt3, 0],    # top-left
        [1.0, -sqrt3, 0],    # bottom-right
        [-1.0, -sqrt3, 0],   # bottom-left
        [3.0, 0, 0],         # far right
        [-3.0, 0, 0],        # far left
        [1.5, 2.598, 0],     # upper far
        [-1.5, 2.598, 0],    # upper far left
        [1.5, -2.598, 0],    # lower far
        [-1.5, -2.598, 0],   # lower far left
    ])

    # Adjust to 12 hexagons (remove last two that exceed count)
    inner_hex_data = inner_hex_data[:12]

    # Recalculate appropriate side length for this arrangement
    outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data)

    # Ensure the outer hexagon is properly centered and oriented
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END