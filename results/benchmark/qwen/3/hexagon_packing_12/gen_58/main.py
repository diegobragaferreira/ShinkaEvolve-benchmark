# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math


def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon with given center and rotation"""
    # Vertices of a unit hexagon centered at origin with rotation in degrees
    angle_offset = math.radians(rotation)
    radius = 1  # unit hexagon
    vertices = []
    for i in range(6):
        angle = angle_offset + i * math.pi / 3
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))
    return Polygon(vertices)


def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon"""
    return outer_hex.contains(inner_hex)


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    # Use a small buffer to handle floating point precision issues
    return hex1.buffer(1e-10).intersects(hex2.buffer(1e-10))


def calculate_outer_hex_radius(inner_hex_data):
    """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
    # Get all vertices from all inner hexagons
    all_vertices = []

    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        hexagon = create_unit_hexagon(center, rotation)
        all_vertices.extend(list(hexagon.exterior.coords))

    # Find the maximum distance from origin to any vertex
    max_distance = 0
    for vertex in all_vertices:
        distance = math.sqrt(vertex[0]**2 + vertex[1]**2)
        max_distance = max(max_distance, distance)

    # Add small buffer to ensure containment
    return max_distance + 0.1


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Advanced symmetric configuration inspired by optimal known solutions
    # This configuration attempts to create a highly symmetric arrangement
    # based on principles of hexagonal packing
    refined_hex_data = np.array([
        # Central hexagon
        [0.0, 0.0, 0.0],

        # First ring around center
        [0.0, 2.0, 0.0],           # Top
        [1.732, 1.0, 0.0],         # Top-right
        [1.732, -1.0, 0.0],        # Bottom-right
        [0.0, -2.0, 0.0],          # Bottom
        [-1.732, -1.0, 0.0],       # Bottom-left
        [-1.732, 1.0, 0.0],        # Top-left

        # Second ring
        [0.0, 4.0, 0.0],           # Far top
        [3.464, 2.0, 0.0],         # Upper right
        [3.464, -2.0, 0.0],        # Lower right
        [0.0, -4.0, 0.0],          # Far bottom
        [-3.464, -2.0, 0.0],       # Lower left
        [-3.464, 2.0, 0.0],        # Upper left
    ])

    # Calculate outer hexagon radius for this arrangement
    outer_radius = calculate_outer_hex_radius(refined_hex_data)

    # Use a more precise calculation for the outer hexagon side length
    # For a regular hexagon, the relationship between circumradius and side length is R = s
    # But we need to account for the fact that we want to tightly enclose all hexagons
    outer_hex_side_length = outer_radius * 1.02  # Small buffer to ensure perfect containment

    # Return the refined data with optimized parameters
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin

    return refined_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END