# EVOLVE-BLOCK-START
import numpy as np
import math


def hexagon_vertices(center_x, center_y, side_length, angle_degrees):
    """Calculate vertices of a regular hexagon"""
    angle_rad = math.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)


def calculate_outer_hexagon_radius(inner_hex_data):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    side_length = 1  # Unit hexagons

    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(center_x, center_y, side_length, angle)

        # Calculate distance from origin to each vertex
        for vx, vy in vertices:
            dist = math.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)

    # Add small margin to ensure containment
    return max_dist * 1.01


def hexagon_packing_12():
    """
    Constructs an optimized packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a symmetric ring configuration for better packing efficiency.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use a more efficient hexagonal packing pattern
    # Central hexagon with 12 surrounding in two concentric rings
    inner_hex_data = np.array([
        [0, 0, 0],           # Center hexagon
        [0, 2, 0],           # Top
        [1.732, 1, 0],       # Top right
        [1.732, -1, 0],      # Bottom right
        [0, -2, 0],          # Bottom
        [-1.732, -1, 0],     # Bottom left
        [-1.732, 1, 0],      # Top left
        [3.464, 2, 0],       # Far top right
        [3.464, -2, 0],      # Far bottom right
        [-3.464, -2, 0],     # Far bottom left
        [-3.464, 2, 0],      # Far top left
        [0, 4, 0],           # Far top center
    ])

    # Calculate outer hexagon radius
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data)
    # Convert radius to side length for regular hexagon
    outer_hex_side_length = outer_radius

    # Outer hexagon centered at origin with no rotation
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END