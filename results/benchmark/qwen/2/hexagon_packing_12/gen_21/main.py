# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 12

    # Generate a more symmetric initial configuration
    # Place 12 hexagons in a pattern that respects hexagonal symmetry
    # Center hexagon + 6 surrounding hexagons + 5 additional hexagons
    inner_hex_data = np.zeros((12, 3))

    # Define hexagon radius (distance from center to corner for unit hexagon)
    hex_radius = 1.0

    # Place center hexagon
    inner_hex_data[0] = [0, 0, 0]

    # Place 6 surrounding hexagons in a ring around center
    for i in range(6):
        angle = i * np.pi / 3
        x = hex_radius * np.cos(angle)
        y = hex_radius * np.sin(angle)
        inner_hex_data[i+1] = [x, y, 0]

    # Place remaining 5 hexagons in a second ring
    # This creates a more symmetric configuration with better space utilization
    for i in range(5):
        angle = i * 2 * np.pi / 5 + np.pi / 5  # Offset to avoid perfect alignment
        distance_from_center = 2 * hex_radius
        x = distance_from_center * np.cos(angle)
        y = distance_from_center * np.sin(angle)
        inner_hex_data[i+7] = [x, y, 0]

    # Set outer hexagon to be centered at origin
    outer_hex_data = np.array([0, 0, 0])

    # Estimate initial outer hexagon size needed to contain all hexagons
    # The maximum distance from center to any hexagon vertex
    max_distance = 3.0  # Conservative estimate based on placement
    outer_hex_side_length = max_distance + 0.1  # Add small margin for safety

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END