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
    # Improved initialization based on tighter packing arrangements
    # Using a combination of central cluster and radial arrangement

    # Define unit hexagon radius (distance from center to corner)
    hex_radius = 1.0

    # Arrange 12 hexagons in a more optimal configuration
    # Central hexagon + surrounding ring with optimized spacing
    inner_hex_data = np.array([
        [0, 0, 0],           # center hexagon

        # First ring around center (6 hexagons)
        [hex_radius * 2, 0, 0],           # right
        [hex_radius, hex_radius * np.sqrt(3), 0],   # upper right
        [-hex_radius, hex_radius * np.sqrt(3), 0],  # upper left
        [-hex_radius * 2, 0, 0],         # left
        [-hex_radius, -hex_radius * np.sqrt(3), 0], # lower left
        [hex_radius, -hex_radius * np.sqrt(3), 0],  # lower right

        # Second ring (6 hexagons)
        [hex_radius * 3, hex_radius * np.sqrt(3), 0],     # upper right
        [hex_radius * 3, -hex_radius * np.sqrt(3), 0],    # lower right
        [0, hex_radius * 2 * np.sqrt(3), 0],              # upper
        [0, -hex_radius * 2 * np.sqrt(3), 0],             # lower
        [-hex_radius * 3, hex_radius * np.sqrt(3), 0],    # upper left
        [-hex_radius * 3, -hex_radius * np.sqrt(3), 0],   # lower left
    ])

    # Compute required outer hexagon size
    # Find maximum distance from origin to any vertex of any inner hexagon
    max_distance = 0

    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        # Each hexagon has 6 vertices at distance 1 from center
        # The maximum distance from origin to any vertex
        distance = np.sqrt(x*x + y*y) + 1  # Add 1 for hexagon radius
        max_distance = max(max_distance, distance)

    # Add a small margin to ensure proper containment
    outer_hex_side_length = max_distance * 1.1  # 10% margin for safety

    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END