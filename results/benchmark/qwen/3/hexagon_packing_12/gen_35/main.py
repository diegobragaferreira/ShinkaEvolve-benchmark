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
    # More efficient hexagonal arrangement based on close packing principles
    # Using a central hexagon surrounded by 6 others in a ring, with 5 more in another ring
    sqrt3 = np.sqrt(3)

    # Positioning for 12 hexagons in a hexagonal pattern
    # Central hexagon (0,0)
    # First ring (6 hexagons at distance 2 from center)
    # Second ring (5 hexagons in a hexagonal pattern)

    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 2.0, 0.0],           # top
        [1.732, 1.0, 0.0],         # top-right
        [1.732, -1.0, 0.0],        # bottom-right
        [0.0, -2.0, 0.0],          # bottom
        [-1.732, -1.0, 0.0],       # bottom-left
        [-1.732, 1.0, 0.0],        # top-left
        [3.464, 2.0, 0.0],         # far top-right
        [3.464, -2.0, 0.0],        # far bottom-right
        [-3.464, -2.0, 0.0],       # far bottom-left
        [-3.464, 2.0, 0.0],        # far top-left
        [0.0, -4.0, 0.0]           # far bottom
    ])

    # Adjust outer hexagon to be tighter fit
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    # The maximum distance from origin to any vertex of the inner hexagons
    # Estimate based on the farthest points
    outer_hex_side_length = 5.0  # This will be refined during optimization

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END