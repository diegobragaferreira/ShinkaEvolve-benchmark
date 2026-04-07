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
    # More sophisticated initial arrangement based on hexagonal close packing
    # Place 12 hexagons in a pattern resembling a hexagonal lattice with 3 concentric rings
    inner_hex_data = np.array([
        [0.0, 0.0, 0],      # center
        [0.0, 2.0, 0],      # top
        [1.732, 1.0, 0],    # top-right
        [1.732, -1.0, 0],   # bottom-right
        [0.0, -2.0, 0],     # bottom
        [-1.732, -1.0, 0],  # bottom-left
        [-1.732, 1.0, 0],   # top-left
        [3.464, 2.0, 0],    # far top-right
        [3.464, -2.0, 0],   # far bottom-right
        [0.0, -4.0, 0],     # far bottom
        [-3.464, -2.0, 0],  # far bottom-left
        [-3.464, 2.0, 0],   # far top-left
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 5.0  # reasonable starting point for this arrangement

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END