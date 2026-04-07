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
    # More sophisticated initial arrangement inspired by hexagonal packing principles
    # Using a pattern that places hexagons in concentric rings around center
    inner_hex_data = np.array([
        [0, 0, 0],           # center
        [1.732, 0, 0],       # right
        [-1.732, 0, 0],      # left
        [0.866, 1.5, 0],     # top-right
        [-0.866, 1.5, 0],    # top-left
        [0.866, -1.5, 0],    # bottom-right
        [-0.866, -1.5, 0],   # bottom-left
        [2.598, 1.5, 0],     # far top-right
        [-2.598, 1.5, 0],    # far top-left
        [2.598, -1.5, 0],    # far bottom-right
        [-2.598, -1.5, 0],   # far bottom-left
        [0, -3, 0],          # bottom-center
    ])

    # Initial estimate for outer hexagon size - should be refined by optimization later
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 4.0  # Estimate based on arrangement

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END