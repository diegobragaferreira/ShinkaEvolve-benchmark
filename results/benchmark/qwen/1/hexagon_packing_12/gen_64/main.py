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
    # More sophisticated initial configuration based on hexagonal packing
    # Arrange in concentric rings around center
    inner_hex_data = np.array([
        [0.0, 0.0, 0],      # center
        [1.0, 0.0, 0],      # right
        [0.5, 0.866, 0],    # upper right
        [-0.5, 0.866, 0],   # upper left
        [-1.0, 0.0, 0],     # left
        [-0.5, -0.866, 0],  # lower left
        [0.5, -0.866, 0],   # lower right
        [1.5, 1.732, 0],    # far upper right
        [0.0, 1.732, 0],    # upper center
        [-1.5, 1.732, 0],   # far upper left
        [-1.5, -1.732, 0],  # far lower left
        [0.0, -1.732, 0],   # lower center
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 4.0  # adjusted to be reasonable for initial configuration

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END