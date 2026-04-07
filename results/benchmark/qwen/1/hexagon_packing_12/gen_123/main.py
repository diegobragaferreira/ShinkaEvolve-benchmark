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
    # More sophisticated arrangement based on hexagonal packing principles
    # Arrange 12 hexagons in a pattern that minimizes the outer hexagon size

    # Calculate the positions for 12 hexagons in a compact arrangement
    # Using a combination of central, radial and edge placement

    # Distance between centers of adjacent hexagons in a tight packing
    # For unit hexagons, this is 2 (center-to-center distance for touching hexes)
    hex_radius = 1.0
    hex_center_distance = 2.0 * hex_radius

    # Positioning parameters
    # 1 center hexagon
    # 6 surrounding hexagons in first ring
    # 5 additional hexagons in second ring (arranged to minimize outer size)

    inner_hex_data = np.zeros((12, 3))

    # Center hexagon
    inner_hex_data[0] = [0, 0, 0]

    # First ring of 6 hexagons (around center)
    for i in range(6):
        angle = i * np.pi / 3  # 60 degrees increments
        x = hex_center_distance * np.cos(angle)
        y = hex_center_distance * np.sin(angle)
        inner_hex_data[i+1] = [x, y, 0]

    # Second ring of 5 hexagons - placed more strategically to reduce outer size
    # These are positioned to create a tighter packing
    offset_angle = np.pi / 6  # Half of 60 degrees
    for i in range(5):
        # Position in second ring
        angle = i * 2 * np.pi / 5 + offset_angle  # 72 degree increments with offset
        radius = hex_center_distance * np.sqrt(3)  # Approximate spacing for second ring
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        inner_hex_data[i+7] = [x, y, 0]

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    # Estimate outer hexagon side length based on extreme positions
    # We need to compute the maximum distance from center to any vertex of any hexagon
    max_dist = 0
    for i in range(12):
        x, y = inner_hex_data[i][:2]
        # The furthest point of a hexagon from center includes its radius plus the center offset
        # For unit hexagons, the distance from center to vertex is 1
        dist_from_center = np.sqrt(x*x + y*y) + 1.0  # Adding 1 for hexagon radius
        max_dist = max(max_dist, dist_from_center)

    # Outer hexagon side length needs to be at least max_dist for containment
    outer_hex_side_length = max_dist * 1.1  # Add some padding

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END