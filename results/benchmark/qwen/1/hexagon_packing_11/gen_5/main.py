# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math
import time


def hexagon_vertices(center_x, center_y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_degrees)
    # Vertices of a regular hexagon centered at origin with side_length 1
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = side_length * math.cos(theta)
        y = side_length * math.sin(theta)
        base_vertices.append((x, y))

    # Apply translation and rotation
    rotated_vertices = []
    for x, y in base_vertices:
        new_x = center_x + x
        new_y = center_y + y
        rotated_vertices.append((new_x, new_y))

    return np.array(rotated_vertices)


def hexagon_contains_point(hex_vertices, point):
    """Check if a point is inside a hexagon using ray casting method."""
    # Simplified version using distance to edges
    # For a regular hexagon, this is more reliable than complex polygon tests
    center_x = np.mean(hex_vertices[:, 0])
    center_y = np.mean(hex_vertices[:, 1])

    # Check distance to center
    dist_to_center = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)

    # For a unit hexagon, max distance from center to vertex is 1
    # If point is closer to center than radius, it might be inside
    return dist_to_center <= 1.0


def hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using distance calculation."""
    # Calculate centroids
    centroid1 = np.mean(hex1_vertices, axis=0)
    centroid2 = np.mean(hex2_vertices, axis=0)

    # Distance between centroids
    dist = np.linalg.norm(centroid1 - centroid2)

    # Two unit hexagons overlap if their centers are closer than 2 units apart
    return dist < 2.0


def is_valid_arrangement(inner_positions, outer_side_length):
    """Check if an arrangement is valid: no overlaps and all inside outer hexagon."""
    # Generate vertices for all inner hexagons
    inner_hexes = []
    for i, (x, y, angle) in enumerate(inner_positions):
        vertices = hexagon_vertices(x, y, angle)
        inner_hexes.append(vertices)

    # Check for overlaps between any pair of inner hexagons
    for i in range(len(inner_hexes)):
        for j in range(i+1, len(inner_hexes)):
            if hexagon_overlap(inner_hexes[i], inner_hexes[j]):
                return False

    # Check if all inner hexagons fit within outer hexagon
    # Outer hexagon center is at (0, 0), side_length is given
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)

    # Check if all vertices of inner hexagons are within outer hexagon
    for vertices in inner_hexes:
        for vertex in vertices:
            # Use simpler containment check based on distance to center
            dist_to_center = math.sqrt(vertex[0]**2 + vertex[1]**2)
            if dist_to_center > outer_side_length:
                return False

    return True


def objective_function(params):
    """Objective function to minimize: negative of 1/outer_hex_side_length"""
    # Parameters: [x1, y1, angle1, x2, y2, angle2, ..., x11, y11, angle11, outer_side_length]
    n_inner = 11
    inner_params = params[:-1]
    outer_side_length = params[-1]

    # Reshape into positions array
    inner_positions = np.array(inner_params).reshape(-1, 3)

    # Check validity and return penalty if invalid
    if not is_valid_arrangement(inner_positions, outer_side_length):
        # Large penalty for invalid configurations
        return 1e10

    # Return negative inverse of outer side length (to be minimized)
    return -1.0 / outer_side_length


def optimize_hexagon_packing():
    """Use optimization to find the best arrangement of 11 unit hexagons."""
    n_inner = 11
    bounds = []

    # Add bounds for inner hexagon positions and angles
    # Positions: x, y in range [-5, 5] (reasonable bounds)
    for _ in range(n_inner):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])

    # Outer hexagon side length bound: reasonable range
    bounds.append((1.0, 10.0))  # Minimum 1.0 to ensure hexagons can fit

    # Starting point: initial guess
    initial_guess = []
    # Center hexagon
    initial_guess.extend([0, 0, 0])
    # Arrange others in a pattern similar to what we know works well
    positions = [
        (-2, 0, 0), (2, 0, 0),
        (-1, 1.732, 0), (1, 1.732, 0),
        (-1, -1.732, 0), (1, -1.732, 0),
        (-3, 0, 0), (3, 0, 0),
        (-2, 3.464, 0), (2, 3.464, 0),
        (-2, -3.464, 0), (2, -3.464, 0)
    ]

    for x, y, angle in positions:
        initial_guess.extend([x, y, angle])

    # Last parameter: outer side length
    initial_guess.append(5.0)

    # Optimization with bounds
    result = differential_evolution(objective_function, bounds, seed=42,
                                   maxiter=200, popsize=15, disp=False)

    if result.success:
        final_params = result.x
        inner_positions = np.array(final_params[:-1]).reshape(-1, 3)
        outer_side_length = final_params[-1]
        return inner_positions, outer_side_length
    else:
        # Fallback to simple arrangement if optimization fails
        inner_positions = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0]
        ])
        outer_side_length = 8.0
        return inner_positions, outer_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Find optimized solution
    inner_positions, outer_side_length = optimize_hexagon_packing()

    # Set the outer hexagon at the origin with no rotation
    outer_hex_data = np.array([0, 0, 0])

    # Ensure we don't exceed time limits
    elapsed = time.time() - start_time
    if elapsed > 175:  # Leave some buffer
        raise TimeoutError("Optimization exceeded time limit")

    return inner_positions, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END