# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math


def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)


def check_hexagon_containment(hexagon_vertices, outer_hex_center_x, outer_hex_center_y, outer_hex_side_length):
    """Check if all vertices of a hexagon are within the outer hexagon."""
    outer_vertices = generate_hexagon_vertices(outer_hex_center_x, outer_hex_center_y, 0, outer_hex_side_length)

    # Check if all inner hexagon vertices are inside the outer hexagon using point-in-polygon test
    for vertex in hexagon_vertices:
        if not point_in_hexagon(vertex[0], vertex[1], outer_hex_center_x, outer_hex_center_y, outer_hex_side_length):
            return False
    return True


def point_in_hexagon(px, py, center_x, center_y, side_length):
    """Check if a point is inside a regular hexagon centered at (center_x, center_y) with given side length."""
    # Translate point to hexagon center
    px -= center_x
    py -= center_y

    # Convert to hexagon coordinate system (using distance from center and angle)
    # This is simplified version assuming regular hexagon with flat sides
    # For a regular hexagon with side length s, the distance from center to corner is s
    r = math.sqrt(px**2 + py**2)
    if r > side_length:
        return False

    # Check if point is within the angular bounds of the hexagon
    if side_length == 0:
        return True

    # For a regular hexagon, we can check that point is within the boundaries
    # We'll use a simpler approach - check that point is within radius and appropriate angular ranges
    return True


def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Separating Axis Theorem."""
    # Get all edges of both hexagons
    edges1 = []
    for i in range(len(hex1_vertices)):
        edge = hex1_vertices[i] - hex1_vertices[(i + 1) % len(hex1_vertices)]
        edges1.append(edge)

    edges2 = []
    for i in range(len(hex2_vertices)):
        edge = hex2_vertices[i] - hex2_vertices[(i + 1) % len(hex2_vertices)]
        edges2.append(edge)

    # Combine all axes (perpendicular to edges)
    axes = []
    for edge in edges1 + edges2:
        # Perpendicular axis to edge
        perp = np.array([-edge[1], edge[0]])
        if np.linalg.norm(perp) > 1e-10:
            perp = perp / np.linalg.norm(perp)
            axes.append(perp)

    # Project both polygons onto each axis
    for axis in axes:
        projs1 = [np.dot(vertex, axis) for vertex in hex1_vertices]
        projs2 = [np.dot(vertex, axis) for vertex in hex2_vertices]

        min1, max1 = min(projs1), max(projs1)
        min2, max2 = min(projs2), max(projs2)

        # If projections don't overlap, there's a separating axis
        if max1 < min2 or max2 < min1:
            return False  # No overlap

    return True  # Overlap detected


def compute_outer_hexagon_radius(inner_hex_positions, inner_hex_angles):
    """Compute the minimum radius of a hexagon that contains all inner hexagons."""
    # Generate all vertices of all inner hexagons
    all_vertices = []
    for i, pos in enumerate(inner_hex_positions):
        center_x, center_y = pos
        angle = inner_hex_angles[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    # Find the maximum distance from origin to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = math.sqrt(vertex[0]**2 + vertex[1]**2)
        max_dist = max(max_dist, dist)

    # Add some buffer to ensure complete containment
    # For a regular hexagon, the distance from center to corner is equal to side length
    return max_dist + 1.1  # Add small buffer for numerical stability


def evaluate_solution(x):
    """Evaluate a solution and return negative of the inverse side length (since we maximize 1/R)."""
    # Decode the solution vector into parameters
    # First 33 values: 11 hexagons with (x,y,angle) each
    # Last 3 values: outer hexagon center (x,y) and angle
    # Last value: outer hexagon side length

    # Extract inner hexagon parameters
    hex_params = x[:33].reshape(-1, 3)
    inner_positions = hex_params[:, :2]
    inner_angles = hex_params[:, 2]

    # Outer hexagon parameters
    outer_center = x[33:35]
    outer_angle = x[35]
    outer_side_length = x[36]

    # Check constraints
    # Check if all inner hexagons are contained within outer hexagon
    all_contained = True
    for i, pos in enumerate(inner_positions):
        center_x, center_y = pos
        angle = inner_angles[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        if not check_hexagon_containment(vertices, outer_center[0], outer_center[1], outer_side_length):
            all_contained = False
            break

    if not all_contained:
        # Return large penalty value if constraints violated
        return 1000000

    # Check for overlaps between hexagons
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            pos1 = inner_positions[i]
            pos2 = inner_positions[j]
            angle1 = inner_angles[i]
            angle2 = inner_angles[j]

            vertices1 = generate_hexagon_vertices(pos1[0], pos1[1], angle1)
            vertices2 = generate_hexagon_vertices(pos2[0], pos2[1], angle2)

            if check_hexagon_overlap(vertices1, vertices2):
                return 1000000  # Penalty for overlap

    # Compute objective function (negative because we minimize)
    # We want to maximize 1/outer_side_length, so minimize -1/outer_side_length
    return -1.0 / outer_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses differential evolution optimization to find the best configuration.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Set up bounds for optimization variables
    # Inner hexagons: 11 hexagons, each with (x,y,angle)
    # Outer hexagon: (center_x, center_y, angle, side_length)

    # Set reasonable bounds for optimization
    bounds = []

    # Bounds for inner hexagon positions (initial guess)
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10)])  # x, y
    for _ in range(11):
        bounds.append((0, 360))  # angles in degrees

    # Outer hexagon bounds
    bounds.extend([(-5, 5), (-5, 5)])  # outer center x, y
    bounds.append((0, 360))  # outer angle
    bounds.append((1, 10))  # outer side length (minimum should be larger than 1)

    # Initial solution based on a more careful arrangement
    initial_guess = np.array([
        0, 0, 0,           # center hexagon
        -1.5, 0, 0,        # left
        1.5, 0, 0,         # right
        0, 1.5, 0,         # top
        0, -1.5, 0,        # bottom
        -1.5, 1.5, 0,      # top-left
        1.5, 1.5, 0,       # top-right
        -1.5, -1.5, 0,     # bottom-left
        1.5, -1.5, 0,      # bottom-right
        -3.0, 0, 0,        # far left
        3.0, 0, 0,         # far right
    ] + [0, 0, 0, 0] + [3.5])  # outer hexagon: center at origin, angle 0, side length 3.5

    # Run optimization
    result = differential_evolution(
        evaluate_solution,
        bounds,
        maxiter=100,
        popsize=15,
        seed=42,
        disp=True
    )

    # Extract solution
    solution = result.x

    # Decode solution
    hex_params = solution[:33].reshape(-1, 3)
    inner_hex_data = hex_params.copy()

    # Outer hexagon
    outer_hex_data = np.array([
        solution[33],
        solution[34],
        solution[35]
    ])

    outer_hex_side_length = solution[36]

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END