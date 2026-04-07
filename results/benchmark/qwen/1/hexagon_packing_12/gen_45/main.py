# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math

# Helper function to generate hexagon vertices
def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon centered at (center_x, center_y) with rotation angle_deg"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

# Helper function to check if a point is inside a hexagon
def point_in_hexagon(point, hexagon_vertices):
    """Check if a point is inside a hexagon using ray casting"""
    x, y = point
    n = len(hexagon_vertices)
    inside = False
    p1x, p1y = hexagon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = hexagon_vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Helper function to check if two hexagons overlap
def hexagons_overlap(vertices1, vertices2):
    """Check if two hexagons represented by their vertices overlap using separating axis theorem"""
    # For unit hexagons, we can do a simpler check
    # Get all edge normals for both polygons
    edges1 = [vertices1[i] - vertices1[(i+1)%6] for i in range(6)]
    edges2 = [vertices2[i] - vertices2[(i+1)%6] for i in range(6)]

    # Check all possible separating axes
    axes = edges1 + edges2

    for axis in axes:
        # Project both polygons onto this axis
        proj1 = [np.dot(vertex, axis) for vertex in vertices1]
        proj2 = [np.dot(vertex, axis) for vertex in vertices2]

        # Check if projections overlap
        if max(proj1) < min(proj2) or max(proj2) < min(proj1):
            return False  # Found separating axis, no overlap

    return True  # Overlapping

# Function to calculate objective and constraints
def evaluate_configuration(params):
    """Evaluate a configuration of 12 hexagons plus outer hexagon parameters"""
    # Extract inner hexagon parameters (x, y, angle for each of 12 hexagons)
    inner_positions_angles = params[:36].reshape(12, 3)

    # Extract outer hexagon parameters (x, y, angle)
    outer_center_x, outer_center_y, outer_angle = params[36:39]

    # Fixed outer hexagon side length
    outer_radius = 5.0  # Start with reasonable value

    # Calculate bounding box for inner hexagons
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    total_penalty = 0.0

    # Check containment and collect boundary info
    for i in range(12):
        x, y, angle = inner_positions_angles[i]
        vertices = hexagon_vertices(x, y, angle)

        # Check containment in outer hexagon
        outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_radius)

        # Check if all vertices are inside the outer hexagon
        for vertex in vertices:
            if not point_in_hexagon(vertex, outer_vertices):
                # Apply penalty for containment violations
                distance = min(np.linalg.norm(vertex - v) for v in outer_vertices)
                total_penalty += distance * 1000  # Large penalty

        # Update bounding box
        for vertex in vertices:
            min_x = min(min_x, vertex[0])
            max_x = max(max_x, vertex[0])
            min_y = min(min_y, vertex[1])
            max_y = max(max_y, vertex[1])

    # Check for overlaps between inner hexagons
    for i in range(12):
        for j in range(i+1, 12):
            x1, y1, angle1 = inner_positions_angles[i]
            x2, y2, angle2 = inner_positions_angles[j]

            vertices1 = hexagon_vertices(x1, y1, angle1)
            vertices2 = hexagon_vertices(x2, y2, angle2)

            if hexagons_overlap(vertices1, vertices2):
                # Apply penalty for overlap
                total_penalty += 10000000  # Large penalty

    # Calculate required outer hexagon size
    # We need to make sure that our inner hexagons fit properly
    required_size = max(abs(max_x), abs(min_x), abs(max_y), abs(min_y)) + 1.5  # Add margin

    # Objective: minimize outer hexagon size (maximize 1/size)
    # However, we want to be generous with the size calculation here since
    # we're mainly focused on finding a good arrangement
    objective = -required_size  # Negative because we want to maximize 1/size

    # Add penalties to objective
    final_objective = objective + total_penalty

    return final_objective

# Main optimization function
def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimization to find near-optimal arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Initial guess based on a better arrangement than the naive grid
    # Using a pattern that has been found to work well for hexagon packing
    initial_positions = np.array([
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom
        [1.732, 1, 0],  # top-right
        [-1.732, 1, 0], # top-left
        [1.732, -1, 0], # bottom-right
        [-1.732, -1, 0],# bottom-left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3, 0],  # top-top-right
        [-1.732, 3, 0], # top-top-left
        [0, 4, 0],      # far top center
    ])

    # Flatten the initial positions for differential evolution
    initial_params = initial_positions.flatten()

    # Add outer hexagon parameters
    initial_params = np.concatenate([initial_params, [0, 0, 0]])  # outer center and angle

    # Bounds for parameters
    # Inner hexagons: x,y in [-5,5], angle in [0,360)
    bounds = [(-5, 5)] * 24 + [(0, 360)] * 12 + [(-5, 5)] * 2 + [(0, 360)]  # outer hex parameters

    # Run optimization
    result = differential_evolution(evaluate_configuration, bounds, seed=42, maxiter=100, popsize=15)

    # Extract results
    final_params = result.x
    inner_positions_angles = final_params[:36].reshape(12, 3)
    outer_center_x, outer_center_y, outer_angle = final_params[36:39]

    # Calculate the actual outer hexagon side length needed
    # Find the maximum distance from center to any inner hexagon vertex
    max_dist = 0
    for i in range(12):
        x, y, angle = inner_positions_angles[i]
        vertices = hexagon_vertices(x, y, angle)
        for vertex in vertices:
            dist = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
            max_dist = max(max_dist, dist)

    # Add some padding for safe containment
    outer_hex_side_length = max_dist + 1.0

    # Create outer hexagon data
    outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])

    # Return results
    return inner_positions_angles, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END