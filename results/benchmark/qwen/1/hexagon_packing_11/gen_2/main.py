# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def create_hexagon_vertices(center, side_length, angle_degrees):
    """Create vertices of a regular hexagon given center, side length, and rotation."""
    angle_rad = math.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center[0] + side_length * math.cos(angle)
        y = center[1] + side_length * math.sin(angle)
        vertices.append([x, y])
    return np.array(vertices)

def check_hexagon_intersection(hex1_vertices, hex2_vertices):
    """Check if two hexagons intersect using separating axis theorem."""
    # Get edges of both hexagons
    edges1 = []
    edges2 = []

    for i in range(6):
        edge1 = hex1_vertices[(i+1)%6] - hex1_vertices[i]
        edge2 = hex2_vertices[(i+1)%6] - hex2_vertices[i]
        edges1.append(edge1)
        edges2.append(edge2)

    # Test all potential separating axes
    all_axes = []
    for edge in edges1 + edges2:
        # Normalize perpendicular vector
        perp = np.array([-edge[1], edge[0]])
        if np.linalg.norm(perp) > 1e-10:
            perp = perp / np.linalg.norm(perp)
            all_axes.append(perp)

    # Check projection overlap
    for axis in all_axes:
        projections1 = np.dot(hex1_vertices, axis)
        projections2 = np.dot(hex2_vertices, axis)

        min1, max1 = np.min(projections1), np.max(projections1)
        min2, max2 = np.min(projections2), np.max(projections2)

        # If projections don't overlap, there's separation
        if max1 < min2 or max2 < min1:
            return False

    return True

def check_containment(inner_hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    from shapely.geometry import Polygon

    outer_polygon = Polygon(outer_hex_vertices)

    # Check if all vertices of inner hex are inside outer hex
    for vertex in inner_hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def calculate_outer_hexagon_side_length(inner_hex_data, outer_hex_side_length_guess):
    """Calculate minimum required outer hexagon size."""
    # Set up the problem to find the minimal outer hexagon
    # Use a constraint-based approach
    return outer_hex_side_length_guess

def evaluate_solution(params):
    """Evaluate a solution given parameters."""
    # params: [x1,y1,a1,x2,y2,a2,...,xn,yn,an,R]
    n = 11
    # Extract positions and angles for inner hexagons (first 3n parameters)
    inner_positions_angles = params[:3*n].reshape(n, 3)
    # Outer hexagon side length (last parameter)
    R = params[3*n]

    # Create inner hexagons
    inner_hexes = []
    for i in range(n):
        center = inner_positions_angles[i][:2]
        angle = inner_positions_angles[i][2]
        hex_verts = create_hexagon_vertices(center, 1.0, angle)
        inner_hexes.append(hex_verts)

    # Create outer hexagon with side length R
    outer_center = [0, 0]
    outer_hex_verts = create_hexagon_vertices(outer_center, R, 0)

    # Check for intersections
    penalty = 0.0

    # Check pairwise intersections
    for i in range(n):
        for j in range(i+1, n):
            if check_hexagon_intersection(inner_hexes[i], inner_hexes[j]):
                penalty += 1000.0  # Large penalty for overlap

    # Check containment - all inner vertices must be inside outer hexagon
    for i in range(n):
        if not check_containment(inner_hexes[i], outer_hex_verts):
            penalty += 1000.0  # Large penalty for containment violation

    # Objective: minimize -1/R (since we want to maximize 1/R)
    obj_value = -1.0/R + penalty

    return obj_value

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11

    # Better initial configuration inspired by known dense packings
    # Start with a more symmetric arrangement
    initial_positions_angles = np.array([
        [0, 0, 0],       # center
        [-1.5, 0, 0],    # left
        [1.5, 0, 0],     # right
        [0, 2.6, 0],     # top
        [0, -2.6, 0],    # bottom
        [-1.5, 2.6, 0],  # top-left
        [1.5, 2.6, 0],   # top-right
        [-1.5, -2.6, 0], # bottom-left
        [1.5, -2.6, 0],  # bottom-right
        [-3.0, 0, 0],    # far left
        [3.0, 0, 0],     # far right
    ])

    # Initial guess for outer hexagon side length
    initial_R = 5.0

    # Flatten initial parameters
    initial_params = np.concatenate([initial_positions_angles.flatten(), [initial_R]])

    # Define bounds for optimization (positions, angles, outer radius)
    bounds = []
    # Positions: [-10, 10] for x and y
    for _ in range(n):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    # Outer radius: [1, 20]
    bounds.append((1, 20))

    # Optimization options
    try:
        result = minimize(evaluate_solution, initial_params, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 1000})

        if result.success:
            final_params = result.x
            # Extract results
            final_positions_angles = final_params[:3*n].reshape(n, 3)
            outer_hex_side_length = final_params[3*n]

            # Return the optimized solution
            inner_hex_data = final_positions_angles
            outer_hex_data = np.array([0, 0, 0])  # centered at origin
        else:
            # Fall back to the original configuration if optimization fails
            inner_hex_data = initial_positions_angles
            outer_hex_data = np.array([0, 0, 0])
            outer_hex_side_length = 5.0

    except Exception as e:
        # If optimization fails, use initial configuration
        inner_hex_data = initial_positions_angles
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 5.0

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END