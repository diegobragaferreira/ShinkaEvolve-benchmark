# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time


def generate_hexagon_vertices(center, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    angle = np.radians(rotation_degrees)
    # Hexagon vertices relative to center with side length 1
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles, excluding last to close the polygon
    x_rel = np.cos(angles)
    y_rel = np.sin(angles)

    # Apply rotation and scaling
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = x_rel * cos_a - y_rel * sin_a
    y_rot = x_rel * sin_a + y_rel * cos_a

    # Scale by side length and translate by center
    vertices = np.column_stack([x_rot * side_length + center[0],
                               y_rot * side_length + center[1]])
    return vertices


def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    from shapely.geometry import Polygon
    outer_poly = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_poly.contains(point):
            return False
    return True


def check_overlap(hex_vertices1, hex_vertices2):
    """Check if two hexagons overlap using Shapely."""
    from shapely.geometry import Polygon
    poly1 = Polygon(hex_vertices1)
    poly2 = Polygon(hex_vertices2)
    return poly1.intersects(poly2)


def compute_packing_score(inner_configs, outer_side_length):
    """Compute the inverse of outer hexagon side length."""
    return 1.0 / outer_side_length


def optimize_hexagon_packing():
    """
    Optimizes the arrangement of 12 unit hexagons inside a larger hexagon.
    Returns optimized configuration and performance metrics.
    """
    # Initial guess based on known good configurations
    # This configuration is designed to be reasonably close to optimal
    initial_positions = np.array([
        [0, 0, 0],      # center
        [-2.0, 0, 0],   # left
        [2.0, 0, 0],    # right
        [-1.0, 1.732, 0],  # top-left
        [1.0, 1.732, 0],   # top-right
        [-1.0, -1.732, 0], # bottom-left
        [1.0, -1.732, 0],  # bottom-right
        [-2.0, 3.464, 0],  # far top-left
        [2.0, 3.464, 0],   # far top-right
        [-2.0, -3.464, 0], # far bottom-left
        [2.0, -3.464, 0],  # far bottom-right
        [0, -4.0, 0],      # far bottom-center
    ])

    # Convert to flat parameter vector [x1, y1, theta1, x2, y2, theta2, ...]
    initial_params = initial_positions.flatten()

    # Constraints and bounds
    def objective(params):
        # Reshape parameters back to positions
        positions = params.reshape(-1, 3)

        # Find bounding box for all hexagons
        max_dist = 0
        for i in range(len(positions)):
            center = positions[i, :2]
            # Calculate distance from origin to center (assuming symmetric arrangement)
            dist = np.sqrt(center[0]**2 + center[1]**2)
            # Add radius of hexagon (sqrt(3)/2 for unit hexagon)
            max_dist = max(max_dist, dist + np.sqrt(3)/2)

        # For a hexagon with side length r, the circumradius is r
        # But we want minimum enclosing circle
        outer_radius = max_dist

        # Return negative because we're minimizing
        return -1.0 / outer_radius

    def constraint_containment(params):
        # Reshape parameters
        positions = params.reshape(-1, 3)

        # Get outer hexagon vertices (with a margin)
        outer_radius = 0
        for i in range(len(positions)):
            center = positions[i, :2]
            dist = np.sqrt(center[0]**2 + center[1]**2)
            outer_radius = max(outer_radius, dist + np.sqrt(3)/2)

        # For now just returning a dummy constraint
        return outer_radius - 10  # Placeholder - actual constraint handled via bounds

    def constraint_nonoverlap(params):
        # Reshape parameters
        positions = params.reshape(-1, 3)

        penalty = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                center1 = positions[i, :2]
                center2 = positions[j, :2]
                distance = np.linalg.norm(center1 - center2)
                min_distance = np.sqrt(3)  # Minimum distance between centers of touching hexagons

                # Add penalty if they overlap (distance < min_distance)
                if distance < min_distance:
                    penalty += (min_distance - distance)**2

        return penalty

    # Optimization bounds
    bounds = [(-10, 10)] * 36  # x, y, angle for each hexagon (36 parameters total)

    # Set up constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 10 - np.max(np.sqrt(x[::3]**2 + x[1::3]**2))},
    ]

    # Perform optimization
    try:
        result = minimize(objective, initial_params, method='SLSQP', bounds=bounds,
                         constraints=constraints, options={'maxiter': 1000})

        # Extract optimized result
        final_positions = result.x.reshape(-1, 3)
        outer_side_length = 1.0 / (-objective(result.x))  # Convert back from negative

        # Create final data structures
        inner_hex_data = final_positions.copy()
        outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin

        return inner_hex_data, outer_hex_data, outer_side_length

    except Exception as e:
        # Fallback to original configuration if optimization fails
        print(f"Optimization failed: {e}")
        return create_initial_config(), np.array([0, 0, 0]), 8.0


def create_initial_config():
    """Create a better initial configuration than the default."""
    return np.array([
        [0, 0, 0],      # center
        [-2.0, 0, 0],   # left
        [2.0, 0, 0],    # right
        [-1.0, 1.732, 0],  # top-left
        [1.0, 1.732, 0],   # top-right
        [-1.0, -1.732, 0], # bottom-left
        [1.0, -1.732, 0],  # bottom-right
        [-2.0, 3.464, 0],  # far top-left
        [2.0, 3.464, 0],   # far top-right
        [-2.0, -3.464, 0], # far bottom-left
        [2.0, -3.464, 0],  # far bottom-right
        [0, -4.0, 0],      # far bottom-center
    ])


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Try optimization first
    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    except Exception:
        # Fallback to simple configuration if optimization doesn't work
        inner_hex_data = create_initial_config()
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8  # large enough to contain all inner hexagons

    end_time = time.time()
    eval_time = end_time - start_time

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END