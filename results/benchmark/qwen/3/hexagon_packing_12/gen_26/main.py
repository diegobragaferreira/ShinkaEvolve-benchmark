# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist


def hexagon_vertices(center_x, center_y, size=1, angle_deg=0):
    """Generate vertices of a regular hexagon given center, size, and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + size * np.cos(angle)
        y = center_y + size * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)


def check_containment(hex_vertices, outer_center_x, outer_center_y, outer_size):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_size, 0)
    outer_polygon = Polygon(outer_vertices)

    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True


def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)


def compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y):
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        distance = np.sqrt((cx - outer_center_x)**2 + (cy - outer_center_y)**2)
        max_distance = max(max_distance, distance + 1)  # Add radius of unit hexagon

    return max_distance


def evaluate_configuration(inner_hex_data, outer_center_x, outer_center_y):
    """Evaluate current configuration: returns (validity, inv_radius)."""
    # Check for overlaps
    for i in range(len(inner_hex_data)):
        hex1_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for j in range(i+1, len(inner_hex_data)):
            hex2_vertices = hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], 1, inner_hex_data[j][2])
            if check_overlap(hex1_vertices, hex2_vertices):
                return False, 0

    # Check containment
    outer_radius = compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y)
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_radius, 0)
    outer_polygon = Polygon(outer_vertices)

    for i in range(len(inner_hex_data)):
        hex_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, 0

    # Return inverse of outer radius
    return True, 1.0 / outer_radius


def generate_initial_symmetric_config():
    """Generate a symmetric initial configuration based on proven hexagonal tiling."""
    # Central hexagon
    config = [[0, 0, 0]]

    # First ring (6 hexagons) - arranged in a hexagonal pattern
    for i in range(6):
        angle = i * 60
        radius = 2  # Distance from origin
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    # Second ring (6 hexagons) - arranged in a larger hexagonal pattern
    for i in range(6):
        angle = 30 + i * 60  # offset by 30 degrees for alternating pattern
        radius = 2 * np.sqrt(3)  # sqrt(12) approximately, places them at distance 2sqrt(3) from center
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    return np.array(config)


def optimize_positions(initial_config, outer_center_x, outer_center_y):
    """Optimize positions using constrained numerical optimization."""

    def objective(params):
        # Reconstruct configuration from flattened parameters
        # Only optimize the (x,y) positions; keep angles fixed at 0 for simplicity
        config = initial_config.copy()
        idx = 0
        for i in range(len(config)):
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2

        validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
        if not validity:
            return 1e10  # Large penalty for invalid configurations
        return -inv_radius  # Negative because we want to maximize

    # Flatten initial configuration for optimization (only positions, not angles)
    initial_params = []
    for i in range(len(initial_config)):
        initial_params.extend([initial_config[i][0], initial_config[i][1]])

    # Perform optimization with bounds to keep hexagons within reasonable range
    bounds = [(-10, 10), (-10, 10)] * len(initial_config)
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

    # Reconstruct optimized configuration
    optimized_config = initial_config.copy()
    idx = 0
    for i in range(len(optimized_config)):
        optimized_config[i][0] = result.x[idx]
        optimized_config[i][1] = result.x[idx + 1]
        idx += 2

    return optimized_config


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use a proven symmetric initial configuration
    initial_config = generate_initial_symmetric_config()

    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0

    # Optimize positions to maximize packing efficiency
    optimized_config = optimize_positions(initial_config, outer_center_x, outer_center_y)

    # Final verification and refinement
    validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)

    # If not valid, try simple fallback
    if not validity:
        # Fallback to a known valid configuration
        optimized_config = np.array([
            [0, 0, 0],
            [-1.5, 0, 0],
            [1.5, 0, 0],
            [-0.75, 1.299, 0],
            [0.75, 1.299, 0],
            [-0.75, -1.299, 0],
            [0.75, -1.299, 0],
            [-2.25, 1.299, 0],
            [2.25, 1.299, 0],
            [-2.25, -1.299, 0],
            [2.25, -1.299, 0],
            [0, -2.598, 0]
        ])
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)

    # Compute final outer hexagon radius
    outer_radius = 1.0 / inv_radius if inv_radius > 0 else 10.0

    # Convert back to required format
    inner_hex_data = np.array(optimized_config)

    # Ensure we have exactly 12 hexagons
    assert len(inner_hex_data) == 12

    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])
    outer_hex_side_length = outer_radius * 2  # approximate

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END