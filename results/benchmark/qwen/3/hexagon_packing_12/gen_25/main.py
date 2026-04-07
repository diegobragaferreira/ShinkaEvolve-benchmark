# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import math


def create_hexagon_vertices(center_x, center_y, side_length=1, rotation=0):
    """Create vertices of a regular hexagon"""
    vertices = []
    for i in range(6):
        angle = rotation + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices


def hexagon_to_polygon(center_x, center_y, side_length=1, rotation=0):
    """Convert hexagon data to shapely polygon"""
    vertices = create_hexagon_vertices(center_x, center_y, side_length, rotation)
    return Polygon(vertices)


def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)


def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained in outer hexagon"""
    return outer_hex.contains(inner_hex)


def compute_outer_hexagon_size(inner_hex_data):
    """Compute minimum outer hexagon size that contains all inner hexagons"""
    # Create all inner hexagons
    inner_polygons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        poly = hexagon_to_polygon(center_x, center_y, 1, angle)
        inner_polygons.append(poly)

    # Find bounding box of all hexagon vertices
    all_vertices = []
    for poly in inner_polygons:
        all_vertices.extend(list(poly.exterior.coords))

    # Compute centroid of all vertices
    avg_x = sum(v[0] for v in all_vertices) / len(all_vertices)
    avg_y = sum(v[1] for v in all_vertices) / len(all_vertices)

    # Find maximum distance from centroid to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - avg_x)**2 + (y - avg_y)**2)
        max_dist = max(max_dist, dist)

    # Estimate outer hexagon side length based on maximum distance
    # For a regular hexagon, side length = distance to corner / sqrt(3)
    outer_side_length = max_dist / math.sqrt(3)

    return outer_side_length, (avg_x, avg_y)


def evaluate_layout(inner_hex_data):
    """Evaluate the layout and return penalty and outer hex side length"""
    # Create outer hexagon
    outer_side_length, outer_center = compute_outer_hexagon_size(inner_hex_data)

    # Create outer hexagon polygon
    outer_hex = hexagon_to_polygon(outer_center[0], outer_center[1], outer_side_length, 0)

    # Check overlaps between inner hexagons
    penalty = 0
    inner_polygons = []

    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_poly = hexagon_to_polygon(center_x, center_y, 1, angle)
        inner_polygons.append(inner_poly)

        # Check containment
        if not check_containment(inner_poly, outer_hex):
            penalty += 1000  # Large penalty for containment violation

        # Check overlap with all other hexagons
        for j in range(i+1, len(inner_hex_data)):
            if check_overlap(inner_poly, inner_polygons[j]):
                penalty += 1000  # Large penalty for overlap

    # Return negative penalty (since we want to minimize) and outer hex side length
    return penalty, outer_side_length


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Initial configuration based on known good arrangement
    initial_inner_hex_data = np.array([
        [0, 0, 0],  # center
        [-2.5, 0, 0],  # left
        [2.5, 0, 0],  # right
        [-1.25, 2.17, 0],  # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],  # bottom-right
        [-3.75, 2.17, 0],  # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],  # far bottom-right,
        [0, -4, 0],  # far bottom-center
    ])

    # Convert to flat array for optimization
    initial_params = initial_inner_hex_data.flatten()

    def objective(params):
        # Reshape params back to hexagon data
        hex_data = params.reshape(-1, 3)

        # Evaluate the layout
        penalty, outer_side_length = evaluate_layout(hex_data)

        # Objective is to minimize outer hex side length (maximize 1/outer_side_length)
        # But add penalty if there are overlaps or containment issues
        return outer_side_length + penalty

    # Run local optimization
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-6})
        optimized_params = result.x

        # Reshape back to hexagon data
        optimized_hex_data = optimized_params.reshape(-1, 3)
    except Exception as e:
        # Fallback to original if optimization fails
        optimized_hex_data = initial_inner_hex_data

    # Final evaluation
    _, outer_side_length = evaluate_layout(optimized_hex_data)

    # Create outer hexagon data
    outer_center_x, outer_center_y = compute_outer_hexagon_size(optimized_hex_data)[1]
    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])

    return optimized_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END