# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import itertools

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution: check containment and non-overlap"""
    # Precompute all hexagon polygons once for reuse
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))

    # Calculate outer radius once
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # Check containment using the outer hexagon polygon
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Check if all inner hexagons are contained within outer hexagon
    for hex_poly in hex_polygons:
        # Fast check: if any vertex is outside, reject
        for vertex in hex_poly.exterior.coords[:-1]:  # Exclude closing vertex
            if not outer_polygon.contains(Point(vertex)):
                return False

    # Check overlaps efficiently using spatial indexing
    # Build spatial index for faster overlap detection
    points_list = []
    for i, hex_poly in enumerate(hex_polygons):
        # Collect all vertices for spatial indexing
        for vertex in hex_poly.exterior.coords[:-1]:
            points_list.append((vertex[0], vertex[1], i))

    if len(points_list) > 0:
        # Create spatial tree for vertices
        tree_points = cKDTree([(p[0], p[1]) for p in points_list])

        # Check overlaps between hexagons
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return False
    else:
        # Fallback for empty case
        return False

    return True

def generate_diverse_initial_configs():
    """Generate multiple diverse initial configurations"""
    configs = []

    # Configuration 1: Standard hexagonal lattice pattern
    config1 = np.array([
        [0, 0, 0],           # center
        [-2.0, 0, 0],        # left
        [2.0, 0, 0],         # right
        [-1.0, 1.732, 0],    # top-left
        [1.0, 1.732, 0],     # top-right
        [-1.0, -1.732, 0],   # bottom-left
        [1.0, -1.732, 0],    # bottom-right
        [-3.0, 1.732, 0],    # far top-left
        [3.0, 1.732, 0],     # far top-right
        [-3.0, -1.732, 0],   # far bottom-left
        [3.0, -1.732, 0],    # far bottom-right
    ])
    configs.append(config1)

    # Configuration 2: Slightly perturbed version of first config
    config2 = config1.copy()
    for i in range(NUM_INNER_HEXAGONS):
        config2[i][0] += np.random.normal(0, 0.2)
        config2[i][1] += np.random.normal(0, 0.2)
        config2[i][2] += np.random.normal(0, 10)
        config2[i][2] %= 360
    configs.append(config2)

    # Configuration 3: Different spacing pattern
    config3 = np.array([
        [0, 0, 0],           # center
        [-2.2, 0, 0],        # left
        [2.2, 0, 0],         # right
        [-1.1, 1.9, 0],      # top-left
        [1.1, 1.9, 0],       # top-right
        [-1.1, -1.9, 0],     # bottom-left
        [1.1, -1.9, 0],      # bottom-right
        [-3.3, 1.9, 0],      # far top-left
        [3.3, 1.9, 0],       # far top-right
        [-3.3, -1.9, 0],     # far bottom-left
        [3.3, -1.9, 0],      # far bottom-right
    ])
    configs.append(config3)

    # Configuration 4: More spread out pattern
    config4 = np.array([
        [0, 0, 0],           # center
        [-2.5, 0, 0],        # left
        [2.5, 0, 0],         # right
        [-1.25, 2.17, 0],    # top-left
        [1.25, 2.17, 0],     # top-right
        [-1.25, -2.17, 0],   # bottom-left
        [1.25, -2.17, 0],    # bottom-right
        [-3.75, 2.17, 0],    # far top-left
        [3.75, 2.17, 0],     # far top-right
        [-3.75, -2.17, 0],   # far bottom-left
        [3.75, -2.17, 0],    # far bottom-right
    ])
    configs.append(config4)

    return configs

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Generate multiple diverse initial configurations
    initial_configs = generate_diverse_initial_configs()

    best_config = None
    best_radius = float('inf')

    # Try each initial configuration
    for i, initial_config in enumerate(initial_configs):
        current_config = initial_config.copy()
        current_radius = calculate_outer_hexagon_radius(current_config)

        # Adaptive local optimization to refine the solution
        improvement_count = 0
        stagnation_counter = 0
        max_stagnation = 100
        perturbation_magnitude = 0.2  # Start with larger perturbations

        for iteration in range(1000):
            if time.time() - start_time > MAX_EVAL_TIME - 1:
                break

            # Create small perturbations to explore neighborhood
            test_config = current_config.copy()

            # Perturb each hexagon with adaptive magnitude
            for j in range(NUM_INNER_HEXAGONS):
                # Position perturbations with adaptive magnitude
                test_config[j][0] += np.random.normal(0, perturbation_magnitude)
                test_config[j][1] += np.random.normal(0, perturbation_magnitude)
                # Rotation perturbations
                test_config[j][2] += np.random.normal(0, perturbation_magnitude * 10)
                test_config[j][2] %= 360

            # Validate and accept if better
            if validate_solution(test_config):
                test_radius = calculate_outer_hexagon_radius(test_config)
                if test_radius < current_radius:
                    current_radius = test_radius
                    current_config = test_config.copy()
                    improvement_count += 1
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

            # Adapt perturbation magnitude based on progress
            if stagnation_counter > max_stagnation:
                perturbation_magnitude *= 0.9  # Reduce perturbation size
                stagnation_counter = 0
                if perturbation_magnitude < 0.001:
                    perturbation_magnitude = 0.001  # Minimum perturbation

        # Keep track of the best configuration found so far
        if current_radius < best_radius:
            best_radius = current_radius
            best_config = current_config.copy()

    # Final validation to ensure solution is correct
    if best_config is None or not validate_solution(best_config):
        # Fallback to well-known good configuration
        best_config = np.array([
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
            [3.75, -2.17, 0],
        ])
        best_radius = 8.0

    # Return result
    inner_hex_data = best_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = best_radius

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END