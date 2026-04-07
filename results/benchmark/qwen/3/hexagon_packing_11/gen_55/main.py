# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from itertools import combinations

# Precompute hexagon vertices for unit hexagon (centered at origin)
def get_unit_hexagon_vertices():
    """Return vertices of a unit regular hexagon centered at origin."""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles, skip last to close the polygon
    vertices = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
    return vertices

UNIT_HEX_VERTICES = get_unit_hexagon_vertices()

def transform_hexagon_vertices(vertices, center_x, center_y, angle_deg):
    """Transform hexagon vertices by translation and rotation."""
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Rotation matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Apply rotation and translation
    rotated_vertices = vertices @ rotation_matrix.T
    translated_vertices = rotated_vertices + np.array([center_x, center_y])

    return translated_vertices

def create_hexagon_polygon(center_x, center_y, angle_deg):
    """Create a Shapely polygon representing a unit hexagon at given position and rotation."""
    vertices = transform_hexagon_vertices(UNIT_HEX_VERTICES, center_x, center_y, angle_deg)
    return Polygon(vertices)

def is_contained(hex_polygon, outer_hex_polygon):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_polygon.contains(hex_polygon)

def check_overlap(hex1_polygon, hex2_polygon):
    """Check if two hexagons overlap."""
    return hex1_polygon.intersects(hex2_polygon)

def compute_outer_hexagon_radius(inner_configs, outer_center=(0,0), outer_angle=0):
    """
    Compute minimum radius needed to contain all inner hexagons.
    Uses binary search to find tightest fit.
    """
    # Start with a reasonable upper bound
    max_dist = 0
    for i in range(len(inner_configs)):
        cx, cy, angle = inner_configs[i]
        # Calculate distance from center to furthest vertex of hexagon
        dist = np.sqrt(cx**2 + cy**2) + 1.0  # plus radius of unit hexagon
        max_dist = max(max_dist, dist)

    # Binary search bounds
    min_radius = 0.1
    max_radius = max_dist * 2

    # Check if outer hexagon of current max_radius contains all inner hexagons
    def test_radius(radius):
        outer_vertices = transform_hexagon_vertices(
            UNIT_HEX_VERTICES, outer_center[0], outer_center[1], outer_angle)
        outer_hex = Polygon(outer_vertices)

        for i in range(len(inner_configs)):
            cx, cy, angle = inner_configs[i]
            inner_hex = create_hexagon_polygon(cx, cy, angle)

            if not is_contained(inner_hex, outer_hex):
                return False

        return True

    # Binary search for tightest fit
    while max_radius - min_radius > 0.001:
        mid_radius = (min_radius + max_radius) / 2
        if test_radius(mid_radius):
            max_radius = mid_radius
        else:
            min_radius = mid_radius

    return max_radius

def evaluate_fitness(config):
    """
    Evaluate fitness of a configuration.
    Returns negative of 1/outer_hex_side_length to maximize 1/outer_hex_side_length.
    """
    # Reshape config into (11, 3) array
    configs = config.reshape(-1, 3)

    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(11):
        cx, cy, angle = configs[i]
        inner_polygons.append(create_hexagon_polygon(cx, cy, angle))

    # Check for overlaps
    for i, j in combinations(range(11), 2):
        if check_overlap(inner_polygons[i], inner_polygons[j]):
            return 1e6  # Large penalty for overlaps

    # Create outer hexagon polygon
    outer_radius = compute_outer_hexagon_radius(configs)

    # Return negative inverse of outer radius (to maximize 1/outer_radius)
    return -1.0 / outer_radius

def evolutionary_search():
    """
    Use evolutionary algorithm to find optimal packing.
    """
    # Initial guess based on good known configuration
    initial_guess = np.array([
        [0, 0, 0],      # center
        [-2.5, 0, 0],   # left
        [2.5, 0, 0],    # right
        [-1.25, 2.17, 0],  # top-left
        [1.25, 2.17, 0],   # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0],  # bottom-right
        [-3.75, 2.17, 0],  # far top-left
        [3.75, 2.17, 0],   # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0],  # far bottom-right
    ]).flatten()

    # Bounds for search: x,y in [-10,10], angle in [0,360]
    bounds = [(-10, 10), (-10, 10), (0, 360)] * 11

    # Run differential evolution
    result = differential_evolution(
        evaluate_fitness,
        bounds,
        maxiter=500,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        tol=1e-6
    )

    configs = result.x.reshape(-1, 3)
    final_radius = compute_outer_hexagon_radius(configs)

    # Final validation check
    inner_polygons = []
    for i in range(11):
        cx, cy, angle = configs[i]
        inner_polygons.append(create_hexagon_polygon(cx, cy, angle))

    # Check for overlaps again
    for i, j in combinations(range(11), 2):
        if check_overlap(inner_polygons[i], inner_polygons[j]):
            raise ValueError("Overlap detected in final result")

    return configs, final_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Run evolutionary search
        inner_configs, outer_radius = evolutionary_search()

        # Convert back to required format
        outer_hex_data = np.array([0, 0, 0])  # centered at origin

        end_time = time.time()

        return inner_configs, outer_hex_data, outer_radius

    except Exception as e:
        print(f"Evolutionary search failed: {e}")
        # Fallback to original configuration
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0],  # far bottom-right
        ])

        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8  # large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
