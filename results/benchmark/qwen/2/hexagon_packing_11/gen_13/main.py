# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
import math


def hexagon_vertices(center_x, center_y, side_length=1, rotation_angle=0):
    """Generate vertices of a regular hexagon."""
    vertices = []
    for i in range(6):
        angle = rotation_angle + i * np.pi / 3
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)


def get_hexagon_polygon(center_x, center_y, side_length=1, rotation_angle=0):
    """Get shapely polygon representation of a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_angle)
    return Polygon(vertices)


def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hexagon_poly)


def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly)


def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        vertices = hexagon_vertices(cx, cy, 1, angle)
        all_vertices.extend(vertices)

    if len(all_vertices) == 0:
        return 1.0

    # Convert to numpy array
    all_vertices = np.array(all_vertices)

    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

    # Calculate diagonal distance from center to farthest vertex
    # For a regular hexagon with side length s, the distance from center to corner is s
    # But we need to consider the actual extent of the packed hexagons
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate max distance from center to any vertex
    distances = np.sqrt((all_vertices[:, 0] - center_x)**2 + (all_vertices[:, 1] - center_y)**2)
    max_distance = np.max(distances)

    # Add some buffer to ensure full containment
    # For hexagons, we need to account for the fact that they're regular
    # The side length of the outer hexagon should be at least the max distance plus padding
    # The theoretical minimum outer hexagon side length for a regular hexagon that contains
    # the bounding circle of inner hexagons would be max_distance / (sqrt(3)/2)
    # However, for simplicity we can just compute it properly
    outer_side_length = max_distance * 2  # Double for safety

    # Actually compute this correctly - for a regular hexagon circumscribing our points,
    # we compute the distance from center to each vertex and take the maximum
    return max_distance * 2


def evaluate_packing_config(params):
    """Evaluate configuration and return negative inverse side length (for minimization)."""
    # Reshape params into 11 hexagons with (x, y, angle)
    hex_params = params.reshape(-1, 3)

    # Convert angles from degrees to radians for computation
    inner_hex_data = hex_params.copy()

    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(len(inner_hex_data)):
        cx, cy, angle_deg = inner_hex_data[i]
        angle_rad = np.deg2rad(angle_deg)
        poly = get_hexagon_polygon(cx, cy, 1, angle_rad)
        inner_polygons.append(poly)

    # Check overlap constraints
    for i in range(len(inner_polygons)):
        for j in range(i+1, len(inner_polygons)):
            if check_overlap(inner_polygons[i], inner_polygons[j]):
                # Return very high penalty if overlapping
                return 1e10

    # Calculate outer hexagon side length needed
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)

    # Check containment constraints
    # Create a large outer hexagon centered at origin to test containment
    outer_center_x, outer_center_y = 0, 0
    outer_angle = 0
    outer_poly = get_hexagon_polygon(outer_center_x, outer_center_y, outer_side_length, outer_angle)

    for poly in inner_polygons:
        if not check_containment(poly, outer_poly):
            # Return penalty if not contained
            return 1e10

    # Return negative of inverse side length (since we want to maximize 1/side_length)
    return -1.0 / outer_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimization to find a better solution than the fixed grid arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start with a better initial configuration
    initial_inner_hex_data = np.array([
        [0, 0, 0],      # center
        [-1.732, 0, 0],   # left
        [1.732, 0, 0],    # right
        [-0.866, 1.5, 0], # top-left
        [0.866, 1.5, 0],  # top-right
        [-0.866, -1.5, 0], # bottom-left
        [0.866, -1.5, 0], # bottom-right
        [-2.598, 1.5, 0], # far top-left
        [2.598, 1.5, 0],  # far top-right
        [-2.598, -1.5, 0], # far bottom-left
        [2.598, -1.5, 0], # far bottom-right
    ])

    # Flatten initial data for optimization
    initial_params = initial_inner_hex_data.flatten()

    # Define bounds for optimization:
    # x, y positions roughly bounded by reasonable values, angles in [0, 360)
    bounds = []
    # Each inner hexagon gets 3 parameters: x, y, angle
    for i in range(11):
        bounds.extend([(None, None), (None, None), (0, 360)])  # x, y, angle

    # Set up optimization
    start_time = time.time()

    # Use differential evolution for global optimization (more robust)
    result = differential_evolution(
        evaluate_packing_config,
        bounds,
        maxiter=200,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=False,
        tol=1e-6
    )

    end_time = time.time()

    # Extract the best solution found
    best_params = result.x.reshape(-1, 3)

    # Create final result with angles in degrees as expected by API
    final_inner_hex_data = best_params.copy()

    # Calculate final outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(final_inner_hex_data)

    # Outer hexagon is centered at origin with no rotation
    outer_hex_data = np.array([0, 0, 0])

    return final_inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END