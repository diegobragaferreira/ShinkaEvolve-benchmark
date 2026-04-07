# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import time


def create_hexagon_vertices(center, side_length, rotation_degrees):
    """Create vertices of a regular hexagon."""
    angle_rad = np.radians(rotation_degrees)
    angle_step = 2 * np.pi / 6
    vertices = []
    for i in range(6):
        angle = angle_step * i + angle_rad
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)


def check_containment(hex_vertices, outer_hex_center, outer_hex_side_length):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    outer_vertices = create_hexagon_vertices(outer_hex_center, outer_hex_side_length, 0)
    outer_polygon = Polygon(outer_vertices)

    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True


def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    hex1_polygon = Polygon(hex1_vertices)
    hex2_polygon = Polygon(hex2_vertices)
    return hex1_polygon.intersects(hex2_polygon)


def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center):
    """Calculate the minimum required outer hexagon side length."""
    max_distance = 0

    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Calculate distance from center of outer hexagon to center of inner hexagon
        distance = np.sqrt((center_x - outer_hex_center[0])**2 + (center_y - outer_hex_center[1])**2)
        # Add radius of inner hexagon (which is 1 for unit hexagons)
        distance_to_outer_edge = distance + 1  # Approximate; actual would be more precise
        max_distance = max(max_distance, distance_to_outer_edge)

    # Convert to side length of outer hexagon
    # For a regular hexagon, the side length equals the radius
    return max_distance * 2  # Approximation


def evaluate_packing(inner_hex_data, outer_hex_center=(0, 0)):
    """Evaluate a packing configuration and return the inverse side length."""
    # Create hexagons with unit side length
    hexagons = []
    for center_x, center_y, angle in inner_hex_data:
        vertices = create_hexagon_vertices((center_x, center_y), 1.0, angle)
        hexagons.append(vertices)

    # Check containment and overlap
    all_contained = True
    any_overlap = False

    # Check containment for each hexagon
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data, outer_hex_center)
    outer_vertices = create_hexagon_vertices(outer_hex_center, outer_side_length, 0)
    outer_polygon = Polygon(outer_vertices)

    for vertices in hexagons:
        for vertex in vertices:
            point = Point(vertex)
            if not outer_polygon.contains(point):
                all_contained = False
                break
        if not all_contained:
            break

    # Check overlaps between all pairs
    if all_contained:
        for i in range(len(hexagons)):
            for j in range(i+1, len(hexagons)):
                if check_overlap(hexagons[i], hexagons[j]):
                    any_overlap = True
                    break
            if any_overlap:
                break

    # If valid packing, return inverse of outer side length
    if all_contained and not any_overlap:
        return 1.0 / outer_side_length
    else:
        # Return very small value for invalid configurations
        return 1e-10


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start with a reasonable initial guess based on known good configurations
    # This is a more symmetric arrangement than the baseline
    initial_guess = np.array([
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [0, -2, 0],     # bottom
        [1.732, 1, 0],  # top right
        [-1.732, 1, 0], # top left
        [1.732, -1, 0], # bottom right
        [-1.732, -1, 0],# bottom left
        [3.464, 0, 0],  # far right
        [-3.464, 0, 0], # far left
        [1.732, 3, 0],  # top far right
        [-1.732, 3, 0], # top far left
        [1.732, -3, 0], # bottom far right
    ])

    # Objective function to minimize (negative of our score to be maximized)
    def objective(x):
        # Reshape x into 12 hexagons with (x, y, angle) each
        hex_data = x.reshape(-1, 3)

        # Evaluate the current configuration
        score = evaluate_packing(hex_data)
        return -score  # Negative because we want to maximize the score

    # Initial guess for optimization (flattened)
    x0 = initial_guess.flatten()

    # Optimization bounds (position limits and angle limits)
    bounds = []
    # X and Y positions (limited to reasonable area)
    for i in range(24):  # 12 hexagons * 2 coordinates each
        bounds.append((-10, 10))  # Reasonable range
    # Angles (0-360 degrees)
    for i in range(12):
        bounds.append((0, 360))

    # Perform optimization
    try:
        start_time = time.time()
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        end_time = time.time()

        # Extract optimized values
        optimized_hex_data = result.x.reshape(-1, 3)

        # Evaluate final result
        final_score = evaluate_packing(optimized_hex_data)

        # If optimization was successful, return the result
        if result.success and final_score > 1e-5:
            # Use the first hexagon's center as outer hexagon center
            outer_hex_center = optimized_hex_data[0][:2]
            outer_hex_side_length = 1.0 / final_score
            outer_hex_data = np.array([outer_hex_center[0], outer_hex_center[1], 0])

            return optimized_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        print(f"Optimization failed: {e}")

    # Fallback to original solution if optimization fails
    print("Using fallback solution")
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
        [3.75, -2.17, 0],  # far bottom-right,
        [0, -4, 0],  # far bottom-center
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END