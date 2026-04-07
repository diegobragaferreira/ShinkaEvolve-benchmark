# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
import time


def create_hexagon_vertices(center_x, center_y, radius, angle_degrees=0):
    """Create vertices of a regular hexagon with given center, radius, and rotation."""
    angle_rad = np.radians(angle_degrees)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices


def check_hexagon_containment(hexagon_vertices, outer_hex_center_x, outer_hex_center_y, outer_hex_radius):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    outer_hex_vertices = create_hexagon_vertices(outer_hex_center_x, outer_hex_center_y, outer_hex_radius)
    outer_polygon = Polygon(outer_hex_vertices)

    for vertex in hexagon_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True


def check_overlap_pair(vertices1, vertices2):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)
    return poly1.intersects(poly2)


def calculate_outer_hexagon_radius(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate minimum outer hexagon radius that contains all inner hexagons."""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        # Get the furthest vertex from center for this hexagon
        hex_vertices = create_hexagon_vertices(center_x, center_y, 1, angle)
        for vx, vy in hex_vertices:
            dist = np.sqrt((vx - outer_hex_center[0])**2 + (vy - outer_hex_center[1])**2)
            max_dist = max(max_dist, dist)
    # Add small margin to ensure full containment
    return max_dist * 1.01


def evaluate_configuration(config, outer_hex_center=(0, 0)):
    """
    Evaluate the configuration by calculating the objective function.
    Returns negative inverse of outer hexagon radius (since we want to maximize 1/R).
    """
    # Parse configuration into inner hexagon data (12 hexagons * 3 parameters = 36 parameters)
    # First 36 parameters: [x1, y1, angle1, x2, y2, angle2, ..., x12, y12, angle12]
    inner_hex_data = config.reshape(-1, 3)

    # Calculate outer hexagon size needed to contain all inner hexagons
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_hex_center)

    # Check for overlaps between any pair of inner hexagons
    total_penalty = 0

    # Check containment and overlaps
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_vertices = create_hexagon_vertices(center_x, center_y, 1, angle)

        # Check containment
        if not check_hexagon_containment(hex_vertices, outer_hex_center[0], outer_hex_center[1], outer_radius):
            total_penalty += 1000000  # Large penalty for containment violation

        # Check overlaps with all other hexagons
        for j in range(i+1, len(inner_hex_data)):
            center_x2, center_y2, angle2 = inner_hex_data[j]
            hex_vertices2 = create_hexagon_vertices(center_x2, center_y2, 1, angle2)

            if check_overlap_pair(hex_vertices, hex_vertices2):
                total_penalty += 1000000  # Large penalty for overlap

    # Return negative inverse radius plus penalties
    # Since we want to maximize 1/R, we minimize -1/R
    if total_penalty > 0:
        return total_penalty + 1000000  # Ensure infeasible solutions have high penalty

    return -1.0 / outer_radius


def optimize_hexagon_packing():
    """
    Optimizes the arrangement of 12 unit hexagons within a larger hexagon.
    Returns:
        inner_hex_data: np.ndarray of shape (12,3), where each row is (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degrees)
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start with a good initial guess based on known optimal arrangements
    # This is a symmetric pattern that should be close to optimal
    initial_guess = np.array([
        [0, 0, 0],      # center
        [-2.0, 0, 0],   # left
        [2.0, 0, 0],    # right
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [-1.0, 1.0, 0], # top-left
        [1.0, 1.0, 0],  # top-right
        [-1.0, -1.0, 0], # bottom-left
        [1.0, -1.0, 0], # bottom-right
        [-2.0, 1.0, 0], # far top-left
        [2.0, 1.0, 0],  # far top-right
        [-2.0, -1.0, 0], # far bottom-left
    ]).flatten()

    # Bounds for optimization: x, y in [-5, 5], angle in [0, 360)
    bounds = []
    for _ in range(12):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])

    # Use differential evolution for global optimization
    result = differential_evolution(
        lambda x: evaluate_configuration(x),
        bounds,
        maxiter=100,
        popsize=15,
        seed=42,
        disp=False
    )

    # Extract final solution
    final_config = result.x.reshape(-1, 3)

    # Calculate final outer hexagon size
    outer_radius = calculate_outer_hexagon_radius(final_config)

    # Create outer hexagon centered at origin
    outer_hex_data = np.array([0, 0, 0])

    return final_config, outer_hex_data, outer_radius


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()

    end_time = time.time()

    # Calculate performance metrics
    inv_outer_hex_side_length = 1.0 / outer_hex_side_length
    benchmark_ratio = inv_outer_hex_side_length / 0.2537

    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    print(f"Inverse outer hex side length: {inv_outer_hex_side_length:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END