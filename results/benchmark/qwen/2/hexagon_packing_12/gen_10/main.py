# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

# Precompute hexagon vertices for unit hexagons
def get_unit_hexagon_vertices(center=(0,0), rotation=0):
    """Get vertices of a unit regular hexagon with given center and rotation."""
    angle_offset = np.radians(rotation)
    radius = 1.0
    angles = np.linspace(0, 2*np.pi, 7) + angle_offset
    vertices = np.column_stack([center[0] + radius * np.cos(angles),
                               center[1] + radius * np.sin(angles)])
    return vertices[:-1]  # Remove last point to close polygon

def check_containment(hex_vertices, outer_hex_center, outer_hex_radius):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    # Create outer hexagon polygon
    outer_vertices = get_unit_hexagon_vertices(outer_hex_center, 0)
    # Scale outer hexagon vertices to actual size
    outer_vertices *= outer_hex_radius

    outer_polygon = Polygon(outer_vertices)

    # Check all vertices of inner hexagon
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def evaluate_packing_config(config):
    """
    Evaluate a configuration of 12 hexagons packed inside a larger hexagon.
    Returns negative inverse side length (since we want to maximize 1/R).
    """
    # Parse configuration: [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12, R]
    positions = config[:24].reshape(-1, 2)  # 12 positions
    rotations = config[24:36]               # 12 rotations
    outer_radius = config[36]               # Outer hexagon radius

    # Generate vertices for all inner hexagons
    inner_hex_vertices = []
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        # Get vertices for unit hexagon at position with rotation
        vertices = get_unit_hexagon_vertices(pos, rot)
        inner_hex_vertices.append(vertices)

    # Check containment
    outer_center = np.array([0, 0])
    for vertices in inner_hex_vertices:
        if not check_containment(vertices, outer_center, outer_radius):
            return 1e6  # Large penalty for containment violations

    # Check overlaps
    for i in range(len(inner_hex_vertices)):
        for j in range(i+1, len(inner_hex_vertices)):
            if check_overlap(inner_hex_vertices[i], inner_hex_vertices[j]):
                return 1e6  # Large penalty for overlaps

    # Return negative inverse of outer radius (to minimize this value = maximize 1/R)
    return -1.0 / outer_radius

def optimize_hexagon_packing():
    """
    Optimize arrangement of 12 unit hexagons inside a regular hexagon.
    Returns:
        inner_hex_data: np.ndarray of shape (12,3) with (x,y,angle) for each hexagon
        outer_hex_data: np.ndarray of shape (3,) with (x,y,angle) for outer hexagon
        outer_hex_side_length: float representing the side length of outer hexagon
    """

    # Initial guess: symmetric configuration
    # Start with a good known arrangement
    initial_positions = np.array([
        [0, 0],           # center
        [-1.732, 0],      # left
        [1.732, 0],       # right
        [-0.866, 1.5],    # top-left
        [0.866, 1.5],     # top-right
        [-0.866, -1.5],   # bottom-left
        [0.866, -1.5],    # bottom-right
        [-2.598, 1.5],    # far top-left
        [2.598, 1.5],     # far top-right
        [-2.598, -1.5],   # far bottom-left
        [2.598, -1.5],    # far bottom-right
        [0, -3.0],        # far bottom-center
    ])

    # Initial rotations (all horizontal for simplicity)
    initial_rotations = np.zeros(12)

    # Initial outer radius (should be about 3.9419123 to achieve target ratio)
    initial_radius = 4.0

    # Concatenate into single config vector: [positions, rotations, radius]
    initial_config = np.concatenate([
        initial_positions.flatten(),
        initial_rotations,
        [initial_radius]
    ])

    # Define bounds for optimization
    # Positions: reasonable bounds around center
    pos_bounds = [(-10, 10)] * 24  # positions
    rot_bounds = [(-180, 180)] * 12  # rotations
    radius_bounds = [(1.0, 10.0)]  # outer radius

    bounds = pos_bounds + rot_bounds + radius_bounds

    # Optimization parameters
    maxiter = 1000
    popsize = 15
    tol = 1e-6

    # Run optimization
    result = differential_evolution(
        evaluate_packing_config,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=42,
        polish=True,
        disp=False
    )

    # Extract results
    final_config = result.x

    positions = final_config[:24].reshape(-1, 2)
    rotations = final_config[24:36]
    outer_radius = final_config[36]

    # Create inner hex data
    inner_hex_data = np.column_stack([positions, rotations])

    # Create outer hex data (centered at origin with no rotation)
    outer_hex_data = np.array([0, 0, 0])

    # Return results
    return inner_hex_data, outer_hex_data, outer_radius

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

    eval_time = time.time() - start_time

    # Calculate performance metrics
    inv_outer_hex_side_length = 1.0 / outer_hex_side_length
    benchmark_ratio = inv_outer_hex_side_length / 0.2537

    print(f"inv_outer_hex_side_length: {inv_outer_hex_side_length:.6f}")
    print(f"benchmark_ratio: {benchmark_ratio:.6f}")
    print(f"eval_time: {eval_time:.4f}s")

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END