# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
import time
import math


def create_regular_hexagon(center=(0, 0), side_length=1, rotation=0):
    """Create a regular hexagon as a shapely polygon."""
    angle_offset = rotation * np.pi / 180
    points = []
    for i in range(6):
        angle = angle_offset + i * np.pi / 3
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        points.append((x, y))
    return Polygon(points)


def get_hexagon_vertices(center, side_length, rotation):
    """Get vertices of a hexagon."""
    angle_offset = rotation * np.pi / 180
    vertices = []
    for i in range(6):
        angle = angle_offset + i * np.pi / 3
        x = center[0] + side_length * np.cos(angle)
        y = center[1] + side_length * np.sin(angle)
        vertices.append((x, y))
    return vertices


def check_containment(hexagon, outer_hexagon):
    """Check if all vertices of hexagon are contained within outer_hexagon."""
    vertices = get_hexagon_vertices(hexagon[:2], 1, hexagon[2])
    for vertex in vertices:
        if not outer_hexagon.contains(Point(vertex)):
            return False
    return True


def check_overlap_pair(hex1, hex2):
    """Check if two hexagons overlap using shapely."""
    h1 = create_regular_hexagon(hex1[:2], 1, hex1[2])
    h2 = create_regular_hexagon(hex2[:2], 1, hex2[2])
    return h1.intersects(h2)


def check_all_overlaps(hex_data, outer_hexagon):
    """Check all pairwise overlaps and containment."""
    n = len(hex_data)

    # Check containment first
    for i in range(n):
        if not check_containment(hex_data[i], outer_hexagon):
            return True  # Overlap due to containment failure

    # Check pairwise overlaps
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap_pair(hex_data[i], hex_data[j]):
                return True  # Overlap detected

    return False  # No overlaps or containment issues


def calculate_objective(hex_data, outer_hex_side_length):
    """Calculate the objective function to maximize: 1/outer_hex_side_length"""
    # Convert outer hexagon side length to proper scale
    # This is our objective to maximize
    return 1.0 / outer_hex_side_length if outer_hex_side_length > 0 else 0.0


def pack_hexagons(hex_data, outer_hex_side_length):
    """Pack hexagons into an outer hexagon of given side length."""
    outer_hex = create_regular_hexagon((0, 0), outer_hex_side_length, 0)
    return check_all_overlaps(hex_data, outer_hex)


def objective_function(params):
    """Objective function to minimize (negative of 1/outer_hex_side_length)."""
    # Parse parameters
    # First 12*3 parameters: positions and rotations for 12 hexagons
    # Last 1 parameter: outer hexagon side length

    hex_positions_and_angles = params[:-1].reshape(-1, 3)
    outer_side_length = params[-1]

    # Ensure outer side length is positive
    if outer_side_length <= 0:
        return 1e10  # Large penalty for invalid side length

    # Create outer hexagon
    outer_hex = create_regular_hexagon((0, 0), outer_side_length, 0)

    # Check constraints
    if check_all_overlaps(hex_positions_and_angles, outer_hex):
        return 1e10  # Large penalty for overlap/containment violation

    # Objective: maximize 1/outer_side_length (minimize negative of it)
    return -1.0 / outer_side_length


def generate_initial_guess():
    """Generate a good initial guess for the hexagon packing."""
    # A better initial configuration based on hexagonal packing principles
    # Center hexagon + surrounding ring of 6 hexagons + outer ring of 5 hexagons

    # Hexagon radius (distance from center to corner)
    hex_radius = 1

    # Positions arranged in a hexagonal pattern
    positions = [
        [0, 0, 0],          # center
        [2, 0, 0],          # right
        [-2, 0, 0],         # left
        [1, 1.732, 0],      # upper right (sqrt(3) = 1.732)
        [-1, 1.732, 0],     # upper left
        [1, -1.732, 0],     # lower right
        [-1, -1.732, 0],    # lower left
        [3, 0, 0],          # far right
        [-3, 0, 0],         # far left
        [0, 3, 0],          # top
        [0, -3, 0],         # bottom
        [2, 1.732, 0],      # far upper right
        [-2, 1.732, 0],     # far upper left
        [2, -1.732, 0],     # far lower right
        [-2, -1.732, 0],    # far lower left
    ]

    # Use first 12 hexagons, remove extra ones
    initial_positions = positions[:12]

    # Flatten and add outer hexagon side length (initial guess)
    flat_params = []
    for pos in initial_positions:
        flat_params.extend(pos)  # x, y, angle

    # Initial guess for outer hexagon side length (should be much smaller than this)
    outer_side_length_guess = 6.0
    flat_params.append(outer_side_length_guess)

    return np.array(flat_params)


# Global counters for performance tracking
constraint_evaluations = 0
overlap_checks = 0
containment_checks = 0


def log_performance_stats():
    """Log performance statistics."""
    print(f"Constraint Evaluations: {constraint_evaluations}")
    print(f"Overlap Checks: {overlap_checks}")
    print(f"Containment Checks: {containment_checks}")


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    print("Starting hexagon packing optimization...")

    # Generate initial guess
    x0 = generate_initial_guess()

    # Set bounds for parameters
    # Each hexagon has x,y ([-10,10]) and angle (0,360)
    # Outer hexagon side length (1, 20)
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle for each hexagon
    bounds.append((1e-6, 20))  # outer hexagon side length

    # Run optimization
    start_time = time.time()

    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=100,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        disp=True
    )

    end_time = time.time()
    eval_time = end_time - start_time

    # Extract results
    final_params = result.x
    hex_positions_and_angles = final_params[:-1].reshape(-1, 3)
    outer_side_length = final_params[-1]

    # Return results
    # Note: We're returning the optimized inner hexagon positions
    inner_hex_data = hex_positions_and_angles.copy()

    # Outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])

    # Calculate final objective value
    inv_outer_side_length = 1.0 / outer_side_length if outer_side_length > 0 else 0.0

    # Benchmark ratio (0.2537 is the target SOTA)
    benchmark_ratio = inv_outer_side_length / 0.2537

    print(f"Optimization completed in {eval_time:.2f} seconds")
    print(f"Final inverse outer hex side length: {inv_outer_side_length:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"Outer hex side length: {outer_side_length:.6f}")

    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END