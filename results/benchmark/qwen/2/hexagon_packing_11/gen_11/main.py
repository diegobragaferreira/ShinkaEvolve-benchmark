# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time

# Constants
UNIT_HEX_RADIUS = 1.0  # radius of unit hexagon (distance from center to corner)
UNIT_HEX_WIDTH = 2.0  # width of unit hexagon (distance between parallel sides)
UNIT_HEX_HEIGHT = math.sqrt(3.0)  # height of unit hexagon

def get_hexagon_vertices(center_x, center_y, angle_deg, radius=UNIT_HEX_RADIUS):
    """Get vertices of a regular hexagon."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def hexagon_to_polygon(center_x, center_y, angle_deg, radius=UNIT_HEX_RADIUS):
    """Convert hexagon to Shapely polygon."""
    vertices = get_hexagon_vertices(center_x, center_y, angle_deg, radius)
    return Polygon(vertices)

def check_hexagon_containment(hex_center_x, hex_center_y, angle_deg, outer_radius, radius=UNIT_HEX_RADIUS):
    """Check if hexagon is fully contained in outer hexagon."""
    outer_hex = hexagon_to_polygon(0, 0, 0, outer_radius)
    inner_hex = hexagon_to_polygon(hex_center_x, hex_center_y, angle_deg, radius)
    return outer_hex.contains(inner_hex)

def check_hexagon_overlap(hex1_center_x, hex1_center_y, hex1_angle_deg,
                         hex2_center_x, hex2_center_y, hex2_angle_deg, radius=UNIT_HEX_RADIUS):
    """Check if two hexagons overlap."""
    hex1 = hexagon_to_polygon(hex1_center_x, hex1_center_y, hex1_angle_deg, radius)
    hex2 = hexagon_to_polygon(hex2_center_x, hex2_center_y, hex2_angle_deg, radius)
    return hex1.intersects(hex2)

def evaluate_layout(positions_and_angles, outer_radius):
    """Evaluate whether layout is valid and compute penalty."""
    n = len(positions_and_angles) // 3  # number of hexagons
    penalty = 0

    # Check containment of all inner hexagons
    for i in range(n):
        center_x, center_y, angle_deg = positions_and_angles[3*i:3*i+3]
        if not check_hexagon_containment(center_x, center_y, angle_deg, outer_radius):
            penalty += 1000  # Large penalty for containment violation

    # Check overlaps between all pairs
    for i in range(n):
        for j in range(i+1, n):
            center_x1, center_y1, angle_deg1 = positions_and_angles[3*i:3*i+3]
            center_x2, center_y2, angle_deg2 = positions_and_angles[3*j:3*j+3]
            if check_hexagon_overlap(center_x1, center_y1, angle_deg1,
                                   center_x2, center_y2, angle_deg2):
                penalty += 1000  # Large penalty for overlap violation

    return penalty

def objective_function(params):
    """Minimize outer hexagon radius while maintaining valid packing."""
    n = 11
    # First parameter is the outer radius
    outer_radius = params[0]

    # Remaining parameters are positions and angles of inner hexagons (11*3 = 33 params)
    positions_and_angles = params[1:]

    # Evaluate layout quality
    penalty = evaluate_layout(positions_and_angles, outer_radius)

    # Return negative of outer radius (we want to maximize 1/outer_radius)
    # plus penalty for constraint violations
    return -outer_radius + penalty

def get_initial_guess():
    """Generate a good initial guess for the optimization."""
    # Start with a reasonable configuration based on hexagonal packing principles
    # This is a known good starting point - arrange in a honeycomb pattern

    initial_positions_and_angles = []

    # Center hexagon
    initial_positions_and_angles.extend([0.0, 0.0, 0.0])

    # Surrounding hexagons in 2 layers
    # Layer 1 (6 hexagons around center)
    layer1_angles = [0, 60, 120, 180, 240, 300]
    for angle in layer1_angles:
        rad = UNIT_HEX_WIDTH  # distance from center
        x = rad * math.cos(math.radians(angle))
        y = rad * math.sin(math.radians(angle))
        initial_positions_and_angles.extend([x, y, 0.0])

    # Layer 2 (additional hexagons)
    layer2_angles = [30, 90, 150, 210, 270, 330]
    for angle in layer2_angles:
        rad = 2 * UNIT_HEX_WIDTH
        x = rad * math.cos(math.radians(angle))
        y = rad * math.sin(math.radians(angle))
        initial_positions_and_angles.extend([x, y, 0.0])

    # Add some randomness to prevent getting stuck in local minima
    np.random.seed(42)
    for i in range(len(initial_positions_and_angles)):
        if i % 3 != 2:  # Don't perturb angles
            initial_positions_and_angles[i] += np.random.uniform(-0.1, 0.1)

    return initial_positions_and_angles

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Initial guess
    initial_params = [6.0] + get_initial_guess()  # Start with outer radius = 6

    # Define bounds for optimization
    bounds = [(3.0, 10.0)]  # outer radius bounds
    for i in range(11):  # 11 hexagons
        bounds.extend([(None, None), (None, None), (-180, 180)])  # x, y, angle bounds

    # Use L-BFGS-B for optimization
    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-6})

    # Extract results
    final_outer_radius = result.x[0]
    positions_and_angles = result.x[1:]

    # Convert to final data structure
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = positions_and_angles[3*i]
        inner_hex_data[i, 1] = positions_and_angles[3*i+1]
        inner_hex_data[i, 2] = positions_and_angles[3*i+2]

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = final_outer_radius

    eval_time = time.time() - start_time

    # Final validation check
    penalty = evaluate_layout(positions_and_angles, final_outer_radius)
    assert penalty < 10, "Final solution has constraint violations"

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END