# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import math


def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = math.radians(angle_deg)
    # Vertices of a regular hexagon with side_length=1 centered at origin
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = math.cos(theta)
        y = math.sin(theta)
        base_vertices.append((x, y))

    # Scale and translate
    vertices = [(center_x + side_length * vx, center_y + side_length * vy) for vx, vy in base_vertices]
    return vertices


def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hexagon_vertices)
    return outer_poly.contains(inner_poly)


def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)


def objective_function(params):
    """Objective function to minimize (negative of inverse of outer hex side length)."""
    # Extract inner hexagon positions and rotations (x,y,theta for each of 11 hexagons)
    # And outer hexagon parameters (center_x, center_y, angle, side_length)
    inner_params = params[:-4]
    outer_center_x, outer_center_y, outer_angle, outer_side_length = params[-4:]

    # Create inner hexagons
    inner_hexagons = []
    for i in range(11):
        x, y, theta = inner_params[3*i:3*i+3]
        vertices = generate_hexagon_vertices(x, y, theta, 1.0)  # unit hexagons
        inner_hexagons.append(vertices)

    # Create outer hexagon
    outer_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_side_length)

    # Check constraints
    penalty = 0

    # Check containment
    for vertices in inner_hexagons:
        if not check_containment(vertices, outer_vertices):
            penalty += 1000000

    # Check overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                penalty += 1000000

    # Return negative of inverse side length plus penalties
    if penalty > 0:
        return penalty + 1.0 / outer_side_length  # Penalty makes it worse than any valid solution
    else:
        return -1.0 / outer_side_length


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses optimization to find better arrangement than simple grid.
    """
    # Initial guess based on a known good configuration (approximate)
    # Inner hexagons positions and rotations (11 * 3 parameters)
    # Outer hexagon center, rotation, and side length (4 parameters)

    # Start with a better initial configuration
    initial_guess = []

    # Center hexagon
    initial_guess.extend([0.0, 0.0, 0.0])

    # Surrounding hexagons in honeycomb pattern (approximate)
    positions = [
        (-1.732, 0, 0),      # Left
        (1.732, 0, 0),       # Right
        (0, 1.732, 0),       # Top
        (0, -1.732, 0),      # Bottom
        (-0.866, 0.866, 0),  # Top-left
        (0.866, 0.866, 0),   # Top-right
        (-0.866, -0.866, 0), # Bottom-left
        (0.866, -0.866, 0),  # Bottom-right
        (-1.732, 1.732, 0),  # Far top-left
        (1.732, 1.732, 0),   # Far top-right
        (-1.732, -1.732, 0), # Far bottom-left
        (1.732, -1.732, 0),  # Far bottom-right
    ]

    # Add all positions and rotations
    for x, y, rot in positions:
        initial_guess.extend([x, y, rot])

    # Add outer hexagon parameters (center, angle, side_length)
    initial_guess.extend([0.0, 0.0, 0.0, 4.0])  # reasonable starting side length

    # Set bounds for optimization
    bounds = []

    # Bounds for inner hexagon positions (x,y) and rotations (theta)
    for _ in range(11):
        bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (-180.0, 180.0)])

    # Bounds for outer hexagon (center x, center y, angle, side_length)
    bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (-180.0, 180.0), (1.0, 10.0)])

    # Perform optimization
    result = differential_evolution(objective_function, bounds, maxiter=100, popsize=15, seed=42)

    # Extract best solution
    best_params = result.x
    inner_params = best_params[:-4]
    outer_center_x, outer_center_y, outer_angle, outer_side_length = best_params[-4:]

    # Format inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]]

    outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])

    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END