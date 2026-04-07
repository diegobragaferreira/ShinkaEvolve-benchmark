# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from math import sqrt, cos, sin, pi
import warnings
warnings.filterwarnings('ignore')


def get_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Get vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = angle_deg * pi / 180
    vertices = []
    for i in range(6):
        theta = angle_rad + i * pi / 3
        x = center_x + side_length * cos(theta)
        y = center_y + side_length * sin(theta)
        vertices.append((x, y))
    return vertices


def is_hexagon_contained(hexagon_vertices, outer_hexagon_vertices):
    """Check if a hexagon is fully contained within another hexagon using Shapely"""
    try:
        inner_poly = Polygon(hexagon_vertices)
        outer_poly = Polygon(outer_hexagon_vertices)
        return outer_poly.contains(inner_poly)
    except:
        return False


def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        return True  # If there's an issue, assume they overlap


def calculate_outer_hexagon_side_length(inner_hex_data, outer_hex_center=(0, 0), outer_angle=0):
    """
    Calculate the minimum side length of outer hexagon that contains all inner hexagons
    """
    # Get vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle_deg = inner_hex_data[i]
        vertices = get_hexagon_vertices(center_x, center_y, angle_deg, 1)
        all_vertices.extend(vertices)

    # Find bounding box of all vertices
    if not all_vertices:
        return 100

    min_x = min(v[0] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)

    # Calculate distance from center to farthest vertex
    center_x, center_y = outer_hex_center
    max_dist = 0
    for x, y in all_vertices:
        dist = sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)

    # For hexagon, side length should be such that it encompasses this distance
    # The radius of circumscribed circle of hexagon with side length s is s
    # So we want s >= max_dist
    return max_dist


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Initial good configuration based on hexagonal lattice packing (approximate)
    # 3 rows: center row, upper row, lower row
    # Row 1 (top): 3 hexagons
    # Row 2 (middle): 3 hexagons
    # Row 3 (bottom): 3 hexagons
    # Center hexagon
    # Plus one additional hexagon to reach 11 total

    # Hexagon diameter (distance across opposite points) = 2
    # Horizontal spacing = sqrt(3) ≈ 1.732
    # Vertical spacing = 1.5

    initial_positions = [
        [0, 0, 0],      # center
        [-sqrt(3), 1.5, 0],   # top-left
        [sqrt(3), 1.5, 0],    # top-right
        [-sqrt(3), -1.5, 0],  # bottom-left
        [sqrt(3), -1.5, 0],   # bottom-right
        [-2*sqrt(3), 0, 0],   # left
        [2*sqrt(3), 0, 0],    # right
        [-sqrt(3)/2, 0, 0],    # middle-left
        [sqrt(3)/2, 0, 0],     # middle-right
        [0, 1.5, 0],           # top-center
        [0, -1.5, 0],          # bottom-center
    ]

    # Convert to numpy array
    inner_hex_data = np.array(initial_positions)

    # Calculate initial outer hexagon side length
    outer_hex_side_length = calculate_outer_hexagon_side_length(inner_hex_data)

    # Refine with optimization
    def objective(params):
        # params: [x1, y1, ..., x11, y11] - positions of 11 hexagons
        # Reshape into 11 (x,y) pairs
        positions = params.reshape(11, 2)

        # Update inner hex data with new positions
        new_inner_data = np.zeros((11, 3))
        for i in range(11):
            new_inner_data[i] = [positions[i][0], positions[i][1], 0]

        # Calculate required outer hexagon side length
        side_length = calculate_outer_hexagon_side_length(new_inner_data)

        # We want to minimize this (maximize 1/side_length)
        return side_length

    def constraint_func(params):
        # params: [x1, y1, ..., x11, y11]
        positions = params.reshape(11, 2)

        # Check overlaps
        penalty = 0

        # Check pairwise overlaps
        for i in range(11):
            for j in range(i+1, 11):
                pos_i = positions[i]
                pos_j = positions[j]

                # Distance between centers
                dx = pos_i[0] - pos_j[0]
                dy = pos_i[1] - pos_j[1]
                dist_sq = dx*dx + dy*dy

                # Minimum distance to avoid overlapping hexagons
                # Two unit hexagons just touching have distance = 2
                if dist_sq < 3.99:  # slightly less than 2^2 = 4 to ensure non-overlapping
                    penalty += (3.99 - dist_sq) * 1000

        return penalty

    # Start with initial positions
    x0 = inner_hex_data[:, :2].flatten()

    # Define bounds for positions - reasonable limits to prevent wild searches
    bounds = [(-10, 10) for _ in range(22)]

    # Optimization settings
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}

    try:
        # Use L-BFGS-B optimizer
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         constraints={'type': 'ineq', 'fun': constraint_func},
                         options=options, tol=1e-6)

        if result.success:
            # Extract optimized positions
            final_positions = result.x.reshape(11, 2)

            # Update data
            inner_hex_data = np.zeros((11, 3))
            for i in range(11):
                inner_hex_data[i] = [final_positions[i][0], final_positions[i][1], 0]

            outer_hex_side_length = calculate_outer_hexagon_side_length(inner_hex_data)
        else:
            # If optimization fails, use initial configuration
            outer_hex_side_length = calculate_outer_hexagon_side_length(inner_hex_data)

    except Exception as e:
        # If optimization fails, fall back to initial configuration
        outer_hex_side_length = calculate_outer_hexagon_side_length(inner_hex_data)

    # Set outer hexagon at center
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END