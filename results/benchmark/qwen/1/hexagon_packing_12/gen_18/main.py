# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Constants
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6
PI = np.pi

def get_hexagon_vertices(center_x, center_y, angle_degrees, radius=UNIT_HEX_RADIUS):
    """Get vertices of a regular hexagon given center, angle, and radius"""
    angle_rad = np.radians(angle_degrees)
    angles = np.linspace(0, 2*np.pi, HEX_VERTICES+1)[:-1] + angle_rad
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return list(zip(x, y))

def create_outer_hexagon_vertices(radius, center_x=0, center_y=0):
    """Create vertices of outer hexagon"""
    angles = np.linspace(0, 2*np.pi, HEX_VERTICES+1)[:-1]
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return list(zip(x, y))

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2) and not poly1.touches(poly2)

def check_containment(hex_vertices, outer_vertices):
    """Check if all hex vertices are inside outer hexagon"""
    outer_polygon = Polygon(outer_vertices)
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def calculate_objective(config):
    """Calculate objective function value (1/outer_radius)"""
    # Extract parameters
    inner_positions = config[:24].reshape(-1, 2)  # 12 hexagons * 2 coordinates
    inner_angles = config[24:36]  # 12 angles
    outer_radius = config[36]

    # Create vertices for all inner hexagons
    inner_hexagons = []
    for i in range(12):
        center_x, center_y = inner_positions[i]
        angle = inner_angles[i]
        vertices = get_hexagon_vertices(center_x, center_y, angle)
        inner_hexagons.append(vertices)

    # Create outer hexagon vertices
    outer_vertices = create_outer_hexagon_vertices(outer_radius)

    # Check constraints
    try:
        # Check containment
        for vertices in inner_hexagons:
            if not check_containment(vertices, outer_vertices):
                return 1e10  # Large penalty for containment violation

        # Check overlaps
        for i in range(12):
            for j in range(i+1, 12):
                if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return 1e10  # Large penalty for overlap violation

        # If all constraints satisfied, return inverse of outer radius
        return 1.0 / outer_radius if outer_radius > 0 else 1e10

    except Exception:
        return 1e10

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define initial configuration based on mathematical insight for dense packing
    # Start with a known good configuration pattern
    initial_positions = np.array([
        [0, 0],          # Center
        [0, 2],          # Top
        [0, -2],         # Bottom
        [1.732, 1],      # Top-right (sqrt(3) for hexagon distance)
        [-1.732, 1],     # Top-left
        [1.732, -1],     # Bottom-right
        [-1.732, -1],    # Bottom-left
        [3.464, 0],      # Far right
        [-3.464, 0],     # Far left
        [1.732, -3],     # Bottom far-right
        [-1.732, -3],    # Bottom far-left
        [0, -4],         # Far bottom
    ])

    initial_angles = np.zeros(12)  # All horizontal for now

    # Initial guess for outer radius (larger than needed)
    initial_outer_radius = 5.0

    # Combine into single configuration vector
    config = np.concatenate([
        initial_positions.flatten(),   # 24 values
        initial_angles,                # 12 values
        [initial_outer_radius]         # 1 value
    ])

    # Use scipy optimization to find better configuration
    def objective(x):
        return calculate_objective(x)

    # Set bounds and constraints
    bounds = [(None, None)] * 37  # All variables unbounded except outer radius which should be positive
    bounds[-1] = (1e-6, None)  # Outer radius must be positive

    # Optimization
    try:
        result = minimize(
            objective,
            config,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        if result.success:
            optimized_config = result.x
        else:
            # If optimization fails, use the initial configuration
            optimized_config = config

    except Exception:
        # Fallback to initial configuration if optimization fails
        optimized_config = config

    # Extract results
    inner_positions = optimized_config[:24].reshape(-1, 2)
    inner_angles = optimized_config[24:36]
    outer_radius = optimized_config[36]

    # Format inner hex data
    inner_hex_data = np.column_stack([inner_positions, inner_angles])

    # Outer hex data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END