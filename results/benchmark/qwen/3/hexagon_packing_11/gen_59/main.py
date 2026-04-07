# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import time

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
    Uses optimized approach with direct geometric calculation.
    """
    # Calculate maximum distance from center to any vertex of any hexagon
    max_dist = 0
    
    for i in range(len(inner_configs)):
        cx, cy, angle = inner_configs[i]
        
        # Get the vertices of this hexagon
        hex_vertices = transform_hexagon_vertices(UNIT_HEX_VERTICES, cx, cy, angle)
        
        # Find maximum distance from center to any vertex
        distances = np.sqrt(np.sum((hex_vertices - np.array(outer_center))**2, axis=1))
        max_dist = max(max_dist, np.max(distances))
    
    # Add small margin for numerical stability
    return max_dist + 0.01

def calculate_overlaps_penalty(inner_configs):
    """Calculate penalty for overlaps between hexagons."""
    penalty = 0
    n = len(inner_configs)
    
    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(n):
        cx, cy, angle = inner_configs[i]
        inner_polygons.append(create_hexagon_polygon(cx, cy, angle))
    
    # Check for overlaps
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(inner_polygons[i], inner_polygons[j]):
                # Return large penalty for overlaps
                return 1e6
    
    return penalty

def evaluate_objective(config, outer_center=(0,0), outer_angle=0):
    """Evaluate objective function: maximize 1/outer_hex_side_length"""
    # Reshape config into (11, 3) array
    configs = config.reshape(-1, 3)
    
    # Check overlaps
    overlap_penalty = calculate_overlaps_penalty(configs)
    if overlap_penalty > 1e5:
        return 1e6  # Large penalty for overlaps
    
    # Compute outer radius
    outer_radius = compute_outer_hexagon_radius(configs, outer_center, outer_angle)
    
    # Return negative inverse of outer radius (to maximize 1/outer_radius)
    return -1.0 / outer_radius

def optimize_positions_and_angles(initial_configs):
    """Optimize positions and angles using scipy minimization"""
    # Flatten initial configuration
    flat_config = initial_configs.flatten()
    
    # Define bounds: x,y in [-10,10], angle in [0,360]
    bounds = [(-10, 10), (-10, 10), (0, 360)] * 11
    
    # Optimization options
    options = {'maxiter': 500, 'disp': False}
    
    # Perform optimization
    result = minimize(
        evaluate_objective,
        flat_config,
        method='L-BFGS-B',
        bounds=bounds,
        options=options,
        tol=1e-6
    )
    
    return result.x.reshape(-1, 3)

def generate_tiling_pattern():
    """Generate an initial tiling pattern based on hexagonal lattice principles"""
    # Inspired by optimal hexagonal packing arrangements
    # Place 11 hexagons in a specific pattern that should be close to optimal
    
    configs = np.array([
        [0, 0, 0],           # center
        [-2.5, 0, 0],        # left
        [2.5, 0, 0],         # right
        [-1.25, 2.17, 0],    # top-left
        [1.25, 2.17, 0],     # top-right
        [-1.25, -2.17, 0],   # bottom-left
        [1.25, -2.17, 0],    # bottom-right
        [-3.75, 2.17, 0],    # far top-left
        [3.75, 2.17, 0],     # far top-right
        [-3.75, -2.17, 0],   # far bottom-left
        [3.75, -2.17, 0],    # far bottom-right
    ])
    
    return configs

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Create initial tiling pattern
    initial_configs = generate_tiling_pattern()
    
    # Apply optimization to refine the configuration
    optimized_configs = optimize_positions_and_angles(initial_configs)
    
    # Final validation
    # Check overlaps one more time
    penalty = calculate_overlaps_penalty(optimized_configs)
    if penalty > 1e5:
        # If still has overlaps, fallback to initial pattern
        optimized_configs = initial_configs
    
    # Compute final outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(optimized_configs)
    
    # Convert to required output format
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    # Return results
    return optimized_configs, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END
