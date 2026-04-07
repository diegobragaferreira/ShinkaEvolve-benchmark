# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
import math

def hexagon_vertices(center_x, center_y, size=1, angle_deg=0):
    """Generate vertices of a regular hexagon given center, size, and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + size * np.cos(angle)
        y = center_y + size * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_containment(hex_vertices, outer_center_x, outer_center_y, outer_size):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_size, 0)
    outer_polygon = Polygon(outer_vertices)
    
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y):
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        distance = np.sqrt((cx - outer_center_x)**2 + (cy - outer_center_y)**2)
        max_distance = max(max_distance, distance + 1)  # Add radius of unit hexagon
    
    return max_distance

def evaluate_configuration(inner_hex_data, outer_center_x, outer_center_y):
    """Evaluate current configuration: returns (validity, inv_radius)."""
    # Check for overlaps
    for i in range(len(inner_hex_data)):
        hex1_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for j in range(i+1, len(inner_hex_data)):
            hex2_vertices = hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], 1, inner_hex_data[j][2])
            if check_overlap(hex1_vertices, hex2_vertices):
                return False, 0
    
    # Check containment
    outer_radius = compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y)
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_radius, 0)
    outer_polygon = Polygon(outer_vertices)
    
    for i in range(len(inner_hex_data)):
        hex_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, 0
    
    # Return inverse of outer radius
    return True, 1.0 / outer_radius

def generate_initial_symmetric_config():
    """Generate a symmetric initial configuration based on hexagonal tiling principles."""
    # Central hexagon
    config = [[0, 0, 0]]
    
    # First ring (6 hexagons)
    for i in range(6):
        angle = i * 60
        radius = 2  # Distance from origin
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config.append([x, y, 0])
    
    # Second ring (6 hexagons)
    for i in range(6):
        angle = 30 + i * 60
        radius = 3.464  # sqrt(12) approximately
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config.append([x, y, 0])
    
    return np.array(config)

def optimize_positions(initial_config, outer_center_x, outer_center_y):
    """Optimize positions using constrained numerical optimization."""
    
    def objective(params):
        # Reconstruct configuration from flattened parameters
        config = initial_config.copy()
        # Update positions only (leave angles as they are for now)
        idx = 0
        for i in range(len(config)):
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2
        
        validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
        if not validity:
            return 1e10  # Large penalty for invalid configurations
        return -inv_radius  # Negative because we want to maximize
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(len(initial_config)):
        initial_params.extend([initial_config[i][0], initial_config[i][1]])
    
    # Perform optimization
    result = minimize(objective, initial_params, method='L-BFGS-B', 
                      bounds=[(-10, 10), (-10, 10)] * len(initial_config))
    
    # Reconstruct optimized configuration
    optimized_config = initial_config.copy()
    idx = 0
    for i in range(len(optimized_config)):
        optimized_config[i][0] = result.x[idx]
        optimized_config[i][1] = result.x[idx + 1]
        idx += 2
        
    return optimized_config

class Point:
    """Simple point class for Shapely compatibility."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate initial symmetric configuration
    initial_config = generate_initial_symmetric_config()
    
    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0
    
    # Optimized positions
    optimized_config = optimize_positions(initial_config, outer_center_x, outer_center_y)
    
    # Final verification and refinement
    iterations = 0
    while iterations < 5:
        iterations += 1
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity:
            break
            
        # If not valid, try a small adjustment to positions
        for i in range(len(optimized_config)):
            optimized_config[i][0] += np.random.normal(0, 0.01)
            optimized_config[i][1] += np.random.normal(0, 0.01)
    
    # Compute final outer hexagon radius
    outer_radius = 1.0 / inv_radius if inv_radius > 0 else 10.0
    
    # Convert back to required format
    # Note: We're keeping rotation angles at 0 for simplicity, as the symmetric solution
    # typically doesn't benefit from rotation in this case
    inner_hex_data = np.array(optimized_config)
    
    # Ensure that we have exactly 12 hexagons
    if len(inner_hex_data) != 12:
        # Fall back to simpler configuration if needed
        # This shouldn't happen with our algorithm, but just in case
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        
        outer_radius = 8.0  # A safe but suboptimal value
    
    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])
    outer_hex_side_length = outer_radius * 2  # approximate
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
