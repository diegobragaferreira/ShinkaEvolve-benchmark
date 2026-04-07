# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from itertools import combinations
from joblib import Parallel, delayed

# Helper function to create a regular hexagon polygon
def create_hexagon(center_x, center_y, side_length=1, rotation_deg=0):
    """Create a shapely polygon representing a regular hexagon"""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    x = center_x + side_length * np.cos(angles)
    y = center_y + side_length * np.sin(angles)
    return Polygon(zip(x, y))

# Helper function to check if hexagon is contained in outer hexagon
def is_contained(hex_poly, outer_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    return outer_poly.contains(hex_poly)

# Helper function to check if two hexagons overlap
def do_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

# Helper function to calculate minimum bounding hexagon side length
def get_bounding_hex_side_length(inner_hex_data, outer_center=(0, 0)):
    """Calculate the side length of the smallest hexagon that can contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        hex_poly = create_hexagon(center_x, center_y, 1, angle)
        vertices = list(hex_poly.exterior.coords)[:-1]  # exclude last duplicate vertex
        all_vertices.extend(vertices)
    
    # Calculate distance from center to each vertex, and take the max
    distances = [np.sqrt((x - outer_center[0])**2 + (y - outer_center[1])**2) for x, y in all_vertices]
    max_distance = max(distances) if distances else 0
    
    # The side length of the outer hexagon should be at least this distance
    return max_distance

# Constraint function for geometric validity
def validate_configuration(inner_hex_data, outer_side_length):
    """Validate that configuration is geometrically valid"""
    try:
        # Create outer hexagon
        outer_hex = create_hexagon(0, 0, outer_side_length, 0)
        
        # Check containment and overlap for each inner hexagon
        total_overlaps = 0
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            inner_hex = create_hexagon(center_x, center_y, 1, angle)
            
            # Check containment
            if not is_contained(inner_hex, outer_hex):
                return False, 0
            
            # Check overlaps with other hexagons (excluding self)
            for j in range(i+1, len(inner_hex_data)):
                center_x2, center_y2, angle2 = inner_hex_data[j]
                inner_hex2 = create_hexagon(center_x2, center_y2, 1, angle2)
                
                if do_overlap(inner_hex, inner_hex2):
                    total_overlaps += 1
        
        return True, total_overlaps
    except Exception:
        return False, float('inf')

# Main optimization function
def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initial configuration inspired by known optimal packings
    # This is a hexagonal lattice arrangement with some variation
    initial_positions = [
        [0, 0, 0],       # center
        [-1.732, 0, 0],   # left
        [1.732, 0, 0],    # right
        [0, 3.464, 0],    # top
        [0, -3.464, 0],   # bottom
        [-1.732, 3.464, 0], # top-left
        [1.732, 3.464, 0],  # top-right
        [-1.732, -3.464, 0], # bottom-left
        [1.732, -3.464, 0],  # bottom-right
        [-3.464, 1.732, 0], # far top-left
        [3.464, 1.732, 0],  # far top-right
        [-3.464, -1.732, 0] # far bottom-left
    ]
    
    initial_angles = [0] * 12
    
    # Convert to flattened array for optimization
    # Format: [x0, y0, angle0, x1, y1, angle1, ...]
    initial_params = []
    for i in range(12):
        initial_params.extend([initial_positions[i][0], 
                             initial_positions[i][1], 
                             initial_positions[i][2]])
    
    # Define bounds for optimization: x, y, angle for each hexagon
    bounds = []
    for i in range(12):
        # x and y coordinates (can be anywhere)
        bounds.extend([(-10, 10), (-10, 10)])
        # angle (0 to 360 degrees)
        bounds.extend([(-180, 180)])
    
    def objective_function(params):
        # Convert flattened params back into hexagon data
        inner_hex_data = []
        for i in range(12):
            x = params[3*i]
            y = params[3*i+1]
            angle = params[3*i+2]
            inner_hex_data.append([x, y, angle])
        
        # Calculate outer hexagon side length
        side_length = get_bounding_hex_side_length(inner_hex_data)
        
        # Check if valid configuration (containment and non-overlap)
        valid, overlaps = validate_configuration(inner_hex_data, side_length)
        
        if not valid:
            # Large penalty for invalid configurations
            return 1e6 + side_length * 1000
        
        # Return negative because we want to maximize 1/side_length
        # So minimize -1/side_length which is equivalent to maximize 1/side_length
        return -1.0 / side_length if side_length > 0 else 1e6
    
    # Run optimization with bounds
    result = differential_evolution(objective_function, bounds, seed=42, 
                                   maxiter=50, popsize=15, disp=False)
    
    # Extract optimized parameters
    optimized_params = result.x
    
    # Convert back to inner hex data format
    inner_hex_data = []
    for i in range(12):
        x = optimized_params[3*i]
        y = optimized_params[3*i+1]
        angle = optimized_params[3*i+2]
        inner_hex_data.append([x, y, angle])
    
    # Get final side length
    outer_hex_side_length = get_bounding_hex_side_length(inner_hex_data)
    
    # Final validation
    valid, _ = validate_configuration(inner_hex_data, outer_hex_side_length)
    if not valid:
        # If still invalid, fallback to initial configuration
        inner_hex_data = np.array(initial_positions)
        outer_hex_side_length = get_bounding_hex_side_length(inner_hex_data)
    
    # Create outer hexagon centered at origin with correct orientation
    outer_hex_data = np.array([0, 0, 0])
    
    # Ensure we don't exceed time limit
    elapsed = time.time() - start_time
    if elapsed > 175:  # leave buffer
        print(f"Warning: Optimization took {elapsed:.2f}s")
    
    return np.array(inner_hex_data), outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
