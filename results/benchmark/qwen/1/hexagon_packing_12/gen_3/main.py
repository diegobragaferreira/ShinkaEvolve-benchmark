# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
import math

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = [(center[0] + side_length * np.cos(a), 
               center[1] + side_length * np.sin(a)) for a in angles]
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hexagon.contains(hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def evaluate_packing(config, outer_side_length):
    """Evaluate the packing quality for given configuration."""
    # Convert config to list of hexagons
    hexagons = []
    for i in range(12):
        x, y, angle = config[i*3:(i+1)*3]
        hexagon = create_regular_hexagon((x, y), 1, angle)
        hexagons.append(hexagon)
    
    # Create outer hexagon
    outer_hex = create_regular_hexagon((0, 0), outer_side_length, 0)
    
    # Check containment and overlap
    total_penalty = 0
    
    # Check containment
    for hexagon in hexagons:
        if not check_containment(hexagon, outer_hex):
            total_penalty += 10000  # Large penalty for containment violation
            
    # Check overlaps
    for i in range(12):
        for j in range(i+1, 12):
            if check_overlap(hexagons[i], hexagons[j]):
                total_penalty += 10000  # Large penalty for overlap
                
    return total_penalty

def objective_function(config_and_side, outer_side_length=None):
    """Objective function to minimize (negative of inverse side length)."""
    config = config_and_side[:-1]
    side_length = config_and_side[-1]
    
    # Calculate penalty based on constraint violations
    penalty = evaluate_packing(config, side_length)
    
    # Return negative inverse side length plus penalty
    if penalty > 0:
        return penalty + 1e6  # Add large penalty for constraint violations
    
    # Return negative inverse side length to maximize 1/side_length
    return -1.0 / side_length

def optimize_hexagon_packing():
    """Optimize the 12 hexagon packing."""
    # Initial good configuration based on known solutions
    initial_config = np.array([
        [0, 0, 0],         # center
        [-2.0, 0, 0],      # left
        [2.0, 0, 0],       # right
        [0, 2.0, 0],       # top
        [0, -2.0, 0],      # bottom
        [-1.0, 1.0, 0],    # top-left
        [1.0, 1.0, 0],     # top-right
        [-1.0, -1.0, 0],   # bottom-left
        [1.0, -1.0, 0],    # bottom-right
        [-1.5, 1.5, 0],    # further top-left
        [1.5, 1.5, 0],     # further top-right
        [-1.5, -1.5, 0],   # further bottom-left
    ]).flatten()
    
    # Scale initial configuration to fit in expected region
    initial_config *= 0.7
    initial_side_length = 4.0
    
    # Optimization bounds
    bounds = [(-10, 10)] * 36 + [(2.0, 8.0)]
    
    # Perform optimization
    result = minimize(objective_function, 
                      np.append(initial_config, initial_side_length),
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 1000})
    
    return result.x[:-1], result.x[-1]

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start timing
    start_time = time.time()
    
    # Optimize the packing
    try:
        optimized_config, final_side_length = optimize_hexagon_packing()
    except Exception as e:
        # Fallback to a reasonable configuration if optimization fails
        print(f"Optimization failed: {e}")
        optimized_config = np.array([
            0, 0, 0,           # center
            -1.7, 0, 0,        # left
            1.7, 0, 0,         # right
            0, 1.7, 0,         # top
            0, -1.7, 0,        # bottom
            -1.7, 1.7, 0,      # top-left
            1.7, 1.7, 0,       # top-right
            -1.7, -1.7, 0,     # bottom-left
            1.7, -1.7, 0,      # bottom-right
            -2.1, 2.1, 0,      # further top-left
            2.1, 2.1, 0,       # further top-right
            -2.1, -2.1, 0,     # further bottom-left
        ])
        final_side_length = 4.0
    
    # Format the output data
    inner_hex_data = optimized_config.reshape(12, 3)
    outer_hex_data = np.array([0, 0, 0])
    
    # Calculate performance metrics
    end_time = time.time()
    eval_time = end_time - start_time
    inv_outer_hex_side_length = 1.0 / final_side_length
    benchmark_ratio = inv_outer_hex_side_length / 0.2537
    
    # Print some debug info
    print(f"Final side length: {final_side_length:.6f}")
    print(f"Inverse side length: {inv_outer_hex_side_length:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")
    print(f"Eval time: {eval_time:.6f}")
    
    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END
