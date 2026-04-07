# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import itertools
from scipy.optimize import differential_evolution
import math

def create_regular_hexagon(center=(0, 0), side_length=1, rotation_deg=0):
    """Create a regular hexagon as a shapely polygon"""
    rotation_rad = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center[0] + side_length * math.cos(angle)
        y = center[1] + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon"""
    return outer_hexagon.contains(hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

def calculate_outer_hex_side_length(inner_hex_data, outer_center=(0, 0)):
    """Calculate the minimum outer hexagon side length that contains all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        
        # Create temporary hexagon to get vertices
        temp_hex = create_regular_hexagon(pos, 1, rot)
        vertices = list(temp_hex.exterior.coords)[:-1]  # Exclude duplicate last point
        
        # Find max distance from center to any vertex
        for vertex in vertices:
            dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # Add some margin to ensure containment
    return max_dist * 1.1

def evaluate_configuration(inner_hex_data, outer_center=(0, 0)):
    """Evaluate a configuration and return penalty and side length"""
    # Create inner hexagons
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        pos = (inner_hex_data[i][0], inner_hex_data[i][1])
        rot = inner_hex_data[i][2]
        hexagon = create_regular_hexagon(pos, 1, rot)
        inner_hexagons.append(hexagon)

    # Create outer hexagon
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data, outer_center)
    outer_hexagon = create_regular_hexagon(outer_center, outer_side_length, 0)

    # Check containment and overlap constraints
    total_penalty = 0

    # Check containment for all inner hexagons
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hexagon):
            total_penalty += 1000  # Large penalty for containment violation

    # Check overlaps between all pairs of inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                total_penalty += 1000  # Large penalty for overlap violation

    return outer_side_length, total_penalty

def generate_hexagon_grid_configurations():
    """Generate initial configurations using hexagonal grid pattern"""
    configurations = []
    
    # Base positions in hexagonal grid pattern
    base_positions = [
        (0, 0),    # center
        (2, 0),    # right
        (-2, 0),   # left
        (1, math.sqrt(3)),    # top-right
        (-1, math.sqrt(3)),   # top-left
        (1, -math.sqrt(3)),   # bottom-right
        (-1, -math.sqrt(3)),  # bottom-left
        (3, math.sqrt(3)),    # far top-right
        (-3, math.sqrt(3)),   # far top-left
        (3, -math.sqrt(3)),   # far bottom-right
        (-3, -math.sqrt(3))   # far bottom-left
    ]
    
    # Systematic perturbation of positions
    for i in range(5):
        config = []
        for j, (x, y) in enumerate(base_positions):
            # Add small random perturbation
            dx = np.random.uniform(-0.3, 0.3)
            dy = np.random.uniform(-0.3, 0.3)
            config.extend([x + dx, y + dy, np.random.uniform(0, 360)])
        configurations.append(config)
    
    return configurations

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid approach combining discrete grid sampling and continuous optimization.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    
    # Generate multiple starting configurations
    initial_configs = generate_hexagon_grid_configurations()
    
    best_config = None
    best_side_length = float('inf')
    
    # Try multiple starting points
    for config in initial_configs:
        # Convert flat config to structured format
        inner_hex_data = []
        for i in range(n):
            x = config[3*i]
            y = config[3*i+1]
            angle = config[3*i+2]
            inner_hex_data.append([x, y, angle])
        
        # Evaluate this configuration
        side_length, penalty = evaluate_configuration(inner_hex_data)
        
        if penalty == 0 and side_length < best_side_length:
            best_side_length = side_length
            best_config = inner_hex_data.copy()
    
    # If we didn't find a valid configuration, fallback to simple arrangement
    if best_config is None:
        best_config = [
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ]
        best_side_length = 8
    
    # Run local optimization on the best configuration
    def objective(params):
        # Reshape parameters
        inner_hex_data = []
        for i in range(n):
            inner_hex_data.append([params[3*i], params[3*i+1], params[3*i+2]])
        
        # Evaluate current configuration
        side_length, penalty = evaluate_configuration(inner_hex_data)
        
        # Return (inverse of side length) + penalty
        # We want to maximize 1/side_length, so minimize -1/side_length
        if penalty > 0:
            return 1000 + penalty  # Penalty for constraint violations
        else:
            return -1.0 / side_length  # Want to maximize 1/side_length
    
    # Set up bounds
    bounds = []
    for i in range(n):
        bounds.extend([(-8, 8), (-8, 8), (0, 360)])
    
    # Start from our best configuration
    initial_params = []
    for hex_data in best_config:
        initial_params.extend(hex_data)
    
    # Optimize using differential evolution with multiple restarts
    try:
        # Multiple restarts with different random seeds
        best_result = None
        best_value = float('inf')
        
        for seed in [42, 123, 456, 789]:
            try:
                np.random.seed(seed)
                result = differential_evolution(
                    objective,
                    bounds,
                    seed=seed,
                    maxiter=200,
                    popsize=15,
                    tol=1e-6,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False
                )
                
                # Check if this is better
                if result.fun < best_value:
                    best_value = result.fun
                    best_result = result
            except:
                continue
        
        if best_result is not None and best_result.success:
            # Extract optimized parameters
            optimized_params = best_result.x
            optimized_config = []
            for i in range(n):
                optimized_config.append([optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]])
            
            # Re-evaluate final configuration
            final_side_length, final_penalty = evaluate_configuration(optimized_config)
            
            if final_penalty == 0:
                best_config = optimized_config
                best_side_length = final_side_length
            else:
                # If constraints violated, fall back to previous best
                pass
                
    except Exception as e:
        # If optimization fails, keep the best configuration found
        pass
    
    # Final validation
    inner_hex_data = np.array(best_config)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = best_side_length
    
    # Ensure we have a valid configuration
    try:
        # Double-check constraints one more time
        side_length, penalty = evaluate_configuration(best_config)
        if penalty > 0:
            # Revert to simple configuration if constraints still violated
            inner_hex_data = np.array([
                [0, 0, 0],  # center
                [-2.5, 0, 0],  # left
                [2.5, 0, 0],  # right
                [-1.25, 2.17, 0],  # top-left
                [1.25, 2.17, 0],  # top-right
                [-1.25, -2.17, 0],  # bottom-left
                [1.25, -2.17, 0],  # bottom-right
                [-3.75, 2.17, 0],  # far top-left
                [3.75, 2.17, 0],  # far top-right
                [-3.75, -2.17, 0],  # far bottom-left
                [3.75, -2.17, 0],  # far bottom-right
            ])
            outer_hex_side_length = 8
            outer_hex_data = np.array([0, 0, 0])
    except:
        # Fallback to simple configuration on error
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_side_length = 8
        outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
