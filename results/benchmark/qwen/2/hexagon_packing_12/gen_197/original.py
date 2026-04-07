# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time
import math
from scipy.spatial.distance import cdist

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def outer_hexagon_vertices(side_length):
    """Generate vertices of outer hexagon centered at origin."""
    return hexagon_vertices(0, 0, 0, side_length)

def check_containment_distance_only(hexagon_vertices_list, outer_side_length):
    """Fast containment check using distance bounds."""
    # For a hexagon with side length 1, apothem is sqrt(3)/2
    # For outer hexagon with side length R, distance from center to edge is R * sqrt(3)/2
    outer_apothem = outer_side_length * math.sqrt(3) / 2
    
    for vertices in hexagon_vertices_list:
        # Check distance of each vertex from origin
        for vertex in vertices:
            x, y = vertex
            distance = math.sqrt(x*x + y*y)
            if distance > outer_apothem:
                return False
    return True

def check_overlap_fast(hexagon_vertices_list):
    """Fast overlap detection using distance bounds."""
    # Calculate centers and apothems for quick bounding check
    centers = []
    apothems = []
    
    for vertices in hexagon_vertices_list:
        # Calculate center
        cx = sum(v[0] for v in vertices) / 6
        cy = sum(v[1] for v in vertices) / 6
        centers.append((cx, cy))
        # Approximate apothem (distance from center to edge)
        apothems.append(1.0 * math.sqrt(3) / 2)
    
    # Check distances between centers
    centers_array = np.array(centers)
    distances = cdist(centers_array, centers_array)
    
    # Any pair that are closer than sum of apothems overlap
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            if distances[i,j] < (apothems[i] + apothems[j]):
                # Do precise check for suspected overlap
                try:
                    p1 = Polygon(hexagon_vertices_list[i])
                    p2 = Polygon(hexagon_vertices_list[j])
                    if p1.intersects(p2):
                        return False
                except:
                    return False
    return True

def check_overlap_shapely(hexagon_vertices_list):
    """Precise overlap detection using Shapely."""
    polygons = [Polygon(vertices) for vertices in hexagon_vertices_list]
    try:
        union = Polygon.union_all(polygons)
        total_area = sum(polygon.area for polygon in polygons)
        union_area = union.area
        return abs(total_area - union_area) < 1e-10
    except:
        # Fallback for complex cases
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return False
        return True

def evaluate_configuration(config, outer_side_length):
    """Evaluate a configuration of 12 hexagons."""
    # Parse configuration into 12 hexagons (x, y, angle)
    hexagons = config.reshape(12, 3)

    # Get vertices for all hexagons
    hexagon_vertices_list = []
    for i in range(12):
        x, y, angle = hexagons[i]
        vertices = hexagon_vertices(x, y, angle)
        hexagon_vertices_list.append(vertices)

    # Fast containment check first
    if not check_containment_distance_only(hexagon_vertices_list, outer_side_length):
        return False, float('inf')  # Invalid configuration

    # Fast overlap check
    if not check_overlap_fast(hexagon_vertices_list):
        return False, float('inf')  # Overlapping hexagons

    # Final precise validation
    if not check_overlap_shapely(hexagon_vertices_list):
        return False, float('inf')
    
    return True, 0  # Valid configuration

def generate_good_initial_config():
    """Generate a good initial configuration based on known optimal patterns."""
    # Start with a known good symmetric pattern
    # This pattern is designed to be close to optimal with some space for improvement
    initial_config = np.array([
        # Center hexagon
        [0.0, 0.0, 0.0],
        # First ring around center (approximate spacing)
        [-1.732, 0.0, 0.0],  # Left
        [1.732, 0.0, 0.0],   # Right
        [0.0, 1.732, 0.0],   # Top
        [0.0, -1.732, 0.0],  # Bottom
        [-0.866, 0.866, 0.0],  # Top-left
        [0.866, 0.866, 0.0],   # Top-right
        [-0.866, -0.866, 0.0], # Bottom-left
        [0.866, -0.866, 0.0],  # Bottom-right
        # Outer ring
        [-2.598, 0.0, 0.0],   # Far left
        [2.598, 0.0, 0.0],    # Far right
        [0.0, 2.598, 0.0],    # Far top
    ])
    
    # Add small random noise to get initial variety
    initial_config += np.random.normal(0, 0.05, initial_config.shape)
    
    return initial_config.flatten()

def local_refinement(config, outer_side_length, max_iter=50):
    """Perform local refinement to improve configuration."""
    # Simple gradient-like approach
    best_config = config.copy()
    best_valid, best_penalty = evaluate_configuration(best_config, outer_side_length)
    
    if not best_valid:
        return config
    
    # Try small perturbations to improve
    for iteration in range(max_iter):
        # Try small modifications to positions
        current_config = best_config.copy()
        # Randomly modify one hexagon at a time
        hex_idx = np.random.randint(0, 12)
        # Small random perturbation
        current_config[hex_idx*3:hex_idx*3+2] += np.random.normal(0, 0.01, 2)
        # Keep angle within [0, 360]
        current_config[hex_idx*3+2] = current_config[hex_idx*3+2] % 360
        
        valid, penalty = evaluate_configuration(current_config, outer_side_length)
        if valid and penalty < best_penalty:
            best_config = current_config
            best_penalty = penalty
    
    return best_config

def optimize_hexagon_positions():
    """Main optimization routine using Monte Carlo sampling."""
    best_outer_side_length = 3.9419123  # Start with target SOTA
    best_config = None
    best_valid = False
    
    # Number of samples to try
    num_samples = 10000
    
    # Generate initial configuration
    initial_config = generate_good_initial_config()
    
    # Sample random configurations
    for sample in range(num_samples):
        # Generate random configuration
        config = np.zeros(36)  # 12 hexagons * 3 params each
        
        # Random positions within reasonable bounds
        for i in range(12):
            config[i*3] = np.random.uniform(-4.0, 4.0)  # x
            config[i*3+1] = np.random.uniform(-4.0, 4.0)  # y
            config[i*3+2] = np.random.uniform(0, 360)     # angle
        
        # Try to improve with local refinement for this config
        refined_config = local_refinement(config, best_outer_side_length)
        
        # Test if this configuration works with current boundary
        valid, penalty = evaluate_configuration(refined_config, best_outer_side_length)
        
        if valid:
            # Try to fit with smaller outer hexagon
            for test_side in np.linspace(3.8, best_outer_side_length, 20)[::-1]:
                valid_test, penalty_test = evaluate_configuration(refined_config, test_side)
                if valid_test:
                    if test_side < best_outer_side_length:
                        best_outer_side_length = test_side
                        best_config = refined_config.copy()
                        best_valid = True
                        break
    
    # If nothing worked, use fallback
    if not best_valid:
        # Use the initial good configuration as fallback
        best_config = initial_config
        best_outer_side_length = 4.0
    
    # Final validation
    final_valid, _ = evaluate_configuration(best_config, best_outer_side_length)
    if not final_valid:
        # Use simple grid as last resort
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_config.reshape(12, 3), np.array([0, 0, 0]), best_outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_positions()
    
    # Calculate actual score
    inv_side_length = 1.0 / outer_hex_side_length
    eval_time = time.time() - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END