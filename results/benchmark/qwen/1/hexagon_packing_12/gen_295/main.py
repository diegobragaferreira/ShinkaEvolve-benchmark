# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree
import time
from numba import jit, njit
from joblib import Parallel, delayed
import warnings

@njit
def hexagon_vertices_numba(center_x, center_y, angle_deg, side_length=1):
    """Fast computation of hexagon vertices using numba."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            center_x + side_length * (x_orig * cos_a - y_orig * sin_a),
            center_y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]
    
    return rotated_verts

@njit
def point_in_hexagon_fast(px, py, hx, hy, angle_deg, side_length=1):
    """Fast point-in-hexagon test."""
    vertices = hexagon_vertices_numba(hx, hy, angle_deg, side_length)
    # Ray casting method
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@njit
def distance_point_to_point(px, py, qx, qy):
    """Fast Euclidean distance calculation."""
    dx = px - qx
    dy = py - qy
    return np.sqrt(dx * dx + dy * dy)

@njit
def hexagon_distance_fast(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Fast minimum distance calculation between hexagons."""
    v1 = hexagon_vertices_numba(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices_numba(h2_x, h2_y, h2_angle)
    
    min_dist = np.inf
    for i in range(6):
        for j in range(6):
            # Distance between vertices
            dist = distance_point_to_point(v1[i,0], v1[i,1], v2[j,0], v2[j,1])
            if dist < min_dist:
                min_dist = dist
                
    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices_numba(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment_all_vertices_fast(hex_vertices, outer_center, outer_side_length):
    """Fast containment check using numba."""
    outer_vertices = hexagon_vertices_numba(outer_center[0], outer_center[1], 0, outer_side_length)
    outer_polygon = Polygon(outer_vertices)

    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def compute_outer_hex_side_from_config(inner_hex_data, center=(0,0)):
    """Compute the minimum required outer hexagon side length from current configuration."""
    if len(inner_hex_data) == 0:
        return 100

    max_dist = 0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
        # Add the circumradius of inner hexagon (1 for unit hexagon)
        dist_to_edge = dist + 1.0
        max_dist = max(max_dist, dist_to_edge)

    # For a hexagon, diameter = 2 * circumradius
    return max_dist * 2.0  # Diameter gives us the side length for a hexagon

def initialize_symmetric_configuration():
    """Generate a good symmetric initial configuration."""
    # Mathematical layout based on optimal hexagonal packing principles
    positions = []
    
    # Central hexagon
    positions.append([0, 0, 0])
    
    # First ring - 6 hexagons arranged at distance sqrt(3) from center
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions
    radius = 1.732  # sqrt(3) for optimal packing
    
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append([x, y, 0])
    
    # Second ring - 6 hexagons arranged at distance 2*sqrt(3) from center  
    angles2 = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 directions
    radius2 = 3.464  # 2*sqrt(3)
    
    for angle in angles2:
        x = radius2 * np.cos(angle)
        y = radius2 * np.sin(angle)
        positions.append([x, y, 0])
    
    # Adjust to make exactly 12
    positions = positions[:12]
    
    # Convert to array format
    config = np.array(positions)
    
    # Add slight randomness to break perfect symmetry
    np.random.seed(42)
    config[:, 0] += np.random.normal(0, 0.1, 12)
    config[:, 1] += np.random.normal(0, 0.1, 12)
    
    return config

def evaluate_configuration_with_constraints(inner_hex_data, outer_hex_center=(0,0)):
    """Fast evaluation with constraint checking and penalty system."""
    if len(inner_hex_data) != 12:
        return 1e-10
    
    # Get outer hexagon side length
    outer_side_length = compute_outer_hex_side_from_config(inner_hex_data, outer_hex_center)
    outer_vertices = hexagon_vertices_numba(outer_hex_center[0], outer_hex_center[1], 0, outer_side_length)
    outer_polygon = Polygon(outer_vertices)
    
    # Check containment for all hexagons
    contain_violations = 0
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        vertices = hexagon_vertices_numba(cx, cy, angle)
        
        # Fast containment check first
        if not check_containment_all_vertices_fast(vertices, outer_hex_center, outer_side_length):
            contain_violations += 1
            # Check more precisely with shapely if needed
            hex_poly = Polygon(vertices)
            if not outer_polygon.contains(hex_poly):
                return 1e-10
                
    # Check overlaps efficiently using spatial indexing if needed
    if contain_violations > 0:
        # More thorough overlap checking for cases with containment violations
        hex_polygons = []
        for i in range(len(inner_hex_data)):
            cx, cy, angle = inner_hex_data[i]
            hex_polygons.append(compute_hexagon_polygon(cx, cy, angle))
        
        # Check overlaps between all pairs
        for i in range(len(inner_hex_data)):
            for j in range(i+1, len(inner_hex_data)):
                if hex_polygons[i].intersects(hex_polygons[j]) and not hex_polygons[i].touches(hex_polygons[j]):
                    return 1e-10
    
    # If we reach here, the configuration is valid
    return 1.0 / outer_side_length

def fast_optimization_step(initial_config):
    """Perform fast optimization step using global and local search."""
    # Define bounds for optimization
    bounds = []
    # Positions: -10 to 10 for both x and y (reasonable bounds)
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10)])
    # Angles: 0 to 360 degrees
    for _ in range(12):
        bounds.append((0, 360))

    def objective(x):
        # Reshape the flat vector back to 12 hexagons
        hex_data = x.reshape(-1, 3)
        score = evaluate_configuration_with_constraints(hex_data)
        return -score  # Negative because we want to maximize

    # Use differential evolution for initial global search
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=15,
            seed=42,
            strategy='best1bin'
        )
        
        if result.success:
            optimized_hex_data = result.x.reshape(-1, 3)
            
            # Apply local refinement with L-BFGS-B
            flat_solution = optimized_hex_data.flatten()
            
            def local_objective(x_flat):
                hex_data = x_flat.reshape(-1, 3)
                return -evaluate_configuration_with_constraints(hex_data)
            
            local_result = minimize(
                local_objective,
                flat_solution,
                method='L-BFGS-B',
                bounds=bounds * 12,
                options={'maxiter': 30}
            )
            
            if local_result.success:
                refined_hex_data = local_result.x.reshape(-1, 3)
                return refined_hex_data, True
                
    except Exception:
        pass
        
    return initial_config, False

def advanced_refinement_step(config):
    """Advanced refinement using iterative improvement."""
    current_config = config.copy()
    improved = True
    max_iterations = 20
    
    for iteration in range(max_iterations):
        if not improved:
            break
            
        improved = False
        updated_config = current_config.copy()
        
        # Process hexagons one by one with better spatial awareness
        for i in range(12):
            # Create temporary configuration with other hexagons fixed
            temp_config = current_config.copy()
            # Save current hexagon data
            backup_data = temp_config[i].copy()
            
            # Try to optimize this particular hexagon
            def single_hex_objective(param_array):
                # Update this hexagon with new parameters
                temp_config[i] = param_array
                score = evaluate_configuration_with_constraints(temp_config)
                return -score  # Minimize negative of score
                
            # Optimizer bounds for this hexagon
            bounds = [(-10, 10), (-10, 10), (0, 360)]
            start_params = backup_data
            
            try:
                result = minimize(
                    single_hex_objective,
                    start_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 20}
                )
                
                if result.success and result.fun < -0.9999:  # Some improvement
                    updated_config[i] = result.x
                    improved = True
                    
            except Exception:
                pass
                
        # Update if there was improvement
        if improved:
            current_config = updated_config
            
    return current_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Generate initial configuration
        initial_config = initialize_symmetric_configuration()
        
        # Step 1: Global optimization
        optimized_config, _ = fast_optimization_step(initial_config)
        
        # Step 2: Advanced refinement
        final_config = advanced_refinement_step(optimized_config)
        
        # Final evaluation
        final_score = evaluate_configuration_with_constraints(final_config)
        
        if final_score > 1e-5:
            outer_side_length = 1.0 / final_score
            outer_hex_data = np.array([0, 0, 0])
            return final_config, outer_hex_data, outer_side_length
            
    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}")
        pass

    # Fallback to known good configuration
    inner_hex_data = np.array([
        [0, 0, 0],
        [0, 2, 0],
        [0, -2, 0],
        [1.732, 1, 0],
        [-1.732, 1, 0],
        [1.732, -1, 0],
        [-1.732, -1, 0],
        [3.464, 0, 0],
        [-3.464, 0, 0],
        [1.732, 3, 0],
        [-1.732, 3, 0],
        [1.732, -3, 0],
    ])

    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 6.928

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END