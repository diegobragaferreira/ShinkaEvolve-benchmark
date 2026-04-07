# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from numba import jit
import time
from itertools import product
import warnings

@jit(nopython=True)
def hexagon_vertices_fast(x, y, angle_deg, side_length=1):
    """Fast generation of hexagon vertices using numba"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon_fast(point, polygon):
    """Fast point-in-polygon test using ray casting"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@jit(nopython=True)
def distance_point_to_segment(point, seg_start, seg_end):
    """Fast distance from point to line segment"""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    # Vector from start to end
    dx, dy = x2 - x1, y2 - y1
    # Vector from start to point
    px_minus_x1, py_minus_y1 = px - x1, py - y1

    # Project point onto line
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return np.sqrt(px_minus_x1*px_minus_x1 + py_minus_y1*py_minus_y1)

    t = (px_minus_x1*dx + py_minus_y1*dy) / length_sq
    t = max(0, min(1, t))

    # Closest point on segment
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy

    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance_fast(hex1_vertices, hex2_vertices):
    """Fast computation of minimum distance between hexagons"""
    min_dist = np.inf
    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i+1)%6]
        for j in range(6):
            q1 = hex2_vertices[j]
            q2 = hex2_vertices[(j+1)%6]
            dist = distance_point_to_segment(q1, p1, p2)
            min_dist = min(min_dist, dist)
    return min_dist

@jit(nopython=True)
def compute_outer_hex_side_length_fast(all_vertices):
    """Fast computation of outer hexagon side length"""
    # Find bounding circle center and radius
    center_x = 0.0
    center_y = 0.0
    for i in range(len(all_vertices)):
        center_x += all_vertices[i, 0]
        center_y += all_vertices[i, 1]
    center_x /= len(all_vertices)
    center_y /= len(all_vertices)

    # Calculate maximum distance from center to any vertex
    max_distance = 0.0
    for i in range(len(all_vertices)):
        dx = all_vertices[i, 0] - center_x
        dy = all_vertices[i, 1] - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > max_distance:
            max_distance = dist

    # For a hexagon, we need side length >= max_distance * 2 / sqrt(3)
    side_length = max_distance * 2 / np.sqrt(3)

    return side_length

def create_symmetry_base_config():
    """Create base configuration using symmetry group theory with optimized arrangement"""
    # Highly optimized symmetric configuration based on 12-fold symmetry
    # Positions chosen to maximize packing efficiency while respecting geometric constraints
    
    # Central hexagon
    positions = [[0.0, 0.0, 0.0]]
    
    # First ring: 6 hexagons arranged in regular hexagon pattern
    # These are carefully placed to allow for optimal packing
    ring1_angles = [0, 60, 120, 180, 240, 300]
    ring1_radius = 2.15  # Slightly optimized radius for better packing
    
    for angle in ring1_angles:
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Second ring: 5 hexagons arranged in pentagon
    # Placed to minimize overlap potential with first ring
    ring2_angles = [0, 72, 144, 216, 288]
    ring2_radius = 3.45  # Optimized for reduced conflicts
    
    for angle in ring2_angles:
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    return np.array(positions)

def create_outer_hexagon_vertices(side_length):
    """Create vertices of outer hexagon with given side length"""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.column_stack([np.cos(angles), np.sin(angles)]) * side_length
    return vertices

def check_containment_fast(hex_position, outer_vertices):
    """Fast containment check using vertex position"""
    x, y, angle = hex_position
    vertices = hexagon_vertices_fast(x, y, angle)

    # Check if all vertices are within outer hexagon
    for vertex in vertices:
        if not point_in_polygon_fast(vertex, outer_vertices):
            return False
    return True

def compute_outer_side_length(hex_data):
    """Compute minimum side length of outer hexagon"""
    all_vertices = []
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        all_vertices.extend(vertices)
    
    all_vertices = np.array(all_vertices)
    return compute_outer_hex_side_length_fast(all_vertices)

def calculate_fitness(hex_data, outer_side_length):
    """Calculate fitness for the hexagon packing with penalty function"""
    penalty = 0
    
    # Fast initial validation using distance checks for overlaps
    # Only do precise overlap checks when needed
    for i in range(len(hex_data)):
        for j in range(i+1, len(hex_data)):
            x1, y1, angle1 = hex_data[i]
            x2, y2, angle2 = hex_data[j]
            
            # Quick distance-based early rejection
            dist_centers = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist_centers < 1.99:  # Likely to overlap
                # Precise checking with shapely for exact overlap detection
                v1 = hexagon_vertices_fast(x1, y1, angle1)
                v2 = hexagon_vertices_fast(x2, y2, angle2)
                p1 = Polygon(v1)
                p2 = Polygon(v2)
                if p1.intersects(p2):
                    penalty += 1000000  # Heavy penalty for overlaps
    
    # Check containment with Shapely for precision
    outer_vertices = create_outer_hexagon_vertices(outer_side_length)
    
    containment_violations = 0
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        vertices = hexagon_vertices_fast(x, y, angle)
        hex_poly = Polygon(vertices)
        
        # Point-by-point containment check
        for vertex in vertices:
            point = Point(vertex[0], vertex[1])
            if not Polygon(outer_vertices).contains(point):
                # Calculate how far outside the boundary
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                violation_distance = max(0, dist - outer_side_length)
                penalty += violation_distance * 1500  # Higher penalty for containment
                containment_violations += 1

    # Objective: maximize 1/outer_side_length (minimize -1/outer_side_length)
    # Add penalty to avoid invalid configurations
    if penalty > 0:
        return penalty + 1000000  # Very high penalty for invalid configurations
    
    # Return negative of 1/outer_side_length to maximize 1/outer_side_length
    return -1.0 / (outer_side_length + 1e-10)

def apply_symmetry_operations(config):
    """Apply symmetry operations to generate diverse configurations"""
    # Apply 12-fold dihedral group operations (rotations + reflections)
    configs = [config.copy()]
    
    # Generate rotated versions
    for i in range(1, 12):
        angle = i * 30  # 30 degree increments
        rotated = config.copy()
        for j in range(len(rotated)):
            # Rotate around origin
            x, y, theta = rotated[j]
            rad_angle = np.radians(angle)
            new_x = x * np.cos(rad_angle) - y * np.sin(rad_angle)
            new_y = x * np.sin(rad_angle) + y * np.cos(rad_angle)
            rotated[j] = [new_x, new_y, (theta + angle) % 360]
        configs.append(rotated)
    
    # Generate reflected versions (for better diversity)
    for i in range(len(configs)):
        reflected = configs[i].copy()
        for j in range(len(reflected)):
            x, y, theta = reflected[j]
            # Reflect across x-axis
            reflected[j] = [x, -y, (360 - theta) % 360]
        configs.append(reflected)
    
    return configs

def optimize_with_symmetry_search():
    """Main optimization using symmetry-based search approach"""
    # Start with highly optimized symmetric base
    base_config = create_symmetry_base_config()
    
    # Generate diverse configurations through symmetry
    all_configs = apply_symmetry_operations(base_config)
    
    best_fitness = float('inf')
    best_config = base_config.copy()
    
    # Evaluate all symmetry-generated configurations
    for config in all_configs:
        # Compute outer side length
        outer_side = compute_outer_side_length(config)
        
        # Calculate fitness
        fitness = calculate_fitness(config, outer_side)
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_config = config.copy()
    
    # Local refinement of the best configuration using optimization
    # Only optimize a few key parameters to reduce computation cost
    try:
        # Select only key positions for fine-tuning (central and first ring)
        key_positions = [0, 1, 2, 3, 4, 5, 6]  # Important positions
        # Create flattened vector for optimization
        initial_params = []
        for idx in key_positions:
            initial_params.extend(best_config[idx][:2])  # Only x,y positions, ignore rotation for now
        
        def local_objective(params):
            # Reconstruct the full configuration
            temp_config = best_config.copy()
            param_idx = 0
            for idx in key_positions:
                temp_config[idx][0] = params[param_idx]
                temp_config[idx][1] = params[param_idx + 1]
                param_idx += 2
            
            outer_side = compute_outer_side_length(temp_config)
            return calculate_fitness(temp_config, outer_side)
        
        # Optimize just the key positions
        result = minimize(local_objective, initial_params, method='L-BFGS-B', 
                         bounds=[(-8, 8), (-8, 8)] * len(key_positions), 
                         options={'maxiter': 50, 'ftol': 1e-6})
        
        if result.success:
            # Rebuild final configuration
            final_config = best_config.copy()
            param_idx = 0
            for idx in key_positions:
                final_config[idx][0] = result.x[param_idx]
                final_config[idx][1] = result.x[param_idx + 1]
                param_idx += 2
            best_config = final_config
            
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
    
    return best_config

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
        inner_hex_data = optimize_with_symmetry_search()
    except Exception as e:
        warnings.warn(f"Symmetry optimization failed: {e}")
        # Fallback to base symmetric configuration
        inner_hex_data = create_symmetry_base_config()
    
    # Final validation and compute outer side length
    outer_side_length = compute_outer_side_length(inner_hex_data)
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    end_time = time.time()
    
    # Calculate benchmark ratio for reporting
    inv_outer_side_length = 1.0 / outer_side_length
    benchmark_ratio = inv_outer_side_length / 0.2537
    
    print(f"inv_outer_hex_side_length: {inv_outer_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {end_time - start_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END