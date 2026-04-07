# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from numba import jit
import time
import random
from collections import defaultdict

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given position, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.arange(0, 6) * np.pi / 3
    vertices = np.zeros((6, 2))
    for i in range(6):
        vertices[i, 0] = x + side_length * np.cos(angles[i] + angle_rad)
        vertices[i, 1] = y + side_length * np.sin(angles[i] + angle_rad)
    return vertices

@jit(nopython=True)
def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting"""
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
def distance_point_to_line_segment(point, line_start, line_end):
    """Calculate distance from point to line segment"""
    A = point[0] - line_start[0]
    B = point[1] - line_start[1]
    C = line_end[0] - line_start[0]
    D = line_end[1] - line_start[1]
    
    dot = A*C + B*D
    len_sq = C*C + D*D
    if len_sq == 0:
        return np.sqrt(A*A + B*B)
    param = dot / len_sq
    param = max(0, min(1, param))
    xx = line_start[0] + param * C
    yy = line_start[1] + param * D
    dx = point[0] - xx
    dy = point[1] - yy
    return np.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def point_to_hexagon_distance(point, hex_vertices):
    """Calculate minimum distance from point to hexagon boundary"""
    min_dist = np.inf
    for i in range(6):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i+1)%6]
        dist = distance_point_to_line_segment(point, p1, p2)
        min_dist = min(min_dist, dist)
    return min_dist

@jit(nopython=True)
def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using separating axis theorem"""
    # Check if any vertex of hex1 is inside hex2
    for v in hex1_vertices:
        if point_in_polygon(v, hex2_vertices):
            return True
    # Check if any vertex of hex2 is inside hex1  
    for v in hex2_vertices:
        if point_in_polygon(v, hex1_vertices):
            return True
    return False

def create_spatial_hash(hex_vertices_list, cell_size=2.0):
    """Create spatial hash grid for fast overlap checking"""
    hash_grid = defaultdict(list)
    for i, vertices in enumerate(hex_vertices_list):
        # Get bounding box of hexagon
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)
        
        # Add to all relevant cells
        start_col = int(min_x // cell_size)
        end_col = int(max_x // cell_size) + 1
        start_row = int(min_y // cell_size)
        end_row = int(max_y // cell_size) + 1
        
        for col in range(start_col, end_col + 1):
            for row in range(start_row, end_row + 1):
                hash_grid[(col, row)].append(i)
    return hash_grid

def get_overlapping_indices(hash_grid, hex_index, hex_vertices, cell_size=2.0):
    """Get indices of potentially overlapping hexagons using spatial hash"""
    overlapping = set()
    # Get bounding box of hexagon
    min_x = min(v[0] for v in hex_vertices)
    max_x = max(v[0] for v in hex_vertices)
    min_y = min(v[1] for v in hex_vertices)
    max_y = max(v[1] for v in hex_vertices)
    
    # Check all relevant cells
    start_col = int(min_x // cell_size)
    end_col = int(max_x // cell_size) + 1
    start_row = int(min_y // cell_size)
    end_row = int(max_y // cell_size) + 1
    
    for col in range(start_col, end_col + 1):
        for row in range(start_row, end_row + 1):
            if (col, row) in hash_grid:
                for idx in hash_grid[(col, row)]:
                    if idx != hex_index:
                        overlapping.add(idx)
    return overlapping

def calculate_penalty_for_hexagon(hex_index, hex_data, hex_vertices_list, outer_radius):
    """Calculate penalty for a single hexagon"""
    penalty = 0
    vertices = hex_vertices_list[hex_index]
    
    # Check containment penalty
    outer_hex_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    for vx, vy in vertices:
        point = np.array([vx, vy])
        if not point_in_polygon(point, outer_hex_vertices):
            dist = np.sqrt(vx*vx + vy*vy)
            penalty += (dist - outer_radius + 0.5)**2
    
    return penalty

def calculate_total_penalty(hex_data, outer_radius):
    """Calculate total penalty for all hexagons using spatial hashing"""
    n = len(hex_data)
    
    # Precompute vertices for all hexagons
    hex_vertices_list = [hexagon_vertices(hex_data[i][0], hex_data[i][1], hex_data[i][2]) for i in range(n)]
    
    # Create spatial hash
    hash_grid = create_spatial_hash(hex_vertices_list)
    
    total_penalty = 0
    
    # Check containment for each hexagon
    for i in range(n):
        total_penalty += calculate_penalty_for_hexagon(i, hex_data, hex_vertices_list, outer_radius)
    
    # Check overlaps using spatial hashing
    for i in range(n):
        # Get potentially overlapping hexagons
        overlapping_indices = get_overlapping_indices(hash_grid, i, hex_vertices_list[i])
        
        # Check actual overlaps
        for j in overlapping_indices:
            if i < j:  # Avoid double counting
                vertices1 = hex_vertices_list[i]
                vertices2 = hex_vertices_list[j]
                
                if check_hexagon_overlap(vertices1, vertices2):
                    total_penalty += 1000000  # Large penalty for overlaps
    
    return total_penalty

def get_outer_hexagon_radius(inner_hex_data):
    """Compute the minimum radius required to contain all hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(x, y, angle)
        for vx, vy in vertices:
            dist = np.sqrt(vx*vx + vy*vy)
            max_dist = max(max_dist, dist)
    return max_dist + 1.0  # Add small margin

def create_symmetric_initial_config():
    """Create a highly symmetric initial configuration inspired by group theory"""
    # More mathematically informed arrangement based on proven packing patterns
    positions = [
        [0.0, 0.0, 0.0],     # Center
        [0.0, 2.1, 0.0],     # Up
        [0.0, -2.1, 0.0],    # Down
        [1.8, 1.0, 0.0],     # Up-right 
        [-1.8, 1.0, 0.0],    # Up-left
        [1.8, -1.0, 0.0],    # Down-right
        [-1.8, -1.0, 0.0],   # Down-left
        [3.6, 0.0, 0.0],     # Far right
        [-3.6, 0.0, 0.0],    # Far left
        [0.0, 3.6, 0.0],     # Far up
        [0.0, -3.6, 0.0],    # Far down
        [1.8, 3.1, 0.0],     # Far upper right
    ]
    
    # Add slight randomness to avoid getting stuck in poor local minima
    positions = np.array(positions)
    for i in range(1, len(positions)):
        positions[i][0] += np.random.normal(0, 0.05)
        positions[i][1] += np.random.normal(0, 0.05)
        
    return positions

def adaptive_local_refinement(initial_config, max_iterations=100, method='L-BFGS-B'):
    """Apply local refinement to improve the configuration using scipy optimization"""
    from scipy.optimize import minimize
    
    # Convert to flat parameter array for optimization
    params = initial_config.flatten()
    
    def objective(flat_params):
        config = flat_params.reshape(-1, 3)
        outer_radius = get_outer_hexagon_radius(config)
        penalty = calculate_total_penalty(config, outer_radius)
        # Objective is to minimize penalty and maximize 1/R 
        return penalty - 1.0/(outer_radius + 1e-6)  # Small epsilon for numerical stability
    
    # Use bounds for parameters to prevent extreme positions
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)] * 12
    
    try:
        result = minimize(objective, params, method=method, bounds=bounds, 
                         options={'maxiter': max_iterations, 'gtol': 1e-8})
        if result.success:
            # Return optimized configuration
            final_config = result.x.reshape(-1, 3)
            return final_config
    except Exception as e:
        # If optimization fails, return original
        print(f"Local refinement failed: {e}")
    
    return initial_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Phase 1: Start with mathematical symmetric configuration
    inner_hex_data = create_symmetric_initial_config()
    
    # Phase 2: Apply coarse local optimization to refine initial configuration
    inner_hex_data = adaptive_local_refinement(inner_hex_data, max_iterations=50)
    
    # Phase 3: Apply fine-tuning with more aggressive local optimization
    inner_hex_data = adaptive_local_refinement(inner_hex_data, max_iterations=75)
    
    # Compute final outer hexagon size
    outer_hex_side_length = get_outer_hexagon_radius(inner_hex_data)
    
    # Create outer hexagon data (centered, no rotation)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin, no rotation
    
    # Final validation and cleanup
    final_penalty = calculate_total_penalty(inner_hex_data, outer_hex_side_length)
    if final_penalty > 50000:  # If there are significant violations
        # Fallback to known good solution with slightly better parameters
        inner_hex_data = np.array([
            [0, 0, 0],          # center
            [-2.4, 0, 0],       # left
            [2.4, 0, 0],        # right
            [-1.2, 2.1, 0],     # top-left
            [1.2, 2.1, 0],      # top-right
            [-1.2, -2.1, 0],    # bottom-left
            [1.2, -2.1, 0],     # bottom-right
            [-3.6, 2.1, 0],     # far top-left
            [3.6, 2.1, 0],      # far top-right
            [-3.6, -2.1, 0],    # far bottom-left
            [3.6, -2.1, 0],     # far bottom-right
            [0, -3.8, 0],       # far bottom-center
        ])
        outer_hex_side_length = 7.8  # Updated to match better fit
    
    end_time = time.time()
    
    # Calculate benchmark ratio
    benchmark_ratio = 1.0 / outer_hex_side_length / 0.2537
    
    # Print diagnostic information for tracking progress
    print(f"inv_outer_hex_side_length: {1.0 / outer_hex_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {end_time - start_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
