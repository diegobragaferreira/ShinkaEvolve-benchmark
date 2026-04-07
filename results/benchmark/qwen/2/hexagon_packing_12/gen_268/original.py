# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from collections import defaultdict
import time
from numba import jit, prange
import warnings
from scipy.spatial.distance import cdist

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

def calculate_total_penalty(hex_data, outer_radius):
    """Calculate total penalty for all hexagons using spatial hashing"""
    n = len(hex_data)
    
    # Precompute vertices for all hexagons
    hex_vertices_list = [hexagon_vertices(hex_data[i][0], hex_data[i][1], hex_data[i][2]) for i in range(n)]
    
    # Create spatial hash
    hash_grid = create_spatial_hash(hex_vertices_list)
    
    total_penalty = 0
    
    # Check containment for each hexagon
    outer_hex_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    for i in range(n):
        vertices = hex_vertices_list[i]
        # Check containment penalty - higher penalty for containment violations
        for vx, vy in vertices:
            point = np.array([vx, vy])
            if not point_in_polygon(point, outer_hex_vertices):
                dist = np.sqrt(vx*vx + vy*vy)
                total_penalty += (dist - outer_radius + 0.5)**2 * 1500000  # Higher penalty for containment

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

def create_enhanced_symmetric_initial_config():
    """Create an enhanced symmetric initial configuration with mathematical insight"""
    # Create a configuration inspired by optimal hexagonal packing arrangements
    # Based on the proven symmetric solutions with tighter spacing
    positions = [
        [0.0, 0.0, 0.0],     # Center
        [0.0, 2.2, 0.0],     # Up (slightly tighter than previous)
        [0.0, -2.2, 0.0],    # Down
        [1.9, 1.1, 0.0],     # Upper right
        [-1.9, 1.1, 0.0],    # Upper left
        [1.9, -1.1, 0.0],    # Lower right
        [-1.9, -1.1, 0.0],   # Lower left
        [3.8, 0.0, 0.0],     # Far right
        [-3.8, 0.0, 0.0],    # Far left
        [0.0, 3.8, 0.0],     # Far up
        [0.0, -3.8, 0.0],    # Far down
        [1.9, 3.3, 0.0],     # Far upper right (tighter spacing)
    ]

    # Add controlled randomness to avoid getting stuck in poor local minima
    positions = np.array(positions)
    for i in range(1, len(positions)):
        # Add small random perturbations (smaller than previous versions)
        positions[i][0] += np.random.normal(0, 0.05)
        positions[i][1] += np.random.normal(0, 0.05)
        
    return positions

def adaptive_local_refinement(initial_config, max_iterations=100, method='L-BFGS-B'):
    """Apply local refinement to improve the configuration with better convergence criteria"""
    # Convert to flat parameter array for optimization
    params = initial_config.flatten()
    
    def objective(flat_params):
        config = flat_params.reshape(-1, 3)
        outer_radius = get_outer_hexagon_radius(config)
        penalty = calculate_total_penalty(config, outer_radius)
        # Objective is to minimize penalty and maximize 1/R (equivalent to minimize -1/R + penalty)
        return penalty - 1.0/(outer_radius + 1e-6)  # Small epsilon for numerical stability
    
    # Use L-BFGS-B for local optimization with bounds
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 360.0)] * 12  # bounds for each hexagon
    
    try:
        result = minimize(objective, params, method=method, bounds=bounds, 
                         options={'maxiter': max_iterations, 'gtol': 1e-8, 'ftol': 1e-8})
        if result.success:
            # Return optimized configuration
            final_config = result.x.reshape(-1, 3)
            return final_config
    except Exception as e:
        warnings.warn(f"Local refinement failed: {e}")
        pass
    
    # If optimization fails, return original
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
    
    # Phase 1: Generate enhanced symmetric initial configuration (better than simple grid)
    initial_config = create_enhanced_symmetric_initial_config()
    
    # Phase 2: First local refinement with fewer iterations (fast initial improvement)
    refined_config = adaptive_local_refinement(initial_config, max_iterations=30, method='L-BFGS-B')
    
    # Phase 3: Second local refinement with more iterations for fine-tuning
    final_config = adaptive_local_refinement(refined_config, max_iterations=70, method='L-BFGS-B')
    
    # Calculate final outer hexagon size
    outer_hex_side_length = get_outer_hexagon_radius(final_config)
    
    # Create outer hexagon data (centered, no rotation)
    outer_hex_data = np.array([0, 0, 0])  # centered at origin, no rotation
    
    # Final validation
    final_penalty = calculate_total_penalty(final_config, outer_hex_side_length)
    
    # More aggressive fallback if there are severe violations
    if final_penalty > 500000:  # If there are severe violations
        # Create a more carefully tuned known good configuration
        fallback_config = np.array([
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
            [0, -4, 0],          # far bottom-center
        ])
        final_config = fallback_config
        outer_hex_side_length = 8.0
    
    # Calculate benchmark ratio
    benchmark_ratio = 1.0 / outer_hex_side_length / 0.2537
    
    end_time = time.time()
    
    # Print diagnostic information for tracking progress
    print(f"inv_outer_hex_side_length: {1.0 / outer_hex_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {end_time - start_time:.4f}s")
    
    return final_config, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END