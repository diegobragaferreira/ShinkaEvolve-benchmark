# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# Constants
HEXAGON_RADIUS = 1.0
HEXAGON_SIDE_LENGTH = 1.0

@jit(nopython=True)
def hexagon_vertices_numba(center_x, center_y, rotation_deg):
    """Compute vertices of a unit regular hexagon - JIT compiled"""
    vertices = np.empty((6, 2))
    angle_step = np.pi / 3
    rotation_rad = np.radians(rotation_deg)
    for i in range(6):
        angle = rotation_rad + i * angle_step
        x = center_x + HEXAGON_RADIUS * np.cos(angle)
        y = center_y + HEXAGON_RADIUS * np.sin(angle)
        vertices[i] = (x, y)
    return vertices

@jit(nopython=True)
def point_in_hexagon_numba(px, py, vertices):
    """Check if point is inside hexagon using ray casting - JIT compiled"""
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

@jit(nopython=True)
def distance_point_to_point_numba(px, py, qx, qy):
    """Euclidean distance between two points - JIT compiled"""
    dx = px - qx
    dy = py - qy
    return np.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def distance_point_to_hexagon_edge_numba(px, py, vertices):
    """Minimum distance from point to hexagon edges - JIT compiled"""
    min_dist = 1e10
    n = len(vertices)
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        
        # Compute distance from point to line segment
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:
            param = dot / len_sq
        
        if param < 0:
            xx = x1
            yy = y1
        elif param > 1:
            xx = x2
            yy = y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        dx = px - xx
        dy = py - yy
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = min(min_dist, dist)
    return min_dist

def create_unit_hexagon(center=(0, 0), rotation=0):
    """Create a unit regular hexagon as a Shapely polygon"""
    angle_step = np.pi / 3
    points = []
    for i in range(6):
        angle = rotation + i * angle_step
        x = center[0] + HEXAGON_RADIUS * np.cos(angle)
        y = center[1] + HEXAGON_RADIUS * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if a hexagon is fully contained within outer hexagon"""
    return outer_hexagon.contains(hexagon)

def check_overlap_fast(hex1, hex2):
    """Fast overlap check using spatial indexing - early termination"""
    # Quick bounding box check first
    if not hex1.bounds[2] < hex2.bounds[0] and \
       not hex1.bounds[0] > hex2.bounds[2] and \
       not hex1.bounds[3] < hex2.bounds[1] and \
       not hex1.bounds[1] > hex2.bounds[3]:
        # If bounding boxes intersect, do detailed check
        return hex1.intersects(hex2)
    return False

@jit(nopython=True)
def calculate_outer_radius_from_positions_numba(inner_params):
    """Calculate minimum outer hexagon radius required to contain all inner hexagons - JIT compiled"""
    max_dist = 0.0
    for i in range(len(inner_params)):
        center_x, center_y, _ = inner_params[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        # Hexagon diagonal is sqrt(3) * radius
        hex_diag = HEXAGON_RADIUS * np.sqrt(3)
        max_dist = max(max_dist, dist_to_center + hex_diag)
    return max_dist

@jit(nopython=True)
def validate_positions_numba(inner_params, outer_radius):
    """Validate that positions are within bounds and don't overlap - JIT compiled"""
    # Check containment
    for i in range(len(inner_params)):
        center_x, center_y, _ = inner_params[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        if dist_to_center + HEXAGON_RADIUS * np.sqrt(3) > outer_radius:
            return False, 0.0
    
    # Check overlaps with simple distance check
    for i in range(len(inner_params)):
        for j in range(i+1, len(inner_params)):
            x1, y1, _ = inner_params[i]
            x2, y2, _ = inner_params[j]
            dist = distance_point_to_point_numba(x1, y1, x2, y2)
            if dist < 2.0:  # Overlap threshold
                return False, 0.0
    
    return True, 0.0

def calculate_outer_hexagon_radius(inner_hex_data):
    """Calculate minimum outer hexagon radius required to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        # Hexagon diagonal is sqrt(3) * radius
        hex_diag = HEXAGON_RADIUS * np.sqrt(3)
        max_dist = max(max_dist, dist_to_center + hex_diag)
    return max_dist

def evaluate_solution(params):
    """
    Optimized evaluation using symmetry and analytical geometry - JIT compiled portions
    params: array of shape (7,) = [r1, r2, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, outer_radius]
            where r1,r2 are radii of rings, alphas are angles for first ring, outer_radius is the outer hexagon radius
    """
    # Reduce parameters: r1, r2, 6 angles for first ring, 6 angles for second ring, outer_radius = 12 + 1 = 13
    # But we'll use symmetry to reduce to 7 parameters: r1, r2, 6 angles for first ring, outer_radius
    # The second ring will be symmetrically derived
    
    # This assumes 6 positions at first ring, 6 positions at second ring with symmetry
    # We'll use simpler approach with just 7 parameters for 12 positions total
    
    # Parameter mapping: 
    # r1, r2, 6 angles for first ring, 6 angles for second ring (same as first but shifted), outer_radius
    # Simplification: use 6 angles for first ring, 6 derived from symmetry, r1 and r2 parameters
    
    # For optimal symmetry, let's work with just 6 independent parameters:
    # r1 (radius of first ring), r2 (radius of second ring), 6 angles for first ring
    # Second ring is symmetrically determined
    
    r1 = params[0]  # First ring radius
    r2 = params[1]  # Second ring radius  
    angles1 = params[2:8]  # 6 angles for first ring
    
    # Generate 12 positions with symmetry
    inner_params = np.zeros((12, 3))
    
    # Center position (index 0)
    inner_params[0] = [0.0, 0.0, 0.0]
    
    # First ring (indices 1-6)
    for i in range(6):
        angle = angles1[i]
        inner_params[i+1] = [r1 * np.cos(np.radians(angle)), r1 * np.sin(np.radians(angle)), 0.0]
    
    # Second ring (indices 7-12) - offset by 30 degrees
    for i in range(6):
        angle = angles1[i] + 30.0  # Offset by 30 degrees for hexagonal tiling
        inner_params[i+7] = [r2 * np.cos(np.radians(angle)), r2 * np.sin(np.radians(angle)), 0.0]
    
    # Calculate outer radius
    outer_radius = calculate_outer_radius_from_positions_numba(inner_params)
    
    # Validate positions
    valid, _ = validate_positions_numba(inner_params, outer_radius)
    
    # Penalty for constraint violations  
    if not valid:
        return 100000  # Large penalty for constraint violations
    
    # Return negative of 1/outer_radius (we want to maximize 1/outer_radius)
    return -(1.0 / outer_radius)

def generate_symmetric_initial_guess():
    """Generate highly symmetric initial configuration based on mathematically optimal arrangements"""
    # Based on mathematical analysis of tight hexagonal packings
    # Central hexagon + 6 surrounding in first ring + 6 in second ring
    
    # Parameters for optimal symmetric arrangement
    r1 = 1.732  # sqrt(3) - distance that maximizes packing density
    r2 = 3.464  # 2*sqrt(3) - second ring distance
    
    # Angles for first ring - evenly distributed
    angles1 = np.linspace(0, 360, 6, endpoint=False)
    
    # Create parameters
    params = [r1, r2] + angles1.tolist()
    
    # Add small random perturbations to escape local minima
    params = np.array(params) + np.random.normal(0, 0.05, len(params))
    
    return np.array(params)

def optimize_hexagon_arrangement():
    """
    Advanced symmetric optimization with mathematically informed parameterization
    """
    # Phase 1: Global search with symmetry-aware parameterization
    bounds = [
        (1.0, 4.0),      # r1 - first ring radius
        (2.0, 6.0),      # r2 - second ring radius
    ]
    
    # Add 6 angle bounds for first ring (0 to 360 degrees)
    for i in range(6):
        bounds.append((0, 360))
    
    # Initial guess with mathematical symmetry
    initial_guess = generate_symmetric_initial_guess()
    
    # Global optimization with less iterations due to time constraints
    try:
        result1 = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=30,       # Reduced iterations to stay within time limits
            popsize=10,       # Smaller population
            seed=42,
            disp=False,
            polish=False
        )
        
        # Phase 2: Refinement with higher quality method
        if result1.success:
            optimized_params = result1.x
        else:
            optimized_params = initial_guess
            
    except Exception as e:
        optimized_params = initial_guess

    # Phase 3: Final refinement with L-BFGS-B
    try:
        result_final = minimize(
            evaluate_solution,
            optimized_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20},  # Reduced iterations
            callback=None
        )
        
        if result_final.success:
            optimized_params = result_final.x
    except Exception as e:
        pass

    # Extract final configuration
    r1 = optimized_params[0]
    r2 = optimized_params[1]
    angles1 = optimized_params[2:8]
    
    # Generate final 12 hexagon positions with symmetry
    inner_hex_data = np.zeros((12, 3))
    
    # Center position
    inner_hex_data[0] = [0.0, 0.0, 0.0]
    
    # First ring
    for i in range(6):
        angle = angles1[i]
        inner_hex_data[i+1] = [r1 * np.cos(np.radians(angle)), r1 * np.sin(np.radians(angle)), 0.0]
    
    # Second ring - offset by 30 degrees for hexagonal tiling
    for i in range(6):
        angle = angles1[i] + 30.0
        inner_hex_data[i+7] = [r2 * np.cos(np.radians(angle)), r2 * np.sin(np.radians(angle)), 0.0]
    
    # Calculate final outer hexagon side length
    outer_radius = calculate_outer_radius_from_positions_numba(inner_hex_data)
    outer_hex_side_length = outer_radius * np.sqrt(3)  # Convert from radius to side length

    # Outer hexagon centered at origin with 0 rotation
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Use optimized symmetric approach
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_arrangement()

    end_time = time.time()
    eval_time = end_time - start_time

    # Debug output
    inv_outer_side_length = 1.0 / outer_hex_side_length
    benchmark_ratio = inv_outer_side_length / 0.2537

    print(f"Eval time: {eval_time:.4f}s")
    print(f"Inv outer side length: {inv_outer_side_length:.6f}")
    print(f"Benchmark ratio: {benchmark_ratio:.6f}")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END