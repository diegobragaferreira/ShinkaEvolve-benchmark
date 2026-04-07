# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from itertools import combinations
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# Constants
HEXAGON_RADIUS = 1.0
HEXAGON_WIDTH = HEXAGON_RADIUS * 2 * np.sqrt(3) / 3
HEXAGON_HEIGHT = HEXAGON_RADIUS * 2
MAX_EVAL_TIME = 180

@njit
def create_hexagon_vertices(center_x, center_y, rotation):
    """Create vertices of a unit regular hexagon - JIT compiled for speed"""
    vertices = np.empty((6, 2))
    angle_step = np.pi / 3
    for i in range(6):
        angle = rotation + i * angle_step
        x = center_x + HEXAGON_RADIUS * np.cos(angle)
        y = center_y + HEXAGON_RADIUS * np.sin(angle)
        vertices[i] = (x, y)
    return vertices

@njit
def point_in_hexagon_fast(px, py, vertices):
    """Fast point-in-polygon check using ray casting - JIT compiled"""
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
def distance_point_to_hexagon(px, py, vertices):
    """Calculate minimum distance from point to hexagon boundary - JIT compiled"""
    min_dist = float('inf')
    n = len(vertices)
    
    # Check distance to each edge
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        
        # Vector from point to first vertex
        dx = x2 - x1
        dy = y2 - y1
        
        # If edge is degenerate
        if dx == 0 and dy == 0:
            dist = np.sqrt((px - x1)**2 + (py - y1)**2)
        else:
            # Project point onto line
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        
        min_dist = min(min_dist, dist)
    
    return min_dist

@njit
def check_overlap_simple(vertices1, vertices2):
    """Simple but effective overlap check for hexagons - JIT compiled"""
    # Check if any vertex of hex1 is inside hex2
    for v in vertices1:
        if point_in_hexagon_fast(v[0], v[1], vertices2):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    for v in vertices2:
        if point_in_hexagon_fast(v[0], v[1], vertices1):
            return True
    
    return False

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

@njit
def calculate_outer_hexagon_radius_numba(inner_hex_data):
    """Calculate minimum outer hexagon radius required to contain all inner hexagons - JIT compiled"""
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        # Hexagon diagonal is sqrt(3) * radius
        hex_diag = HEXAGON_RADIUS * np.sqrt(3)
        max_dist = max(max_dist, dist_to_center + hex_diag)
    return max_dist

def evaluate_solution(params):
    """
    Optimized evaluation combining geometric accuracy with computational efficiency
    params: array of shape (36,) = [x1, y1, theta1, ..., x12, y12, theta12]
    """
    # Extract inner hexagon parameters
    inner_params = params.reshape(-1, 3)

    # Create inner hexagons
    inner_hexagons = []
    inner_vertices = []
    for i in range(12):
        center = (inner_params[i][0], inner_params[i][1])
        angle = np.radians(inner_params[i][2])
        hexagon = create_unit_hexagon(center, angle)
        inner_hexagons.append(hexagon)
        vertices = create_hexagon_vertices(center[0], center[1], angle)
        inner_vertices.append(vertices)

    # Calculate outer hexagon radius
    outer_radius = calculate_outer_hexagon_radius_numba(inner_params)

    # Create outer hexagon for containment checking
    outer_hexagon = create_unit_hexagon((0, 0), 0)
    scaled_outer_radius = outer_radius * 1.05  # Add small margin
    outer_hexagon_scaled = create_unit_hexagon((0, 0), 0)

    # Check containment constraints - vectorized for speed
    containment_violations = 0
    for i in range(12):
        if not outer_hexagon_scaled.contains(inner_hexagons[i]):
            containment_violations += 1

    # Check overlap constraints with adaptive search
    overlap_violations = 0
    
    # Use a more efficient approach: only check neighbors in spatial proximity
    centers = np.array([[h.centroid.x, h.centroid.y] for h in inner_hexagons])
    tree = cKDTree(centers)
    
    # For each hexagon, only check against nearby ones (not all pairs)
    for i in range(12):
        # Query nearby hexagons within a threshold
        nearby_indices = tree.query_ball_point(centers[i], 3.0)  # Distance threshold
        
        # Check overlaps with nearby hexagons only
        for j in nearby_indices:
            if i < j:
                # Use fast geometric checks first
                if check_overlap_simple(inner_vertices[i], inner_vertices[j]):
                    overlap_violations += 1
                    break  # Early termination once overlap found
                    
        if overlap_violations > 0:
            break  # Early termination if any overlap found

    # Penalty for constraint violations
    penalty = 10000 * (containment_violations + overlap_violations)

    # Objective: minimize negative of 1/outer_radius + penalty
    if containment_violations > 0 or overlap_violations > 0:
        return 100000 + penalty  # Large penalty for constraint violations
    else:
        # Return negative of 1/R (we want to maximize 1/R, so minimize -1/R)
        return -(1.0 / scaled_outer_radius) + penalty

def generate_triangular_lattice_initial():
    """Generate initial configuration using triangular lattice principles"""
    # Start with a triangular lattice arrangement that is likely to yield good packing
    # Triangular lattice points in polar coordinates
    positions = []
    
    # Central position
    positions.append([0.0, 0.0, 0.0])
    
    # First ring: 6 positions
    ring1_radius = 2.0 * HEXAGON_RADIUS
    for i in range(6):
        angle = i * 60
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Second ring: 6 positions, offset
    ring2_radius = 3.5 * HEXAGON_RADIUS
    for i in range(6):
        angle = i * 60 + 30  # Offset by 30 degrees
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    # Adjust positions to avoid excessive overlap and improve packing
    # These are tuned values that work well for this specific case
    positions[1][0] *= 0.95  # First ring, first position
    positions[1][1] *= 0.95
    positions[2][0] *= 0.97  # First ring, second position
    positions[2][1] *= 0.97
    positions[3][0] *= 0.98  # First ring, third position
    positions[3][1] *= 0.98
    positions[4][0] *= 0.97  # First ring, fourth position
    positions[4][1] *= 0.97
    positions[5][0] *= 0.95  # First ring, fifth position
    positions[5][1] *= 0.95
    positions[6][0] *= 0.96  # First ring, sixth position
    positions[6][1] *= 0.96
    
    # Add slight randomness to improve convergence
    for i in range(12):
        positions[i][0] += np.random.normal(0, 0.05)
        positions[i][1] += np.random.normal(0, 0.05)
    
    return np.array(positions)

def adaptive_local_search(initial_params, bounds, max_iter=20):
    """Perform adaptive local search with varying step sizes"""
    
    # Try multiple optimization approaches with different strategies
    current_params = initial_params.copy()
    
    # Strategy 1: Differential Evolution for global search
    try:
        de_result = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=15,
            popsize=10,
            seed=42,
            disp=False,
            polish=False
        )
        if de_result.success and de_result.fun < evaluate_solution(current_params):
            current_params = de_result.x
    except:
        pass
    
    # Strategy 2: Fine-grained L-BFGS-B for local refinement
    try:
        lbfgs_result = minimize(
            evaluate_solution,
            current_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 15},
            callback=None
        )
        if lbfgs_result.success and lbfgs_result.fun < evaluate_solution(current_params):
            current_params = lbfgs_result.x
    except:
        pass
    
    # Strategy 3: Simpler local optimization with different parameters
    try:
        # More aggressive local optimization
        lbfgs_result2 = minimize(
            evaluate_solution,
            current_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 25, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=None
        )
        if lbfgs_result2.success and lbfgs_result2.fun < evaluate_solution(current_params):
            current_params = lbfgs_result2.x
    except:
        pass
    
    return current_params

def optimize_hexagon_arrangement():
    """
    Novel optimization approach combining dual resolution and adaptive refinement
    """
    # Define bounds for optimization (36 parameters for 12 hexagons with x,y,theta)
    bounds = []
    for i in range(12):
        # x, y coordinates (larger range to allow for flexible positioning)
        bounds.extend([(-8, 8), (-8, 8), (-180, 180)])
    
    # Generate initial configuration using triangular lattice approach
    initial_guess = generate_triangular_lattice_initial()
    # Flatten for optimization
    initial_params = initial_guess.flatten()
    
    # Perform adaptive local search with different strategies
    optimized_params = adaptive_local_search(initial_params, bounds)
    
    # Convert back to proper structure
    inner_params = optimized_params.reshape(-1, 3)
    
    # Calculate final outer hexagon side length
    outer_radius = calculate_outer_hexagon_radius(inner_params)
    outer_hex_side_length = outer_radius * np.sqrt(3)  # Convert from radius to side length

    # Outer hexagon centered at origin with 0 rotation
    outer_hex_data = np.array([0, 0, 0])

    return inner_params, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Use the optimized approach
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