# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from numba import njit

# Constants
HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180

@njit
def hexagon_vertices_numba(center_x, center_y, rotation):
    """Create vertices of a unit regular hexagon - JIT compiled"""
    vertices = np.empty((6, 2))
    angle_step = np.pi / 3
    for i in range(6):
        angle = rotation + i * angle_step
        x = center_x + HEXAGON_RADIUS * np.cos(angle)
        y = center_y + HEXAGON_RADIUS * np.sin(angle)
        vertices[i] = (x, y)
    return vertices

@njit
def distance_point_to_point_numba(x1, y1, x2, y2):
    """Fast Euclidean distance calculation"""
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)

@njit
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

@njit
def fast_hexagon_overlap_check_numba(vertices1, vertices2):
    """Fast overlap check using bounding circle approximation"""
    # Get bounding circles
    cx1, cy1 = np.mean(vertices1[:, 0]), np.mean(vertices1[:, 1])
    cx2, cy2 = np.mean(vertices2[:, 0]), np.mean(vertices2[:, 1])
    
    # Calculate max distances from centers to vertices (approximate radius)
    max_dist1 = 0
    max_dist2 = 0
    for i in range(6):
        d1 = distance_point_to_point_numba(cx1, cy1, vertices1[i, 0], vertices1[i, 1])
        d2 = distance_point_to_point_numba(cx2, cy2, vertices2[i, 0], vertices2[i, 1])
        max_dist1 = max(max_dist1, d1)
        max_dist2 = max(max_dist2, d2)
    
    # Fast circle-circle overlap test
    dist_centers = distance_point_to_point_numba(cx1, cy1, cx2, cy2)
    if dist_centers > (max_dist1 + max_dist2):
        return False
    
    # If circles overlap, do exact check
    # Simplified: check if any vertex of one hexagon is inside other
    for i in range(6):
        if point_in_hexagon_numba(vertices1[i, 0], vertices1[i, 1], vertices2):
            return True
    for i in range(6):
        if point_in_hexagon_numba(vertices2[i, 0], vertices2[i, 1], vertices1):
            return True
    return False

@njit
def calculate_outer_radius_numba(inner_hex_data):
    """Calculate minimum outer hexagon radius required to contain all inner hexagons - JIT compiled"""
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        center_x, center_y, _ = inner_hex_data[i]
        dist_to_center = np.sqrt(center_x**2 + center_y**2)
        # Hexagon diagonal is sqrt(3) * radius
        hex_diag = HEXAGON_RADIUS * np.sqrt(3)
        max_dist = max(max_dist, dist_to_center + hex_diag)
    return max_dist

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

def fast_containment_check(vertices, outer_vertices):
    """Fast containment check using bounding circle"""
    # Center of hexagon
    cx = np.mean(vertices[:, 0])
    cy = np.mean(vertices[:, 1])
    
    # Bounding radius (distance from center to furthest vertex)
    max_dist = 0
    for i in range(6):
        dist = distance_point_to_point_numba(cx, cy, vertices[i, 0], vertices[i, 1])
        max_dist = max(max_dist, dist)
    
    # Check if center is inside outer hexagon
    outer_center = np.mean(outer_vertices[:, 0]), np.mean(outer_vertices[:, 1])
    dist_center_to_outer = distance_point_to_point_numba(cx, cy, outer_center[0], outer_center[1])
    
    # If center is too far or if any vertex is outside, return False
    if dist_center_to_outer + max_dist > 15.0:  # rough outer hexagon radius
        return False
    
    # More precise check - if center is inside, check if all vertices are inside
    for i in range(6):
        if not point_in_hexagon_numba(vertices[i, 0], vertices[i, 1], outer_vertices):
            return False
    return True

def evaluate_solution(params):
    """
    Optimized evaluation with fast pre-checks and smart early termination
    params: array of shape (12,) = [r1, theta1, r2, theta2, ..., r6, theta6] 
            where r_i, theta_i are radial and angular parameters for 6 unique positions
            and we mirror these for the full 12 hexagons
    """
    # Parameter mapping: 6 unique positions (radial, angular), then mirrored
    # This reduces parameters from 36 to 12 while preserving symmetry
    n_positions = 6
    inner_params = np.zeros((12, 3))
    
    # Fill first 6 positions (0-5)
    for i in range(n_positions):
        inner_params[i][0] = params[2*i]  # radius
        inner_params[i][1] = params[2*i+1]  # angle (in radians)
        inner_params[i][2] = 0  # rotation (fixed for now)
    
    # Mirror positions for the 2nd ring (6-11)
    for i in range(n_positions):
        # Mirror radially and add pi to angle (180 degree rotation)
        inner_params[i+6][0] = params[2*i]  # same radius
        inner_params[i+6][1] = params[2*i+1] + np.pi  # opposite angle
        inner_params[i+6][2] = 0  # rotation
    
    # Convert radial/angular to cartesian coordinates
    for i in range(12):
        r = inner_params[i][0]
        theta = inner_params[i][1]
        inner_params[i][0] = r * np.cos(theta)  # x
        inner_params[i][1] = r * np.sin(theta)  # y
    
    # Pre-check: early rejection using bounding circle
    max_dist = 0
    for i in range(12):
        x, y, _ = inner_params[i]
        dist = np.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist)
    
    # Conservative estimate of outer hexagon needed
    outer_radius_estimate = max_dist + 2.0  # Add some margin for hexagon size
    
    # Fast bounding circle rejection
    if outer_radius_estimate > 15.0:
        return 1e9  # Too large, reject quickly
    
    # Create hexagon objects and check overlaps
    hexagon_objects = []
    for i in range(12):
        center = (inner_params[i][0], inner_params[i][1])
        angle = np.radians(inner_params[i][2])
        hexagon = create_unit_hexagon(center, angle)
        hexagon_objects.append(hexagon)
    
    # Calculate outer hexagon radius
    outer_radius = calculate_outer_radius_numba(inner_params)
    
    # Create outer hexagon polygon (centered at origin)
    outer_hexagon = create_unit_hexagon((0, 0), 0)
    
    # Check containment for all inner hexagons - early termination
    containment_violations = 0
    outer_vertices = hexagon_vertices_numba(0, 0, 0)
    
    for i in range(12):
        hex_poly = hexagon_objects[i]
        hex_vertices = hexagon_vertices_numba(inner_params[i][0], inner_params[i][1], np.radians(inner_params[i][2]))
        
        # Fast containment check
        if not fast_containment_check(hex_vertices, outer_vertices):
            containment_violations += 1
            break  # Early exit
    
    # Check overlap between all pairs
    overlap_violations = 0
    
    # Use fast numba-based overlap checking for the most likely overlapping pairs
    for i in range(12):
        for j in range(i+1, 12):
            if abs(inner_params[i][0] - inner_params[j][0]) > 3.0 or \
               abs(inner_params[i][1] - inner_params[j][1]) > 3.0:
                continue  # Not close enough to potentially overlap
            
            # Use fast numba-based overlap checking
            vertices1 = hexagon_vertices_numba(inner_params[i][0], inner_params[i][1], np.radians(inner_params[i][2]))
            vertices2 = hexagon_vertices_numba(inner_params[j][0], inner_params[j][1], np.radians(inner_params[j][2]))
            
            if fast_hexagon_overlap_check_numba(vertices1, vertices2):
                overlap_violations += 1
                break  # Early exit if any overlap found
        if overlap_violations > 0:
            break  # Early exit if any overlap found
    
    # Adaptive penalty scaling based on constraint violations
    base_penalty = 1000
    containment_penalty = base_penalty * containment_violations
    overlap_penalty = base_penalty * overlap_violations
    
    penalty = containment_penalty + overlap_penalty
    
    # Objective: minimize negative of 1/outer_radius + penalty
    if containment_violations > 0 or overlap_violations > 0:
        # Very large penalty if constraints violated
        return 1000000 + penalty
    else:
        # Return negative of 1/R (we want to maximize 1/R, so minimize -1/R)
        return -(1.0 / (outer_radius * 1.05)) + penalty

def generate_improved_initial_placement():
    """Generate an improved initial configuration based on mathematical insights"""
    # This uses a 3-layer hexagonal arrangement with strategic placements
    
    # Layer 1: central hexagon
    positions = [[0, 0, 0]]
    
    # Layer 2: 6 hexagons in first ring
    # Place them at distance of approx 2*radius from center
    for i in range(6):
        angle = i * 60.0  # degrees
        dist = 2.0  # This should allow good packing
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        positions.append([x, y, 0])
    
    # Layer 3: 6 hexagons in second ring, offset to create density
    # These go further out and offset for better packing
    for i in range(6):
        angle = i * 60.0 + 30.0  # offset by 30 degrees
        dist = 3.5  # further out
        x = dist * np.cos(np.radians(angle))
        y = dist * np.sin(np.radians(angle))
        positions.append([x, y, 0])
    
    # Adjust positions for better packing and reduced overlap risk
    positions[1][0] *= 0.95  # Slightly pull inner ring hexagons inward
    positions[1][1] *= 0.95
    positions[2][0] *= 0.95
    positions[2][1] *= 0.95
    positions[3][0] *= 0.95
    positions[3][1] *= 0.95
    
    # Final tuning to reduce overlap risk
    positions[7][0] -= 0.15  # Adjust top-right
    positions[7][1] += 0.15
    positions[8][0] += 0.15  # Adjust top-left
    positions[8][1] += 0.15
    positions[9][0] -= 0.15  # Adjust bottom-right
    positions[9][1] -= 0.15
    positions[10][0] += 0.15  # Adjust bottom-left
    positions[10][1] -= 0.15
    
    # Convert to polar coordinates for the new parameter scheme (r, theta pairs)
    polar_coords = []
    for i in range(6):  # First ring positions
        x, y, _ = positions[i+1]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        polar_coords.extend([r, theta])
    
    # Add second ring positions (also convert to polar)
    for i in range(6):  # Second ring positions
        x, y, _ = positions[i+7]
        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)
        polar_coords.extend([r, theta])
        
    return np.array(polar_coords)

def optimize_hexagon_arrangement():
    """
    Multi-stage optimization with smart initial placement and targeted parameter tuning
    """
    # Generate and use improved initial placement
    initial_guess = generate_improved_initial_placement()
    
    # Define bounds for each parameter
    bounds = []
    # Radius bounds (positive values) and angle bounds  
    for i in range(6):
        bounds.extend([(0.5, 5.0), (-np.pi, np.pi)])  # r, theta pairs
    for i in range(6):
        bounds.extend([(0.5, 5.0), (-np.pi, np.pi)])  # r, theta pairs for second ring
    
    # Phase 1: Global search with reduced precision for speed
    try:
        result1 = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=25,
            popsize=10,
            seed=42,
            disp=False,
            polish=False
        )
        
        # Phase 2: Refinement with higher precision polish
        result2 = differential_evolution(
            evaluate_solution,
            bounds,
            maxiter=20,
            popsize=8,
            seed=43,
            disp=False,
            polish=True
        )
        
        # Use the better result
        if result1.fun < result2.fun:
            optimized_params = result1.x
        else:
            optimized_params = result2.x
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        optimized_params = initial_guess

    # Phase 3: Local refinement using L-BFGS-B
    try:
        result_final = minimize(
            evaluate_solution,
            optimized_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 15, 'ftol': 1e-6},
            callback=None
        )
        
        if result_final.success:
            optimized_params = result_final.x
    except Exception as e:
        print(f"Final optimization failed: {e}")

    # Convert final parameters back to standard format
    # Create final 12 hexagon parameters (this version builds the full structure)
    inner_params = np.zeros((12, 3))
    
    # Fill first 6 positions (0-5)
    for i in range(6):
        inner_params[i][0] = optimized_params[2*i]  # radius
        inner_params[i][1] = optimized_params[2*i+1]  # angle (in radians)
        inner_params[i][2] = 0  # rotation
    
    # Mirror positions for the 2nd ring (6-11)
    for i in range(6):
        # Mirror radially and add pi to angle (180 degree rotation)
        inner_params[i+6][0] = optimized_params[2*i]  # same radius
        inner_params[i+6][1] = optimized_params[2*i+1] + np.pi  # opposite angle
        inner_params[i+6][2] = 0  # rotation
    
    # Convert radial/angular to cartesian coordinates
    for i in range(12):
        r = inner_params[i][0]
        theta = inner_params[i][1]
        inner_params[i][0] = r * np.cos(theta)  # x
        inner_params[i][1] = r * np.sin(theta)  # y
    
    # Build final data structure
    inner_hex_data = inner_params.copy()
    
    # Calculate final outer hexagon side length
    outer_radius = calculate_outer_radius_numba(inner_params)
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
    
    # Initialize optimization engine
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