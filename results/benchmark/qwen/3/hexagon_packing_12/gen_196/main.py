# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from scipy.spatial.distance import cdist
import random
from numba import jit, prange

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of segment
    length_sq = dx*dx + dy*dy
    
    # If segment is actually a point
    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp t to [0,1]
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def hexagon_distance(h1_center_x, h1_center_y, h1_angle, h2_center_x, h2_center_y, h2_angle):
    """Fast approximation of minimum distance between hexagons"""
    # Get vertices of both hexagons
    v1 = get_hexagon_vertices(h1_center_x, h1_center_y, h1_angle)
    v2 = get_hexagon_vertices(h2_center_x, h2_center_y, h2_angle)
    
    min_dist = 1e10
    # Check distance from each vertex of hexagon 1 to edges of hexagon 2
    for i in range(6):
        p1 = v1[i]
        for j in range(6):
            p2_start = v2[j]
            p2_end = v2[(j+1)%6]
            dist = distance_point_to_segment(p1[0], p1[1], p2_start[0], p2_start[1], p2_end[0], p2_end[1])
            min_dist = min(min_dist, dist)
    
    # Also check reverse direction
    for i in range(6):
        p1 = v2[i]
        for j in range(6):
            p2_start = v1[j]
            p2_end = v1[(j+1)%6]
            dist = distance_point_to_segment(p1[0], p1[1], p2_start[0], p2_start[1], p2_end[0], p2_end[1])
            min_dist = min(min_dist, dist)
    
    return min_dist

def create_regular_hexagon(center=(0,0), side_length=1, rotation=0):
    """Create a regular hexagon as a Shapely polygon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    x = center[0] + side_length * np.cos(angles)
    y = center[1] + side_length * np.sin(angles)
    return Polygon(list(zip(x, y)))

def hexagon_to_polygon(x, y, angle_deg, radius=1.0):
    """Convert hexagon parameters to shapely polygon"""
    vertices = get_hexagon_vertices(x, y, angle_deg, radius)
    return Polygon(vertices)

def check_overlap_fast(hex1_poly, hex2_poly):
    """Fast overlap check using Shapely with buffer for numerical stability"""
    return hex1_poly.buffer(1e-10).intersects(hex2_poly.buffer(1e-10)) and not hex1_poly.touches(hex2_poly)

def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon"""
    return outer_hex.contains(inner_hex) or outer_hex.covers(inner_hex)

def compute_outer_hexagon_radius(inner_hex_data):
    """Compute minimum outer hexagon radius that contains all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 0.0

    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        all_vertices.extend(vertices)

    if len(all_vertices) == 0:
        return 0.0

    # Compute centroid
    centroid_x = np.mean([v[0] for v in all_vertices])
    centroid_y = np.mean([v[1] for v in all_vertices])

    # Find maximum distance from centroid to any vertex
    max_distance = 0.0
    for x, y in all_vertices:
        distance = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        max_distance = max(max_distance, distance)

    # Add buffer for hexagon radius calculation
    return max_distance + UNIT_HEX_RADIUS + 1e-10

def evaluate_constraint_violations(inner_hex_data, outer_hex_data):
    """Evaluate constraint violations for a given configuration"""
    violations = []
    
    # Create outer hexagon
    outer_x, outer_y, outer_angle = outer_hex_data
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)
    outer_hex = hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)
    
    # Check each inner hexagon for containment
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        inner_hex = hexagon_to_polygon(x, y, angle)
        
        if not check_containment(inner_hex, outer_hex):
            violations.append(f"Inner hexagon {i} not contained")
    
    # Check overlaps between all pairs
    for i in range(len(inner_hex_data)):
        x1, y1, angle1 = inner_hex_data[i]
        hex1_poly = hexagon_to_polygon(x1, y1, angle1)
        
        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            hex2_poly = hexagon_to_polygon(x2, y2, angle2)
            
            if check_overlap_fast(hex1_poly, hex2_poly):
                violations.append(f"Overlapping hexagons {i} and {j}")
    
    return violations

def compute_objective_function(hex_data):
    """Compute negative of 1/outer_hex_side_length (to minimize instead of maximize)"""
    # Check if hex_data is valid
    if len(hex_data) != 12:
        return 1e10
    
    # Compute outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(hex_data)
    
    # If outer radius is invalid, penalize heavily
    if outer_radius <= 0:
        return 1e10
    
    # Return negative of 1/outer_radius (for minimization)
    return -1.0 / outer_radius

def evaluate_solution(hex_data, outer_hex_data):
    """Comprehensive evaluation of solution validity and quality"""
    # Basic constraint checking
    violations = evaluate_constraint_violations(hex_data, outer_hex_data)
    
    if violations:
        return False, 1e10, violations
    
    # Compute objective value
    obj_value = compute_objective_function(hex_data)
    return True, obj_value, []

def generate_better_initial_solution():
    """Generate a high-quality initial solution based on mathematical insights"""
    # Based on known good configurations for hexagonal packing
    # This configuration is designed to be closer to optimal than naive approaches
    
    # Hexagonal close-packed arrangement with strategic positioning
    hex_data = []
    
    # Center hexagon
    hex_data.append([0.0, 0.0, 0.0])
    
    # First ring (6 hexagons) - arranged in a hexagonal pattern
    for i in range(6):
        angle = i * 60  # degrees
        rad = np.radians(angle)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        hex_data.append([x, y, 0.0])
    
    # Second ring (6 hexagons) - positioned to create dense packing
    for i in range(6):
        angle = i * 60 + 30  # degrees (offset)
        rad = np.radians(angle)
        # Place at distance of approx sqrt(12) to form a tight cluster
        x = 3.464 * np.cos(rad)  # approx sqrt(12) = 3.464
        y = 3.464 * np.sin(rad)
        hex_data.append([x, y, 0.0])
    
    # Ensure exactly 12 hexagons
    while len(hex_data) < 12:
        hex_data.append([0.0, 0.0, 0.0])
    hex_data = hex_data[:12]
    
    # Add small random perturbations to escape symmetric local minima
    for i in range(12):
        hex_data[i][0] += random.uniform(-0.1, 0.1)
        hex_data[i][1] += random.uniform(-0.1, 0.1)
        hex_data[i][2] += random.uniform(-2, 2)
    
    return np.array(hex_data)

def optimize_single_configuration(initial_hex_data):
    """Perform optimization on a single configuration using L-BFGS-B"""
    
    # Flatten initial data for optimization
    initial_flat = initial_hex_data.flatten()
    
    # Bounds: positions (-10, 10), angles (0, 360)
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    def objective_and_gradient(params):
        # Reshape parameters back to hex data format
        hex_data = params.reshape(12, 3)
        
        # Evaluate objective function
        obj_value = compute_objective_function(hex_data)
        
        # Approximate gradient using finite differences
        epsilon = 1e-6
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            hex_data_plus = params_plus.reshape(12, 3)
            obj_plus = compute_objective_function(hex_data_plus)
            grad[i] = (obj_plus - obj_value) / epsilon
        
        return obj_value, grad
    
    # Optimize using L-BFGS-B
    try:
        # Use a more aggressive optimization approach with progressive tightening
        result = minimize(
            objective_and_gradient,
            initial_flat,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-12,
                'maxls': 50
            },
            tol=1e-12
        )
        
        if result.success:
            optimized_data = result.x.reshape(12, 3)
            return optimized_data
    except Exception:
        pass
    
    return initial_hex_data

def multi_stage_optimization(initial_params):
    """Perform multi-stage optimization for better results with adaptive parameters"""
    start_time = time.time()
    best_params = initial_params.copy()
    best_score = compute_objective_function(initial_params.reshape(12, 3))

    # Stage 1: Coarse optimization with relaxed tolerances
    try:
        result1 = minimize(
            lambda x: (compute_objective_function(x.reshape(12, 3)), None)[0],
            initial_params,
            method='L-BFGS-B',
            bounds=[(-10.0, 10.0)] * 24 + [(0.0, 360.0)] * 12,
            options={'maxiter': 50, 'ftol': 1e-3, 'gtol': 1e-3},
            callback=lambda x: time.time() - start_time > MAX_EVAL_TIME - 20
        )

        if result1.success:
            current_params = result1.x
            current_score = compute_objective_function(current_params.reshape(12, 3))
            if current_score < best_score:
                best_params = current_params
                best_score = current_score
        else:
            current_params = initial_params

    except Exception as e:
        current_params = initial_params

    # Stage 2: Refinement with moderate tolerances
    try:
        result2 = minimize(
            lambda x: (compute_objective_function(x.reshape(12, 3)), None)[0],
            current_params,
            method='L-BFGS-B',
            bounds=[(-10.0, 10.0)] * 24 + [(0.0, 360.0)] * 12,
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6},
            callback=lambda x: time.time() - start_time > MAX_EVAL_TIME - 15
        )

        if result2.success:
            current_params = result2.x
            current_score = compute_objective_function(current_params.reshape(12, 3))
            if current_score < best_score:
                best_params = current_params
                best_score = current_score

    except Exception as e:
        pass

    # Stage 3: Fine-tuning with tight tolerances
    try:
        result3 = minimize(
            lambda x: (compute_objective_function(x.reshape(12, 3)), None)[0],
            best_params,
            method='L-BFGS-B',
            bounds=[(-10.0, 10.0)] * 24 + [(0.0, 360.0)] * 12,
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9},
            callback=lambda x: time.time() - start_time > MAX_EVAL_TIME - 5
        )

        if result3.success:
            final_params = result3.x
            final_score = compute_objective_function(final_params.reshape(12, 3))
            if final_score < best_score:
                best_params = final_params
                best_score = final_score

    except Exception as e:
        pass

    return best_params

def multi_start_optimization():
    """Run multiple optimization starts with different initial configurations"""
    best_score = float('inf')
    best_solution = None
    
    # Multiple random restarts
    for restart in range(20):  # Increased from 5 to 20 for better exploration
        # Generate slightly different initial configurations
        if restart == 0:
            # First restart: better initial configuration
            initial_hex_data = generate_better_initial_solution()
        else:
            # Later restarts: slightly perturbed versions of previous best
            if best_solution is not None:
                initial_hex_data = best_solution.copy()
                # Add small random perturbations
                for i in range(12):
                    initial_hex_data[i, 0] += random.uniform(-0.5, 0.5)
                    initial_hex_data[i, 1] += random.uniform(-0.5, 0.5)
                    initial_hex_data[i, 2] += random.uniform(-5, 5)
            else:
                initial_hex_data = generate_better_initial_solution()
        
        # Flatten for optimization
        initial_flat = initial_hex_data.flatten()
        
        # Multi-stage optimization
        optimized_flat = multi_stage_optimization(initial_flat)
        
        # Reshape back to 12 hexagons
        optimized_hex_data = optimized_flat.reshape(12, 3)
        
        # Evaluate this solution
        valid, obj_value, violations = evaluate_solution(optimized_hex_data, [0, 0, 0])
        
        if valid and obj_value < best_score:
            best_score = obj_value
            best_solution = optimized_hex_data
    
    return best_solution if best_solution is not None else generate_better_initial_solution()

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Get best solution through multi-start optimization
    best_hex_data = multi_start_optimization()
    
    # Final validation
    valid, obj_value, violations = evaluate_solution(best_hex_data, [0, 0, 0])
    
    if not valid:
        # Fallback to a known good configuration if optimization fails
        fallback_config = np.array([
            [0, 0, 0],              # center
            [-2.5, 0, 0],           # left
            [2.5, 0, 0],            # right
            [-1.25, 2.17, 0],       # top-left
            [1.25, 2.17, 0],        # top-right
            [-1.25, -2.17, 0],      # bottom-left
            [1.25, -2.17, 0],       # bottom-right
            [-3.75, 2.17, 0],       # far top-left
            [3.75, 2.17, 0],        # far top-right
            [-3.75, -2.17, 0],      # far bottom-left
            [3.75, -2.17, 0],       # far bottom-right
            [0, -4, 0],             # far bottom-center
        ])
        return fallback_config, np.array([0, 0, 0]), 8.0
    
    # Compute final outer hexagon radius
    final_radius = compute_outer_hexagon_radius(best_hex_data)
    
    return best_hex_data, np.array([0, 0, 0]), final_radius

# EVOLVE-BLOCK-END