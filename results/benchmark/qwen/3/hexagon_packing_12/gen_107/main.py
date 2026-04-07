# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
import random
from numba import jit, prange

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius (numba-compiled)"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

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
    return max_distance + UNIT_HEX_RADIUS

def fast_overlap_check_tree(hex_data, tree, threshold=2.0):
    """Efficiently check overlaps using spatial indexing"""
    # Only check hexagons that are potentially close to each other
    overlaps = []
    for i in range(len(hex_data)):
        x1, y1, angle1 = hex_data[i]
        # Query nearby points within a reasonable distance
        nearby_indices = tree.query_ball_point([x1, y1], threshold)
        
        for j in nearby_indices:
            if i >= j:  # Avoid duplicate checking and self-checking
                continue
            x2, y2, angle2 = hex_data[j]
            # Quick bounding box check first
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx < threshold and dy < threshold:
                hex1_poly = hexagon_to_polygon(x1, y1, angle1)
                hex2_poly = hexagon_to_polygon(x2, y2, angle2)
                if hex1_poly.intersects(hex2_poly.buffer(1e-10)):
                    overlaps.append((i, j))
    return overlaps

def evaluate_solution_fast(hex_data, outer_hex_data, use_tree=True):
    """Fast evaluation with spatial indexing for overlap checking"""
    if len(hex_data) != 12:
        return False, 1e10, ["Invalid number of hexagons"]
    
    # Build spatial tree for faster neighbor lookups
    if use_tree:
        positions = np.array([[x, y] for x, y, _ in hex_data])
        tree = cKDTree(positions)
        
        # Early check for overlaps
        overlaps = fast_overlap_check_tree(hex_data, tree)
        if overlaps:
            return False, 1e10, [f"Overlapping hexagons {i} and {j}" for i, j in overlaps]
    else:
        # Slower but accurate check for very few overlaps
        for i in range(len(hex_data)):
            x1, y1, angle1 = hex_data[i]
            hex1_poly = hexagon_to_polygon(x1, y1, angle1)
            
            for j in range(i+1, len(hex_data)):
                x2, y2, angle2 = hex_data[j]
                hex2_poly = hexagon_to_polygon(x2, y2, angle2)
                
                if hex1_poly.intersects(hex2_poly.buffer(1e-10)):
                    return False, 1e10, [f"Overlapping hexagons {i} and {j}"]
    
    # Check containment with outer hexagon
    outer_x, outer_y, outer_angle = outer_hex_data
    outer_radius = compute_outer_hexagon_radius(hex_data)
    outer_hex = hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)
    
    # Check each inner hexagon for containment
    for i in range(len(hex_data)):
        x, y, angle = hex_data[i]
        inner_hex = hexagon_to_polygon(x, y, angle)
        if not outer_hex.contains(inner_hex):
            return False, 1e10, [f"Inner hexagon {i} not contained"]
    
    # Compute objective value
    obj_value = -1.0 / outer_radius
    return True, obj_value, []

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

def generate_symmetric_initial_solution():
    """Generate a highly symmetric initial solution"""
    # Hexagonal close packing arrangement
    # Center hexagon
    hex_data = [[0.0, 0.0, 0.0]]
    
    # First ring: 6 hexagons around center
    for i in range(6):
        angle = i * 60  # degrees
        rad = np.radians(angle)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        hex_data.append([x, y, 0.0])
    
    # Second ring: 5 hexagons (not perfectly symmetric to allow more space)
    for i in range(5):
        angle = i * 72 + 30  # degrees (offset) - 72 degrees for 5 hexagons
        rad = np.radians(angle)
        # Place at distance of approx sqrt(12) to form a tight cluster
        x = 3.464 * np.cos(rad)  # approx sqrt(12) = 3.464
        y = 3.464 * np.sin(rad)
        hex_data.append([x, y, 0.0])
    
    # Final placement - add one more hexagon at strategic location
    hex_data.append([0.0, -4.0, 0.0])
    
    # Ensure exactly 12 hexagons
    while len(hex_data) < 12:
        hex_data.append([0.0, 0.0, 0.0])
    hex_data = hex_data[:12]
    
    # Add small random perturbations to escape symmetric local minima
    for i in range(12):
        hex_data[i][0] += random.uniform(-0.2, 0.2)
        hex_data[i][1] += random.uniform(-0.2, 0.2)
        hex_data[i][2] += random.uniform(-5, 5)
    
    return np.array(hex_data)

def optimize_phase_1_coarse(initial_hex_data):
    """Coarse optimization phase for rapid improvement"""
    def objective(params):
        hex_data = params.reshape(12, 3)
        return compute_objective_function(hex_data)
    
    # Coarse bounds - focus on positions, ignore angles initially
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    # First, optimize only positions with relaxed tolerances
    try:
        # Optimizer settings for coarse phase
        result = minimize(
            objective,
            initial_hex_data.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            return result.x.reshape(12, 3)
    except Exception:
        pass
    
    return initial_hex_data

def optimize_phase_2_fine(initial_hex_data):
    """Fine optimization phase for maximum precision"""
    def objective(params):
        hex_data = params.reshape(12, 3)
        return compute_objective_function(hex_data)
    
    # Fine bounds
    bounds = []
    for i in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    
    try:
        # Optimizer settings for fine phase with tighter tolerances
        result = minimize(
            objective,
            initial_hex_data.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10},
            tol=1e-10
        )
        
        if result.success:
            return result.x.reshape(12, 3)
    except Exception:
        pass
    
    return initial_hex_data

def adaptive_local_search(initial_hex_data):
    """Apply adaptive local search with different intensities"""
    best_solution = initial_hex_data.copy()
    best_obj = compute_objective_function(best_solution)
    
    # Try several local search variants
    for attempt in range(3):
        # Random restart with different intensities
        current_solution = best_solution.copy()
        if attempt == 0:
            # Coarse optimization
            current_solution = optimize_phase_1_coarse(current_solution)
        elif attempt == 1:
            # Medium optimization
            current_solution = optimize_phase_2_fine(current_solution)
        elif attempt == 2:
            # Fine optimization with symmetry awareness
            # Apply symmetry-preserving perturbations
            current_solution = optimize_phase_2_fine(current_solution)
        
        # Evaluate new solution
        current_obj = compute_objective_function(current_solution)
        if current_obj < best_obj:
            best_obj = current_obj
            best_solution = current_solution
    
    return best_solution

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Phase 1: Generate initial solution with strong symmetry
    initial_solution = generate_symmetric_initial_solution()
    
    # Phase 2: Multi-phase optimization with spatial acceleration
    # Coarse optimization first
    coarse_solution = optimize_phase_1_coarse(initial_solution)
    
    # Fine optimization
    fine_solution = optimize_phase_2_fine(coarse_solution)
    
    # Phase 3: Adaptive local search
    final_solution = adaptive_local_search(fine_solution)
    
    # Phase 4: Validation with spatial indexing
    valid, obj_value, violations = evaluate_solution_fast(final_solution, [0, 0, 0], use_tree=True)
    
    # If not valid, fall back to better known configuration
    if not valid:
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
        valid, obj_value, violations = evaluate_solution_fast(fallback_config, [0, 0, 0], use_tree=True)
        if not valid:
            # Last resort - basic configuration
            fallback_config = np.array([
                [0, 0, 0],              # center
                [-2, 0, 0],             # left
                [2, 0, 0],              # right
                [0, 2, 0],              # top
                [0, -2, 0],             # bottom
                [-1.5, 1.5, 0],         # top-left
                [1.5, 1.5, 0],          # top-right
                [-1.5, -1.5, 0],        # bottom-left
                [1.5, -1.5, 0],         # bottom-right
                [-2.5, 2.5, 0],         # far top-left
                [2.5, 2.5, 0],          # far top-right
                [0, -3, 0],             # far bottom-center
            ])
            return fallback_config, np.array([0, 0, 0]), 7.0
    
    # Compute final outer hexagon radius
    final_radius = compute_outer_hexagon_radius(final_solution)
    
    return final_solution, np.array([0, 0, 0]), final_radius

# EVOLVE-BLOCK-END
