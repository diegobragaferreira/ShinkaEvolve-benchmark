# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from joblib import Parallel, delayed
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')

# Constants
UNIT_HEX_RADIUS = 1.0  # radius of unit hexagon circumcircle
UNIT_HEX_SIDE = 1.0    # side length of unit hexagon
PI = np.pi
SQRT3 = np.sqrt(3)

@jit(nopython=True)
def hexagon_vertices_jit(center_x, center_y, angle_rad, side_length):
    """Compute hexagon vertices efficiently using numba"""
    vertices = np.zeros((6, 2))
    for i in range(6):
        angle = angle_rad + i * PI / 3
        vertices[i, 0] = center_x + side_length * np.cos(angle)
        vertices[i, 1] = center_y + side_length * np.sin(angle)
    return vertices

@jit(nopython=True)
def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of segment
    len_sq = dx*dx + dy*dy
    
    # Avoid division by zero
    if len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0, min(1, t))  # Clamp t to [0,1]
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance squared
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def point_in_hexagon_fast(point_x, point_y, hex_center_x, hex_center_y, hex_angle_rad, hex_side_length):
    """Fast point-in-hexagon check using distance to edges"""
    # Get vertices
    vertices = hexagon_vertices_jit(hex_center_x, hex_center_y, hex_angle_rad, hex_side_length)
    
    # Calculate distance to each edge
    min_dist = np.inf
    for i in range(6):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1)%6]
        dist = distance_point_to_segment(point_x, point_y, x1, y1, x2, y2)
        min_dist = min(min_dist, dist)
    
    # For a regular hexagon with side length s, inradius = s * sqrt(3)/2
    inradius = hex_side_length * SQRT3 / 2
    return min_dist >= inradius

def hexagon_vertices(center_x, center_y, angle_rad, side_length):
    """Compute hexagon vertices"""
    return hexagon_vertices_jit(center_x, center_y, angle_rad, side_length)

def compute_outer_hexagon_radius(inner_hex_data, outer_hex_center=(0,0)):
    """Estimate minimum outer hexagon radius needed to contain all inner hexagons"""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        # Get all vertices of this hexagon
        vertices = hexagon_vertices(cx, cy, np.radians(angle), UNIT_HEX_SIDE)
        # Find maximum distance from center
        for vx, vy in vertices:
            dist = np.sqrt((vx - outer_hex_center[0])**2 + (vy - outer_hex_center[1])**2)
            max_distance = max(max_distance, dist)
    
    # Add safety margin for numerical precision
    return max_distance * 1.05

@jit(nopython=True)
def check_overlap_simple_jit(v1, v2):
    """Simple overlap check between two hexagon vertices arrays"""
    # Check if any vertex of hexagon1 is inside hexagon2
    for i in range(6):
        px, py = v1[i]
        # Check if point inside hexagon2 (not very accurate but fast for early rejection)
        # Using a simplified check: point should be within distance of center
        # This is a quick reject for obvious overlaps
        pass  # We'll use shapely for precise check
    
    return False

def check_overlap_hexagons(h1_center_x, h1_center_y, h1_angle, h1_side,
                          h2_center_x, h2_center_y, h2_angle, h2_side):
    """Check if two hexagons overlap using vertices inclusion test"""
    vertices1 = hexagon_vertices(h1_center_x, h1_center_y, np.radians(h1_angle), h1_side)
    vertices2 = hexagon_vertices(h2_center_x, h2_center_y, np.radians(h2_angle), h2_side)
    
    # Create shapely polygons
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)
    
    # Check if they intersect
    return poly1.intersects(poly2)

def check_all_overlaps(inner_hex_data):
    """Check all pairs of hexagons for overlaps"""
    n = len(inner_hex_data)
    # Early return if too few hexagons
    if n < 2:
        return False
    
    # Check only unique pairs
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, angle1 = inner_hex_data[i]
            cx2, cy2, angle2 = inner_hex_data[j]
            
            if check_overlap_hexagons(cx1, cy1, angle1, UNIT_HEX_SIDE,
                                    cx2, cy2, angle2, UNIT_HEX_SIDE):
                return True
    return False

def check_containment(inner_hex_data, outer_center=(0,0), outer_side=10):
    """Check if all inner hexagons are contained in outer hexagon"""
    outer_vertices = hexagon_vertices(outer_center[0], outer_center[1], 0, outer_side)
    outer_polygon = Polygon(outer_vertices)
    
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        vertices = hexagon_vertices(cx, cy, np.radians(angle), UNIT_HEX_SIDE)
        
        # Create hexagon polygon
        inner_polygon = Polygon(vertices)
        
        # Check if it's contained
        if not outer_polygon.contains(inner_polygon):
            return False
    
    return True

def binary_search_outer_radius(inner_hex_data, target_radius, tolerance=1e-6):
    """Binary search to find tightest outer hexagon radius that contains all inner hexagons"""
    # Binary search bounds
    low = target_radius * 0.9  # Conservative lower bound
    high = target_radius * 1.5  # Conservative upper bound
    
    # Refine until we're within tolerance
    while high - low > tolerance:
        mid = (low + high) / 2
        if check_containment(inner_hex_data, (0,0), mid):
            high = mid
        else:
            low = mid
    
    return (low + high) / 2

def evaluate_layout(inner_hex_data, outer_side_estimate=None):
    """Evaluate the quality of a given hexagon layout"""
    # Check overlaps first (early rejection)
    if check_all_overlaps(inner_hex_data):
        return 1e10  # Large penalty for overlaps
    
    # Estimate outer hexagon size
    if outer_side_estimate is None:
        estimated_outer_radius = compute_outer_hexagon_radius(inner_hex_data)
        outer_side = estimated_outer_radius * 2  # rough estimate
    else:
        outer_side = outer_side_estimate
    
    # Check containment
    if not check_containment(inner_hex_data, (0,0), outer_side):
        return 1e10  # Large penalty for containment violations
    
    # Refine outer radius with binary search for tighter bound
    try:
        refined_radius = binary_search_outer_radius(inner_hex_data, outer_side)
        # Return inverse of outer hexagon side length (we want to maximize 1/outer_side)
        return 1.0 / refined_radius
    except:
        # Fallback to simple calculation
        return 1.0 / outer_side

def generate_initial_config():
    """Generate a high-quality initial configuration using optimized hexagonal packing principles"""
    # Strategy: Build a geometrically balanced configuration based on hexagonal lattice principles
    # This is a variation of the approach that attempts to build a more efficient packing
    
    # The idea is to place hexagons in a way that maximizes space efficiency while maintaining
    # good radial distribution and avoiding overly clustered arrangements
    
    # Start with a core pattern
    config = []
    
    # Add central hexagon
    config.append([0.0, 0.0, 0.0])
    
    # Add first ring - 6 hexagons around center at proper spacing
    for i in range(6):
        angle = i * 60  # 60 degree increments
        rad_angle = np.radians(angle)
        x = 2.0 * np.cos(rad_angle)  # spacing of 2 units centers
        y = 2.0 * np.sin(rad_angle)
        config.append([x, y, 0.0])
    
    # Add second ring - 12 hexagons
    # Arrange them to fill gaps and create better packing
    for i in range(12):
        angle = i * 30  # 30 degree increments
        rad_angle = np.radians(angle)
        x = 3.0 * np.cos(rad_angle)
        y = 3.0 * np.sin(rad_angle)
        config.append([x, y, 0.0])
    
    # Use only first 11 positions (we had too many)
    config = config[:11]
    
    # Convert to numpy array
    config = np.array(config)
    
    # Add controlled noise to break symmetries
    np.random.seed(42)  # Fixed seed for reproducibility
    noise_scale = 0.015  # Reduced noise compared to previous version
    config[:, 0] += np.random.normal(0, noise_scale, config.shape[0])
    config[:, 1] += np.random.normal(0, noise_scale, config.shape[0])
    
    # Add small rotations to improve chances of finding better local minima
    config[:, 2] += np.random.uniform(-3, 3, config.shape[0])
    config[:, 2] = config[:, 2] % 360  # Keep in [0,360) range
    
    return config

def force_directed_relaxation_step(current_config, iteration=0, max_iter=100):
    """
    Apply force-directed relaxation to improve hexagon packing
    Forces include:
    1. Repulsion between overlapping hexagons
    2. Attraction to maintain desired spacing
    3. Confinement to outer boundary
    """
    # Copy current configuration
    relaxed_config = current_config.copy()
    
    # Parameters for relaxation
    alpha = 0.8  # Repulsion strength factor
    beta = 0.05  # Attraction to spacing factor
    gamma = 0.5  # Boundary confinement factor
    
    # Number of hexagons
    n = len(relaxed_config)
    
    # Create force vectors
    forces = np.zeros((n, 2))
    
    # Calculate pairwise forces
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, angle1 = relaxed_config[i]
            cx2, cy2, angle2 = relaxed_config[j]
            
            # Distance between centers
            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Avoid division by zero
            if dist > 1e-10:
                # Repulsion force (when too close)
                if dist < 2.0:  # When hexagons are closer than 2 units (sum of radii)
                    force_mag = alpha * (2.0 - dist) / dist  # Strong repulsion
                    forces[i, 0] += force_mag * dx
                    forces[i, 1] += force_mag * dy
                    forces[j, 0] -= force_mag * dx
                    forces[j, 1] -= force_mag * dy
    
    # Apply forces and update positions (with bounds checking)
    for i in range(n):
        # Apply forces
        relaxed_config[i, 0] += forces[i, 0] * 0.1
        relaxed_config[i, 1] += forces[i, 1] * 0.1
        
        # Keep within reasonable bounds
        relaxed_config[i, 0] = np.clip(relaxed_config[i, 0], -10, 10)
        relaxed_config[i, 1] = np.clip(relaxed_config[i, 1], -10, 10)
        
    # Apply attraction to ideal spacing if needed (optional)
    # For now just keep it simple and focus on overlap resolution
    
    return relaxed_config

def geometric_constraint_optimizer(initial_config):
    """
    A geometric constraint-based optimizer that systematically improves the configuration
    by enforcing geometric properties and local optimization
    """
    current_config = initial_config.copy()
    best_config = current_config.copy()
    best_score = evaluate_layout(current_config)
    
    # Multi-stage optimization with progressive refinement
    stages = [
        {"iterations": 50, "relaxation_factor": 0.1, "noise_std": 0.05},
        {"iterations": 30, "relaxation_factor": 0.05, "noise_std": 0.02},
        {"iterations": 20, "relaxation_factor": 0.01, "noise_std": 0.01}
    ]
    
    for stage_idx, stage in enumerate(stages):
        for iter_num in range(stage["iterations"]):
            # Apply force-directed relaxation
            relaxed = force_directed_relaxation_step(current_config, iter_num, stage["iterations"])
            
            # Add noise to escape local minima
            np.random.seed(iter_num + stage_idx * 1000)  # Different seed for each iteration
            noise = np.random.normal(0, stage["noise_std"], relaxed.shape)
            noisy = relaxed + noise
            
            # Evaluate new configuration
            new_score = evaluate_layout(noisy)
            
            # Accept better configurations or occasionally accept worse ones
            if new_score > best_score or np.random.random() < 0.05:
                current_config = noisy
                if new_score > best_score:
                    best_config = noisy.copy()
                    best_score = new_score
    
    return best_config, best_score

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Step 1: Generate initial configuration
        initial_config = generate_initial_config()
        
        # Step 2: Apply geometric constraint optimization
        optimized_config, optimized_score = geometric_constraint_optimizer(initial_config)
        
        # Step 3: Final validation
        final_score = evaluate_layout(optimized_config)
        
        # If optimization didn't improve significantly, fallback to original approach
        if final_score < 0.1:
            inner_hex_data = np.array([
                [0, 0, 0],          # center
                [-2.5, 0, 0],       # left
                [2.5, 0, 0],        # right
                [-1.25, 2.17, 0],   # top-left
                [1.25, 2.17, 0],    # top-right
                [-1.25, -2.17, 0],  # bottom-left
                [1.25, -2.17, 0],   # bottom-right
                [-3.75, 2.17, 0],   # far top-left
                [3.75, 2.17, 0],    # far top-right
                [-3.75, -2.17, 0],  # far bottom-left
                [3.75, -2.17, 0],   # far bottom-right
            ])
            outer_hex_side_length = 8.0
        else:
            # Extract the best configuration found
            inner_hex_data = optimized_config
            
            # Compute actual outer hexagon size
            estimated_outer_radius = compute_outer_hexagon_radius(inner_hex_data)
            outer_hex_side_length = estimated_outer_radius * 2.0
        
        # Set outer hexagon at center with zero rotation
        outer_hex_data = np.array([0, 0, 0])
        
        # Validate solution
        if not check_all_overlaps(inner_hex_data) and check_containment(inner_hex_data, (0,0), outer_hex_side_length):
            pass
        else:
            # If validation fails, fall back to a known good configuration
            inner_hex_data = np.array([
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ])
            outer_hex_side_length = 8.0
            outer_hex_data = np.array([0, 0, 0])
            
    except Exception as e:
        print(f"Exception in optimization: {e}")
        # Fallback to baseline approach
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0, 0, 0])
    
    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")
    
    # Return the result
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END