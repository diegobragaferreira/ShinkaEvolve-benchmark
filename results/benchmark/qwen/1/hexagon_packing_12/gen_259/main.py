# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import time
from numba import njit
import random

# Constants
HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180

@njit
def hexagon_vertices_numba(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length - JIT compiled."""
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
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

@njit
def point_in_hexagon_numba(px, py, vertices):
    """Fast point-in-hexagon test using ray casting - JIT compiled."""
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
def distance_point_to_point_numba(px, py, qx, qy):
    """Euclidean distance between two points - JIT compiled."""
    return np.sqrt((px - qx)**2 + (py - qy)**2)

@njit
def distance_point_to_hexagon_numba(px, py, vertices):
    """Distance from point to hexagon boundary."""
    min_dist = float('inf')
    n = len(vertices)
    
    # Check distance to each edge
    for i in range(n):
        j = (i + 1) % n
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        
        # Distance from point to line segment
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

@njit
def check_pairwise_overlap_numba(vertices1, vertices2):
    """Check if two hexagon vertices arrays overlap - JIT compiled."""
    # Check if any vertex of hex1 is inside hex2
    for v in vertices1:
        if point_in_hexagon_numba(v[0], v[1], vertices2):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    for v in vertices2:
        if point_in_hexagon_numba(v[0], v[1], vertices1):
            return True
    
    # Check edge intersections (simple approximation)
    return False

def create_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices_numba(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment_fast(vertices, outer_vertices):
    """Fast containment check using vertex inclusion test."""
    for v in vertices:
        if not point_in_hexagon_numba(v[0], v[1], outer_vertices):
            return False
    return True

def monte_carlo_hexagon_packing():
    """Monte Carlo sampling approach for hexagon packing."""
    best_inv_side_length = 0.0
    best_config = None
    best_outer_radius = float('inf')
    
    # Total number of samples to try
    total_samples = 50000
    
    # Pre-computed reference hexagon vertices for common tests
    ref_hex_vertices = hexagon_vertices_numba(0, 0, 0, 1)
    
    # Sample counter
    sample_count = 0
    
    start_time = time.time()
    
    # Main sampling loop
    while sample_count < total_samples and (time.time() - start_time) < MAX_EVAL_TIME:
        sample_count += 1
        
        # Generate random configuration for 12 hexagons
        hex_configs = []
        
        # Generate positions with careful consideration of spacing
        # Strategy: place first few hexagons in concentric rings
        for i in range(12):
            # Concentric ring placements with some randomness
            if i == 0:
                # Central hexagon
                x, y = 0.0, 0.0
                angle = 0.0
            elif i <= 6:
                # First ring - evenly spaced
                ring_radius = 2.0 + np.random.uniform(-0.2, 0.2)
                angle_rad = np.random.uniform(0, 2*np.pi)
                x = ring_radius * np.cos(angle_rad)
                y = ring_radius * np.sin(angle_rad)
                angle = np.random.uniform(0, 360)
            else:
                # Second ring - offset for better packing
                ring_radius = 3.5 + np.random.uniform(-0.3, 0.3)
                angle_rad = np.random.uniform(0, 2*np.pi) + np.pi/6
                x = ring_radius * np.cos(angle_rad)
                y = ring_radius * np.sin(angle_rad)
                angle = np.random.uniform(0, 360)
            
            hex_configs.append((x, y, angle))
        
        # Skip if too many invalid configurations
        if sample_count % 1000 == 0:
            # Periodically check if we're making progress
            pass
        
        # Early geometric pruning to detect obvious violations
        # Check if any hexagon centers are too far from origin (would be likely containment failures)
        max_dist_from_center = 0
        for x, y, _ in hex_configs:
            dist = np.sqrt(x*x + y*y)
            max_dist_from_center = max(max_dist_from_center, dist)
        
        # Estimate minimum required outer radius
        estimated_outer_radius = max_dist_from_center + 1.5  # +1.5 for hexagon radius + margin
        
        # Skip configurations that are clearly too small to contain all inner hexagons
        if estimated_outer_radius < 1.0:
            continue
            
        # Create the hexagon data array
        hex_data = np.array(hex_configs)
        
        # Create outer hexagon vertices for containment testing
        outer_vertices = hexagon_vertices_numba(0, 0, 0, estimated_outer_radius)
        
        # Check containment for all hexagons
        all_contained = True
        for i in range(12):
            x, y, angle = hex_configs[i]
            vertices = hexagon_vertices_numba(x, y, angle)
            if not check_containment_fast(vertices, outer_vertices):
                all_contained = False
                break
        
        if not all_contained:
            # This configuration cannot work, move to next sample
            continue
            
        # Check pairwise overlaps
        has_overlap = False
        for i in range(12):
            for j in range(i+1, 12):
                x1, y1, angle1 = hex_configs[i]
                x2, y2, angle2 = hex_configs[j]
                
                # Quick distance check - if centers too far apart, no overlap
                dist = distance_point_to_point_numba(x1, y1, x2, y2)
                if dist > 2.0:  # Sum of hexagon radii
                    continue
                
                # Precise overlap checking
                vertices1 = hexagon_vertices_numba(x1, y1, angle1)
                vertices2 = hexagon_vertices_numba(x2, y2, angle2)
                
                if check_pairwise_overlap_numba(vertices1, vertices2):
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        if has_overlap:
            # Skip overlapping configurations
            continue
            
        # If we reach here, this is a valid configuration
        # Calculate inverse of side length (objective function)
        inv_side_length = 1.0 / estimated_outer_radius
        
        if inv_side_length > best_inv_side_length:
            best_inv_side_length = inv_side_length
            best_config = hex_data.copy()
            best_outer_radius = estimated_outer_radius
            
        # Occasionally update and report progress
        if sample_count % 5000 == 0:
            if best_config is not None:
                print(f"Sample {sample_count}: Best inv side length = {best_inv_side_length:.6f}")

    # Final validation and cleanup
    if best_config is None:
        # Fall back to a known good configuration
        best_config = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        best_outer_radius = 8.0
        best_inv_side_length = 1.0 / best_outer_radius
        
    return best_config, best_outer_radius

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use Monte Carlo approach
    inner_hex_data, outer_hex_side_length = monte_carlo_hexagon_packing()
    
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
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