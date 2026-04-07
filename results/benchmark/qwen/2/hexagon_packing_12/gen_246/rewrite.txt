# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import time
import math
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices_numba(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon using numba for speed."""
    angle_rad = angle_deg * math.pi / 180.0
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices[i] = (x, y)
    return vertices

@jit(nopython=True)
def point_distance_squared(x1, y1, x2, y2):
    """Calculate squared distance between two points."""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy

@jit(nopython=True)
def fast_hexagon_center_and_apothem(vertices):
    """Fast calculation of center and apothem for a hexagon."""
    # Calculate center
    cx = 0.0
    cy = 0.0
    for i in range(6):
        cx += vertices[i, 0]
        cy += vertices[i, 1]
    cx /= 6.0
    cy /= 6.0

    # Calculate apothem using distance from center to first vertex
    apothem_sq = point_distance_squared(cx, cy, vertices[0, 0], vertices[0, 1])
    return cx, cy, apothem_sq

@jit(nopython=True)
def compute_centers_apothems_numba(hexagon_vertices_list):
    """Numba-accelerated computation of centers and apothems."""
    num_hexagons = len(hexagon_vertices_list)
    centers = np.empty((num_hexagons, 2))
    apothems_sq = np.empty(num_hexagons)

    for i in range(num_hexagons):
        cx, cy, apothem_sq = fast_hexagon_center_and_apothem(hexagon_vertices_list[i])
        centers[i] = (cx, cy)
        apothems_sq[i] = apothem_sq

    return centers, apothems_sq

def generate_deterministic_seed_config():
    """Generate a high-quality deterministic seed configuration based on mathematical insights."""
    # Arrangement inspired by close-packing theory with strategic symmetry
    # Positions chosen to be mathematically meaningful and visually balanced
    return np.array([
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First ring (6 hexagons)
        [0.0, 2.17, 0.0],      # Top
        [1.86, 1.1, 0.0],      # Top-right
        [1.86, -1.1, 0.0],     # Bottom-right
        [0.0, -2.17, 0.0],     # Bottom
        [-1.86, -1.1, 0.0],    # Bottom-left
        [-1.86, 1.1, 0.0],     # Top-left
        # Second ring (6 hexagons)
        [0.0, 4.34, 0.0],      # Far top
        [3.75, 2.17, 0.0],     # Upper right
        [3.75, -2.17, 0.0],    # Lower right
        [0.0, -4.34, 0.0],     # Far bottom
        [-3.75, -2.17, 0.0],   # Lower left
        [-3.75, 2.17, 0.0],    # Upper left
    ])

def check_containment_distance_only_numba(hexagon_vertices_list, outer_side_length):
    """Fast containment check using numba-accelerated distance bounds."""
    # For outer hexagon with side length R, distance from center to edge is R * sqrt(3)/2
    outer_apothem_sq = (outer_side_length * math.sqrt(3) / 2) ** 2

    for vertices in hexagon_vertices_list:
        # Check distance of each vertex from origin (0,0)
        for i in range(6):
            x = vertices[i, 0]
            y = vertices[i, 1]
            distance_sq = x*x + y*y
            if distance_sq > outer_apothem_sq:
                return False
    return True

def check_overlap_fast_numba(hexagon_vertices_list):
    """Fast overlap detection using numba-accelerated distance bounds."""
    # Get centers and apothems
    centers, apothems_sq = compute_centers_apothems_numba(hexagon_vertices_list)

    # Precompute apothem thresholds (sqrt(apothem_sq) * 2 for safety)
    apothems = np.sqrt(apothems_sq) * 2.0

    # Check distances between centers
    num_hexagons = len(hexagon_vertices_list)
    for i in range(num_hexagons):
        for j in range(i+1, num_hexagons):
            # Fast distance calculation
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            distance_sq = dx * dx + dy * dy
            combined_radius = apothems[i] + apothems[j]

            if distance_sq < combined_radius * combined_radius:
                return False
    return True

def check_overlap_shapely(hexagon_vertices_list):
    """Precise overlap detection using Shapely."""
    try:
        polygons = [Polygon(vertices) for vertices in hexagon_vertices_list]
        union = Polygon.union_all(polygons)
        total_area = sum(polygon.area for polygon in polygons)
        union_area = union.area
        return abs(total_area - union_area) < 1e-10
    except:
        # Fallback for complex cases
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return False
        return True

def evaluate_single_configuration(config, outer_side_length):
    """Evaluate a single configuration using a multi-stage constraint checker."""
    # Parse configuration
    hexagons = config.reshape(12, 3)
    
    # Get vertices for all hexagons
    hexagon_vertices_list = []
    for i in range(12):
        x, y, angle = hexagons[i]
        vertices = hexagon_vertices_numba(x, y, angle)
        hexagon_vertices_list.append(vertices)

    # Stage 1: Fast containment check
    if not check_containment_distance_only_numba(hexagon_vertices_list, outer_side_length):
        return False, float('inf')  # Invalid configuration

    # Stage 2: Fast overlap check
    if not check_overlap_fast_numba(hexagon_vertices_list):
        # If fast check indicates possible overlap, do more thorough check
        if not check_overlap_shapely(hexagon_vertices_list):
            return False, float('inf')  # Confirmed overlap

    return True, 0  # Valid configuration

def monte_carlo_sampling_parallel(num_samples=5000, max_workers=4):
    """Monte Carlo sampling with parallel processing."""
    # Generate a high-quality initial seed configuration
    seed_config = generate_deterministic_seed_config().flatten()
    
    best_config = seed_config.copy()
    best_outer_side_length = 5.0
    best_score = 1.0 / best_outer_side_length
    
    # Generate multiple configurations in batches to avoid memory issues
    batch_size = min(num_samples, 1000)
    total_processed = 0
    
    while total_processed < num_samples:
        remaining = min(batch_size, num_samples - total_processed)
        
        # Generate random configurations with mathematical constraints
        configs = []
        for _ in range(remaining):
            # Start with a good configuration and perturb slightly
            config = seed_config.copy()
            
            # Add noise to positions (less than 0.3 to keep in good region)
            config[::3] += np.random.normal(0, 0.2, 12)  # x positions
            config[1::3] += np.random.normal(0, 0.2, 12)  # y positions
            # Keep angles between 0 and 360
            config[2::3] = config[2::3] % 360
            
            configs.append(config)
        
        # Parallel processing of configurations
        def process_config(config_item):
            # Try configurations with decreasing outer hexagon sizes
            for test_size in np.linspace(4.5, 3.8, 10)[::-1]:
                valid, penalty = evaluate_single_configuration(config_item, test_size)
                if valid:
                    return config_item, test_size, True
            return config_item, 5.0, False
        
        # Process in parallel
        results = Parallel(n_jobs=max_workers)(
            delayed(process_config)(config) for config in configs
        )
        
        # Update best configuration found
        for config, size, valid in results:
            if valid:
                score = 1.0 / size
                if score > best_score:
                    best_score = score
                    best_config = config.copy()
                    best_outer_side_length = size
        
        total_processed += remaining
        
        # Early stopping if we're getting close to target
        if best_outer_side_length < 3.95:
            break
    
    return best_config, best_outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Use Monte Carlo sampling with parallel processing
    inner_hex_data, outer_hex_side_length = monte_carlo_sampling_parallel(
        num_samples=10000, max_workers=4
    )
    
    # Convert to proper structure
    inner_hex_data = inner_hex_data.reshape(12, 3)
    
    # Create outer hexagon data (centered, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    # Calculate benchmark ratio
    benchmark_ratio = 1.0 / outer_hex_side_length / 0.2537
    eval_time = time.time() - start_time
    
    print(f"inv_outer_hex_side_length: {1.0 / outer_hex_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {eval_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END