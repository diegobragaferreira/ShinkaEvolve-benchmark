# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import time
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# Global constants
HEX_RADIUS = 1.0
HEX_APO = HEX_RADIUS * np.sqrt(3) / 2  # Apothem of unit hexagon

@jit(nopython=True)
def generate_hexagon_vertices_numba(center_x, center_y, angle_degrees):
    """Generate vertices of a unit regular hexagon given center and rotation - JIT compiled."""
    angle_rad = np.radians(angle_degrees)
    vertices = np.empty((6, 2), dtype=np.float64)
    
    # Precompute trigonometric values for efficiency
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Hexagon vertices in local coordinate system (unit hexagon centered at origin)
    local_vertices = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Apply rotation and translation efficiently
    for i in range(6):
        x, y = local_vertices[i]
        # Rotate
        rot_x = x * cos_angle - y * sin_angle
        rot_y = x * sin_angle + y * cos_angle
        # Translate
        vertices[i, 0] = rot_x + center_x
        vertices[i, 1] = rot_y + center_y
    
    return vertices

@jit(nopython=True)
def point_in_polygon_numba(point, polygon_vertices):
    """Fast point-in-polygon test - JIT compiled."""
    x, y = point
    n = len(polygon_vertices)
    inside = False
    
    p1x, p1y = polygon_vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_vertices[i % n]
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
def distance_point_to_segment(point, seg_start, seg_end):
    """Calculate distance from point to line segment - JIT compiled."""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Vector from seg_start to seg_end
    dx, dy = x2 - x1, y2 - y1
    # Vector from seg_start to point
    fx, fy = px - x1, py - y1
    
    # Project point onto line
    length_sq = dx*dx + dy*dy
    if length_sq == 0:
        return np.sqrt(fx*fx + fy*fy)
    
    t = (fx*dx + fy*dy) / length_sq
    t = max(0, min(1, t))  # Clamp t to [0,1]
    
    # Closest point on segment
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def polygon_distance(polygon1, polygon2):
    """Compute minimum distance between two polygons - JIT compiled."""
    # Check if any vertex of polygon1 is inside polygon2
    for i in range(len(polygon1)):
        if point_in_polygon_numba(polygon1[i], polygon2):
            return 0.0
    
    # Check if any vertex of polygon2 is inside polygon1
    for i in range(len(polygon2)):
        if point_in_polygon_numba(polygon2[i], polygon1):
            return 0.0
    
    # Check edges
    min_dist = np.inf
    for i in range(len(polygon1)):
        for j in range(len(polygon2)):
            p1 = polygon1[i]
            p2 = polygon1[(i+1)%len(polygon1)]
            p3 = polygon2[j]
            p4 = polygon2[(j+1)%len(polygon2)]
            
            # Compute distance between segments
            dist = distance_point_to_segment(p1, p3, p4)
            min_dist = min(min_dist, dist)
            
            dist = distance_point_to_segment(p3, p1, p2)
            min_dist = min(min_dist, dist)
    
    return min_dist

class SpatialHashGrid:
    """Spatial hash grid for efficient neighbor lookups."""
    
    def __init__(self, cell_size=2.0):
        self.cell_size = cell_size
        self.grid = {}
    
    def clear(self):
        self.grid.clear()
    
    def insert(self, hex_id, center_x, center_y):
        """Insert hexagon into spatial hash grid."""
        cell_x = int(center_x // self.cell_size)
        cell_y = int(center_y // self.cell_size)
        key = (cell_x, cell_y)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(hex_id)
    
    def get_neighbors(self, center_x, center_y):
        """Get all hexagons in the same and neighboring cells."""
        cell_x = int(center_x // self.cell_size)
        cell_y = int(center_y // self.cell_size)
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cell_x + dx, cell_y + dy)
                if key in self.grid:
                    neighbors.extend(self.grid[key])
        return neighbors

@jit(parallel=True)
def compute_outer_hexagon_side_length_fast(inner_hex_data):
    """Fast computation of outer hexagon side length using vectorized operations."""
    if len(inner_hex_data) == 0:
        return 1e6
    
    # Get all vertices efficiently
    all_vertices = np.empty((len(inner_hex_data) * 6, 2), dtype=np.float64)
    
    for i in prange(len(inner_hex_data)):
        center_x, center_y, angle_degrees = inner_hex_data[i]
        vertices = generate_hexagon_vertices_numba(center_x, center_y, angle_degrees)
        for j in range(6):
            all_vertices[i*6 + j] = vertices[j]
    
    # Compute bounding box
    min_x = np.min(all_vertices[:, 0])
    max_x = np.max(all_vertices[:, 0])
    min_y = np.min(all_vertices[:, 1])
    max_y = np.max(all_vertices[:, 1])
    
    # Compute center
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Compute max distance squared from center
    max_dist_sq = 0.0
    for i in prange(len(all_vertices)):
        x, y = all_vertices[i]
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        max_dist_sq = max(max_dist_sq, dist_sq)
    
    # Side length = sqrt(max_dist) * 2 / sqrt(3)
    return np.sqrt(max_dist_sq) * 2.0 / np.sqrt(3)

def create_hexagon_polygon(center_x, center_y, angle_degrees):
    """Create Shapely polygon representation of a hexagon."""
    vertices = generate_hexagon_vertices_numba(center_x, center_y, angle_degrees)
    return Polygon(vertices)

def check_containment_fast(hex_poly, outer_hex_poly):
    """Fast containment check using Shapely."""
    return outer_hex_poly.contains(hex_poly)

def check_overlap_fast(hex1_poly, hex2_poly):
    """Fast overlap check using Shapely."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def evaluate_fitness(solution, outer_hex_center=(0, 0), outer_hex_angle=0, penalty_weight=1e5):
    """Evaluate fitness of a solution with optimized overlap checking and adaptive penalties."""
    # Reshape solution back to 12 hexagons with (x, y, angle)
    positions = solution.reshape(-1, 3)
    
    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(len(positions)):
        center_x, center_y, angle_degrees = positions[i]
        poly = create_hexagon_polygon(center_x, center_y, angle_degrees)
        inner_polygons.append(poly)
    
    # Find outer hexagon side length
    side_length = compute_outer_hexagon_side_length_fast(positions)
    
    # Check containment and overlaps efficiently using spatial hashing
    total_penalty = 0
    num_inner = len(inner_polygons)
    
    # Outer hexagon polygon (centered at origin with given angle)
    outer_poly = create_hexagon_polygon(outer_hex_center[0], outer_hex_center[1], outer_hex_angle)
    
    # Build spatial hash for fast neighbor lookup
    # Use slightly smaller cell size for better neighbor detection
    spatial_grid = SpatialHashGrid(cell_size=1.8)
    for i in range(num_inner):
        center_x, center_y, _ = positions[i]
        spatial_grid.insert(i, center_x, center_y)
    
    # Check containment
    for i in range(num_inner):
        if not check_containment_fast(inner_polygons[i], outer_poly):
            # Dynamic penalty based on violation magnitude
            # Calculate how far outside the boundary
            penalty_multiplier = 1.0
            total_penalty += penalty_weight * penalty_multiplier * 1e3  # Large penalty for containment violations
    
    # Check overlaps using spatial hash for efficiency
    overlap_count = 0
    for i in range(num_inner):
        center_x, center_y, _ = positions[i]
        neighbors = spatial_grid.get_neighbors(center_x, center_y)
        
        for j in neighbors:
            if i >= j:  # Avoid duplicate checks and self-checking
                continue
            if check_overlap_fast(inner_polygons[i], inner_polygons[j]):
                # Dynamic penalty increased for overlap violations
                overlap_count += 1
                # Increase penalty based on number of overlaps
                penalty_multiplier = 1.0 + overlap_count * 0.1
                total_penalty += penalty_weight * penalty_multiplier * 1e4
    
    # Objective: maximize 1/side_length
    # So minimize negative log of side_length plus penalties
    if side_length < 1e-6:
        fitness = -1e10 - total_penalty
    else:
        fitness = -np.log(side_length) - total_penalty
    
    return fitness, side_length

def get_symmetric_initial_config(config_type=1):
    """Generate initial configuration with enhanced symmetry properties."""
    positions = []
    
    if config_type == 1:
        # Central hexagon
        positions.append([0, 0, 0])
        
        # First ring: 6 hexagons around center
        angles = np.arange(0, 360, 60)
        radius = 2.0
        for angle in angles:
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Second ring: 6 hexagons at greater distance
        radius = 3.5
        for angle in angles:
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Additional strategic placements for better packing
        additional_positions = [
            [0, -4.5, 0],  # Bottom center
            [4.5, 0, 0],   # Right center
            [-4.5, 0, 0],  # Left center
            [0, 4.5, 0],   # Top center
        ]
        
        positions.extend(additional_positions)
        
    elif config_type == 2:
        # Hexagonal close-packed arrangement
        positions.append([0, 0, 0])
        
        # First ring - 6 hexagons
        angles = np.arange(0, 360, 60)
        radius = 2.1
        for angle in angles:
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Second ring - 12 hexagons in a hexagonal pattern
        radius = 4.2
        for i in range(12):
            angle = i * 30  # Every 30 degrees
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Additional strategic positions
        positions.extend([
            [0, -5.0, 0],
            [5.0, 0, 0],
            [-5.0, 0, 0],
            [0, 5.0, 0]
        ])
        
    elif config_type == 3:
        # Optimized symmetric pattern
        positions.append([0, 0, 0])
        
        # Ring 1 - 6 hexagons
        angles = np.arange(0, 360, 60)
        radius = 1.8
        for angle in angles:
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Ring 2 - 6 hexagons
        radius = 3.2
        for angle in angles:
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
        
        # Ring 3 - 6 hexagons (different spacing)
        radius = 4.5
        for i in range(6):
            angle = 60 * i
            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))
            positions.append([x, y, 0])
            
        # Additional positions
        positions.extend([
            [0, -4.2, 0],
            [4.2, 0, 0],
            [-4.2, 0, 0],
            [0, 4.2, 0]
        ])
    
    return np.array(positions).flatten()

def refine_with_local_search(initial_solution, max_iterations=25, method='L-BFGS-B'):
    """Use local optimization to refine the solution."""
    # Flatten the solution
    x0 = initial_solution.copy()
    
    def objective(x):
        # Reshape for evaluation
        positions = x.reshape(-1, 3)
        # Add small penalty to encourage staying close to initial
        initial_penalty = 0.002 * np.sum((x - initial_solution)**2)
        _, side_length = evaluate_fitness(x)
        return -np.log(side_length) - initial_penalty
    
    bounds = [(-15, 15)] * len(x0)  # Reasonable bounds
    
    try:
        result = minimize(objective, x0, method=method, bounds=bounds, 
                         options={'maxiter': max_iterations}, tol=1e-6)
        return result.x
    except:
        return x0

def optimize_hexagon_packing():
    """Main optimization function with multi-stage strategy."""
    # Multiple initial configurations to avoid local optima
    initial_configs = []
    
    # Configuration 1: Original symmetric pattern
    initial_configs.append(get_symmetric_initial_config(1))
    
    # Configuration 2: Hexagonal packing pattern
    initial_configs.append(get_symmetric_initial_config(2))
    
    # Configuration 3: Optimized symmetric pattern  
    initial_configs.append(get_symmetric_initial_config(3))
    
    # Configuration 4: Perturbed version of the best pattern
    base_config = get_symmetric_initial_config(1).copy()
    perturbation = np.random.normal(0, 0.2, len(base_config))
    initial_configs.append(base_config + perturbation)
    
    # Configuration 5: Alternative arrangement
    positions = []
    positions.append([0, 0, 0])
    
    # Spiral pattern
    for i in range(1, 12):
        angle = i * 30  # 30 degree increments
        radius = i * 0.5
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0])
    
    # Adjust to fit within hexagon
    positions[1:] = np.array(positions[1:]) * 1.2  # Scale down
    initial_configs.append(np.array(positions).flatten())
    
    best_solution = None
    best_side_length = float('inf')
    
    # Multi-stage optimization approach
    stage1_iter = 25
    stage2_iter = 30
    stage3_iter = 20
    
    # Try multiple optimization runs
    for i, initial_pos in enumerate(initial_configs):
        try:
            # Stage 1: Coarse global search with relaxed constraints
            refined_solution = refine_with_local_search(initial_pos, 10, 'L-BFGS-B')
            
            # Stage 2: Medium resolution refinement 
            positions = refined_solution.reshape(-1, 3)
            bounds = [(-12, 12)] * len(refined_solution)
            
            def de_objective(x):
                _, side_length = evaluate_fitness(x, penalty_weight=1e4)
                return -np.log(side_length)  # Minimize negative log side length
            
            # Run differential evolution with moderate iterations
            result = differential_evolution(de_objective, bounds,
                                          maxiter=stage1_iter, popsize=15,
                                          seed=None, disp=False, 
                                          strategy='best1bin')
            
            if result.success:
                _, side_length = evaluate_fitness(result.x, penalty_weight=1e5)
                if side_length < best_side_length:
                    best_side_length = side_length
                    best_solution = result.x
            else:
                # Even if not successful, try with the refined solution
                _, side_length = evaluate_fitness(refined_solution, penalty_weight=1e5)
                if side_length < best_side_length:
                    best_side_length = side_length
                    best_solution = refined_solution
                    
        except Exception as e:
            print(f"Optimization attempt {i} failed: {e}")
            continue
    
    # Stage 3: Fine local optimization on the best solution
    if best_solution is not None:
        final_refined = refine_with_local_search(best_solution, stage3_iter)
        final_positions = final_refined.reshape(-1, 3)
        _, final_side_length = evaluate_fitness(final_refined, penalty_weight=1e6)
    else:
        # Fallback to initial configuration
        positions = initial_configs[0].reshape(-1, 3)
        final_side_length = compute_outer_hexagon_side_length_fast(positions)
        final_positions = positions
    
    return final_positions, final_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Optimize
    inner_hex_data, outer_hex_side_length = optimize_hexagon_packing()
    
    # Ensure we have exactly 12 hexagons
    if len(inner_hex_data) != 12:
        # Fallback to simple grid if optimization fails
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
            [0, -4, 0],
        ])
        outer_hex_side_length = 8
    
    # Set outer hexagon parameters (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    # Final validation and adjustment
    try:
        # Double-check the solution
        _, computed_side_length = evaluate_fitness(inner_hex_data.flatten())
        outer_hex_side_length = computed_side_length
    except:
        pass
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Calculate benchmark ratio (inverse side length vs target)
    benchmark_ratio = 1 / outer_hex_side_length / 0.2537
    
    # Print metrics for verification
    print(f"inv_outer_hex_side_length: {1/outer_hex_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {eval_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END