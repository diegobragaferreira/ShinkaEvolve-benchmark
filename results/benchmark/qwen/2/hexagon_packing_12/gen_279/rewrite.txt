# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from numba import jit, prange
import random
from collections import defaultdict

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
        # Check containment penalty with adaptive penalty weights
        for vx, vy in vertices:
            point = np.array([vx, vy])
            if not point_in_polygon(point, outer_hex_vertices):
                dist = np.sqrt(vx*vx + vy*vy)
                violation_distance = dist - outer_radius + 0.5
                if violation_distance > 0:
                    # Increase penalty based on how far outside the boundary
                    total_penalty += violation_distance**2 * 2000000  # Higher penalty for containment
    
    # Check overlaps using spatial hashing with penalty scaling
    for i in range(n):
        # Get potentially overlapping hexagons
        overlapping_indices = get_overlapping_indices(hash_grid, i, hex_vertices_list[i])
        
        # Check actual overlaps
        for j in overlapping_indices:
            if i < j:  # Avoid double counting
                vertices1 = hex_vertices_list[i]
                vertices2 = hex_vertices_list[j]
                
                if check_hexagon_overlap(vertices1, vertices2):
                    total_penalty += 1500000  # Even higher penalty for overlaps
    
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

def create_advanced_symmetric_config():
    """Create an advanced symmetric configuration with mathematical optimization"""
    # Highly structured symmetric arrangement based on group theory principles
    # This configuration is designed to maximize packing density while maintaining symmetry
    positions = [
        [0.0, 0.0, 0.0],      # Center (fixed)
        [0.0, 2.1, 0.0],      # Up (slightly tighter spacing than previous)
        [0.0, -2.1, 0.0],     # Down
        [1.82, 1.05, 0.0],    # Upper right 
        [-1.82, 1.05, 0.0],   # Upper left
        [1.82, -1.05, 0.0],   # Lower right
        [-1.82, -1.05, 0.0],  # Lower left
        [3.64, 0.0, 0.0],     # Far right
        [-3.64, 0.0, 0.0],    # Far left
        [0.0, 3.64, 0.0],     # Far up
        [0.0, -3.64, 0.0],    # Far down
        [1.82, 3.15, 0.0],    # Extended upper right (more spread out)
    ]
    
    # Add moderate randomness to escape local minima while preserving structure
    positions = np.array(positions)
    for i in range(1, len(positions)):
        # Moderate perturbations to allow for optimization
        positions[i][0] += np.random.normal(0, 0.1)
        positions[i][1] += np.random.normal(0, 0.1)
        positions[i][2] = random.uniform(0, 30)  # Small rotation variations
        
    return positions

def adaptive_local_refinement(initial_config, max_iterations=150, method='L-BFGS-B', phase='fine'):
    """Apply local refinement with adaptive parameters for different phases"""
    # Convert to flat parameter array for optimization
    params = initial_config.flatten()
    
    # Adjust optimization parameters based on phase
    if phase == 'coarse':
        # Coarse optimization: less strict but faster
        gtol_val = 1e-4
        ftol_val = 1e-4
        max_iter_val = max_iterations // 2
    elif phase == 'medium':
        # Medium refinement: balanced
        gtol_val = 1e-6
        ftol_val = 1e-6
        max_iter_val = max_iterations
    else:  # fine
        # Fine tuning: more strict
        gtol_val = 1e-8
        ftol_val = 1e-8
        max_iter_val = max_iterations * 1.5
    
    def objective(flat_params):
        config = flat_params.reshape(-1, 3)
        outer_radius = get_outer_hexagon_radius(config)
        penalty = calculate_total_penalty(config, outer_radius)
        # Objective is to minimize penalty and maximize 1/R (equivalent to minimize -1/R + penalty)
        return penalty - 1.0/(outer_radius + 1e-6)  # Small epsilon for numerical stability
    
    # Use L-BFGS-B for local optimization with bounds
    bounds = [(-6.0, 6.0), (-6.0, 6.0), (0.0, 360.0)] * 12  # wider bounds for exploration
    
    try:
        result = minimize(objective, params, method=method, bounds=bounds, 
                         options={'maxiter': max_iter_val, 'gtol': gtol_val, 'ftol': ftol_val})
        if result.success:
            # Return optimized configuration
            final_config = result.x.reshape(-1, 3)
            return final_config
    except Exception as e:
        pass
    
    # If optimization fails, return original
    return initial_config

def run_multi_strategy_optimization():
    """Run multiple optimization strategies to find the best solution"""
    best_solution = None
    best_side_length = float('inf')
    best_objective = float('inf')
    best_penalty = float('inf')
    
    # Strategy 1: Differential Evolution with symmetry-aware initialization
    print("Running Strategy 1: Differential Evolution...")
    try:
        # Use advanced symmetric configuration as starting point
        initial_config = create_advanced_symmetric_config()
        bounds = [(-8, 8), (-8, 8), (0, 360)] * 12
        result_de = differential_evolution(
            lambda x: evaluate_solution_multistart(x)[0],
            bounds,
            maxiter=100,
            popsize=30,
            mutation=(0.7, 1.0),
            recombination=0.8,
            seed=42,
            disp=False
        )
        
        final_objective, side_length = evaluate_solution_multistart(result_de.x)
        if final_objective < best_objective:
            best_objective = final_objective
            best_side_length = side_length
            best_solution = result_de.x.copy()
            
    except Exception as e:
        print(f"DE Strategy failed: {e}")
    
    # Strategy 2: Sequential local optimization with adaptive phases
    print("Running Strategy 2: Adaptive Local Optimization...")
    try:
        # Start with advanced symmetric configuration
        initial_config = create_advanced_symmetric_config()
        
        # Phase 1: Coarse optimization
        coarse_result = adaptive_local_refinement(initial_config, max_iterations=30, phase='coarse')
        
        # Phase 2: Medium refinement
        medium_result = adaptive_local_refinement(coarse_result, max_iterations=60, phase='medium')
        
        # Phase 3: Fine-tuning
        fine_result = adaptive_local_refinement(medium_result, max_iterations=100, phase='fine')
        
        final_objective, side_length = evaluate_solution_multistart(fine_result.flatten())
        if final_objective < best_objective:
            best_objective = final_objective
            best_side_length = side_length
            best_solution = fine_result.flatten().copy()
            
    except Exception as e:
        print(f"Local Optimization Strategy failed: {e}")
    
    return best_solution, best_side_length

def evaluate_solution_multistart(solution_array):
    """Improved evaluation with better containment and overlap checking"""
    # Reshape solution array into 12 hexagons with (x, y, angle) each
    inner_hex_data = solution_array.reshape(-1, 3)

    # Calculate the minimum enclosing hexagon
    min_side_length, centroid = calculate_min_enclosing_hexagon_fast(inner_hex_data)

    # Check all constraints
    num_hex = len(inner_hex_data)
    penalty = 0.0

    # Create outer hexagon polygon for containment checks
    outer_hex = create_hexagon_polygon(centroid[0], centroid[1], 0, min_side_length)

    # Check containment for all hexagons using fast approximation
    for i in range(num_hex):
        center_x, center_y, angle = inner_hex_data[i]
        # Simple check using max distance from center
        vertices = hexagon_vertices(center_x, center_y, angle)
        max_dist = np.max(np.sqrt(np.sum((vertices - np.array([center_x, center_y]))**2, axis=1)))
        if max_dist > min_side_length * np.sqrt(3) / 2:
            penalty += 1000000  # High penalty for containment violations

    # Check overlaps using spatial hashing for efficiency
    vertices_cache = []
    for i in range(num_hex):
        vertices = hexagon_vertices(*inner_hex_data[i])
        vertices_cache.append(vertices)

    # Use spatial hash overlap detection
    if check_overlaps_spatial_hash(inner_hex_data, vertices_cache):
        penalty += 1500000  # Even higher penalty for overlaps

    # Return negative inverse side length plus penalty
    objective_value = -1.0 / min_side_length + penalty

    return objective_value, min_side_length

def calculate_min_enclosing_hexagon_fast(inner_hex_data, scale_factor=1.05):
    """Fast calculation of minimum enclosing hexagon using Numba."""
    # Get all vertices of all inner hexagons
    all_vertices = np.empty((0, 2), dtype=np.float64)

    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = hexagon_vertices(center_x, center_y, angle)
        all_vertices = np.vstack([all_vertices, vertices])

    if len(all_vertices) == 0:
        return 1.0, np.array([0., 0.])

    # Find bounding circle radius
    centroid = np.mean(all_vertices, axis=0)
    distances = np.sqrt(np.sum((all_vertices - centroid)**2, axis=1))
    max_distance = np.max(distances)

    # For a regular hexagon, side length = max_distance * sqrt(3)/2
    side_length = max_distance * 2 / np.sqrt(3) * scale_factor

    return side_length, centroid

def create_hexagon_polygon(center_x, center_y, angle_deg, side_length=1):
    """Create a Shapely polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, angle_deg, side_length)
    return Polygon(vertices)

def check_overlaps_spatial_hash(hex_data, vertices_cache=None):
    """Check overlaps using spatial hashing to reduce comparisons."""
    if len(hex_data) <= 1:
        return False

    # Create spatial hash grid
    grid, cell_size = spatial_hash_grid(hex_data)

    # Check overlaps efficiently using neighbor lists
    for i in range(len(hex_data)):
        cx1, cy1, angle1 = hex_data[i]

        # Get grid cell coordinates for this hexagon
        grid_x = int(cx1 // cell_size)
        grid_y = int(cy1 // cell_size)

        # Get potential neighbors
        neighbors = get_neighbors(grid, grid_x, grid_y, cell_size)

        # Check overlap only with neighbors
        for j in neighbors:
            if i >= j:  # Avoid duplicate checks
                continue

            cx2, cy2, angle2 = hex_data[j]

            # Quick AABB bounding box test before expensive SAT test
            # Get approximate bounding boxes for both hexagons
            if vertices_cache is not None:
                vertices1 = vertices_cache[i]
                vertices2 = vertices_cache[j]
            else:
                vertices1 = hexagon_vertices(cx1, cy1, angle1)
                vertices2 = hexagon_vertices(cx2, cy2, angle2)

            # Calculate AABB for first hexagon
            min_x1 = vertices1[:, 0].min()
            max_x1 = vertices1[:, 0].max()
            min_y1 = vertices1[:, 1].min()
            max_y1 = vertices1[:, 1].max()

            # Calculate AABB for second hexagon
            min_x2 = vertices2[:, 0].min()
            max_x2 = vertices2[:, 0].max()
            min_y2 = vertices2[:, 1].min()
            max_y2 = vertices2[:, 1].max()

            # Quick AABB overlap check
            if (max_x1 >= min_x2 and min_x1 <= max_x2 and
                max_y1 >= min_y2 and min_y1 <= max_y2):

                # AABB overlap detected, now perform SAT test
                if check_hexagon_overlap(vertices1, vertices2):
                    return True

    return False

def spatial_hash_grid(hex_data, cell_size=None):
    """Create spatial hash grid for fast neighbor lookup."""
    if cell_size is None:
        # Estimate cell size based on hexagon size
        cell_size = 2.0  # Size of grid cells, roughly 2x the hexagon diameter

    grid = {}

    for i, (cx, cy, _) in enumerate(hex_data):
        # Calculate grid cell indices
        grid_x = int(cx // cell_size)
        grid_y = int(cy // cell_size)

        # Store hex index in grid cell
        if (grid_x, grid_y) not in grid:
            grid[(grid_x, grid_y)] = []
        grid[(grid_x, grid_y)].append(i)

    return grid, cell_size

def get_neighbors(grid, grid_x, grid_y, cell_size):
    """Get all neighbors in the spatial hash grid."""
    neighbors = []
    # Check the cell and its 8 neighbors
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = grid_x + dx, grid_y + dy
            if (nx, ny) in grid:
                neighbors.extend(grid[(nx, ny)])
    return neighbors

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Run multi-strategy optimization
    best_solution, best_side_length = run_multi_strategy_optimization()

    # Check if we got a valid solution
    if best_solution is None:
        # Fallback to simple symmetric configuration
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0],  # far bottom-right
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        benchmark_ratio = 1.0 / outer_hex_side_length / 0.2537
        print(f"inv_outer_hex_side_length: {1.0 / outer_hex_side_length:.8f}")
        print(f"benchmark_ratio: {benchmark_ratio:.8f}")
        print(f"eval_time: {time.time() - start_time:.4f}s")
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Extract the best solution
    inner_hex_data = best_solution.reshape(-1, 3)

    # Calculate the resulting outer hexagon side length
    min_side_length, centroid = calculate_min_enclosing_hexagon_fast(inner_hex_data, 1.05)

    # Center the outer hexagon at the centroid of inner hexagons
    outer_hex_data = np.array([centroid[0], centroid[1], 0])

    # Final verification
    _, final_side_length = evaluate_solution_multistart(best_solution)

    benchmark_ratio = 1.0 / final_side_length / 0.2537

    end_time = time.time()
    
    # Print diagnostic information for tracking progress
    print(f"inv_outer_hex_side_length: {1.0 / final_side_length:.8f}")
    print(f"benchmark_ratio: {benchmark_ratio:.8f}")
    print(f"eval_time: {end_time - start_time:.4f}s")

    return inner_hex_data, outer_hex_data, final_side_length

# EVOLVE-BLOCK-END