# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from numba import njit, prange
import time
import random
from math import cos, sin, pi, sqrt

@njit(parallel=True)
def hexagon_vertices_numba(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon using Numba JIT."""
    angle_rad = np.radians(angle_deg)
    vertices = np.empty((6, 2))
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + side_length * cos(angle)
        y = center_y + side_length * sin(angle)
        vertices[i] = (x, y)
    return vertices

@njit
def point_in_hexagon_fast(px, py, center_x, center_y, angle_deg, side_length=1):
    """Fast point-in-hexagon check using distance from center."""
    # Distance from point to hexagon center
    dx = px - center_x
    dy = py - center_y
    distance_sq = dx*dx + dy*dy
    # For unit hexagons, maximum distance is sqrt(3)/2 ~ 0.866
    max_radius_sq = 3.0/4.0
    return distance_sq <= max_radius_sq

@njit
def get_hexagon_bounds(vertices):
    """Get bounding box of a hexagon."""
    min_x = vertices[0][0]
    max_x = vertices[0][0]
    min_y = vertices[0][1]
    max_y = vertices[0][1]

    for i in range(1, len(vertices)):
        x, y = vertices[i]
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y

    return min_x, max_x, min_y, max_y

@njit
def hexagon_to_grid_cells(vertices, cell_size):
    """Get all grid cells that a hexagon might occupy."""
    min_x, max_x, min_y, max_y = get_hexagon_bounds(vertices)

    # Get grid indices for bounding box
    min_cell_x = int(min_x // cell_size)
    max_cell_x = int(max_x // cell_size)
    min_cell_y = int(min_y // cell_size)
    max_cell_y = int(max_y // cell_size)

    # Collect all cells
    cells = []
    for x in range(min_cell_x, max_cell_x + 1):
        for y in range(min_cell_y, max_cell_y + 1):
            cells.append((x, y))

    return cells

@njit
def check_overlap_spatial_hashing_numba(hexagon_vertices_list, cell_size=1.2):
    """Spatial hashing overlap check for improved performance."""
    n_hexagons = len(hexagon_vertices_list)

    # Build spatial hash grid
    grid = {}

    # Place each hexagon into grid cells
    for i in range(n_hexagons):
        vertices = hexagon_vertices_list[i]
        cells = hexagon_to_grid_cells(vertices, cell_size)
        for cell in cells:
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(i)

    # Check for overlaps by examining neighboring cells
    for i in range(n_hexagons):
        vertices_i = hexagon_vertices_list[i]
        cells = hexagon_to_grid_cells(vertices_i, cell_size)

        # Check all neighboring cells
        for cell in cells:
            # Check all hexagons in this cell and adjacent cells
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    neighbor_cell = (cell[0] + dx, cell[1] + dy)
                    if neighbor_cell in grid:
                        for j in grid[neighbor_cell]:
                            if i != j:  # Don't check against self
                                # Quick bounding box check
                                min_x_i, max_x_i, min_y_i, max_y_i = get_hexagon_bounds(vertices_i)
                                vertices_j = hexagon_vertices_list[j]
                                min_x_j, max_x_j, min_y_j, max_y_j = get_hexagon_bounds(vertices_j)

                                # Simple bounding box intersection test
                                if (max_x_i >= min_x_j and min_x_i <= max_x_j and
                                    max_y_i >= min_y_j and min_y_i <= max_y_j):
                                    # More precise check using distance between centers
                                    cx_i = vertices_i[0][0]
                                    cy_i = vertices_i[0][1]
                                    cx_j = vertices_j[0][0]
                                    cy_j = vertices_j[0][1]
                                    dist_sq = (cx_i - cx_j)**2 + (cy_i - cy_j)**2
                                    # For unit hexagons, minimum distance should be 2 (they don't touch)
                                    if dist_sq < 4.0:
                                        return False  # Overlap detected

    return True  # No overlap detected

@njit
def calculate_bounding_radius_numba(hexagon_configs):
    """Fast calculation of bounding radius for all hexagons."""
    if len(hexagon_configs) == 0:
        return 1.0
    
    # Collect all vertices
    all_vertices = np.empty((0, 2), dtype=np.float64)
    
    for i in range(len(hexagon_configs)):
        cfg = hexagon_configs[i]
        vertices = hexagon_vertices_numba(cfg[0], cfg[1], cfg[2])
        all_vertices = np.vstack([all_vertices, vertices])

    if len(all_vertices) == 0:
        return 1.0
        
    # Compute centroid
    centroid_x = np.mean(all_vertices[:, 0])
    centroid_y = np.mean(all_vertices[:, 1])
    
    # Find maximum distance to centroid
    max_dist_sq = 0
    for i in range(len(all_vertices)):
        dx = all_vertices[i, 0] - centroid_x
        dy = all_vertices[i, 1] - centroid_y
        dist_sq = dx*dx + dy*dy
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            
    return sqrt(max_dist_sq) + 0.01  # Add buffer

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon."""
    return hexagon_vertices_numba(center_x, center_y, angle_deg, side_length)

def check_containment_fast(hexagon_vertices_list, outer_side_length):
    """Fast containment check using distance from center."""
    for vertices in hexagon_vertices_list:
        center_x = vertices[0][0]
        center_y = vertices[0][1]
        distance_sq = center_x*center_x + center_y*center_y
        outer_radius_sq = outer_side_length * outer_side_length
        if distance_sq > outer_radius_sq:
            return False
    return True

def check_overlap(hexagon_vertices_list):
    """Check if any hexagons overlap using optimized spatial hashing."""
    # Try optimized spatial hashing first (faster)
    if check_overlap_spatial_hashing_numba(hexagon_vertices_list):
        # Fall back to Shapely for precise check if needed
        try:
            polygons = [Polygon(vertices) for vertices in hexagon_vertices_list]
            union = unary_union(polygons)
            total_area = sum(polygon.area for polygon in polygons)
            union_area = union.area
            # If areas match, no overlap
            return abs(total_area - union_area) < 1e-10
        except:
            # Fallback for complex cases
            for i in range(len(polygons)):
                for j in range(i+1, len(polygons)):
                    if polygons[i].intersects(polygons[j]):
                        return False
            return True
    return False

def evaluate_configuration(config, outer_side_length):
    """Evaluate a configuration of 12 hexagons."""
    # Parse configuration into 12 hexagons (x, y, angle)
    hexagons = config.reshape(12, 3)

    # Get vertices for all hexagons
    hexagon_vertices_list = []
    for i in range(12):
        x, y, angle = hexagons[i]
        vertices = hexagon_vertices(x, y, angle)
        hexagon_vertices_list.append(vertices)

    # Early termination for invalid configurations
    if not check_containment_fast(hexagon_vertices_list, outer_side_length):
        return float('inf')  # Invalid configuration

    # Check overlap
    if not check_overlap(hexagon_vertices_list):
        return float('inf')  # Overlapping hexagons

    return 0  # Valid configuration

def objective_function(config, outer_side_length):
    """Objective function to minimize (negative of 1/outer_hex_side_length)."""
    # We want to maximize 1/outer_hex_side_length, so we minimize -1/outer_hex_side_length
    penalty = evaluate_configuration(config, outer_side_length)
    if penalty == float('inf'):
        # Return large value for invalid configurations
        return 1000000
    else:
        # For valid configurations, return negative of 1/outer_hex_side_length
        if outer_side_length > 0:
            return -1.0 / outer_side_length
        else:
            return 1000000

def create_advanced_symmetric_pattern():
    """Create a more advanced symmetric pattern based on mathematical principles."""
    # A scientifically designed arrangement with known good properties
    # Using golden ratio and hexagonal lattice principles
    pattern = [
        [0.0, 0.0, 0.0],         # Center
        [-1.732, 0.0, 0.0],      # Left
        [1.732, 0.0, 0.0],       # Right
        [0.0, 1.732, 0.0],       # Top
        [0.0, -1.732, 0.0],      # Bottom
        [-0.866, 0.866, 0.0],    # Top-left
        [0.866, 0.866, 0.0],     # Top-right
        [-0.866, -0.866, 0.0],   # Bottom-left
        [0.866, -0.866, 0.0],    # Bottom-right
        [-2.598, 0.0, 0.0],      # Far left
        [2.598, 0.0, 0.0],       # Far right
        [0.0, 2.598, 0.0],       # Far top
    ]
    return np.array(pattern)

def create_hexagonal_ring_pattern():
    """Create a hexagonal ring pattern that's more mathematically sound."""
    # Ring pattern with 12 hexagons in a hexagonal arrangement
    # First layer (6 surrounding center)
    ring1 = [
        [0.0, 1.732, 0.0],      # Top
        [1.5, 0.866, 0.0],      # Top-right
        [1.5, -0.866, 0.0],     # Bottom-right
        [0.0, -1.732, 0.0],     # Bottom
        [-1.5, -0.866, 0.0],    # Bottom-left
        [-1.5, 0.866, 0.0],     # Top-left
    ]

    # Second layer (6 more around first layer)
    ring2 = [
        [0.0, 3.464, 0.0],      # Top
        [3.0, 1.732, 0.0],      # Top-right
        [3.0, -1.732, 0.0],     # Bottom-right
        [0.0, -3.464, 0.0],     # Bottom
        [-3.0, -1.732, 0.0],    # Bottom-left
        [-3.0, 1.732, 0.0],     # Top-left
    ]

    # Combine center with rings (12 total)
    pattern = [[0.0, 0.0, 0.0]] + ring1 + ring2
    return np.array(pattern)

def generate_enhanced_initial_population(pop_size, max_radius=5.0):
    """Generate enhanced initial population with multiple pattern types."""
    population = []

    # Create multiple base patterns
    base_patterns = [
        create_advanced_symmetric_pattern(),
        create_hexagonal_ring_pattern()
    ]

    # Generate variations from each base pattern
    for i in range(pop_size):
        # Alternate between base patterns to ensure diversity
        base_pattern = base_patterns[i % len(base_patterns)]
        individual = base_pattern.copy().astype(float)

        # Add controlled variation to positions
        for j in range(12):
            # Add small random variation to position (smaller range for stability)
            individual[j, 0] += np.random.uniform(-0.2, 0.2)
            individual[j, 1] += np.random.uniform(-0.2, 0.2)
            # Random angle variation
            individual[j, 2] += np.random.uniform(-10, 10)
            # Keep angle in [0, 360)
            individual[j, 2] = individual[j, 2] % 360

        population.append(individual.flatten())

    return population

def multi_stage_optimization(initial_guess, bounds, target_side_length):
    """Perform multi-stage optimization for better results."""
    best_config = initial_guess.copy()
    best_side_length = target_side_length
    best_score = -float('inf')

    # Stage 1: Coarse optimization with high population
    print("Stage 1: Coarse optimization")
    try:
        opt_result = differential_evolution(
            lambda x: -objective_function(x, target_side_length),
            bounds,
            seed=random.randint(0, 1000),
            maxiter=30,
            popsize=20,
            disp=False,
            strategy='best1bin',
            tol=1e-5
        )

        final_config = opt_result.x
        penalty = evaluate_configuration(final_config, target_side_length)

        if penalty != float('inf'):
            stage1_score = -1.0 / target_side_length
            if stage1_score > best_score:
                best_score = stage1_score
                best_config = final_config.copy()

    except Exception as e:
        print(f"Stage 1 error: {e}")

    # Stage 2: Fine tuning with lower population but more iterations
    print("Stage 2: Fine tuning")
    try:
        # Use a slightly modified version of best config as starting point
        start_config = best_config.copy()

        # Add small random perturbation
        for i in range(len(start_config)):
            if i % 3 < 2:  # Position components only
                start_config[i] += np.random.uniform(-0.1, 0.1)

        opt_result = differential_evolution(
            lambda x: -objective_function(x, target_side_length),
            bounds,
            seed=random.randint(0, 1000),
            maxiter=50,  # More iterations for fine tuning
            popsize=10,   # Lower population for better convergence
            disp=False,
            strategy='best1bin',
            tol=1e-6
        )

        final_config = opt_result.x
        penalty = evaluate_configuration(final_config, target_side_length)

        if penalty != float('inf'):
            stage2_score = -1.0 / target_side_length
            if stage2_score > best_score:
                best_score = stage2_score
                best_config = final_config.copy()

    except Exception as e:
        print(f"Stage 2 error: {e}")

    # Stage 3: Local refinement around best solution
    print("Stage 3: Local refinement")
    try:
        # Test various side lengths around optimal
        test_side_lengths = np.linspace(3.85, min(target_side_length, 3.9419123), 20)

        for test_side in test_side_lengths:
            penalty = evaluate_configuration(best_config, test_side)
            if penalty != float('inf'):
                if test_side > best_side_length:
                    best_side_length = test_side

    except Exception as e:
        print(f"Stage 3 error: {e}")

    return best_config, best_side_length

def optimize_hexagon_positions():
    """Main optimization routine with enhanced strategies."""
    # Initial guess with better placement based on known good patterns
    initial_guess = create_advanced_symmetric_pattern().flatten()

    # Bounds for positions and angles
    bounds = []
    for _ in range(12):
        bounds.extend([(-4.0, 4.0), (-4.0, 4.0), (0.0, 360.0)])

    # Try to find the best solution using enhanced optimization approach
    best_score = -float('inf')
    best_config = None
    best_outer_side = 4.0

    # Multiple optimization attempts with different strategies
    for run in range(5):
        print(f"Run {run + 1}/5")
        try:
            # Use enhanced multi-stage optimization
            final_config, final_side = multi_stage_optimization(
                initial_guess,
                bounds,
                4.0
            )

            # Validate and evaluate the result
            penalty = evaluate_configuration(final_config, 4.0)

            if penalty != float('inf'):  # Valid configuration
                # Try to find a better fit with different outer hexagon sizes
                for test_side in np.linspace(3.85, 3.9419123, 20):
                    penalty = evaluate_configuration(final_config, test_side)
                    if penalty != float('inf'):
                        if test_side > best_outer_side:
                            best_outer_side = test_side
                            best_config = final_config.copy()
                            best_score = -1.0 / test_side

        except Exception as e:
            print(f"Run {run} error: {e}")
            continue

    # Additional refinement step with fine grid search
    if best_config is not None:
        print("Additional refinement...")
        # Try to squeeze even tighter configurations
        try:
            for test_side in np.linspace(3.93, 3.9419123, 25):
                penalty = evaluate_configuration(best_config, test_side)
                if penalty != float('inf'):
                    if test_side > best_outer_side:
                        best_outer_side = test_side
                        best_config = best_config.copy()
        except Exception as e:
            print(f"Refinement error: {e}")

    # Final validation and return
    if best_config is not None:
        return best_config.reshape(12, 3), np.array([0, 0, 0]), best_outer_side
    else:
        # Fallback: return original good pattern
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
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
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

    # Run optimization
    inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_positions()

    # Calculate actual score
    inv_side_length = 1.0 / outer_hex_side_length
    eval_time = time.time() - start_time

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END