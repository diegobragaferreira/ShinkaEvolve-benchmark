# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from numba import jit
from joblib import Parallel, delayed
import warnings

# Suppress shapely warnings for cleaner output
warnings.filterwarnings('ignore')

@jit(nopython=True)
def hexagon_vertices_vectorized(x_array, y_array, angle_array, side_length=1):
    """Vectorized computation of hexagon vertices for multiple hexagons."""
    # Precomputed hexagon vertices (unit regular hexagon centered at origin)
    base_verts_x = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    base_verts_y = np.array([0.0, np.sqrt(3)/2, np.sqrt(3)/2, 0.0, -np.sqrt(3)/2, -np.sqrt(3)/2])
    
    # Initialize output arrays
    n_hex = len(x_array)
    vertices = np.empty((n_hex, 6, 2))
    
    for i in range(n_hex):
        angle_rad = np.radians(angle_array[i])
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        for j in range(6):
            x_orig = base_verts_x[j]
            y_orig = base_verts_y[j]
            vertices[i, j, 0] = x_array[i] + side_length * (x_orig * cos_a - y_orig * sin_a)
            vertices[i, j, 1] = y_array[i] + side_length * (x_orig * sin_a + y_orig * cos_a)
    
    return vertices

def compute_hexagon_polygons_vectorized(hex_data):
    """Vectorized conversion of hexagon parameters to shapely polygons."""
    n = len(hex_data)
    x_array = hex_data[:, 0]
    y_array = hex_data[:, 1]
    angle_array = hex_data[:, 2]
    
    vertices = hexagon_vertices_vectorized(x_array, y_array, angle_array)
    polygons = []
    
    for i in range(n):
        poly = Polygon(vertices[i])
        polygons.append(poly)
    
    return polygons

def check_containment_parallel(hex_polygons, outer_hex_polygon, n_jobs=-1):
    """Parallel check for containment of all inner hexagons."""
    def check_single_containment(i):
        return outer_hex_polygon.contains(hex_polygons[i]) or outer_hex_polygon.covers(hex_polygons[i])
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(check_single_containment)(i) for i in range(len(hex_polygons))
    )
    return results

def check_overlaps_parallel(hex_polygons, n_jobs=-1):
    """Parallel check for overlaps between all pairs of hexagons."""
    n = len(hex_polygons)
    overlap_matrix = np.zeros((n, n), dtype=bool)
    
    def check_pair_overlap(i, j):
        return hex_polygons[i].intersects(hex_polygons[j]) and not hex_polygons[i].touches(hex_polygons[j])
    
    # Use nested parallelism carefully
    overlap_pairs = []
    for i in range(n):
        for j in range(i):
            overlap_pairs.append((i, j))
    
    def check_single_overlap(pair):
        i, j = pair
        return check_pair_overlap(i, j)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(check_single_overlap)(pair) for pair in overlap_pairs
    )
    
    # Fill overlap matrix
    idx = 0
    for i in range(n):
        for j in range(i):
            overlap_matrix[i, j] = results[idx]
            overlap_matrix[j, i] = results[idx]
            idx += 1
    
    return overlap_matrix

def evaluate_configuration_fast(params):
    """Fast evaluation with parallel constraint checking."""
    # Extract inner hexagon data and outer radius
    inner_hex_data = params[:-1].reshape(12, 3)
    outer_hex_side_length = params[-1]
    
    # Create outer hexagon polygon (centered at origin)
    outer_hex_vertices = hexagon_vertices_vectorized(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), outer_hex_side_length
    )[0]
    outer_hex_poly = Polygon(outer_hex_vertices)
    
    # Vectorized inner hexagon polygons
    try:
        inner_polys = compute_hexagon_polygons_vectorized(inner_hex_data)
    except Exception:
        return 1e10  # Large penalty for geometric errors
    
    # Check containment in parallel
    try:
        containment_results = check_containment_parallel(inner_polys, outer_hex_poly)
        valid_containment = all(containment_results)
    except Exception:
        valid_containment = False
    
    # Check overlaps in parallel
    try:
        overlap_matrix = check_overlaps_parallel(inner_polys)
        valid_overlaps = not np.any(overlap_matrix)
    except Exception:
        valid_overlaps = False
    
    # Validate overall
    if not (valid_containment and valid_overlaps):
        # Calculate total violation penalty
        penalty = 0.0
        
        # Containment penalties
        if not valid_containment:
            for i, poly in enumerate(inner_polys):
                if not (outer_hex_poly.contains(poly) or outer_hex_poly.covers(poly)):
                    try:
                        diff = outer_hex_poly.difference(poly)
                        if hasattr(diff, 'area'):
                            penalty += diff.area * 1000
                    except:
                        penalty += 10000
        
        # Overlap penalties
        if not valid_overlaps:
            for i in range(12):
                for j in range(i):
                    if overlap_matrix[i, j]:
                        try:
                            overlap = inner_polys[i].intersection(inner_polys[j])
                            if hasattr(overlap, 'area'):
                                penalty += overlap.area * 1000
                        except:
                            penalty += 10000
        
        return penalty + 1e6  # Return with large penalty
    
    # If valid, return inverse of outer hexagon side length (negative for minimization)
    return -1.0 / outer_hex_side_length

def generate_symmetric_initial_guess():
    """Generate a mathematically informed initial symmetric configuration."""
    # Based on known optimal hexagon packings with strategic placements
    positions = [
        [0.0, 0.0, 0.0],     # Center
        [1.732, 0.0, 0.0],   # Right
        [-1.732, 0.0, 0.0],  # Left
        [0.0, 2.17, 0.0],    # Top
        [0.0, -2.17, 0.0],   # Bottom
        [0.866, 1.5, 0.0],   # Top-right
        [-0.866, 1.5, 0.0],  # Top-left
        [0.866, -1.5, 0.0],  # Bottom-right
        [-0.866, -1.5, 0.0], # Bottom-left
        [1.732, 3.0, 0.0],   # Far top-right
        [-1.732, 3.0, 0.0],  # Far top-left
        [0.0, -3.5, 0.0],    # Far bottom center
    ]
    
    return np.array(positions)

def optimize_packing():
    """Main optimization function with enhanced hybrid approach."""
    # Generate initial guess
    initial_guess_inner = generate_symmetric_initial_guess()
    
    # Initial estimate for outer radius based on configuration
    max_dist = 0
    for i in range(12):
        x, y, _ = initial_guess_inner[i]
        dist = np.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist)
    
    # Add margin for hexagon size (hexagon has width approximately 2)
    initial_outer_radius = max_dist + 2.5
    
    # Combine into single parameter vector: [12*3 positions + 1 outer radius]
    initial_params = np.concatenate([initial_guess_inner.flatten(), [initial_outer_radius]])
    
    # Define bounds for optimization
    # Positions: x, y bounded to reasonable range, angle 0-360
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    # Outer radius bound (should be positive)
    bounds.append((0.1, 20.0))
    
    # First, use global optimization to explore the space broadly
    def objective_global(params):
        return evaluate_configuration_fast(params)
    
    # Run differential evolution with more aggressive settings for speed
    try:
        de_result = differential_evolution(
            objective_global,
            bounds,
            maxiter=30,  # Reduced iterations for speed
            popsize=12,   # Smaller population for faster convergence
            seed=42,
            disp=False,
            tol=1e-6
        )
        best_params = de_result.x
    except Exception:
        # Fallback to initial guess if DE fails
        best_params = initial_params.copy()
    
    # Local refinement using L-BFGS-B
    try:
        local_result = minimize(
            evaluate_configuration_fast,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'disp': False, 'ftol': 1e-9},
            jac=False  # No analytical gradients
        )
        if local_result.success:
            best_params = local_result.x
    except Exception:
        # Fall back to previous solution if local optimization fails
        pass
    
    # Extract results
    inner_hex_data = best_params[:-1].reshape(12, 3)
    outer_hex_side_length = best_params[-1]
    
    # Ensure the outer hexagon is actually large enough
    # Recalculate to make sure it contains all hexagons
    min_outer_radius = 0
    for i in range(12):
        x, y, _ = inner_hex_data[i]
        dist = np.sqrt(x*x + y*y) + 1.0  # +1 for hexagon radius
        min_outer_radius = max(min_outer_radius, dist)
    
    # If we computed a smaller radius than needed, adjust it up
    if outer_hex_side_length < min_outer_radius:
        outer_hex_side_length = min_outer_radius * 1.05  # Add small margin
    
    return inner_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Track execution time
    start_time = time.time()
    
    try:
        # Get optimized configuration
        inner_hex_data, outer_hex_side_length = optimize_packing()
        
        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])
        
        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length
        
    except Exception as e:
        # Fallback to original configuration if optimization fails
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

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
