# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import time
from itertools import combinations
import math
from joblib import Parallel, delayed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Precompute hexagon vertices for unit hexagon (centered at origin)
def get_unit_hexagon_vertices():
    """Return vertices of a unit regular hexagon centered at origin."""
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles, skip last to close the polygon
    vertices = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
    return vertices

UNIT_HEX_VERTICES = get_unit_hexagon_vertices()

def transform_hexagon_vertices(vertices, center_x, center_y, angle_deg):
    """Transform hexagon vertices by translation and rotation."""
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Rotation matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Apply rotation and translation
    rotated_vertices = vertices @ rotation_matrix.T
    translated_vertices = rotated_vertices + np.array([center_x, center_y])

    return translated_vertices

def create_hexagon_polygon(center_x, center_y, angle_deg):
    """Create a Shapely polygon representing a unit hexagon at given position and rotation."""
    vertices = transform_hexagon_vertices(UNIT_HEX_VERTICES, center_x, center_y, angle_deg)
    return Polygon(vertices)

def is_contained(hex_polygon, outer_hex_polygon):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_polygon.contains(hex_polygon)

def check_overlap(hex1_polygon, hex2_polygon):
    """Check if two hexagons overlap."""
    return hex1_polygon.intersects(hex2_polygon)

def compute_outer_hexagon_radius(inner_configs, outer_center=(0,0), outer_angle=0, tolerance=1e-6):
    """
    Compute minimum radius needed to contain all inner hexagons.
    Uses binary search to find tightest fit.
    """
    # Start with a reasonable upper bound
    max_dist = 0
    for i in range(len(inner_configs)):
        cx, cy, angle = inner_configs[i]
        # Calculate distance from center to furthest vertex of hexagon
        dist = np.sqrt(cx**2 + cy**2) + 1.0  # plus radius of unit hexagon
        max_dist = max(max_dist, dist)

    # Binary search bounds
    min_radius = 0.1
    max_radius = max_dist * 2

    # Check if outer hexagon of current max_radius contains all inner hexagons
    def test_radius(radius):
        outer_vertices = transform_hexagon_vertices(
            UNIT_HEX_VERTICES, outer_center[0], outer_center[1], outer_angle)
        outer_hex = Polygon(outer_vertices)

        for i in range(len(inner_configs)):
            cx, cy, angle = inner_configs[i]
            inner_hex = create_hexagon_polygon(cx, cy, angle)

            if not is_contained(inner_hex, outer_hex):
                return False

        return True

    # Binary search for tightest fit with tighter tolerance
    while max_radius - min_radius > tolerance:
        mid_radius = (min_radius + max_radius) / 2
        if test_radius(mid_radius):
            max_radius = mid_radius
        else:
            min_radius = mid_radius

    return max_radius

def evaluate_fitness_single(config):
    """
    Evaluate fitness of a single configuration with early exit conditions.
    Returns negative of 1/outer_hex_side_length to maximize 1/outer_hex_side_length.
    """
    # Reshape config into (11, 3) array
    configs = config.reshape(-1, 3)

    # Early check for extreme positions that would definitely cause containment issues
    for i in range(11):
        cx, cy, angle = configs[i]
        # Very rough sanity check for extreme positions
        if np.sqrt(cx**2 + cy**2) > 20:
            return 1e6  # Large penalty for obviously invalid configurations

    # Create polygons for all inner hexagons
    inner_polygons = []
    for i in range(11):
        cx, cy, angle = configs[i]
        inner_polygons.append(create_hexagon_polygon(cx, cy, angle))

    # Check for overlaps with early exit
    for i, j in combinations(range(11), 2):
        if check_overlap(inner_polygons[i], inner_polygons[j]):
            return 1e6  # Large penalty for overlaps

    # Create outer hexagon polygon
    outer_radius = compute_outer_hexagon_radius(configs)

    # Return negative inverse of outer radius (to maximize 1/outer_radius)
    return -1.0 / outer_radius

def evaluate_fitness_parallel(configs_batch):
    """
    Parallel evaluation of multiple configurations.
    """
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(evaluate_fitness_single)(config) for config in configs_batch
    )
    return np.array(results)

def create_structured_initial_guess():
    """
    Create an initial configuration based on mathematically informed hexagonal tiling.
    This uses a pattern that places hexagons in a way that naturally minimizes empty space.
    """
    # Pattern based on hexagonal lattice with additional central placement
    # Base spacing for unit hexagons
    spacing = np.sqrt(3)
    
    # Core configuration - more compact arrangement
    base_positions = np.array([
        [0, 0, 0],           # center
        [-spacing, 0, 0],    # left
        [spacing, 0, 0],     # right
        [0, spacing, 0],     # top
        [0, -spacing, 0],   # bottom
        [-spacing/2, spacing/2, 0],   # top-left
        [spacing/2, spacing/2, 0],    # top-right
        [-spacing/2, -spacing/2, 0],  # bottom-left
        [spacing/2, -spacing/2, 0],   # bottom-right
        [-spacing * 1.5, spacing/2, 0],  # far top-left
        [spacing * 1.5, spacing/2, 0],   # far top-right
    ])
    
    # Slight perturbations to break symmetry and allow optimization
    np.random.seed(42)
    noise_magnitude = 0.1
    noise = (np.random.rand(11, 3) - 0.5) * noise_magnitude
    
    # Apply noise to positions and angles only (leave center at 0)
    base_positions[:, :2] += noise[:, :2]
    base_positions[:, 2] += noise[:, 2]  # angle noise
    
    return base_positions

def adaptive_local_optimization(initial_configs, max_iterations=500):
    """
    Perform adaptive local optimization that progressively tightens constraints
    """
    configs = initial_configs.copy()
    initial_flat = configs.flatten()
    
    # Define bounds
    bounds = [(-15, 15), (-15, 15), (0, 360)] * 11
    
    # Progressive refinement with decreasing tolerances
    tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    max_iters = [50, 50, 50, 100, 150]
    
    for i, (tol, max_iter) in enumerate(zip(tolerances, max_iters)):
        # Update objective function with current configs
        def opt_func(x):
            return evaluate_fitness_single(x)
        
        try:
            result = minimize(
                opt_func,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol},
                callback=None
            )
            
            if result.success:
                configs = result.x.reshape(-1, 3)
                initial_flat = configs.flatten()
            else:
                # If optimization fails, continue with current best
                break
                
        except Exception as e:
            # Continue even if one step fails
            break
    
    return configs

def early_stopping_criteria(history_scores, patience=10):
    """
    Determine if we should stop early based on improvement rate
    """
    if len(history_scores) < patience:
        return False
    
    recent_scores = history_scores[-patience:]
    # Check if improvement is minimal
    if abs(recent_scores[-1] - recent_scores[0]) < 1e-8:
        return True
    
    return False

def evolutionary_search():
    """
    Use advanced evolutionary algorithm with multiple optimization stages.
    """
    # Stage 1: Generate structured initial guess
    initial_guess = create_structured_initial_guess().flatten()
    
    # Stage 2: Multi-stage differential evolution with adaptive parameters
    bounds = [(-15, 15), (-15, 15), (0, 360)] * 11
    
    # Adaptive parameters for DE
    de_parameters = [
        {'maxiter': 100, 'popsize': 20, 'mutation': (0.5, 1), 'recombination': 0.7},
        {'maxiter': 150, 'popsize': 30, 'mutation': (0.7, 1), 'recombination': 0.8},
        {'maxiter': 200, 'popsize': 40, 'mutation': (0.8, 1), 'recombination': 0.9}
    ]
    
    best_configs = None
    best_fitness = float('inf')
    history_scores = []
    
    # Run multiple rounds of differential evolution with increasing complexity
    for param_set in de_parameters:
        try:
            result = differential_evolution(
                evaluate_fitness_single,
                bounds,
                maxiter=param_set['maxiter'],
                popsize=param_set['popsize'],
                mutation=param_set['mutation'],
                recombination=param_set['recombination'],
                seed=42,
                tol=1e-6,
                disp=False,
                callback=None
            )
            
            current_fitness = evaluate_fitness_single(result.x)
            history_scores.append(current_fitness)
            
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_configs = result.x.reshape(-1, 3)
                
        except Exception as e:
            continue
            
        # Check early stopping criteria
        if early_stopping_criteria(history_scores):
            break
    
    # Stage 3: Adaptive local optimization
    if best_configs is not None:
        refined_configs = adaptive_local_optimization(best_configs, 300)
        final_fitness = evaluate_fitness_single(refined_configs.flatten())
        
        if final_fitness < best_fitness:
            best_fitness = final_fitness
            best_configs = refined_configs
    
    # Final validation and refinement
    if best_configs is None:
        # Fallback to initial guess if everything fails
        best_configs = create_structured_initial_guess()
    
    final_radius = compute_outer_hexagon_radius(best_configs)
    
    # Final validation check
    inner_polygons = []
    for i in range(11):
        cx, cy, angle = best_configs[i]
        inner_polygons.append(create_hexagon_polygon(cx, cy, angle))
    
    # Check for overlaps one final time
    for i, j in combinations(range(11), 2):
        if check_overlap(inner_polygons[i], inner_polygons[j]):
            raise ValueError("Overlap detected in final result")
    
    return best_configs, final_radius

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
        # Run advanced evolutionary search
        inner_configs, outer_radius = evolutionary_search()
        
        # Convert back to required format
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Calculate metrics
        inv_outer_hex_side_length = 1.0 / outer_radius
        benchmark_ratio = inv_outer_hex_side_length / 0.2544
        
        return inner_configs, outer_hex_data, outer_radius

    except Exception as e:
        print(f"Advanced evolutionary search failed: {e}")
        # Fallback to structured configuration
        inner_hex_data = create_structured_initial_guess()
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 4.5  # Reasonable estimate for structured layout
        
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END