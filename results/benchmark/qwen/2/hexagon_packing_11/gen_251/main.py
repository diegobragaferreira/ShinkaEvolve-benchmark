# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
from joblib import Parallel, delayed
import multiprocessing as mp

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon as a Shapely Polygon"""
    angle_offset = np.deg2rad(rotation)
    points = []
    for i in range(6):
        angle = angle_offset + i * np.pi/3
        x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
        y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon with robust buffer handling"""
    # Create conservative buffer to handle floating-point precision
    buffered_hexagon = hexagon.buffer(-1e-8)
    try:
        return outer_hexagon.contains(buffered_hexagon)
    except:
        # Fallback for edge cases
        return outer_hexagon.contains(hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap with robust buffer handling"""
    # Use small positive buffer to detect near contacts
    buffered_hex1 = hex1.buffer(1e-8)
    buffered_hex2 = hex2.buffer(1e-8)
    try:
        return buffered_hex1.intersects(buffered_hex2)
    except:
        # Fallback for edge cases
        return hex1.intersects(hex2)

def calculate_outer_hex_radius(inner_params, outer_center=(0,0)):
    """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
    max_dist = 0
    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        center = (x, y)
        dist = np.linalg.norm(np.array(center) - np.array(outer_center))
        # Add distance from center to corner of unit hexagon
        dist += UNIT_HEX_APOGEE
        max_dist = max(max_dist, dist)
    return max_dist

def calculate_tight_outer_radius(inner_params):
    """Calculate tightest possible outer hexagon radius using actual vertex positions"""
    # Get all hexagon vertices and find bounding circle
    all_vertices = []

    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        hexagon = create_unit_hexagon((x, y), angle)
        # Get all vertices of this hexagon
        for point in hexagon.exterior.coords[:-1]:  # exclude closing point
            all_vertices.append(point)

    if not all_vertices:
        return 1.0

    # Convert to numpy array for easier computation
    vertices_array = np.array(all_vertices)

    # Find centroid of all vertices
    centroid = np.mean(vertices_array, axis=0)

    # Calculate distances from centroid to all vertices
    distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))

    # Outer radius is the maximum distance plus a small margin for numerical stability
    outer_radius = np.max(distances) + 1e-6

    return outer_radius

def evaluate_constraints(inner_params, outer_radius):
    """Comprehensive constraint evaluation with early termination"""
    inner_hexagons = []
    
    # Create inner hexagons
    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        hexagon = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(hexagon)
        
    # Create outer hexagon
    outer_hexagon = create_unit_hexagon((0, 0), 0)
    outer_coords = list(outer_hexagon.exterior.coords)
    scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
    outer_hexagon_scaled = Polygon(scaled_coords)
    
    # Check containment (early termination)
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hexagon_scaled):
            return False, False, 0.0  # containment violated
        
    # Check overlaps (early termination)
    for i in range(11):
        for j in range(i+1, 11):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return False, False, 0.0  # overlap violated
                
    # Calculate actual tight radius for better objective function
    actual_tight_radius = calculate_tight_outer_radius(inner_params)
    return True, True, 1.0 / actual_tight_radius  # valid solution

def fitness_function(params):
    """Fitness function that maximizes 1/outer_radius while penalizing constraint violations"""
    # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
    outer_radius = params[-1]
    
    # Extract inner hexagon parameters
    inner_params = params[:-1]
    
    # Check constraints
    containment_ok, overlap_ok, inv_radius = evaluate_constraints(inner_params, outer_radius)
    
    # If any constraint violated, return large penalty
    if not (containment_ok and overlap_ok):
        # Heavy penalty for constraint violations
        return 100000.0 + abs(outer_radius) 
    
    # Return negative of inverse radius to minimize (maximize 1/outer_radius)
    return -inv_radius

def generate_initial_configurations():
    """Generate high-quality initial configurations using multiple strategies"""
    configs = []
    
    # Strategy 1: Honeycomb arrangement with moderate spacing
    honeycomb_positions = [
        (0, 0),           # center
        (-2, 0),          # left
        (2, 0),           # right
        (0, 2),           # top
        (0, -2),          # bottom
        (-1, 1),          # top-left
        (1, 1),           # top-right
        (-1, -1),         # bottom-left
        (1, -1),          # bottom-right
        (-2.5, 1.5),      # far top-left
        (2.5, 1.5),       # far top-right
    ]
    
    # Strategy 2: Optimized clustered arrangement
    clustered_positions = [
        (0, 0),           # center
        (-1.8, 0),        # left
        (1.8, 0),         # right
        (0, 1.8),         # top
        (0, -1.8),        # bottom
        (-1.3, 1.3),      # top-left
        (1.3, 1.3),       # top-right
        (-1.3, -1.3),     # bottom-left
        (1.3, -1.3),      # bottom-right
        (-2.2, 0),        # further left
        (2.2, 0),         # further right
    ]
    
    # Strategy 3: Spiral-like arrangement with wider separation
    spiral_positions = [
        (0, 0),           # center
        (0, 2.2),         # top
        (1.9, 1.1),       # top-right
        (1.9, -1.1),      # bottom-right
        (0, -2.2),        # bottom
        (-1.9, -1.1),     # bottom-left
        (-1.9, 1.1),      # top-left
        (0, 1.6),         # upper-middle
        (0, -1.6),        # lower-middle
        (1.6, 0),         # right-middle
        (-1.6, 0),        # left-middle
    ]
    
    strategies = [honeycomb_positions, clustered_positions, spiral_positions]
    
    # Generate configs from different strategies
    for i, positions in enumerate(strategies):
        for _ in range(10):  # 10 configs per strategy
            config = []
            # Add randomness to positions
            for j, (cx, cy) in enumerate(positions):
                # Add small random variation to avoid exact symmetries
                jitter_x = np.random.normal(0, 0.1)
                jitter_y = np.random.normal(0, 0.1)
                angle = np.random.uniform(0, 360)
                config.extend([cx + jitter_x, cy + jitter_y, angle])
            
            # Add outer radius estimate
            estimated_radius = 0
            for cx, cy in positions:
                dist = np.sqrt(cx**2 + cy**2) + UNIT_HEX_APOGEE
                estimated_radius = max(estimated_radius, dist)
            
            config.append(estimated_radius + 0.3 + np.random.uniform(0, 0.2))
            configs.append(config)
    
    # Add some completely random configurations for diversity
    for _ in range(5):
        config = []
        for _ in range(11):
            # Random positions within a reasonable range
            x = np.random.uniform(-6, 6)
            y = np.random.uniform(-6, 6)
            angle = np.random.uniform(0, 360)
            config.extend([x, y, angle])
        
        # Random outer radius estimate
        config.append(np.random.uniform(3.5, 7.5))
        configs.append(config)
    
    return configs

def refine_with_local_search(initial_params, bounds):
    """Refine solution using local optimization after global search"""
    try:
        # Use L-BFGS-B for fine-tuning with tighter tolerances
        result = minimize(
            fitness_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=lambda x: None
        )
        
        if result.success:
            return result.x
    except Exception as e:
        warnings.warn(f"Local search failed: {str(e)}")
    
    return initial_params

def optimize_single_config(config, bounds):
    """Optimize a single configuration"""
    try:
        # First try differential evolution
        de_result = differential_evolution(
            fitness_function,
            bounds,
            seed=None,
            maxiter=80,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        
        if de_result.success:
            # Refine with local search
            refined_params = refine_with_local_search(de_result.x, bounds)
            
            # Evaluate refined solution
            inner_params = refined_params[:-1]
            outer_radius = refined_params[-1]
            inv_radius, containment_ok, overlap_ok = evaluate_constraints(inner_params, outer_radius)
            
            if containment_ok and overlap_ok and inv_radius > 0.2:
                return refined_params, inv_radius
                
    except Exception as e:
        warnings.warn(f"Single optimization failed: {str(e)}")
    
    return None, 0

def optimize_solution():
    """Main optimization routine using parallel processing"""
    # Generate diverse initial configurations
    initial_configs = generate_initial_configurations()
    
    # Set up bounds for optimization
    bounds = []
    # Bounds for inner hexagon positions and rotations
    for _ in range(11):
        bounds.extend([(-7.0, 7.0), (-7.0, 7.0), (0, 360)])  # x, y, angle
    # Bound for outer radius
    bounds.append((3.0, 9.0))  # Reasonable range for outer radius
    
    # Use parallel processing to evaluate multiple configurations
    n_jobs = min(mp.cpu_count(), 6)  # Limit to 6 jobs for memory efficiency
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(optimize_single_config)(config, bounds) 
        for config in initial_configs
    )
    
    # Find best solution among all attempts
    best_solution = None
    best_inv_radius = 0
    
    for solution, inv_radius in results:
        if solution is not None and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_solution = solution[:]
    
    return best_solution, best_inv_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Run optimization
        final_params, inv_radius = optimize_solution()
        
        if final_params is not None:
            # Extract results
            inner_params = final_params[:-1]
            outer_radius = final_params[-1]
            
            # Validate solution
            containment_ok, overlap_ok, test_inv_radius = evaluate_constraints(inner_params, outer_radius)
            
            if containment_ok and overlap_ok and test_inv_radius > 0.25:
                # Format output
                inner_hex_data = np.zeros((11, 3))
                for i in range(11):
                    inner_hex_data[i] = inner_params[3*i:3*i+3]
                
                outer_hex_data = np.array([0, 0, 0])
                
                return inner_hex_data, outer_hex_data, outer_radius
                
    except Exception as e:
        warnings.warn(f"Error in optimization: {str(e)}")
        pass

    # Fallback to original method if optimization fails
    inner_hex_data = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END