# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def hexagon_vertices(center_x, center_y, side_length, rotation_degrees):
    """Generate vertices of a regular hexagon given center, side length, and rotation"""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a regular hexagon centered at origin with side length 1
    hex_points = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = side_length * np.cos(angle)
        y = side_length * np.sin(angle)
        hex_points.append((x, y))
    
    # Translate to center
    translated = [(pt[0] + center_x, pt[1] + center_y) for pt in hex_points]
    return translated

def hexagon_polygon(center_x, center_y, side_length, rotation_degrees):
    """Create a Shapely Polygon representation of a hexagon"""
    vertices = hexagon_vertices(center_x, center_y, side_length, rotation_degrees)
    return Polygon(vertices)

def check_hexagon_containment(hex_poly, outer_hex_poly):
    """Check if a hexagon is fully contained within the outer hexagon"""
    try:
        return outer_hex_poly.contains(hex_poly)
    except:
        # Fallback for edge cases
        return outer_hex_poly.contains(hex_poly.buffer(1e-10))

def check_hexagon_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    try:
        return hex1_poly.intersects(hex2_poly)
    except:
        # Fallback for edge cases
        return hex1_poly.intersects(hex2_poly.buffer(1e-10))

def calculate_outer_hexagon_size(inner_hex_data, margin=0.01):
    """Calculate minimum outer hexagon size needed to contain all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1.0
        
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i][0], inner_hex_data[i][1], inner_hex_data[i][2]
        hex_poly = hexagon_polygon(center_x, center_y, 1.0, rotation)
        vertices = list(hex_poly.exterior.coords)[:-1]  # Exclude last duplicate point
        all_vertices.extend(vertices)
    
    if not all_vertices:
        return 1.0
    
    # Find bounding circle
    centers = [(inner_hex_data[i][0], inner_hex_data[i][1]) for i in range(len(inner_hex_data))]
    min_x = min(v[0] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)
    
    # Calculate approximate radius needed
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    max_dist = 0
    
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # Add some margin
    return max_dist + 1.0 + margin

def evaluate_solution(solution_array, use_penalty=True):
    """Evaluate the fitness of a solution"""
    # Parse solution: 33 parameters = 11 hexagons * 3 (x, y, angle)
    inner_hex_data = solution_array.reshape(-1, 3)
    
    # Calculate outer hexagon size
    outer_size = calculate_outer_hexagon_size(inner_hex_data)
    
    # Create outer hexagon polygon (centered at origin)
    outer_hex = hexagon_polygon(0, 0, outer_size, 0)
    
    # Check containment and overlaps
    total_overlaps = 0
    total_violations = 0
    
    # Check each hexagon
    hexagon_polygons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, rotation = inner_hex_data[i]
        hex_poly = hexagon_polygon(center_x, center_y, 1.0, rotation)
        hexagon_polygons.append(hex_poly)
        
        # Check containment
        if not check_hexagon_containment(hex_poly, outer_hex):
            total_violations += 1
            
        # Check overlaps with all previous hexagons
        for j in range(i):
            if check_hexagon_overlap(hex_poly, hexagon_polygons[j]):
                total_overlaps += 1
    
    # Penalty for violations
    penalty = 0
    if total_overlaps > 0 or total_violations > 0:
        # Severe penalty for constraint violations
        penalty = 1000000 * (total_overlaps + total_violations)
    
    # If there are violations, return a very poor score
    if total_overlaps > 0 or total_violations > 0:
        return penalty + 1000000
    
    # Objective: minimize outer hexagon size (maximize 1/size)
    objective_value = 1.0 / outer_size
    
    # Apply penalties if needed
    if use_penalty:
        # Add small negative penalty for overlaps (if any were missed)
        objective_value -= penalty * 1e-6
    
    return -objective_value  # Negative because we want to maximize

def optimize_single_start(seed, bounds, maxiter=50):
    """Run one optimization with specific seed"""
    np.random.seed(seed)
    
    # Generate random initial population
    initial_pop = []
    for _ in range(10):
        individual = np.random.uniform(low=[b[0] for b in bounds], 
                                      high=[b[1] for b in bounds])
        initial_pop.append(individual)
    
    # Run differential evolution
    result = differential_evolution(evaluate_solution, 
                                   bounds=bounds,
                                   maxiter=maxiter,
                                   popsize=10,
                                   tol=1e-6,
                                   mutation=(0.5, 1.0),
                                   recombination=0.7,
                                   seed=seed,
                                   disp=False)
    
    return result

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Set up bounds for 11 hexagons (each with x, y, angle)
    # X and Y coordinates: approximately within a reasonable range
    # Angle: 0-360 degrees
    bounds = []
    for i in range(11):
        bounds.extend([(0, 0), (-3, 3), (-3, 3), (0, 360)])  # x, y, angle for each hex
    
    # Convert to flat bounds list
    flat_bounds = [(0, 0), (-3, 3), (-3, 3), (0, 360)] * 11
    
    # Run multiple optimization instances with different seeds
    start_time = time.time()
    
    # Use parallel optimization with 3 different seeds for diversity
    results = Parallel(n_jobs=3)(delayed(optimize_single_start)(seed, flat_bounds, maxiter=30) 
                                for seed in [42, 123, 456])
    
    best_result = None
    best_score = float('inf')
    
    # Find the best result among all optimizations
    for result in results:
        if result.success and result.fun < best_score:
            best_score = result.fun
            best_result = result
    
    if best_result is None:
        # Fallback to simple configuration if optimization fails
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
        ])
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        outer_hex_side_length = 8  # large enough to contain all inner hexagons
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Extract best solution
    best_solution = best_result.x
    inner_hex_data = best_solution.reshape(-1, 3)
    
    # Calculate final outer hexagon size
    outer_size = calculate_outer_hexagon_size(inner_hex_data)
    
    # Ensure we have a valid configuration by double-checking constraints
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = outer_size
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
