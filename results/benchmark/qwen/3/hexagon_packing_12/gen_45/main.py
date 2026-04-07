# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
import time
from numba import jit, prange
import math
from scipy.optimize import minimize
import copy
from scipy.optimize import differential_evolution

# Constants
UNIT_HEX_RADIUS = 1.0
MAX_EVAL_TIME = 180.0  # seconds
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

def hexagon_to_polygon(x, y, angle_deg, radius=1.0):
    """Convert hexagon parameters to shapely polygon"""
    vertices = get_hexagon_vertices(x, y, angle_deg, radius)
    return Polygon(vertices)

def check_overlap_fast(hex1_poly, hex2_poly):
    """Fast overlap check using bounding boxes"""
    # Quick bounding box check first
    bbox1 = hex1_poly.bounds
    bbox2 = hex2_poly.bounds
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def compute_outer_hexagon_radius(inner_hex_data):
    """Compute minimum outer hexagon radius that contains all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 0.0
    
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        all_vertices.extend(vertices)
    
    if len(all_vertices) == 0:
        return 0.0
    
    # Compute centroid
    centroid_x = np.mean([v[0] for v in all_vertices])
    centroid_y = np.mean([v[1] for v in all_vertices])
    
    # Find maximum distance from centroid to any vertex
    max_distance = 0.0
    for x, y in all_vertices:
        distance = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        max_distance = max(max_distance, distance)
    
    # Add buffer for hexagon radius calculation
    return max_distance + UNIT_HEX_RADIUS

def validate_solution_basic(inner_hex_data):
    """Basic validation without expensive containment checks"""
    if len(inner_hex_data) != 12:
        return False, "Wrong number of hexagons"
    
    # Check for overlaps between any pair of hexagons
    # Use efficient pairwise overlap checking with early exit
    for i in range(len(inner_hex_data)):
        x1, y1, angle1 = inner_hex_data[i]
        hex1_poly = hexagon_to_polygon(x1, y1, angle1)
        
        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            hex2_poly = hexagon_to_polygon(x2, y2, angle2)
            
            if check_overlap_fast(hex1_poly, hex2_poly):
                return False, f"Overlapping hexagons {i} and {j}"
    
    return True, "Valid solution"

def validate_solution_complete(inner_hex_data, outer_hex_data):
    """Complete validation including containment"""
    if len(inner_hex_data) != 12:
        return False, "Wrong number of hexagons"
    
    # Create outer hexagon
    outer_x, outer_y, outer_angle = outer_hex_data
    outer_radius = compute_outer_hexagon_radius(inner_hex_data)
    outer_hex = hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)
    
    # Check each inner hexagon
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        inner_hex = hexagon_to_polygon(x, y, angle)
        
        # Check containment
        if not outer_hex.contains(inner_hex):
            return False, f"Inner hexagon {i} not contained"
        
        # Check overlaps with others
        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            inner_hex2 = hexagon_to_polygon(x2, y2, angle2)
            
            if check_overlap_fast(inner_hex, inner_hex2):
                return False, f"Overlapping hexagons {i} and {j}"
    
    return True, "Valid solution"

def evaluate_fitness_simple(hex_data):
    """Simple fitness evaluation - used for preliminary checks"""
    # Check overlap constraints
    valid, msg = validate_solution_basic(hex_data)
    if not valid:
        return -1e10  # Penalize invalid solutions heavily
    
    # Fitness = 1/outer_radius (higher is better)
    outer_radius = compute_outer_hexagon_radius(hex_data)
    if outer_radius <= 0:
        return -1e10
    
    return 1.0 / outer_radius

def evaluate_fitness_with_penalty(hex_data):
    """Fitness with soft penalties for constraint violations"""
    # Check overlap constraints
    valid, msg = validate_solution_basic(hex_data)
    
    # Fitness = 1/outer_radius + penalty for violations
    outer_radius = compute_outer_hexagon_radius(hex_data)
    if outer_radius <= 0:
        return -1e10
    
    fitness = 1.0 / outer_radius
    
    # Penalty for overlap violations
    if not valid:
        fitness -= 1e6
    
    return fitness

def generate_symmetric_initial_solution():
    """Generate a highly symmetric initial solution"""
    # Hexagonal close packing arrangement
    # Center hexagon
    hex_data = [[0.0, 0.0, 0.0]]
    
    # Surrounding hexagons in 2 layers
    # First layer: 6 hexagons around center
    angles = [0, 60, 120, 180, 240, 300]
    for angle in angles:
        rad = math.radians(angle)
        x = 2.0 * math.cos(rad)
        y = 2.0 * math.sin(rad)
        hex_data.append([x, y, 0.0])
    
    # Second layer: 6 more hexagons
    angles = [30, 90, 150, 210, 270, 330]
    for angle in angles:
        rad = math.radians(angle)
        x = 3.464 * math.cos(rad)  # approx sqrt(12)
        y = 3.464 * math.sin(rad)
        hex_data.append([x, y, 0.0])
    
    # Ensure exactly 12 hexagons
    while len(hex_data) < 12:
        hex_data.append([0.0, 0.0, 0.0])
    hex_data = hex_data[:12]
    
    return np.array(hex_data)

def local_refinement(hex_data, max_iter=100):
    """Refine solution using local optimization"""
    def objective(params):
        # Reshape back to hex_data format
        new_hex_data = params.reshape(-1, 3)
        return -evaluate_fitness_simple(new_hex_data)  # Negative because we minimize
    
    # Flatten for optimization
    initial_flat = hex_data.flatten()
    
    # Optimize only positions (not rotations initially)
    bounds = [(-10.0, 10.0)] * 36  # 12 hexagons * 3 params each
    options = {'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
    
    try:
        result = minimize(
            objective,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options=options,
            tol=1e-8
        )
        
        if result.success:
            refined_data = result.x.reshape(-1, 3)
            valid, _ = validate_solution_basic(refined_data)
            if valid:
                return refined_data
    except:
        pass
    
    return hex_data

def generate_initial_population(size=20):
    """Generate diverse initial population"""
    population = []
    for _ in range(size):
        # Start with symmetric configuration and add perturbations
        base_config = generate_symmetric_initial_solution()
        individual = []
        for x, y, angle in base_config:
            # Add small random perturbations
            new_x = x + random.uniform(-0.5, 0.5)
            new_y = y + random.uniform(-0.5, 0.5)
            new_angle = angle + random.uniform(-10, 10)
            individual.extend([new_x, new_y, new_angle])
        population.append(individual)
    return population

def hexagon_packing_global_optimization():
    """Use differential evolution for global optimization"""
    # Generate diverse initial population
    initial_pop = generate_initial_population(20)
    
    # Flatten initial population for differential evolution
    flat_pop = [ind for ind in initial_pop]
    
    # Define bounds for optimization (positions and angles)
    bounds = []
    for i in range(12):  # 12 hexagons
        bounds.extend([(-10.0, 10.0), (-10.0, 10.0), (0.0, 360.0)])
    
    def objective(params):
        # Reshape parameters to hexagon data
        hex_data = np.array(params).reshape(-1, 3)
        fitness = evaluate_fitness_simple(hex_data)
        return -fitness  # Return negative since we want to maximize
    
    # Run differential evolution for global search
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            popsize=15,
            maxiter=30,
            tol=1e-6,
            recombination=0.7,
            mutation=(0.5, 1.0),
            disp=False
        )
        
        if result.success:
            best_solution = result.x.reshape(-1, 3)
            valid, _ = validate_solution_basic(best_solution)
            if valid:
                return best_solution
    except Exception as e:
        pass
    
    # Fallback to initial population if DE fails
    return generate_symmetric_initial_solution()

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    try:
        # Step 1: Global optimization with differential evolution
        inner_hex_data = hexagon_packing_global_optimization()
        
        # Step 2: Local refinement to improve quality
        inner_hex_data = local_refinement(inner_hex_data, max_iter=50)
        
        # Step 3: Final validation and refinement
        valid, msg = validate_solution_complete(inner_hex_data, [0, 0, 0])
        
        if not valid:
            # Try one more local optimization
            inner_hex_data = local_refinement(inner_hex_data, max_iter=50)
            valid, msg = validate_solution_complete(inner_hex_data, [0, 0, 0])
        
        if not valid:
            # If still invalid, return the symmetric configuration
            inner_hex_data = generate_symmetric_initial_solution()
            valid, msg = validate_solution_complete(inner_hex_data, [0, 0, 0])
            
        # Calculate actual outer hexagon size
        outer_hex_side_length = compute_outer_hexagon_radius(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])
        
    except Exception as e:
        # Fallback to simple solution if everything fails
        print(f"Fallback due to error: {e}")
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
        outer_hex_side_length = 8
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
