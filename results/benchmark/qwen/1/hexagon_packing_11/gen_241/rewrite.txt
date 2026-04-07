# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random
import time
from itertools import product

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)
    return max_dist

def check_containment(hex_poly, outer_polygon):
    """Check if hexagon is fully contained in outer hexagon"""
    return outer_polygon.contains(hex_poly)

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using shapely"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        return False

def build_spatial_hash(hex_polygons, grid_size=2.5):
    """Build spatial hash for fast overlap detection"""
    grid = {}
    for i, hex_poly in enumerate(hex_polygons):
        min_x, min_y, max_x, max_y = hex_poly.bounds
        min_grid_x = int(min_x // grid_size)
        max_grid_x = int(max_x // grid_size)
        min_grid_y = int(min_y // grid_size)
        max_grid_y = int(max_y // grid_size)
        for gx in range(min_grid_x, max_grid_x + 1):
            for gy in range(min_grid_y, max_grid_y + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)
    return grid

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Fast validation using spatial hashing"""
    # Precompute hexagon polygons
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))

    # Calculate outer radius
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Check containment
    for hex_poly in hex_polygons:
        for vertex in hex_poly.exterior.coords[:-1]:
            if not outer_polygon.contains(Point(vertex)):
                return False

    # Fast overlap check using spatial hash
    grid = build_spatial_hash(hex_polygons)
    
    for i in range(len(hex_polygons)):
        min_x, min_y, max_x, max_y = hex_polygons[i].bounds
        min_grid_x = int(min_x // 2.5)
        max_grid_x = int(max_x // 2.5)
        min_grid_y = int(min_y // 2.5)
        max_grid_y = int(max_y // 2.5)
        
        for gx in range(min_grid_x - 1, max_grid_x + 2):
            for gy in range(min_grid_y - 1, max_grid_y + 2):
                if (gx, gy) in grid:
                    for j in grid[(gx, gy)]:
                        if i < j:
                            if hex_polygons[i].intersects(hex_polygons[j]):
                                return False
    return True

def evaluate_fitness(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness (negative of outer hexagon radius for maximization)"""
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)
    
    if not validate_solution(inner_hex_data, outer_center, outer_angle):
        return -1e10
        
    return -outer_radius

def generate_lattice_points():
    """Generate discrete lattice points for exploration"""
    x_range = np.arange(-6, 7, 0.5)
    y_range = np.arange(-6, 7, 0.5)
    rot_range = np.arange(0, 360, 30)  
    lattice_points = list(product(x_range, y_range, rot_range))
    return np.array(lattice_points)

def construct_initial_config():
    """Construct diverse initial configurations"""
    configs = []
    
    # Pattern 1: Hexagonal packing
    base_positions = [
        [0, 0, 0],           # center
        [-2.5, 0, 0],       # left
        [2.5, 0, 0],        # right
        [-1.25, 2.17, 0],   # top-left
        [1.25, 2.17, 0],    # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],   # bottom-right
        [-3.75, 2.17, 0],   # far top-left
        [3.75, 2.17, 0],    # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],   # far bottom-right
    ]
    
    # Add noise
    config = []
    for pos in base_positions:
        x = pos[0] + random.uniform(-0.3, 0.3)
        y = pos[1] + random.uniform(-0.3, 0.3)
        angle = pos[2] + random.uniform(-10, 10)
        config.append([x, y, angle])
    configs.append(np.array(config))
    
    # Pattern 2: Spiral arrangement
    spiral_config = []
    for i in range(11):
        angle = i * 30
        radius = 1.5 + i * 0.3
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        spiral_config.append([x, y, random.uniform(0, 360)])
    configs.append(np.array(spiral_config))
    
    # Pattern 3: Grid arrangement
    grid_config = []
    grid_positions = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if len(grid_positions) < 11:
                grid_positions.append([i * 2.5, j * 2.5, random.uniform(0, 360)])
    for pos in grid_positions:
        grid_config.append([pos[0], pos[1], pos[2]])
    configs.append(np.array(grid_config))
    
    return configs

def local_optimization(initial_solution):
    """Refine solution using L-BFGS optimization"""
    def objective(params):
        # Reshape params back into hexagon data
        hex_data = params.reshape(-1, 3)
        return -evaluate_fitness(hex_data)
    
    def constraint_func(params):
        # Ensure all positions stay within reasonable bounds
        hex_data = params.reshape(-1, 3)
        bounds_violation = 0
        for i in range(len(hex_data)):
            if abs(hex_data[i][0]) > 10 or abs(hex_data[i][1]) > 10:
                bounds_violation += 1000
        return bounds_violation
    
    # Flatten initial solution
    flat_solution = initial_solution.flatten()
    
    # Optimize
    result = minimize(
        objective,
        flat_solution,
        method='L-BFGS-B',
        bounds=[(-10, 10), (-10, 10), (0, 360)] * 11,
        options={'maxiter': 100, 'ftol': 1e-6},
        callback=None
    )
    
    # Reshape result
    optimized_solution = result.x.reshape(-1, 3)
    return optimized_solution

def hexagon_lattice_optimization():
    """Primary optimization algorithm using lattice search and local refinement"""
    start_time = time.time()
    
    # Generate initial configurations
    initial_configs = construct_initial_config()
    
    best_fitness = float('inf')
    best_solution = None
    
    # Search through lattice points
    lattice_points = generate_lattice_points()
    
    # Try different initial configurations first
    for initial_config in initial_configs:
        if time.time() - start_time > MAX_EVAL_TIME - 1:
            break
            
        # Local optimization of initial configuration
        refined = local_optimization(initial_config)
        fitness = evaluate_fitness(refined)
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_solution = refined.copy()
    
    # Explore lattice points systematically
    for _ in range(500):  # Limit lattice exploration
        if time.time() - start_time > MAX_EVAL_TIME - 1:
            break
            
        # Sample from lattice
        sample_config = []
        for i in range(11):
            if len(lattice_points) > 0:
                idx = random.randint(0, min(100, len(lattice_points)-1))
                sample_config.append(lattice_points[idx])
            else:
                sample_config.append([random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(0, 360)])
        
        sample_array = np.array(sample_config)
        
        # Quick validation
        if validate_solution(sample_array):
            # Local optimization
            refined = local_optimization(sample_array)
            fitness = evaluate_fitness(refined)
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = refined.copy()
    
    # Final local optimization on best solution
    if best_solution is not None:
        final_solution = local_optimization(best_solution)
        final_fitness = evaluate_fitness(final_solution)
        if final_fitness < best_fitness:
            best_fitness = final_fitness
            best_solution = final_solution
    
    return best_solution if best_solution is not None else np.array([
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
        [3.75, -2.17, 0]
    ])

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Run lattice optimization
    best_solution = hexagon_lattice_optimization()
    
    # Final validation
    if not validate_solution(best_solution):
        # Fallback to standard configuration
        best_solution = np.array([
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
            [3.75, -2.17, 0]
        ])
    
    # Calculate optimal outer hexagon size
    outer_radius = -best_fitness if 'best_fitness' in locals() else calculate_outer_hexagon_radius(best_solution)
    
    # Return results
    inner_hex_data = best_solution
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END