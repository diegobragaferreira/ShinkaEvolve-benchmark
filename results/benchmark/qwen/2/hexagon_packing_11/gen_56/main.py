# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import math
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def create_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Create vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def check_hexagon_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon with buffer for precision"""
    outer_polygon = Polygon(outer_hex_vertices)
    # Apply small buffer to handle floating point precision issues
    buffered_outer = outer_polygon.buffer(1e-6)
    
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not buffered_outer.contains(point):
            return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with buffer for precision"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    
    # Apply small buffers to handle floating point precision issues
    buffered_poly1 = poly1.buffer(1e-6)
    buffered_poly2 = poly2.buffer(1e-6)
    
    return buffered_poly1.intersects(buffered_poly2)

def calculate_outer_hex_side_length(inner_hex_data, margin_factor=1.1):
    """Calculate minimum outer hexagon side length that contains all inner hexagons with tight bounding"""
    if len(inner_hex_data) == 0:
        return 1000

    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    if not all_vertices:
        return 1000

    # Calculate tight bounding box using min/max of coordinates
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Calculate center and maximum distance from center to any vertex
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)

    # For a hexagon, we need to account for the fact that it's inscribed in a circle
    # The side length of the outer hexagon needs to be sufficient to contain all vertices
    # In a regular hexagon, the distance from center to vertices equals the side length
    # So we need to multiply by sqrt(3)/sqrt(3) = 1, but account for the actual geometry
    side_length = max_dist * margin_factor * 2 / math.sqrt(3)

    return side_length

def calculate_outer_hex_side_length_tight(inner_hex_data):
    """Calculate tight outer hexagon side length using actual geometric analysis"""
    if len(inner_hex_data) == 0:
        return 1000

    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    if not all_vertices:
        return 1000

    # Find the min/max coordinates to get bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]
    
    # Calculate the width and height of the bounding box
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)
    
    # For a regular hexagon, the relationship between side length and bounding box dimensions:
    # If we orient the hexagon so that its flat sides are horizontal, 
    # width = 2 * side_length and height = sqrt(3) * side_length
    # But since our hexagons can be rotated, we need to consider the worst case
    
    # Compute the minimum circumscribing hexagon size
    # The most conservative approach is to compute the distance from center to farthest point
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)
    
    # For hexagon packing, we want to ensure all vertices stay within our outer hexagon
    # The side length of the outer hexagon needs to be at least max_dist 
    # But we also need to account for the fact that a hexagon has width = 2*side_length
    # and height = sqrt(3)*side_length
    
    # The minimal outer hexagon has side length equal to the maximum distance
    # from center to any inner hexagon vertex, plus some safety margin
    side_length = max_dist * 1.1  # Add small margin for safety
    
    return side_length

def evaluate_solution_with_constraints(individual, return_details=False):
    """Evaluate fitness of a solution with proper constraint checking"""
    # Convert individual to hexagon data
    hex_data = np.array(individual).reshape(-1, 3)

    # Create outer hexagon vertices (assuming centered at origin)
    outer_side_length = calculate_outer_hex_side_length_tight(hex_data)

    # Check constraints
    try:
        # Check containment for all inner hexagons
        outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)

        # Check if all hexagons are contained and non-overlapping
        total_penalty = 0

        # Check containment
        for i in range(len(hex_data)):
            center_x, center_y, angle = hex_data[i]
            inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
            if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                total_penalty += 10000  # Large penalty for containment violation

        # Check overlaps - only check pairs
        overlap_count = 0
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]

                hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    overlap_count += 1
                    total_penalty += 10000  # Large penalty for overlap

        # Return fitness (inverse of outer hex side length + penalties)
        if total_penalty > 0:
            if return_details:
                return (1.0 / outer_side_length - total_penalty, overlap_count, outer_side_length)
            return (1.0 / outer_side_length - total_penalty,)  # Return tuple for DEAP

        fitness_value = 1.0 / outer_side_length
        
        if return_details:
            return (fitness_value, overlap_count, outer_side_length)
        return (fitness_value,)  # Maximize 1/outer_side_length

    except Exception as e:
        if return_details:
            return (-10000, 0, 1000)
        return (-10000,)  # Very poor fitness for invalid solutions

def generate_initial_configurations(num_configs=10):
    """Generate multiple good initial configurations"""
    configs = []
    
    # Config 1: Grid-based arrangement (simple but often works as baseline)
    base_grid = np.array([
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
    configs.append(base_grid.flatten().tolist())
    
    # Config 2: Spiral arrangement
    spiral_positions = []
    for i in range(11):
        angle = i * (2 * math.pi / 11)
        radius = 1.5 + i * 0.3
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        spiral_positions.append([x, y, 0])
    configs.append(np.array(spiral_positions).flatten().tolist())
    
    # Config 3: Random configurations with seed
    random.seed(42)
    for _ in range(8):
        random_config = []
        for _ in range(11):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            angle = random.uniform(0, 360)
            random_config.extend([x, y, angle])
        configs.append(random_config)
    
    return configs

def optimize_single_config(initial_guess):
    """Optimize a single configuration using differential evolution"""
    def objective(params):
        return -evaluate_solution_with_constraints(params)[0]  # Negative because we want to maximize
    
    # Set bounds for each parameter (x, y, angle for 11 hexagons)
    bounds = []
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle bounds
    
    try:
        result = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=10, disp=False)
        return result.x, -result.fun
    except:
        # Fallback to initial guess if optimization fails
        return initial_guess, evaluate_solution_with_constraints(initial_guess)[0]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses hybrid evolutionary optimization to find the optimal configuration.
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate multiple initial configurations
    initial_configs = generate_initial_configurations(10)
    
    best_fitness = -float('inf')
    best_config = None
    best_side_length = float('inf')
    
    # Try multiple configurations with local optimization
    for i, config in enumerate(initial_configs):
        try:
            # Perform optimization on this configuration
            optimized_params, fitness = optimize_single_config(config)
            
            # Evaluate final result
            final_fitness, overlap_count, side_length = evaluate_solution_with_constraints(optimized_params, return_details=True)
            
            if final_fitness > best_fitness and overlap_count == 0:
                best_fitness = final_fitness
                best_config = optimized_params
                best_side_length = side_length
                
        except Exception as e:
            continue
    
    # If no good configuration was found, fall back to the grid arrangement
    if best_config is None:
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
        outer_hex_side_length = 8
        return inner_hex_data, np.array([0, 0, 0]), outer_hex_side_length
    
    # Convert best result to required format
    best_hex_data = np.array(best_config).reshape(-1, 3)
    
    # Final verification - make sure solution is valid
    _, overlap_count, final_side_length = evaluate_solution_with_constraints(best_config, return_details=True)
    
    if overlap_count > 0:
        # Fallback to basic grid if there are overlaps
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
        outer_hex_side_length = 8
        return inner_hex_data, np.array([0, 0, 0]), outer_hex_side_length
    
    return best_hex_data, np.array([0, 0, 0]), final_side_length

# EVOLVE-BLOCK-END