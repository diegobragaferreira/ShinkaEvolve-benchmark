# EVOLVE-BLOCK-START
import numpy as np
import random
from shapely.geometry import Polygon, Point
import math
from scipy.optimize import minimize
import time

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
    """Check if all vertices of inner hexagon are within outer hexagon with buffer to avoid floating point issues"""
    outer_polygon = Polygon(outer_hex_vertices)
    # Use a small buffer to prevent floating point precision issues
    buffered_outer = outer_polygon.buffer(1e-6)
    for vertex in hex_vertices:
        if not buffered_outer.contains(Point(vertex[0], vertex[1])):
            return False
    return True

def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast bounding box overlap check"""
    # Get bounding boxes
    x1 = [v[0] for v in hex1_vertices]
    y1 = [v[1] for v in hex1_vertices]
    x2 = [v[0] for v in hex2_vertices]
    y2 = [v[1] for v in hex2_vertices]

    min_x1, max_x1 = min(x1), max(x1)
    min_y1, max_y1 = min(y1), max(y1)
    min_x2, max_x2 = min(x2), max(x2)
    min_y2, max_y2 = min(y2), max(y2)

    # Check if bounding boxes overlap
    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with buffer for precision"""
    # Fast preliminary check
    if not check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
        return False

    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    # Use a small buffer to prevent floating point precision issues
    buffered_poly1 = poly1.buffer(1e-6)
    buffered_poly2 = poly2.buffer(1e-6)
    return buffered_poly1.intersects(buffered_poly2)

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length that contains all inner hexagons using precise geometric calculation"""
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

    # Calculate tight bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Calculate center of bounding box
    bbox_center_x = (min_x + max_x) / 2
    bbox_center_y = (min_y + max_y) / 2

    # Calculate the maximum distance from center to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - bbox_center_x)**2 + (y - bbox_center_y)**2)
        max_dist = max(max_dist, dist)

    # For a regular hexagon, if we know the maximum distance from center to any point
    # we can determine the required side length more accurately
    # The circumscribed circle radius is max_dist
    # For a hexagon, the radius of circumscribed circle equals the side length
    side_length = max_dist

    # Add margin for safe containment and hexagon orientation variations
    side_length *= 1.05

    return side_length

def generate_initial_heuristic_config():
    """Generate an initial high-quality configuration using geometric insights"""
    # Start with a known good arrangement - hexagonal lattice pattern
    # Place 11 hexagons in a pattern inspired by optimal packings
    config = np.array([
        [0.0, 0.0, 0],         # center
        [-1.73, 0.0, 0],       # left
        [1.73, 0.0, 0],        # right
        [0.87, 1.51, 0],       # top-right
        [-0.87, 1.51, 0],      # top-left
        [0.87, -1.51, 0],      # bottom-right
        [-0.87, -1.51, 0],     # bottom-left
        [-2.60, 1.51, 0],      # far top-left
        [2.60, 1.51, 0],       # far top-right
        [-2.60, -1.51, 0],     # far bottom-left
        [2.60, -1.51, 0],      # far bottom-right
    ])
    return config

def evaluate_solution(hex_data):
    """Evaluate fitness of a solution - maximize 1/outer_hex_side_length"""
    # Create outer hexagon vertices (assuming centered at origin)
    outer_side_length = calculate_outer_hex_side_length(hex_data)

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

        # Check overlaps
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]

                hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    total_penalty += 10000  # Large penalty for overlap

        # Return fitness (inverse of outer hex side length + penalties)
        if total_penalty > 0:
            return 1.0 / outer_side_length - total_penalty

        return 1.0 / outer_side_length

    except Exception as e:
        return -10000  # Very poor fitness for invalid solutions

def monte_carlo_hexagon_pack():
    """Monte Carlo optimization approach for hexagon packing"""
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Start with a good initial configuration
    best_solution = generate_initial_heuristic_config()
    best_fitness = evaluate_solution(best_solution)
    
    # Parameters for Monte Carlo optimization
    max_iterations = 10000
    temperature_start = 1.0
    temperature_end = 0.0001
    cooling_rate = 0.9995
    
    # Simulated annealing parameters
    current_solution = best_solution.copy()
    current_fitness = best_fitness
    
    # Track the best solution found so far
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # Temperature schedule
    temperature = temperature_start
    
    # Iterative optimization process
    for iteration in range(max_iterations):
        # Generate candidate solution by perturbing current solution
        candidate_solution = current_solution.copy()
        
        # Perturb one hexagon at a time (randomly select which one)
        hex_to_perturb = random.randint(0, 10)
        
        # Small random perturbations
        delta_x = random.uniform(-0.2, 0.2)
        delta_y = random.uniform(-0.2, 0.2)
        delta_angle = random.uniform(-10, 10)
        
        candidate_solution[hex_to_perturb][0] += delta_x
        candidate_solution[hex_to_perturb][1] += delta_y
        candidate_solution[hex_to_perturb][2] += delta_angle
        
        # Evaluate candidate solution
        candidate_fitness = evaluate_solution(candidate_solution)
        
        # Accept or reject based on Metropolis criterion
        if candidate_fitness > current_fitness:
            # Always accept better solutions
            current_solution = candidate_solution
            current_fitness = candidate_fitness
        else:
            # Accept worse solutions with probability based on temperature
            delta_e = candidate_fitness - current_fitness
            acceptance_prob = math.exp(delta_e / temperature)
            if random.random() < acceptance_prob:
                current_solution = candidate_solution
                current_fitness = candidate_fitness
        
        # Update best solution if necessary
        if current_fitness > best_fitness:
            best_solution = current_solution.copy()
            best_fitness = current_fitness
        
        # Cool down temperature
        temperature = max(temperature_end, temperature * cooling_rate)
        
        # Occasionally perform local refinement
        if iteration % 500 == 0 and iteration > 0:
            # Try to locally optimize using L-BFGS
            try:
                def objective(x_flat):
                    hex_data = x_flat.reshape(-1, 3)
                    return -evaluate_solution(hex_data)  # Negative because we minimize
                
                x0 = current_solution.flatten()
                bounds = [(None, None)] * len(x0)  # Unbounded bounds
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 20})
                
                if result.success:
                    refined_solution = result.x.reshape(-1, 3)
                    refined_fitness = evaluate_solution(refined_solution)
                    if refined_fitness > current_fitness:
                        current_solution = refined_solution
                        current_fitness = refined_fitness
                        if refined_fitness > best_fitness:
                            best_solution = refined_solution.copy()
                            best_fitness = refined_fitness
            except:
                pass  # Continue with standard approach if local optimization fails

    return best_solution, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses Monte Carlo optimization with simulated annealing to find the optimal configuration.
    """
    # Run Monte Carlo optimization
    start_time = time.time()
    best_solution, best_fitness = monte_carlo_hexagon_pack()
    end_time = time.time()
    
    # Calculate final outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(best_solution)
    
    # Return results
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return best_solution, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END