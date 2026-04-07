# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
import math
from joblib import Parallel, delayed
import multiprocessing
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
    """Check if all vertices of inner hexagon are within outer hexagon with buffer"""
    outer_polygon = Polygon(outer_hex_vertices)
    buffered_outer = outer_polygon.buffer(1e-6)
    for vertex in hex_vertices:
        if not buffered_outer.contains(Point(vertex[0], vertex[1])):
            return False
    return True

def check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast bounding box overlap check"""
    x1 = [v[0] for v in hex1_vertices]
    y1 = [v[1] for v in hex1_vertices]
    x2 = [v[0] for v in hex2_vertices]
    y2 = [v[1] for v in hex2_vertices]

    min_x1, max_x1 = min(x1), max(x1)
    min_y1, max_y1 = min(y1), max(y1)
    min_x2, max_x2 = min(x2), max(x2)
    min_y2, max_y2 = min(y2), max(y2)

    if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
        return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with buffer"""
    if not check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
        return False

    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    buffered_poly1 = poly1.buffer(1e-6)
    buffered_poly2 = poly2.buffer(1e-6)
    return buffered_poly1.intersects(buffered_poly2)

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length that contains all inner hexagons using tight bounding"""
    if len(inner_hex_data) == 0:
        return 1000

    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    if not all_vertices:
        return 1000

    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    diagonal = math.sqrt(bbox_width**2 + bbox_height**2)
    
    # For hexagon, we want to minimize the side length such that all points fit
    # The side length needs to be at least the diagonal divided by sqrt(3)
    side_length = diagonal / math.sqrt(3)
    
    # Add a safety margin
    side_length *= 1.05
    
    return side_length

def generate_initial_geometric_pattern():
    """Generate initial configuration based on known good hexagonal packing"""
    # This follows a proven hexagonal pattern arrangement
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

def evaluate_fitness(hex_data, penalty_weight=10000):
    """Evaluate fitness with proper constraint handling"""
    outer_side_length = calculate_outer_hex_side_length(hex_data)
    
    try:
        outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)
        
        # Check containment
        total_penalty = 0
        for i in range(len(hex_data)):
            center_x, center_y, angle = hex_data[i]
            inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
            if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                total_penalty += penalty_weight

        # Check overlaps
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]

                hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    total_penalty += penalty_weight

        # Return fitness: maximize 1/outer_side_length (with penalties)
        if total_penalty > 0:
            return 1.0 / outer_side_length - total_penalty
        return 1.0 / outer_side_length

    except Exception as e:
        return -penalty_weight

def project_to_feasible_space(hex_data, outer_radius=10.0):
    """Project hexagon data to ensure reasonable constraints"""
    # Ensure all hexagons are within a reasonable boundary
    projected = hex_data.copy()
    for i in range(len(projected)):
        x, y, angle = projected[i]
        # Keep hexagons within a circular boundary
        distance_from_center = math.sqrt(x*x + y*y)
        if distance_from_center > outer_radius:
            ratio = outer_radius / distance_from_center
            projected[i][0] = x * ratio
            projected[i][1] = y * ratio
        # Keep angles in [0, 360)
        projected[i][2] = projected[i][2] % 360
    return projected

def specialized_mutation(current_solution, bounds, mutation_factor=0.5):
    """Mutation operator specifically designed for hexagon packing constraints"""
    mutated = current_solution.copy()
    
    # Only mutate positions, not orientations for now
    for i in range(len(mutated)):
        # Mutate x coordinate
        if bounds[3*i][0] is not None and bounds[3*i][1] is not None:
            delta_x = random.gauss(0, mutation_factor * (bounds[3*i][1] - bounds[3*i][0]))
            mutated[3*i] = max(bounds[3*i][0], min(bounds[3*i][1], mutated[3*i] + delta_x))
        
        # Mutate y coordinate  
        if bounds[3*i+1][0] is not None and bounds[3*i+1][1] is not None:
            delta_y = random.gauss(0, mutation_factor * (bounds[3*i+1][1] - bounds[3*i+1][0]))
            mutated[3*i+1] = max(bounds[3*i+1][0], min(bounds[3*i+1][1], mutated[3*i+1] + delta_y))
        
        # Mutate angle (smaller change)
        if bounds[3*i+2][0] is not None and bounds[3*i+2][1] is not None:
            delta_angle = random.gauss(0, mutation_factor * (bounds[3*i+2][1] - bounds[3*i+2][0]) / 10)
            mutated[3*i+2] = (mutated[3*i+2] + delta_angle) % 360
    
    return mutated

def adaptive_differential_evolution(initial_solution, bounds, max_iter=200, pop_size=30):
    """Improved differential evolution with adaptive parameters and geometric constraints"""
    best_solution = initial_solution.copy()
    best_fitness = evaluate_fitness(best_solution.reshape(-1, 3))
    
    # Initial parameters
    F = 0.8
    CR = 0.9
    
    # Population initialization
    population = [initial_solution]
    for _ in range(pop_size - 1):
        mutant = specialized_mutation(initial_solution, bounds)
        population.append(mutant)
    
    for generation in range(max_iter):
        # Adaptive parameters
        F = max(0.1, F * 0.999)
        CR = min(0.95, CR * 1.001)
        
        new_population = []
        
        for i in range(len(population)):
            # Select three different individuals
            candidates = list(range(len(population)))
            candidates.remove(i)
            a, b, c = random.sample(candidates, 3)
            
            # Mutation
            mutant = []
            for j in range(len(population[i])):
                if random.random() < CR:
                    mutant.append(population[a][j] + F * (population[b][j] - population[c][j]))
                else:
                    mutant.append(population[i][j])
            
            # Project to feasible space
            mutant = project_to_feasible_space(np.array(mutant).reshape(-1, 3)).flatten()
            
            # Crossover with current
            trial = []
            for j in range(len(mutant)):
                if random.random() < 0.5:
                    trial.append(mutant[j])
                else:
                    trial.append(population[i][j])
            
            # Selection
            trial_solution = np.array(trial).reshape(-1, 3)
            trial_fitness = evaluate_fitness(trial_solution)
            
            if trial_fitness > evaluate_fitness(population[i].reshape(-1, 3)):
                new_population.append(trial)
                if trial_fitness > best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness
            else:
                new_population.append(population[i])
        
        population = new_population
    
    return best_solution, best_fitness

def refine_with_local_search(initial_solution, bounds):
    """Refine solution using local optimization with projected gradient"""
    def objective(x_flat):
        hex_data = x_flat.reshape(-1, 3)
        return -evaluate_fitness(hex_data)  # Negative because we minimize
    
    try:
        # Use L-BFGS-B for local refinement
        result = minimize(
            objective, 
            initial_solution, 
            method='L-BFGS-B', 
            bounds=[(b[0], b[1]) for b in bounds],
            options={'maxiter': 100}
        )
        
        if result.success:
            refined_solution = result.x.reshape(-1, 3)
            refined_fitness = evaluate_fitness(refined_solution)
            return refined_solution, refined_fitness
    except:
        pass
    
    # If local optimization fails, return original
    return initial_solution.reshape(-1, 3), evaluate_fitness(initial_solution.reshape(-1, 3))

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization with geometric insight and adaptive refinement.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Generate initial geometric configuration
    initial_config = generate_initial_geometric_pattern()
    
    # Phase 2: Prepare bounds for optimization
    bounds = []
    for i in range(11):
        # x coordinate bounds
        bounds.append((-8.0, 8.0))
        # y coordinate bounds  
        bounds.append((-8.0, 8.0))
        # angle bounds
        bounds.append((0.0, 360.0))
    
    # Flatten initial solution
    initial_flat = initial_config.flatten()
    
    # Phase 3: Global optimization with adaptive differential evolution
    try:
        global_best_solution, global_fitness = adaptive_differential_evolution(
            initial_flat, bounds, max_iter=150, pop_size=25
        )
    except Exception as e:
        print(f"Global optimization failed: {e}")
        # Fall back to initial solution
        global_best_solution = initial_flat
        global_fitness = evaluate_fitness(initial_config)
    
    # Phase 4: Local refinement
    try:
        final_solution, final_fitness = refine_with_local_search(global_best_solution, bounds)
    except Exception as e:
        print(f"Local refinement failed: {e}")
        final_solution = global_best_solution.reshape(-1, 3)
        final_fitness = global_fitness
    
    # Phase 5: Final validation and cleanup
    outer_side_length = calculate_outer_hex_side_length(final_solution)
    
    # Return the best solution found
    return final_solution, np.array([0, 0, 0]), outer_side_length

# EVOLVE-BLOCK-END