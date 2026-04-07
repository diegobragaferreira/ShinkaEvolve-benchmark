# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
from shapely.geometry import Polygon, Point
import math
from joblib import Parallel, delayed
import multiprocessing
from collections import defaultdict

# Memoization cache for expensive computations
_vertex_cache = {}
_distance_cache = {}

def create_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Create vertices of a regular hexagon given center, rotation, and side length with caching"""
    key = (center_x, center_y, angle_deg, side_length)
    if key in _vertex_cache:
        return _vertex_cache[key]
    
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    
    _vertex_cache[key] = vertices
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

def check_hexagon_overlap_sat(hex1_vertices, hex2_vertices):
    """Efficient SAT-based overlap check optimized for regular hexagons"""
    if not check_hexagon_overlap_fast(hex1_vertices, hex2_vertices):
        return False

    # For regular hexagons, we can exploit symmetry to reduce axes to check
    # Only need to check 6 axes (along edges) instead of 12 (6+6) 
    axes = []
    
    # Get normals for first hexagon edges (only 6 unique directions needed for regular hexagon)
    for i in range(6):
        p1 = hex1_vertices[i]
        p2 = hex1_vertices[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        length = math.sqrt(normal[0]**2 + normal[1]**2)
        if length > 1e-10:
            normal = (normal[0]/length, normal[1]/length)
        axes.append(normal)

    # Get normals for second hexagon edges (same approach)
    for i in range(6):
        p1 = hex2_vertices[i]
        p2 = hex2_vertices[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        length = math.sqrt(normal[0]**2 + normal[1]**2)
        if length > 1e-10:
            normal = (normal[0]/length, normal[1]/length)
        axes.append(normal)

    # Check each axis
    for axis in axes:
        proj1 = [p[0]*axis[0] + p[1]*axis[1] for p in hex1_vertices]
        proj2 = [p[0]*axis[0] + p[1]*axis[1] for p in hex2_vertices]

        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        if max1 < min2 or max2 < min1:
            return False

    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using SAT + Shapely fallback for precision"""
    if not check_hexagon_overlap_sat(hex1_vertices, hex2_vertices):
        return False

    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    buffered_poly1 = poly1.buffer(1e-6)
    buffered_poly2 = poly2.buffer(1e-6)
    return buffered_poly1.intersects(buffered_poly2)

def distance_point_to_line(point, line_start, line_end):
    """Compute distance from point to line segment with caching"""
    key = (tuple(point), tuple(line_start), tuple(line_end))
    if key in _distance_cache:
        return _distance_cache[key]
    
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        result = math.sqrt((px - x1)**2 + (py - y1)**2)
    else:
        # Project point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        result = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    _distance_cache[key] = result
    return result

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum outer hexagon side length using geometric insight"""
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

    # Calculate centroid of all vertices
    avg_x = sum(v[0] for v in all_vertices) / len(all_vertices)
    avg_y = sum(v[1] for v in all_vertices) / len(all_vertices)

    # Calculate maximum distance from centroid
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - avg_x)**2 + (y - avg_y)**2)
        max_dist = max(max_dist, dist)

    # Add safety margin - account for hexagon geometry and rotations
    side_length = max_dist * 1.05  # 5% safety margin

    return side_length

def generate_force_directed_initialization():
    """Generate initial configuration using force-directed-like approach"""
    # Start with central hexagon
    config = [[0.0, 0.0, 0.0]]
    
    # Add surrounding hexagons in concentric rings
    ring_positions = [
        # First ring (6 hexagons)
        [2.0, 0.0, 0.0],
        [1.0, 1.73, 0.0],
        [-1.0, 1.73, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, -1.73, 0.0],
        [1.0, -1.73, 0.0]
    ]
    
    # Add outer ring hexagons  
    outer_ring = [
        [3.0, 0.0, 0.0],
        [1.5, 2.59, 0.0],
        [-1.5, 2.59, 0.0],
        [-3.0, 0.0, 0.0],
        [-1.5, -2.59, 0.0],
        [1.5, -2.59, 0.0]
    ]
    
    # Combine to make 11 total
    config.extend(ring_positions[:5])
    config.extend(outer_ring[:5])
    
    # Trim to exactly 11
    config = config[:11]
    
    # Add small random offsets for diversity
    for i in range(len(config)):
        config[i][0] += random.uniform(-0.1, 0.1)
        config[i][1] += random.uniform(-0.1, 0.1)
        config[i][2] += random.uniform(-5, 5)
    
    return np.array(config)

def generate_hexagon_configurations():
    """Generate diverse geometric configurations"""
    configs = []

    # Configuration 1: Central cluster with peripheral
    config1 = np.array([
        [0, 0, 0],          # center
        [-2.0, 0, 0],       # left
        [2.0, 0, 0],        # right
        [0, 2.0, 0],        # top
        [0, -2.0, 0],       # bottom
        [1.73, 1.0, 0],     # top-right
        [-1.73, 1.0, 0],    # top-left
        [1.73, -1.0, 0],    # bottom-right
        [-1.73, -1.0, 0],   # bottom-left
        [-3.0, 0, 0],       # far left
        [3.0, 0, 0],        # far right
    ])
    configs.append(config1)

    # Configuration 2: Spiral pattern
    config2 = np.array([
        [0, 0, 0],          # center
        [-1.8, 0, 0],       # left
        [1.8, 0, 0],        # right
        [0, 1.8, 0],        # top
        [0, -1.8, 0],       # bottom
        [1.5, 1.2, 0],      # top-right
        [-1.5, 1.2, 0],     # top-left
        [1.5, -1.2, 0],     # bottom-right
        [-1.5, -1.2, 0],    # bottom-left
        [-2.5, 1.5, 0],     # far top-left
        [2.5, 1.5, 0],      # far top-right
    ])
    configs.append(config2)

    # Configuration 3: Hexagonal lattice with gaps
    config3 = np.array([
        [0, 0, 0],          # center
        [2.0, 0, 0],        # right
        [1.0, 1.73, 0],     # upper-right
        [-1.0, 1.73, 0],    # upper-left
        [-2.0, 0, 0],       # left
        [-1.0, -1.73, 0],   # lower-left
        [1.0, -1.73, 0],    # lower-right
        [3.0, 0, 0],        # far right
        [2.0, 3.46, 0],     # far upper-right
        [-2.0, 3.46, 0],    # far upper-left
        [-3.0, 0, 0],       # far left
    ])
    configs.append(config3)

    return configs

def evaluate_solution(individual):
    """Enhanced evaluation with early termination and caching"""
    # Convert individual to hexagon data
    hex_data = np.array(individual).reshape(-1, 3)

    # Create outer hexagon vertices (assuming centered at origin)
    outer_side_length = calculate_outer_hex_side_length(hex_data)

    # Check constraints - early termination optimizations
    try:
        outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)

        # Check containment - early exit if violated
        for i in range(len(hex_data)):
            center_x, center_y, angle = hex_data[i]
            inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
            if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                return (1.0 / outer_side_length - 10000,),  # Large penalty

        # Check overlaps with early termination
        for i in range(len(hex_data)):
            for j in range(i+1, len(hex_data)):
                center_x1, center_y1, angle1 = hex_data[i]
                center_x2, center_y2, angle2 = hex_data[j]

                hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                    return (1.0 / outer_side_length - 10000,),  # Large penalty

        return (1.0 / outer_side_length,),  # Maximize 1/outer_side_length

    except Exception as e:
        return (-10000,),  # Very poor fitness for invalid solutions

def run_single_optimization(toolbox, num_hexagons, max_generations=100):
    """Run single optimization with hybrid approach"""
    try:
        # Generate population with diverse starting points
        pop = []
        
        # Add force-directed initialization 
        force_config = generate_force_directed_initialization()
        individual = []
        for j in range(num_hexagons):
            x, y, angle = force_config[j]
            x += random.uniform(-0.2, 0.2)
            y += random.uniform(-0.2, 0.2)
            angle += random.uniform(-10, 10)
            individual.extend([x, y, angle])
        pop.append(individual)
        
        # Add hexagon configurations
        hex_configs = generate_hexagon_configurations()
        for i, config in enumerate(hex_configs[:2]):
            individual = []
            for j in range(num_hexagons):
                x, y, angle = config[j]
                x += random.uniform(-0.3, 0.3)
                y += random.uniform(-0.3, 0.3)
                angle += random.uniform(-20, 20)
                individual.extend([x, y, angle])
            pop.append(individual)
        
        # Fill with random individuals
        while len(pop) < 50:
            individual = toolbox.individual()
            pop.append(individual)

        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run evolutionary algorithm
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3,
                                          ngen=max_generations, stats=stats, halloffame=hof, verbose=False)

        # Get best individual
        if len(hof) > 0:
            best_individual = hof[0]
            best_hex_data = np.array(best_individual).reshape(-1, 3)

            # Hybrid local refinement
            def objective(x_flat):
                hex_data = x_flat.reshape(-1, 3)
                outer_side_length = calculate_outer_hex_side_length(hex_data)
                outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)

                total_penalty = 0
                for i in range(len(hex_data)):
                    center_x, center_y, angle = hex_data[i]
                    inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
                    if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                        total_penalty += 10000

                for i in range(len(hex_data)):
                    for j in range(i+1, len(hex_data)):
                        center_x1, center_y1, angle1 = hex_data[i]
                        center_x2, center_y2, angle2 = hex_data[j]
                        hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                        hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)
                        if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                            total_penalty += 10000

                if total_penalty > 0:
                    return 1.0 / outer_side_length - total_penalty
                return 1.0 / outer_side_length

            x0 = best_hex_data.flatten()
            bounds = []
            for i in range(len(x0)):
                if i % 3 == 0:  # x coordinate
                    bounds.append((-6.0, 6.0))
                elif i % 3 == 1:  # y coordinate
                    bounds.append((-6.0, 6.0))
                else:  # angle
                    bounds.append((0.0, 360.0))

            # Try multiple local optimization methods
            refined_solution = best_hex_data
            refined_fitness = 1.0 / calculate_outer_hex_side_length(best_hex_data)
            
            try:
                # Try L-BFGS-B first
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 30})
                if result.success:
                    refined_solution = result.x.reshape(-1, 3)
                    refined_fitness = 1.0 / calculate_outer_hex_side_length(refined_solution)
            except:
                pass

            # If still no improvement, try Nelder-Mead as fallback  
            if refined_fitness <= 1.0 / calculate_outer_hex_side_length(best_hex_data) + 1e-6:
                try:
                    result_nm = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 30})
                    if result_nm.success:
                        refined_solution_nm = result_nm.x.reshape(-1, 3)
                        refined_fitness_nm = 1.0 / calculate_outer_hex_side_length(refined_solution_nm)
                        if refined_fitness_nm > refined_fitness:
                            refined_solution = refined_solution_nm
                            refined_fitness = refined_fitness_nm
                except:
                    pass
                    
            outer_side_length = calculate_outer_hex_side_length(refined_solution)
            return (refined_solution, np.array([0, 0, 0]), outer_side_length, refined_fitness)

    except Exception as e:
        return None
    
    return None

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses hybrid evolutionary and local optimization to find the optimal configuration.
    """
    # Set up evolutionary algorithm
    random.seed(42)
    np.random.seed(42)

    # Define the problem: 11 hexagons, each with (x, y, angle)
    NUM_HEXAGONS = 11
    IND_SIZE = NUM_HEXAGONS * 3

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Gene ranges
    toolbox.register("attr_float_x", random.uniform, -6, 6)
    toolbox.register("attr_float_y", random.uniform, -6, 6)
    toolbox.register("attr_float_angle", random.uniform, 0, 360)

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: [toolbox.attr_float_x(), toolbox.attr_float_y(), toolbox.attr_float_angle()],
                     n=NUM_HEXAGONS)

    toolbox.register("evaluate", evaluate_solution)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.8, indpb=0.2)
    
    def sel_tournament_fitness(individuals, k):
        selected = tools.selTournament(individuals, k, tournsize=3)
        selected.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        return selected[:k]

    toolbox.register("select", sel_tournament_fitness)

    # Run multiple optimization attempts in parallel with enhanced diversity
    results = Parallel(n_jobs=min(6, multiprocessing.cpu_count()))(
        delayed(run_single_optimization)(toolbox, NUM_HEXAGONS, 100) 
        for _ in range(6)
    )
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x[3])  # Compare by fitness
        return best_result[0], best_result[1], best_result[2]
    else:
        # Fallback to known good configuration
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

# EVOLVE-BLOCK-END