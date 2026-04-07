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
    """Calculate minimum outer hexagon side length that contains all inner hexagons using tight bounding"""
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

    # Calculate the diagonal of the bounding box
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    # For a hexagon, we need to ensure all points fit within the hexagon's circumscribed circle
    # The diagonal of the bounding box gives us half the side length needed
    # But we also consider the width and height separately
    diagonal = math.sqrt(bbox_width**2 + bbox_height**2)

    # For a regular hexagon, the relationship between circumradius and side length:
    # Circumradius = side_length
    # We want to fit our bounding box diagonal into the hexagon
    # So the outer hexagon's side length should be at least diagonal/sqrt(3)
    side_length = diagonal / math.sqrt(3)

    # Add a small margin to ensure full containment
    side_length *= 1.1

    return side_length

def generate_hexagon_grid_configurations():
    """Generate multiple grid-based configurations for diverse initialization"""
    configs = []

    # Configuration 1: Simple grid
    config1 = np.array([
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
    configs.append(config1)

    # Configuration 2: Spiral pattern offset
    config2 = np.array([
        [0, 0, 0],  # center
        [-1.5, 0, 0],  # left
        [1.5, 0, 0],  # right
        [-0.75, 1.30, 0],  # top-left
        [0.75, 1.30, 0],  # top-right
        [-0.75, -1.30, 0],  # bottom-left
        [0.75, -1.30, 0],  # bottom-right
        [-2.25, 1.30, 0],  # far top-left
        [2.25, 1.30, 0],  # far top-right
        [-2.25, -1.30, 0],  # far bottom-left
        [2.25, -1.30, 0],  # far bottom-right
    ])
    configs.append(config2)

    # Configuration 3: Hexagonal ring pattern
    config3 = np.array([
        [0, 0, 0],  # center
        [2.0, 0, 0],  # right
        [1.0, 1.73, 0],  # upper-right
        [-1.0, 1.73, 0],  # upper-left
        [-2.0, 0, 0],  # left
        [-1.0, -1.73, 0],  # lower-left
        [1.0, -1.73, 0],  # lower-right
        [3.0, 0, 0],  # far right
        [2.0, 3.46, 0],  # far upper-right
        [-2.0, 3.46, 0],  # far upper-left
        [-3.0, 0, 0],  # far left
    ])
    configs.append(config3)

    # Configuration 4: Optimized spiral arrangement
    config4 = np.array([
        [0, 0, 0],  # center
        [-1.75, 0, 0],  # left
        [1.75, 0, 0],  # right
        [-0.875, 1.51, 0],  # top-left
        [0.875, 1.51, 0],  # top-right
        [-0.875, -1.51, 0],  # bottom-left
        [0.875, -1.51, 0],  # bottom-right
        [-2.625, 1.51, 0],  # far top-left
        [2.625, 1.51, 0],  # far top-right
        [-2.625, -1.51, 0],  # far bottom-left
        [2.625, -1.51, 0],  # far bottom-right
    ])
    configs.append(config4)

    return configs

def generate_spiral_configuration():
    """Generate a spiral-based initial configuration"""
    # Core pattern: 1 center + 6 around + 4 more in outer ring
    centers = [
        [0, 0, 0],      # center
        [-2.0, 0, 0],   # left
        [2.0, 0, 0],    # right
        [0, 2.0, 0],    # top
        [0, -2.0, 0],   # bottom
        [1.73, 1.0, 0], # top-right
        [-1.73, 1.0, 0], # top-left
        [1.73, -1.0, 0], # bottom-right
        [-1.73, -1.0, 0], # bottom-left
        [3.46, 0, 0],   # far right
        [0, 3.46, 0],   # far top
    ]

    # Adjust to match exact count
    centers = centers[:11]
    return np.array(centers)

def generate_random_configuration():
    """Generate a random valid configuration"""
    config = []
    for i in range(11):
        # Random positions within a reasonable area
        x = random.uniform(-5.0, 5.0)
        y = random.uniform(-5.0, 5.0)
        angle = random.uniform(0, 360)
        config.append([x, y, angle])
    return np.array(config)

def evaluate_solution(individual):
    """Evaluate fitness of a solution - maximize 1/outer_hex_side_length"""
    # Convert individual to hexagon data
    hex_data = np.array(individual).reshape(-1, 3)

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

        # Check overlaps with early termination
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
            return (1.0 / outer_side_length - total_penalty,),  # Return tuple for DEAP

        return (1.0 / outer_side_length,),  # Maximize 1/outer_side_length

    except Exception as e:
        return (-10000,),  # Very poor fitness for invalid solutions

def initialize_population_with_heuristics(toolbox, num_individuals, num_hexagons):
    """Initialize population with both random and heuristic configurations"""
    population = []

    # Add heuristic configurations
    heuristic_configs = generate_hexagon_grid_configurations()
    for i in range(min(len(heuristic_configs), num_individuals // 3)):
        config = heuristic_configs[i]
        # Add noise to heuristic configurations
        individual = []
        for j in range(num_hexagons):
            x, y, angle = config[j]
            # Add small random noise
            x += random.uniform(-0.3, 0.3)
            y += random.uniform(-0.3, 0.3)
            angle += random.uniform(-20, 20)
            individual.extend([x, y, angle])
        population.append(individual)

    # Add spiral configuration
    spiral_config = generate_spiral_configuration()
    individual = []
    for j in range(num_hexagons):
        x, y, angle = spiral_config[j]
        # Add small random noise
        x += random.uniform(-0.2, 0.2)
        y += random.uniform(-0.2, 0.2)
        angle += random.uniform(-15, 15)
        individual.extend([x, y, angle])
    population.append(individual)

    # Fill remaining slots with random individuals
    while len(population) < num_individuals:
        individual = toolbox.individual()
        population.append(individual)

    return population

def run_single_optimization(toolbox, num_hexagons, max_generations=150):
    """Run a single optimization attempt"""
    try:
        # Generate population with both random and heuristic starting points
        pop = initialize_population_with_heuristics(toolbox, 60, num_hexagons)

        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run the evolutionary algorithm with adaptive mutation
        for generation in range(max_generations):
            # Adapt mutation rate - decrease over generations
            current_mutation_rate = max(0.1, 0.8 * (1 - generation / max_generations))
            toolbox.mutate = tools.mutGaussian(mu=0, sigma=current_mutation_rate, indpb=0.2)

            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.3,
                                              ngen=1, stats=stats, halloffame=hof, verbose=False)

            # Early termination if convergence achieved
            if len(logbook.select("max")) >= 10:
                recent_max = logbook.select("max")[-10:]
                if len(set(recent_max)) == 1:
                    break

        # Get the best individual found
        if len(hof) > 0:
            best_individual = hof[0]
            best_hex_data = np.array(best_individual).reshape(-1, 3)

            # Local refinement with adaptive optimization
            def objective(x_flat):
                # Reshape flat array back to hex data
                hex_data = x_flat.reshape(-1, 3)

                # Calculate outer hexagon side length
                outer_side_length = calculate_outer_hex_side_length(hex_data)

                # Check constraints and apply penalties
                outer_hex_vertices = create_hexagon_vertices(0, 0, 0, outer_side_length)

                # Check containment
                total_penalty = 0
                for i in range(len(hex_data)):
                    center_x, center_y, angle = hex_data[i]
                    inner_hex_vertices = create_hexagon_vertices(center_x, center_y, angle)
                    if not check_hexagon_containment(inner_hex_vertices, outer_hex_vertices):
                        total_penalty += 10000

                # Check overlaps
                for i in range(len(hex_data)):
                    for j in range(i+1, len(hex_data)):
                        center_x1, center_y1, angle1 = hex_data[i]
                        center_x2, center_y2, angle2 = hex_data[j]

                        hex1_vertices = create_hexagon_vertices(center_x1, center_y1, angle1)
                        hex2_vertices = create_hexagon_vertices(center_x2, center_y2, angle2)

                        if check_hexagon_overlap(hex1_vertices, hex2_vertices):
                            total_penalty += 10000

                # Return fitness (inverse of outer hex side length + penalties)
                if total_penalty > 0:
                    return 1.0 / outer_side_length - total_penalty
                return 1.0 / outer_side_length

            # Flatten the best solution for optimization
            x0 = best_hex_data.flatten()

            # Define bounds for optimization (same as original constraints)
            bounds = []
            for i in range(len(x0)):
                if i % 3 == 0:  # x coordinate
                    bounds.append((-6.0, 6.0))
                elif i % 3 == 1:  # y coordinate
                    bounds.append((-6.0, 6.0))
                else:  # angle
                    bounds.append((0.0, 360.0))

            # Apply local refinement with fallback strategies
            try:
                # First try L-BFGS-B (more efficient for smooth functions)
                result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
                refined_solution = result.x.reshape(-1, 3)
                refined_outer_side_length = calculate_outer_hex_side_length(refined_solution)
                refined_fitness = 1.0 / refined_outer_side_length

                # If the improvement is significant, return this result
                if refined_fitness > 1.0 / calculate_outer_hex_side_length(best_hex_data) + 1e-6:
                    return (refined_solution, np.array([0, 0, 0]), refined_outer_side_length, refined_fitness)
            except:
                # If L-BFGS fails, try Nelder-Mead as fallback
                try:
                    result_nm = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 30, 'adaptive': True})
                    refined_solution_nm = result_nm.x.reshape(-1, 3)
                    refined_outer_side_length_nm = calculate_outer_hex_side_length(refined_solution_nm)
                    refined_fitness_nm = 1.0 / refined_outer_side_length_nm

                    # Return the better of L-BFGS and Nelder-Mead results
                    if refined_fitness_nm > 1.0 / calculate_outer_hex_side_length(best_hex_data) + 1e-6:
                        return (refined_solution_nm, np.array([0, 0, 0]), refined_outer_side_length_nm, refined_fitness_nm)
                except:
                    pass

            # If all optimization attempts fail, return the original best solution
            outer_side_length = calculate_outer_hex_side_length(best_hex_data)
            current_fitness = 1.0 / outer_side_length
            return (best_hex_data, np.array([0, 0, 0]), outer_side_length, current_fitness)

    except Exception as e:
        return None

    return None

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find the optimal configuration.
    """
    # Set up evolutionary algorithm
    random.seed(42)
    np.random.seed(42)

    # Define the problem: 11 hexagons, each with (x, y, angle)
    NUM_HEXAGONS = 11
    IND_SIZE = NUM_HEXAGONS * 3  # x, y, angle for each hexagon

    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define gene ranges - more constrained for better optimization
    # x, y: constrained to reasonable bounds based on hexagon size
    # angle: 0 to 360 degrees
    toolbox.register("attr_float_x", random.uniform, -6, 6)
    toolbox.register("attr_float_y", random.uniform, -6, 6)
    toolbox.register("attr_float_angle", random.uniform, 0, 360)

    # Individual is a list of attributes
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: [toolbox.attr_float_x(), toolbox.attr_float_y(), toolbox.attr_float_angle()],
                     n=NUM_HEXAGONS)

    # Register evaluation function
    toolbox.register("evaluate", evaluate_solution)

    # Register genetic operators with tuned parameters
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.8, indpb=0.2)  # Increased sigma initially

    # Use a combination of selection methods for better exploration
    def sel_tournament_fitness(individuals, k):
        # Select based on both tournament and fitness
        selected = tools.selTournament(individuals, k, tournsize=3)
        # Sort by fitness to prefer higher fitness individuals
        selected.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        return selected[:k]

    toolbox.register("select", sel_tournament_fitness)

    # Run multiple optimization attempts in parallel
    results = Parallel(n_jobs=min(4, multiprocessing.cpu_count()))(
        delayed(run_single_optimization)(toolbox, NUM_HEXAGONS, 150)
        for _ in range(4)
    )

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    if valid_results:
        # Find the best result among all attempts
        best_result = max(valid_results, key=lambda x: x[3])  # Compare by fitness
        return best_result[0], best_result[1], best_result[2]
    else:
        # Fallback to original configuration if optimization fails
        print("Using fallback configuration")
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