# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import random
import time
from numba import jit
from deap import base, creator, tools, algorithms
from joblib import Parallel, delayed
from scipy.optimize import minimize

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6
ROTATION_STEPS = 12  # 30 degree increments

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a regular hexagon given position and angle - JIT compiled"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment - JIT compiled"""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1

    # Length squared of line segment
    length_sq = dx*dx + dy*dy

    # Handle degenerate case
    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line segment
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp t to [0, 1]

    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def estimate_min_outer_radius(inner_hex_data):
    """Estimate minimal outer hexagon radius that can contain all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1000.0

    # Get all vertices
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        for vertex in vertices:
            all_vertices.append(vertex)

    all_vertices = np.array(all_vertices)

    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

    # Calculate center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Find maximum distance from center to any vertex
    distances = np.sqrt((all_vertices[:, 0] - center_x)**2 + (all_vertices[:, 1] - center_y)**2)
    max_distance = np.max(distances)

    # Add small margin to account for hexagon shape
    return max_distance * 1.01  # 1% margin for safety

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 1000.0

    max_distance = 0.0
    center_x, center_y = outer_hex_center

    # For each inner hexagon, check all 6 vertices
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)

        # Calculate distance from center to each vertex
        for vertex in vertices:
            distance = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
            max_distance = max(max_distance, distance)

    # Account for hexagon radius
    # The outer hexagon needs to be large enough so that any vertex of inner hexagons
    # lies inside the outer hexagon
    return max_distance * 2.0 / np.sqrt(3)  # Convert circumradius to side length

def check_containment(hex_vertices, outer_center=(0, 0), outer_radius=1000.0):
    """Check if hexagon vertices are within the outer hexagon"""
    outer_center_x, outer_center_y = outer_center
    # Check if all vertices are within the outer hexagon
    # Outer hexagon circumscribed circle has radius = outer_radius * sqrt(3)/2
    outer_circumradius = outer_radius * np.sqrt(3) / 2

    for vertex in hex_vertices:
        dist_from_center = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
        if dist_from_center > outer_circumradius:
            return False
    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using bounding box pre-check then detailed check"""
    # Fast bounding box check
    min1_x = min(v[0] for v in hex1_vertices)
    max1_x = max(v[0] for v in hex1_vertices)
    min1_y = min(v[1] for v in hex1_vertices)
    max1_y = max(v[1] for v in hex1_vertices)

    min2_x = min(v[0] for v in hex2_vertices)
    max2_x = max(v[0] for v in hex2_vertices)
    min2_y = min(v[1] for v in hex2_vertices)
    max2_y = max(v[1] for v in hex2_vertices)

    # If bounding boxes don't overlap, no collision
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False

    # Detailed check using Shapely
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        return False

def evaluate_individual_parallel(individual):
    """Parallel evaluation with improved error handling"""
    try:
        # Reshape individual into (11, 3) array of (x, y, angle)
        hex_data = np.array(individual).reshape(-1, 3)

        # Calculate required outer hex side length
        outer_side_length = calculate_outer_hex_side_length(hex_data)

        # Initialize penalty
        penalty = 0.0

        # Check containment constraints
        # Outer hex is centered at origin with calculated side length
        outer_radius = outer_side_length * np.sqrt(3) / 2  # Circumradius

        # Check each hexagon for containment
        for i in range(NUM_INNER_HEX):
            x, y, angle = hex_data[i]
            vertices = get_hexagon_vertices(x, y, angle)
            if not check_containment(vertices, (0, 0), outer_radius):
                penalty += 1000000.0  # Heavy penalty

        # Check for overlaps between hexagons using fast method
        for i in range(NUM_INNER_HEX):
            for j in range(i+1, NUM_INNER_HEX):
                x1, y1, angle1 = hex_data[i]
                x2, y2, angle2 = hex_data[j]

                vertices1 = get_hexagon_vertices(x1, y1, angle1)
                vertices2 = get_hexagon_vertices(x2, y2, angle2)

                if check_overlap_fast(vertices1, vertices2):
                    penalty += 1000000.0  # Heavy penalty

        # Fitness is negative inverse of side length plus penalties
        # We want to minimize side length, so maximize 1/side_length
        fitness = -1.0 / outer_side_length
        if penalty > 0:
            fitness -= penalty  # Add penalty for constraint violations

        return (fitness,)
    except Exception as e:
        return (-1000000.0,)

def create_initial_individual():
    """Create a smarter initial individual using hexagonal packing principle"""
    individual = []

    # Start with a known good hexagonal arrangement
    # Center hexagon
    individual.extend([0.0, 0.0, 0.0])

    # First ring around center (6 hexagons)
    # Using the standard hexagonal packing pattern: 2 units apart
    ring_1_angles = [0, 60, 120, 180, 240, 300]
    ring_1_radius = 2.0
    for angle in ring_1_angles:
        x = ring_1_radius * np.cos(np.radians(angle))
        y = ring_1_radius * np.sin(np.radians(angle))
        individual.extend([x, y, random.uniform(0, 360)])

    # Second ring (12 hexagons) - but we only need 11 total, so we'll use 5 more
    # This creates a more complex but efficient arrangement
    ring_2_angles = [30, 90, 150, 210, 270, 330]  # Between first ring angles
    ring_2_radius = 3.0
    for angle in ring_2_angles:
        x = ring_2_radius * np.cos(np.radians(angle))
        y = ring_2_radius * np.sin(np.radians(angle))
        individual.extend([x, y, random.uniform(0, 360)])

    # Trim to exactly 11 elements
    individual = individual[:33]  # 11 hexagons * 3 parameters

    # Add small random perturbations for diversity
    for i in range(0, len(individual), 3):
        individual[i] += random.uniform(-0.15, 0.15)  # x position
        individual[i+1] += random.uniform(-0.15, 0.15)  # y position
        individual[i+2] += random.uniform(-10, 10)  # angle

    return individual

def create_initial_population(size=50):
    """Create an initial population with better diversity"""
    population = []
    for _ in range(size):
        individual = create_initial_individual()
        population.append(individual)
    return population

def mutate_individual(individual, indpb=0.1, generation=0, max_generations=50):
    """Enhanced mutation operator with adaptive step sizes"""
    # Decrease mutation strength as optimization progresses
    adaptive_mutation_rate = indpb * (1.0 - (generation / max_generations) * 0.7)

    for i in range(len(individual)):
        if random.random() < adaptive_mutation_rate:
            if i % 3 == 0:  # x coordinate
                individual[i] += random.uniform(-0.2, 0.2)
            elif i % 3 == 1:  # y coordinate
                individual[i] += random.uniform(-0.2, 0.2)
            else:  # angle
                individual[i] += random.uniform(-15, 15)
                individual[i] %= 360
    return individual,

def crossover_individuals(ind1, ind2):
    """Improved crossover that swaps complete hexagon configurations"""
    # Swap entire hexagon blocks (3 parameters each) instead of individual genes
    size = min(len(ind1), len(ind2))
    # Swap complete hexagons (every 3 parameters)
    for i in range(0, size, 3):
        if random.random() < 0.5:
            # Swap the whole hexagon block
            ind1[i:i+3], ind2[i:i+3] = ind2[i:i+3], ind1[i:i+3]
    return ind1, ind2

def local_refinement(individual, max_iter=50):
    """Apply local refinement to improve the best solution found"""
    # Convert individual to flat array
    x0 = np.array(individual)

    # Define bounds for optimization
    bounds = []
    for i in range(0, len(x0), 3):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle bounds

    def objective(x_flat):
        # Convert flat array to hex data structure
        try:
            hex_data = x_flat.reshape(-1, 3)
            outer_side_length = calculate_outer_hex_side_length(hex_data)
            # Return negative inverse of side length (we minimize)
            return -1.0 / outer_side_length
        except:
            return -1000000.0

    def constraint_func(x_flat):
        try:
            hex_data = x_flat.reshape(-1, 3)
            # Check containment
            outer_radius = calculate_outer_hex_side_length(hex_data) * np.sqrt(3) / 2
            for i in range(NUM_INNER_HEX):
                x, y, angle = hex_data[i]
                vertices = get_hexagon_vertices(x, y, angle)
                if not check_containment(vertices, (0, 0), outer_radius):
                    return 1000000.0  # Penalty for containment violation

            # Check overlaps
            for i in range(NUM_INNER_HEX):
                for j in range(i+1, NUM_INNER_HEX):
                    x1, y1, angle1 = hex_data[i]
                    x2, y2, angle2 = hex_data[j]

                    vertices1 = get_hexagon_vertices(x1, y1, angle1)
                    vertices2 = get_hexagon_vertices(x2, y2, angle2)

                    if check_overlap_fast(vertices1, vertices2):
                        return 1000000.0  # Penalty for overlap

            return 0.0  # No penalty
        except:
            return 1000000.0

    # Use scipy's minimize for local refinement
    try:
        # Note: We're using L-BFGS-B which works better for this continuous optimization
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         constraints={'type': 'ineq', 'fun': constraint_func},
                         options={'maxiter': max_iter, 'disp': False})

        if result.success:
            return result.x.tolist()
    except:
        # If local refinement fails, just return original
        pass

    return individual

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Set up DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_initial_individual)
    toolbox.register("population", create_initial_population)
    toolbox.register("evaluate", evaluate_individual_parallel)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", lambda ind, indpb, gen: mutate_individual(ind, indpb, gen, 50))
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population with better seeding
    pop = toolbox.population(n=40)  # Larger population for better diversity

    # Run evolution for limited time
    start_time = time.time()
    max_time = 160  # seconds

    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evolution loop with adaptive parameters
    best_fitness = -float('inf')
    best_individual = None

    try:
        # Run evolution for up to max_time seconds with adaptive parameters
        for generation in range(60):  # Maximum 60 generations
            if time.time() - start_time > max_time:
                break

            # Evolve one generation
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)

            # Evaluate fitness in parallel for speedup
            fits = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            # Elitism: Keep best individuals
            elite = tools.selBest(pop, k=8)  # Keep top 8
            pop = toolbox.select(offspring, k=len(offspring) - 8)  # Select rest
            pop.extend(elite)  # Add elites back

            # Track best individual
            current_best = max(pop, key=lambda x: x.fitness.values[0])
            if current_best.fitness.values[0] > best_fitness:
                best_fitness = current_best.fitness.values[0]
                best_individual = current_best[:]

    except Exception as e:
        print(f"Genetic algorithm error: {e}")
        pass  # Continue with whatever we found

    # Apply local refinement to the best solution
    if best_individual is not None:
        try:
            refined_individual = local_refinement(best_individual)
            refined_fitness, _ = evaluate_individual_parallel(refined_individual)
            if refined_fitness[0] > best_fitness:
                best_fitness = refined_fitness[0]
                best_individual = refined_individual
        except Exception as e:
            print(f"Local refinement error: {e}")
            pass

    # Convert best individual to desired format
    if best_individual is None:
        # Fallback to simple grid if nothing worked
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
    else:
        # Convert best individual to hex data
        inner_hex_data = np.array(best_individual).reshape(-1, 3)
        outer_hex_side_length = calculate_outer_hex_side_length(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])  # outer hexagon centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END