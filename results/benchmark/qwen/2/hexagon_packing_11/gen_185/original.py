# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import math
import warnings
import random
from deap import base, creator, tools, algorithms
from joblib import Parallel, delayed
import time
warnings.filterwarnings('ignore')

# Constants
HEX_RADIUS = 1.0  # Unit hexagon radius
HEX_APO = HEX_RADIUS * math.sqrt(3)/2  # Apothem of unit hexagon
HEX_SIDE = HEX_RADIUS  # Side length of unit hexagon

def get_hexagon_vertices(center_x, center_y, angle_deg, radius=HEX_RADIUS):
    """Get vertices of a regular hexagon given center, rotation, and radius."""
    angle_rad = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi/3
        x = center_x + radius * math.cos(theta)
        y = center_y + radius * math.sin(theta)
        vertices.append((x, y))
    return vertices

def check_containment_single(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Check if all vertices of hexagon are within the outer hexagon (optimized version)."""
    # Fast check by calculating distance from center to vertices
    outer_center = (outer_center_x, outer_center_y)
    for vertex in hex_vertices:
        dist = math.sqrt((vertex[0] - outer_center[0])**2 + (vertex[1] - outer_center[1])**2)
        # Consider a small buffer due to floating point errors
        if dist > outer_radius + 1e-6:
            return False
    return True

def check_overlap_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        # Fallback in case of geometric errors
        return False

def estimate_min_outer_radius(inner_configs):
    """Estimate minimum outer hexagon radius using tighter bounding approach."""
    # Collect all vertices from all hexagons
    all_vertices = []
    for center_x, center_y, angle_deg in inner_configs:
        vertices = get_hexagon_vertices(center_x, center_y, angle_deg)
        all_vertices.extend(vertices)

    # Find center of all vertices (centroid)
    if not all_vertices:
        return 1000.0

    avg_x = sum(v[0] for v in all_vertices) / len(all_vertices)
    avg_y = sum(v[1] for v in all_vertices) / len(all_vertices)

    # Find maximum distance from centroid to any vertex
    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - avg_x)**2 + (y - avg_y)**2)
        max_dist = max(max_dist, dist)

    # Add small padding to ensure complete containment
    return max_dist * 1.05  # 5% padding

def evaluate_individual(individual):
    """Evaluate fitness of an individual (set of hexagon parameters)."""
    # Reshape individual into 11 hexagon configurations
    inner_positions = np.array(individual).reshape(-1, 3)

    # Estimate outer hexagon radius more accurately
    outer_radius = estimate_min_outer_radius(inner_positions)

    # Initialize penalty
    penalty = 0

    # Check constraints
    # 1. All hexagons must be within outer hexagon
    for i, (center_x, center_y, angle_deg) in enumerate(inner_positions):
        hex_vertices = get_hexagon_vertices(center_x, center_y, angle_deg)

        # Check containment
        if not check_containment_single(hex_vertices, 0, 0, outer_radius):
            penalty += 1e6  # Large penalty for containment violation

        # Check overlaps with all others
        for j in range(i+1, len(inner_positions)):
            other_center_x, other_center_y, other_angle_deg = inner_positions[j]
            other_hex_vertices = get_hexagon_vertices(other_center_x, other_center_y, other_angle_deg)

            if check_overlap_single(hex_vertices, other_hex_vertices):
                penalty += 1e6  # Large penalty for overlap

    # Return fitness value (negative inverse of outer radius + penalty)
    # We minimize the penalty and maximize 1/R, so we return negative value of total cost
    fitness = (-1.0 / outer_radius) - penalty
    return (fitness,)

def evaluate_population_parallel(population, n_jobs=-1):
    """Evaluate a population in parallel."""
    fitnesses = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_individual)(ind) for ind in population
    )
    return fitnesses

def create_individual():
    """Create a new individual (random hexagon configuration)."""
    individual = []
    # Start with strategic placement for better convergence
    # Center hexagon
    individual.extend([0, 0, 0])

    # Surrounding hexagons in a more structured pattern
    # Positions arranged in a circular pattern around center
    positions = [
        (2.5, 0), (-2.5, 0), (0, 2.5), (0, -2.5),
        (1.25, 2.17), (-1.25, 2.17), (1.25, -2.17), (-1.25, -2.17),
        (3.75, 2.17), (3.75, -2.17)
    ]

    for i, (x, y) in enumerate(positions):
        # Add some randomness to improve exploration
        individual.extend([
            x + random.uniform(-0.5, 0.5),
            y + random.uniform(-0.5, 0.5),
            random.uniform(0, 360)
        ])
    return individual

def create_initial_population(size=100):
    """Create initial population with more intelligent seeding."""
    population = []
    for _ in range(size):
        individual = create_individual()
        population.append(individual)
    return population

def crossover(ind1, ind2):
    """Custom crossover operator."""
    # Uniform crossover for hexagon parameters
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def mutate(individual, mu=0.1, sigma=0.5):
    """Custom mutation operator."""
    for i in range(len(individual)):
        if random.random() < mu:
            # For x,y coordinates, add Gaussian noise
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.gauss(0, sigma)
            else:  # angle
                individual[i] = (individual[i] + random.gauss(0, sigma)) % 360
    return individual,

def optimize_hexagon_packing():
    """Optimize the arrangement of 11 unit hexagons using evolutionary algorithm with enhancements."""
    start_time = time.time()

    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population with better seeding
    pop = create_initial_population(50)

    # Statistics and hall of fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    # Evolve with time limit
    max_time = 150  # seconds
    try:
        gen_count = 0
        while time.time() - start_time < max_time and gen_count < 100:
            # Evolve one generation
            offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)

            # Evaluate population with parallel processing
            fits = evaluate_population_parallel(offspring, n_jobs=-1)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            # Select next generation
            pop = toolbox.select(offspring, k=len(pop))

            # Track best individual
            current_best = max(pop, key=lambda x: x.fitness.values[0])
            if len(hof) == 0 or current_best.fitness.values[0] > hof[0].fitness.values[0]:
                hof.update([current_best])

            gen_count += 1
            # Print progress every 10 generations
            if gen_count % 10 == 0:
                print(f"Generation {gen_count}: Best fitness = {hof[0].fitness.values[0]}")

    except Exception as e:
        print(f"Evolutionary algorithm failed with error: {e}")
        # Return fallback solution
        return None, None, None

    # Get best individual
    best_individual = hof[0]
    best_positions = np.array(best_individual).reshape(-1, 3)

    # Calculate final outer hexagon radius with better estimation
    final_outer_radius = estimate_min_outer_radius(best_positions)

    # Create outer hexagon data (centered at origin)
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return best_positions, outer_hex_data, final_outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = optimize_hexagon_packing()
        if inner_hex_data is None or outer_hex_side_length is None:
            raise Exception("Evolutionary algorithm returned invalid solution")
    except Exception as e:
        # Fallback to original configuration if optimization fails
        print(f"Optimization failed with error: {e}")
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


# EVOLVE-BLOCK-END