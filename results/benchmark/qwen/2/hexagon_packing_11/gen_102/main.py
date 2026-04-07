# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import math
import warnings
import random
from deap import base, creator, tools, algorithms
from scipy.optimize import differential_evolution
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

def check_containment_fast(hex_vertices, outer_radius):
    """Fast containment check using distance from center."""
    # All vertices of a regular hexagon are at distance <= radius from center
    # So we just check that the center of the hexagon is within the outer hexagon
    # The outer hexagon has circumradius = outer_radius * sqrt(3)/2
    # So we check if the distance from center of outer hexagon (origin) to center of inner hexagon is within bounds

    # Check if all vertices are within the outer hexagon by checking the distance
    # We use a more direct approach: check if the center of hexagon is within outer bounds
    for vertex in hex_vertices:
        dist = math.sqrt(vertex[0]**2 + vertex[1]**2)
        # For outer hexagon with circumradius R, vertices are at distance R from center
        # The center of inner hexagon should be at distance <= R - radius
        # Actually let's do the proper check: outer hexagon radius = outer_radius * sqrt(3)/2
        outer_circumradius = outer_radius * math.sqrt(3)/2
        if dist > outer_circumradius:
            return False
    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using distance between centers."""
    # Calculate centers of both hexagons
    center1 = [sum(v[0] for v in hex1_vertices)/6, sum(v[1] for v in hex1_vertices)/6]
    center2 = [sum(v[0] for v in hex2_vertices)/6, sum(v[1] for v in hex2_vertices)/6]

    # Distance between centers
    dist_centers = math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

    # Two hexagons overlap if distance between centers is less than sum of their radii
    # For unit hexagons, radius = 1
    return dist_centers < 2.0

def calculate_outer_hex_radius(inner_configs, padding=0.01):
    """Calculate minimal outer hexagon radius needed to contain all inner hexagons."""
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for center_x, center_y, angle_deg in inner_configs:
        vertices = get_hexagon_vertices(center_x, center_y, angle_deg)
        for x, y in vertices:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    # Calculate distance from center to farthest point
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    max_dist = 0
    for center_x, center_y, angle_deg in inner_configs:
        vertices = get_hexagon_vertices(center_x, center_y, angle_deg)
        for x, y in vertices:
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = max(max_dist, dist)

    # Add some padding and convert to hexagon radius
    outer_radius = max_dist * (1 + padding)
    return outer_radius

def evaluate_individual(individual):
    """Evaluate fitness of an individual (set of hexagon parameters)."""
    # Reshape individual into 11 hexagon configurations
    inner_positions = np.array(individual).reshape(-1, 3)

    # Calculate outer hexagon radius
    outer_radius = calculate_outer_hex_radius(inner_positions)

    # Initialize penalty
    penalty = 0

    # Check constraints
    # 1. All hexagons must be within outer hexagon (using fast containment check)
    for i, (center_x, center_y, angle_deg) in enumerate(inner_positions):
        hex_vertices = get_hexagon_vertices(center_x, center_y, angle_deg)
        # Use fast containment check
        if not check_containment_fast(hex_vertices, outer_radius):
            penalty += 1e8  # Large penalty for containment violation

        # Check overlaps with all others (using fast overlap check)
        for j in range(i+1, len(inner_positions)):
            other_center_x, other_center_y, other_angle_deg = inner_positions[j]
            other_hex_vertices = get_hexagon_vertices(other_center_x, other_center_y, other_angle_deg)

            if check_overlap_fast(hex_vertices, other_hex_vertices):
                penalty += 1e8  # Large penalty for overlap

    # Return fitness value (negative inverse of outer radius + penalty)
    # We minimize the penalty and maximize 1/R, so we return negative value of total cost
    fitness = -1.0 / outer_radius
    if penalty > 0:
        fitness -= penalty
    return (fitness,)

def create_initial_individual():
    """Create an initial individual with better heuristic placement."""
    individual = []

    # Place center hexagon at origin
    individual.extend([0.0, 0.0, 0.0])

    # Place 10 surrounding hexagons in a pattern that's likely to work
    # Arrange them in a hexagonal pattern around the center
    spacing = 2.0  # Distance between centers of adjacent hexagons

    # Positions around the center (hexagonal pattern)
    positions = [
        (spacing, 0, 0),           # right
        (-spacing, 0, 0),          # left
        (spacing/2, spacing * math.sqrt(3)/2, 0),   # top-right
        (-spacing/2, spacing * math.sqrt(3)/2, 0),  # top-left
        (spacing/2, -spacing * math.sqrt(3)/2, 0),  # bottom-right
        (-spacing/2, -spacing * math.sqrt(3)/2, 0), # bottom-left
        (spacing, spacing * math.sqrt(3), 0),       # far top-right
        (-spacing, spacing * math.sqrt(3), 0),      # far top-left
        (spacing, -spacing * math.sqrt(3), 0),      # far bottom-right
        (-spacing, -spacing * math.sqrt(3), 0),     # far bottom-left
    ]

    # Add positions with slight randomization to avoid perfect symmetry
    for x, y, angle in positions:
        x += random.uniform(-0.2, 0.2)
        y += random.uniform(-0.2, 0.2)
        angle = random.uniform(0, 360)
        individual.extend([x, y, angle])

    return individual

def create_population(n=50):
    """Create initial population with better seeding."""
    return [create_initial_individual() for _ in range(n)]

def crossover(ind1, ind2):
    """Custom crossover operator."""
    # Uniform crossover for hexagon parameters
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def mutate(individual, mu=0.1, sigma=0.3):
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
    """Optimize the arrangement of 11 unit hexagons using evolutionary algorithm."""

    # Try multiple approaches:
    # 1. DEAP genetic algorithm with good initialization
    # 2. Differential Evolution as backup

    best_individual = None
    best_fitness = float('-inf')
    best_radius = float('inf')

    # First try DEAP approach
    try:
        # Set up DEAP framework
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", create_initial_individual)
        toolbox.register("population", create_population)
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create initial population
        pop = toolbox.population(n=30)

        # Statistics and hall of fame
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(1)

        # Evolve with early stopping
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                          ngen=50, stats=stats, halloffame=hof,
                                          verbose=False)

        # Get best individual from EA
        ea_individual = hof[0]
        ea_positions = np.array(ea_individual).reshape(-1, 3)
        ea_radius = calculate_outer_hex_radius(ea_positions)
        ea_fitness = -1.0 / ea_radius

        if ea_fitness > best_fitness:
            best_individual = ea_individual
            best_fitness = ea_fitness
            best_radius = ea_radius

    except Exception as e:
        print(f"DEAP approach failed: {e}")

    # Try differential evolution as backup
    try:
        # Define bounds for positions: x,y in [-10, 10], angle in [0, 360]
        bounds = [(-10, 10), (-10, 10), (0, 360)] * 11

        # Use differential evolution
        result = differential_evolution(
            lambda x: -evaluate_individual(x)[0],
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            tol=1e-6
        )

        if result.success:
            de_fitness = -result.fun
            if de_fitness > best_fitness:
                best_individual = result.x
                best_fitness = de_fitness
                de_positions = np.array(result.x).reshape(-1, 3)
                best_radius = calculate_outer_hex_radius(de_positions)

    except Exception as e:
        print(f"Differential Evolution failed: {e}")

    # If still no good solution, fall back to initial configuration
    if best_individual is None:
        # Use a well-known good configuration that works
        initial_positions = [
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
        ]
        best_positions = np.array(initial_positions)
        best_radius = calculate_outer_hex_radius(best_positions)
        best_individual = best_positions.flatten()
        best_fitness = -1.0 / best_radius

    # Convert best individual to desired format
    best_positions = np.array(best_individual).reshape(-1, 3)

    # Create outer hexagon data (centered at origin)
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return best_positions, outer_hex_data, best_radius

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
            raise Exception("Optimization returned invalid solution")
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