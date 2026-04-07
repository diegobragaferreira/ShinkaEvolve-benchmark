# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution: check containment and non-overlap"""
    # Precompute all hexagon polygons once for reuse
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))

    # Calculate outer radius once
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # Check containment using the outer hexagon polygon
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Check if all inner hexagons are contained within outer hexagon
    for hex_poly in hex_polygons:
        # Fast check: if any vertex is outside, reject
        for vertex in hex_poly.exterior.coords[:-1]:  # Exclude closing vertex
            if not outer_polygon.contains(Point(vertex)):
                return False

    # Check overlaps efficiently using spatial indexing
    # Build spatial index for faster overlap detection
    points_list = []
    for i, hex_poly in enumerate(hex_polygons):
        # Collect all vertices for spatial indexing
        for vertex in hex_poly.exterior.coords[:-1]:
            points_list.append((vertex[0], vertex[1], i))

    if len(points_list) > 0:
        # Create spatial tree for vertices
        tree_points = cKDTree([(p[0], p[1]) for p in points_list])

        # Check overlaps between hexagons
        for i in range(len(hex_polygons)):
            for j in range(i+1, len(hex_polygons)):
                if hex_polygons[i].intersects(hex_polygons[j]):
                    return False
    else:
        # Fallback for empty case
        return False

    return True

def evaluate_fitness(individual, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness (negative of outer hexagon radius for maximization)"""
    # Calculate minimum outer radius needed
    outer_radius = calculate_outer_hexagon_radius(individual, outer_center, outer_angle)

    # If solution is invalid, penalize heavily
    if not validate_solution(individual, outer_center, outer_angle):
        return -1e10  # Very poor fitness

    # Return negative radius (we want to minimize radius, so maximize negative value)
    return -outer_radius

def create_initial_individual():
    """Create a better initial individual using a structured approach"""
    # Start with a known good arrangement and add slight randomness
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

    # Add small random perturbations
    individual = []
    for pos in base_positions:
        x = pos[0] + random.uniform(-0.3, 0.3)
        y = pos[1] + random.uniform(-0.3, 0.3)
        angle = pos[2] + random.uniform(-10, 10)
        individual.append([x, y, angle])

    return np.array(individual)

def create_initial_population(pop_size):
    """Create initial population with better starting solutions"""
    population = []

    # Add a few structured individuals
    for _ in range(pop_size // 2):
        individual = create_initial_individual()
        population.append(individual)

    # Fill remaining with random individuals
    for _ in range(pop_size // 2):
        individual = []
        for _ in range(NUM_INNER_HEXAGONS):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            angle = random.uniform(0, 360)
            individual.append([x, y, angle])
        population.append(np.array(individual))

    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select parent using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx]

def crossover(parent1, parent2):
    """Improved crossover for hexagon positions and rotations"""
    # Uniform crossover for positions and angles with some structure preservation
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover with selective copying
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

    return child1, child2

def mutate(individual, mutation_rate=0.1, max_step=0.3):
    """Enhanced mutation with geometric awareness"""
    mutated = individual.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # More sophisticated mutation approach
            # Mutate position with adaptive step sizes
            mutated[i][0] += random.uniform(-max_step, max_step)
            mutated[i][1] += random.uniform(-max_step, max_step)

            # Mutate angle with smaller steps for fine-tuning
            mutated[i][2] += random.uniform(-5, 5)
            mutated[i][2] %= 360

    return mutated

def get_best_individual(population, fitnesses):
    """Return the best individual and its fitness"""
    best_idx = np.argmax(fitnesses)
    return population[best_idx], fitnesses[best_idx]

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Multi-start approach
    best_overall_fitness = float('-inf')
    best_overall_individual = None

    # Run multiple independent optimizations
    num_starts = 5
    for start_num in range(num_starts):
        # Configuration parameters - adjust for better performance
        pop_size = 40
        generations = 800
        elite_size = 5
        tournament_size = 5

        # Create initial population
        population = create_initial_population(pop_size)

        best_fitness = float('-inf')
        best_individual = None

        # Evolution loop
        for gen in range(generations):
            if time.time() - start_time > MAX_EVAL_TIME - 1:  # Leave 1 second for final processing
                break

            # Adaptive mutation rate scheduling
            current_mutation_rate = 0.8 - (0.7 * gen / generations)  # Linear decay from 0.8 to 0.1
            current_mutation_rate = max(current_mutation_rate, 0.1)  # Ensure minimum rate

            # Evaluate fitness for entire population
            fitnesses = []
            for individual in population:
                fit = evaluate_fitness(individual)
                fitnesses.append(fit)

            # Track best solution
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            elites = [population[i] for i in elite_indices]

            # Create new population
            new_population = elites.copy()

            # Fill rest with offspring
            while len(new_population) < pop_size:
                parent1 = tournament_selection(population, fitnesses, tournament_size)
                parent2 = tournament_selection(population, fitnesses, tournament_size)

                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate=current_mutation_rate)
                child2 = mutate(child2, mutation_rate=current_mutation_rate)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:pop_size]

        # Keep track of best overall solution across all starts
        if best_individual is not None and best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_individual = best_individual.copy()

    # Validate final best solution
    if best_overall_individual is None:
        # Return fallback if we couldn't find anything good
        fallback = create_initial_individual()
        best_overall_individual = fallback

    # Final validation and calculation of outer hexagon parameters
    outer_radius = -best_overall_fitness if best_overall_fitness != float('-inf') else 10.0  # fallback if not found

    # Return result
    inner_hex_data = best_overall_individual
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = outer_radius

    # Final validation to ensure correctness
    if not validate_solution(inner_hex_data):
        # Revert to a reasonable fallback
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0],
        ])
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END