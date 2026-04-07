# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0  # Distance from center to corner for unit hexagon
UNIT_HEXAGON_WIDTH = 2.0  # Diameter of unit hexagon
MAX_EVAL_TIME = 180.0  # seconds

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

def is_contained_in_outer_hexagon(hexagon_poly, outer_center, outer_angle, outer_radius):
    """Check if hexagon is fully contained in outer hexagon"""
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Check if all vertices of inner hexagon are inside outer hexagon
    for vertex in hexagon_poly.exterior.coords[:-1]:  # Exclude closing vertex
        if not outer_polygon.contains(Point(vertex)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap using Shapely"""
    try:
        poly1 = Polygon(hex1)
        poly2 = Polygon(hex2)
        return poly1.intersects(poly2)
    except:
        return False

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
    # First check if all hexagons are contained
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygon = Polygon(vertices)

        if not is_contained_in_outer_hexagon(hex_polygon, outer_center, outer_angle, calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)):
            return False

    # Then check for overlaps
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = inner_hex_data[i][:2]
            angle1 = np.radians(inner_hex_data[i][2])
            center2 = inner_hex_data[j][:2]
            angle2 = np.radians(inner_hex_data[j][2])

            vertices1 = hexagon_vertices(center1, angle1, UNIT_HEXAGON_RADIUS)
            vertices2 = hexagon_vertices(center2, angle2, UNIT_HEXAGON_RADIUS)

            if check_overlap(vertices1, vertices2):
                return False

    return True

def evaluate_fitness(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness (negative of outer hexagon radius for maximization)"""
    # Calculate minimum outer radius needed
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # If solution is invalid, penalize heavily
    if not validate_solution(inner_hex_data, outer_center, outer_angle):
        return -1e10  # Very poor fitness

    # Return negative radius (we want to minimize radius, so maximize negative value)
    return -outer_radius

def create_individual():
    """Create a random valid individual"""
    # Generate random positions and angles for 11 hexagons
    individual = []
    for _ in range(NUM_INNER_HEXAGONS):
        x = random.uniform(-5, 5)  # Reasonable bounds
        y = random.uniform(-5, 5)
        angle = random.uniform(0, 360)
        individual.append([x, y, angle])
    return np.array(individual)

def create_initial_population(pop_size):
    """Create initial population with valid individuals"""
    population = []
    for _ in range(pop_size):
        individual = create_individual()
        # Ensure it's valid (not strictly necessary but good practice)
        if validate_solution(individual):
            population.append(individual)
        else:
            # Try again up to 10 times
            for _ in range(10):
                individual = create_individual()
                if validate_solution(individual):
                    population.append(individual)
                    break
            else:
                # If we couldn't make a valid one, just add it anyway
                population.append(individual)
    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select parent using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx]

def crossover(parent1, parent2):
    """Custom crossover for hexagon positions and rotations"""
    # Uniform crossover for positions and angles
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

    return child1, child2

def mutate(individual, mutation_rate=0.1, max_step=0.5):
    """Custom mutation for hexagon positions and rotations"""
    mutated = individual.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position slightly
            mutated[i][0] += random.uniform(-max_step, max_step)
            mutated[i][1] += random.uniform(-max_step, max_step)

            # Mutate angle (keep within 0-360)
            mutated[i][2] += random.uniform(-15, 15)
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

    # Configuration parameters
    pop_size = 50
    generations = 1000
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
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

    # Validate final best solution
    if best_individual is None:
        # Return fallback if we couldn't find anything good
        fallback = create_individual()
        best_individual = fallback

    # Final validation and calculation of outer hexagon parameters
    outer_radius = -best_fitness if best_fitness != float('-inf') else 10.0  # fallback if not found

    # Return result
    inner_hex_data = best_individual
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