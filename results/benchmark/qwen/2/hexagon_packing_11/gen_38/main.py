# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
import math

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
    """Check if all vertices of inner hexagon are within outer hexagon"""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        if not outer_polygon.contains(Point(vertex[0], vertex[1])):
            return False
    return True

def check_hexagon_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def calculate_outer_hex_side_length(inner_hex_data, margin_factor=1.1):
    """Calculate minimum outer hexagon side length that contains all inner hexagons"""
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

    # Calculate bounding box
    xs = [v[0] for v in all_vertices]
    ys = [v[1] for v in all_vertices]

    # Calculate the radius needed to contain all points
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2

    max_dist = 0
    for x, y in all_vertices:
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = max(max_dist, dist)

    # Account for hexagon geometry - outer hexagon must be large enough to accommodate
    # the maximum distance plus some margin for the hexagon's width
    side_length = max_dist * margin_factor * 2 / math.sqrt(3)

    return side_length

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
            return (1.0 / outer_side_length - total_penalty,),  # Return tuple for DEAP

        return (1.0 / outer_side_length,),  # Maximize 1/outer_side_length

    except Exception as e:
        return (-10000,),  # Very poor fitness for invalid solutions

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

    # Define gene ranges
    # x, y: -10 to 10 (large enough for exploration)
    # angle: 0 to 360 degrees
    toolbox.register("attr_float_x", random.uniform, -10, 10)
    toolbox.register("attr_float_y", random.uniform, -10, 10)
    toolbox.register("attr_float_angle", random.uniform, 0, 360)

    # Individual is a list of attributes
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: [toolbox.attr_float_x(), toolbox.attr_float_y(), toolbox.attr_float_angle()],
                     n=NUM_HEXAGONS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    toolbox.register("evaluate", evaluate_solution)

    # Register genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run evolution
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        # Run the evolutionary algorithm with timeout protection
        start_time = time.time()
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                          ngen=100, stats=stats, halloffame=hof, verbose=False)
        elapsed_time = time.time() - start_time

        # Get the best individual found
        best_individual = hof[0]
        best_hex_data = np.array(best_individual).reshape(-1, 3)

        # Calculate final outer hexagon size
        outer_side_length = calculate_outer_hex_side_length(best_hex_data)

        return best_hex_data, np.array([0, 0, 0]), outer_side_length

    except Exception as e:
        # Fallback to original configuration if optimization fails
        print(f"Evolutionary optimization failed: {e}")
        # Use the grid arrangement from original version as fallback
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