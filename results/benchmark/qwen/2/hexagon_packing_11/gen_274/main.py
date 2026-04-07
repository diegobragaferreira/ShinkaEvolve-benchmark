# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import math
import warnings
import random
from deap import base, creator, tools, algorithms
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

def get_hexagon_edges(vertices):
    """Get edges of hexagon from vertices (vectors from one vertex to next)."""
    edges = []
    n = len(vertices)
    for i in range(n):
        edge = (vertices[i][0] - vertices[(i+1)%n][0], vertices[i][1] - vertices[(i+1)%n][1])
        edges.append(edge)
    return edges

def get_hexagon_normals(edges):
    """Get normal vectors to hexagon edges."""
    normals = []
    for edge in edges:
        # Normal vector (perpendicular to edge)
        normal = (-edge[1], edge[0])
        # Normalize the normal vector
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 0:
            normal = (normal[0]/norm, normal[1]/norm)
        normals.append(normal)
    return normals

def project_polygon_onto_axis(vertices, axis):
    """Project polygon vertices onto an axis and return min/max projections."""
    projections = []
    for vertex in vertices:
        projection = vertex[0] * axis[0] + vertex[1] * axis[1]
        projections.append(projection)
    return min(projections), max(projections)

def sat_check_overlap(hex1_vertices, hex2_vertices):
    """Check overlap using Separating Axis Theorem - much faster than Shapely."""
    # Get edges for both hexagons
    edges1 = get_hexagon_edges(hex1_vertices)
    edges2 = get_hexagon_edges(hex2_vertices)

    # Get normals for both polygons
    normals1 = get_hexagon_normals(edges1)
    normals2 = get_hexagon_normals(edges2)

    # Check all axes (normals to edges)
    all_normals = normals1 + normals2

    for axis in all_normals:
        min1, max1 = project_polygon_onto_axis(hex1_vertices, axis)
        min2, max2 = project_polygon_onto_axis(hex2_vertices, axis)

        # Check for separation
        if max1 < min2 or max2 < min1:
            return False  # No overlap along this axis

    return True  # Overlap detected

def check_containment_fast(hex_vertices, outer_center_x, outer_center_y, outer_radius):
    """Fast containment check using distance from center."""
    # Check if all vertices are within the outer hexagon using approximate distance
    # This is faster than using Shapely for the containment check
    for vertex in hex_vertices:
        # Fast distance check vs outer circle
        dist_sq = (vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2
        # Compare with squared radius for efficiency
        if dist_sq > (outer_radius * 0.999)**2:  # Slight buffer to be safe
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using SAT for efficiency."""
    # Quick bounding box check first for early rejection
    min1_x = min(v[0] for v in hex1_vertices)
    max1_x = max(v[0] for v in hex1_vertices)
    min1_y = min(v[1] for v in hex1_vertices)
    max1_y = max(v[1] for v in hex1_vertices)

    min2_x = min(v[0] for v in hex2_vertices)
    max2_x = max(v[0] for v in hex2_vertices)
    min2_y = min(v[1] for v in hex2_vertices)
    max2_y = max(v[1] for v in hex2_vertices)

    # Quick rejection: if bounding boxes don't overlap, no hexagon overlap
    if max1_x < min2_x or max2_x < min1_x or max1_y < min2_y or max2_y < min1_y:
        return False

    # Use SAT for final precise check
    return sat_check_overlap(hex1_vertices, hex2_vertices)

def calculate_outer_hex_radius(inner_configs, padding=0.01):
    """Calculate minimal outer hexagon radius needed to contain all inner hexagons."""
    # Precompute bounds by checking just the extreme positions rather than all vertices
    # For hexagons, we can compute bounds more efficiently

    # Since we're dealing with unit hexagons, their vertices are at most 1 unit away from center
    # So we can compute the maximal bounds directly from center positions
    centers_x = [config[0] for config in inner_configs]
    centers_y = [config[1] for config in inner_configs]

    if not centers_x or not centers_y:
        return 100.0

    min_x, max_x = min(centers_x), max(centers_x)
    min_y, max_y = min(centers_y), max(centers_y)

    # Since hexagons are regular, we need to account for their size
    # Maximum distance from origin to any hexagon center + hexagon radius
    center_distances = [math.sqrt(x*x + y*y) for x, y in zip(centers_x, centers_y)]
    max_center_dist = max(center_distances) if center_distances else 0

    # The furthest point from origin will be at max_center_dist + HEX_RADIUS
    outer_radius = max_center_dist + HEX_RADIUS * (1 + padding)
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
    # 1. All hexagons must be within outer hexagon
    # We'll do fast containment check first
    for i, (center_x, center_y, angle_deg) in enumerate(inner_positions):
        # Fast check vs bounding circle
        dist_to_origin_sq = center_x**2 + center_y**2
        if dist_to_origin_sq > (outer_radius - HEX_RADIUS)**2:
            penalty += 1e6  # Large penalty for containment violation
            continue

        # For detailed check, we can check against outer hexagon more precisely
        hex_vertices = get_hexagon_vertices(center_x, center_y, angle_deg)

        # Check containment using fast method
        if not check_containment_fast(hex_vertices, 0, 0, outer_radius):
            penalty += 1e6  # Large penalty for containment violation

        # Check overlaps with all others
        for j in range(i+1, len(inner_positions)):
            other_center_x, other_center_y, other_angle_deg = inner_positions[j]
            other_hex_vertices = get_hexagon_vertices(other_center_x, other_center_y, other_angle_deg)

            if check_overlap(hex_vertices, other_hex_vertices):
                penalty += 1e6  # Large penalty for overlap

    # Return fitness value (negative inverse of outer radius + penalty)
    # We minimize the penalty and maximize 1/R, so we return negative value of total cost
    if penalty > 0:
        return (-1.0 / outer_radius) - penalty,
    else:
        return (-1.0 / outer_radius),

def create_individual():
    """Create a new individual (random hexagon configuration)."""
    individual = []
    for _ in range(11):  # 11 hexagons
        # Random x, y position within reasonable bounds
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        # Random angle between 0 and 360 degrees
        angle = random.uniform(0, 360)
        individual.extend([x, y, angle])
    return individual

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
    """Optimize the arrangement of 11 unit hexagons using evolutionary algorithm."""

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

    # Create initial population
    pop = toolbox.population(n=50)

    # Statistics and hall of fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)

    # Evolve
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                          ngen=100, stats=stats, halloffame=hof,
                                          verbose=False)
    except Exception as e:
        print(f"Evolutionary algorithm failed with error: {e}")
        # Return fallback solution
        return None, None, None

    # Get best individual
    best_individual = hof[0]
    best_positions = np.array(best_individual).reshape(-1, 3)

    # Calculate final outer hexagon radius
    final_outer_radius = calculate_outer_hex_radius(best_positions)

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