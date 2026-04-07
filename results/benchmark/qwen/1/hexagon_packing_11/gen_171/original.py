# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

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

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def is_contained_in_outer_hexagon(hexagon_vertices_list, outer_center, outer_angle, outer_radius):
    """Check if hexagon is fully contained in outer hexagon using optimized approach"""
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Fast check: test if all vertices are inside outer polygon
    for vertex in hexagon_vertices_list:
        if not point_in_polygon(vertex, outer_polygon):
            return False
    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using spatial indexing and bounding box pruning"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)

        # Quick bounding box check first
        bbox1 = poly1.bounds
        bbox2 = poly2.bounds

        if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
            bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
            return False

        return poly1.intersects(poly2)
    except:
        # Fallback for degenerate cases
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

def validate_solution_with_spatial_indexing(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution with improved spatial indexing for overlap detection"""
    # Check containment first
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate outer radius based on this solution to check containment properly
        outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

        if not is_contained_in_outer_hexagon(vertices, outer_center, outer_angle, outer_radius):
            return False

    # Create list of all hexagon polygons for spatial indexing
    hex_polygons = []
    centroids = []
    
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))
        # Use centroid for spatial indexing
        centroid = np.mean(vertices, axis=0)
        centroids.append(centroid)

    # Create spatial index (KDTree) for efficient neighbor search
    centroids_array = np.array(centroids)
    tree = cKDTree(centroids_array)

    # Find candidates for overlap checking using distance threshold
    # Only check pairs that are close enough to potentially overlap
    pairs = tree.query_pairs(r=UNIT_HEXAGON_WIDTH * 2.0, eps=0)
    
    # Check actual overlaps for candidate pairs
    for i, j in pairs:
        if i < j:  # Avoid checking pairs twice
            if hex_polygons[i].intersects(hex_polygons[j]):
                return False

    return True

def evaluate_fitness(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness (negative of outer hexagon radius for maximization)"""
    # Calculate minimum outer radius needed
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # If solution is invalid, penalize heavily
    if not validate_solution_with_spatial_indexing(inner_hex_data, outer_center, outer_angle):
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

def create_diverse_initial_individuals(num_configs=8):
    """Create multiple diverse initial configurations"""
    configs = []
    
    # Base structured configurations
    base_configs = [
        # Configuration 1: Basic hexagonal arrangement
        [
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
        ],
        # Configuration 2: Spiral-like arrangement
        [
            [0, 0, 0],           # center
            [-1.5, 0, 0],       # left
            [1.5, 0, 0],        # right
            [0, 1.5, 0],        # top
            [0, -1.5, 0],       # bottom
            [-1.5, 1.5, 0],     # top-left
            [1.5, 1.5, 0],      # top-right
            [-1.5, -1.5, 0],    # bottom-left
            [1.5, -1.5, 0],     # bottom-right
            [-3, 1.5, 0],       # far top-left
            [3, 1.5, 0],        # far top-right
        ],
        # Configuration 3: Grid arrangement with offsets
        [
            [0, 0, 0],           # center
            [-2, 0, 0],         # left
            [2, 0, 0],          # right
            [0, 2, 0],          # top
            [0, -2, 0],         # bottom
            [-2, 2, 0],         # top-left
            [2, 2, 0],          # top-right
            [-2, -2, 0],        # bottom-left
            [2, -2, 0],         # bottom-right
            [-3, 2, 0],         # far top-left
            [3, 2, 0],          # far top-right
        ]
    ]
    
    # Add base configurations with random variations
    for i, base_config in enumerate(base_configs):
        individual = []
        for pos in base_config:
            x = pos[0] + random.uniform(-0.2, 0.2)
            y = pos[1] + random.uniform(-0.2, 0.2)
            angle = pos[2] + random.uniform(-15, 15)
            individual.append([x, y, angle])
        configs.append(np.array(individual))
        
    # Add purely random configurations
    for _ in range(num_configs - len(base_configs)):
        individual = []
        for _ in range(NUM_INNER_HEXAGONS):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            angle = random.uniform(0, 360)
            individual.append([x, y, angle])
        configs.append(np.array(individual))
    
    return configs

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

def adaptive_mutation_rate(generation, max_generations, initial_rate=0.8, final_rate=0.1):
    """Adaptive mutation rate that decreases over generations"""
    # Linear decay from initial_rate to final_rate
    return initial_rate - (initial_rate - final_rate) * (generation / max_generations)

def optimize_single_run(population_size=50, generations=1200, elite_size=5, tournament_size=5, seed=None):
    """Run a single optimization run with given parameters"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create initial population
    population = create_diverse_initial_individuals(population_size // 2)
    
    # Fill remaining with random individuals
    for _ in range(population_size // 2):
        individual = []
        for _ in range(NUM_INNER_HEXAGONS):
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            angle = random.uniform(0, 360)
            individual.append([x, y, angle])
        population.append(np.array(individual))

    best_fitness = float('-inf')
    best_individual = None

    # Evolution loop
    for gen in range(generations):
        # Calculate adaptive mutation rate
        mut_rate = adaptive_mutation_rate(gen, generations)
        
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
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate=mut_rate)
            child2 = mutate(child2, mutation_rate=mut_rate)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:population_size]

    return best_individual, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Multi-start approach with different seeds
    best_overall_fitness = float('-inf')
    best_overall_individual = None
    best_seed = 0

    # Run multiple independent optimizations with different seeds
    num_starts = 6
    seeds = [42, 123, 456, 789, 999, 1001]
    
    for start_num in range(num_starts):
        if time.time() - start_time > MAX_EVAL_TIME - 1:  # Leave 1 second for final processing
            break
            
        seed = seeds[start_num] if start_num < len(seeds) else start_num * 100
        
        try:
            best_individual, best_fitness = optimize_single_run(
                population_size=60,
                generations=1500,
                elite_size=8,
                tournament_size=6,
                seed=seed
            )
            
            if best_individual is not None and best_fitness > best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual.copy()
                best_seed = seed
                
        except Exception as e:
            continue  # Skip this run if it fails

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
    if not validate_solution_with_spatial_indexing(inner_hex_data):
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