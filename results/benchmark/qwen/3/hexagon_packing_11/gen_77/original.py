# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.optimize import differential_evolution, minimize
from joblib import Parallel, delayed
import warnings

def generate_hexagon_vertices(center_x, center_y, side_length=1, rotation_deg=0):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices = np.column_stack([
        center_x + side_length * np.cos(angles),
        center_y + side_length * np.sin(angles)
    ])
    return vertices[:-1]  # Remove duplicate last vertex

def check_containment(hexagon_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are within outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def validate_configuration(inner_hex_data, outer_hex_data, outer_radius):
    """Validate that all hexagons fit properly within the outer hexagon."""
    # Generate outer hexagon vertices
    outer_vertices = generate_hexagon_vertices(
        outer_hex_data[0], outer_hex_data[1], outer_radius, outer_hex_data[2]
    )

    # Validate each inner hexagon
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_vertices = generate_hexagon_vertices(center_x, center_y, 1, angle)

        # Check containment
        if not check_containment(inner_vertices, outer_vertices):
            return False

        # Check overlaps with all other hexagons
        for j in range(i+1, len(inner_hex_data)):
            center_x2, center_y2, angle2 = inner_hex_data[j]
            inner_vertices2 = generate_hexagon_vertices(center_x2, center_y2, 1, angle2)

            if check_overlap(inner_vertices, inner_vertices2):
                return False

    return True

def compute_outer_hexagon_radius(inner_hex_data, outer_hex_data):
    """Compute minimum radius needed to contain all inner hexagons."""
    # Generate outer hexagon vertices
    outer_vertices = generate_hexagon_vertices(
        outer_hex_data[0], outer_hex_data[1], 1000, outer_hex_data[2]
    )

    # Find max distance from outer center to any inner hexagon vertex
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_vertices = generate_hexagon_vertices(center_x, center_y, 1, angle)
        distances = cdist([[outer_hex_data[0], outer_hex_data[1]]], inner_vertices)[0]
        max_dist = max(max_dist, np.max(distances))

    return max_dist + 0.1  # Add small buffer

def evaluate_fitness(config, outer_center=(0, 0), outer_angle=0):
    """Evaluate fitness of a configuration - higher is better."""
    # Extract inner hexagon data
    inner_hex_data = config.reshape(-1, 3)  # Each row: [x, y, angle]

    # Compute outer hexagon radius
    outer_radius = compute_outer_hexagon_radius(inner_hex_data, np.array([outer_center[0], outer_center[1], outer_angle]))

    # Validate configuration
    valid = validate_configuration(inner_hex_data, np.array([outer_center[0], outer_center[1], outer_angle]), outer_radius)

    # Return fitness (inverse of radius if valid, very negative otherwise)
    if valid:
        return 1.0 / outer_radius
    else:
        return -1e6

def generate_initial_config(n_hexagons=11):
    """Generate an initial configuration with sensible starting points."""
    # Start with a known good configuration and add some variation
    base_positions = [
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

    # Add random perturbations
    np.random.seed(42)
    initial_config = np.array(base_positions)
    # Perturb positions slightly
    for i in range(len(initial_config)):
        initial_config[i, 0] += np.random.uniform(-0.2, 0.2)
        initial_config[i, 1] += np.random.uniform(-0.2, 0.2)
        initial_config[i, 2] += np.random.uniform(-10, 10)

    return initial_config.flatten()

def local_optimization_step(config, bounds, outer_center=(0, 0), outer_angle=0):
    """Perform local optimization on a configuration."""
    def objective(x):
        # Reshape to proper format
        test_config = x.reshape(-1, 3)
        fitness = evaluate_fitness(test_config, outer_center, outer_angle)
        return -fitness  # Negative because we want to maximize

    # Use L-BFGS-B for local refinement
    try:
        result = minimize(objective, config, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})
        return result.x
    except:
        # If optimization fails, return original config
        return config

def evolutionary_search(n_generations=50, population_size=20, n_hexagons=11, bounds=None):
    """Evolutionary algorithm for hexagon packing optimization."""
    if bounds is None:
        # Create bounds for each parameter
        bounds_list = []
        for _ in range(n_hexagons):
            bounds_list.extend([(None, None), (None, None), (None, None)])  # x, y, angle
        bounds = bounds_list

    # Initialize population
    population = []
    for _ in range(population_size):
        individual = generate_initial_config(n_hexagons)
        population.append(individual)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None
    best_radius = float('inf')

    for gen in range(n_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        evaluated_population = []

        for individual in population:
            try:
                # Convert to 2D array
                config = individual.reshape(-1, 3)
                fitness = evaluate_fitness(config, (0, 0), 0)
                fitness_scores.append(fitness)
                evaluated_population.append((individual, fitness))
            except Exception as e:
                fitness_scores.append(-1e6)
                evaluated_population.append((individual, -1e6))

        # Find best individual
        max_fitness_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[max_fitness_idx]
        current_best_individual = evaluated_population[max_fitness_idx][0]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()

            # Compute actual radius for this configuration
            config = best_individual.reshape(-1, 3)
            radius = compute_outer_hexagon_radius(config, np.array([0, 0, 0]))
            best_radius = radius

        # Early stopping check
        if gen > 10 and abs(best_fitness - current_best_fitness) < 1e-6:
            break

        # Create new population through selection and mutation
        # Tournament selection
        new_population = []
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order

        # Elitism: keep top 2
        for i in range(2):
            new_population.append(population[sorted_indices[i]])

        # Generate rest by crossover and mutation
        while len(new_population) < population_size:
            # Select parents
            parent1_idx = sorted_indices[np.random.randint(0, int(population_size/2))]
            parent2_idx = sorted_indices[np.random.randint(0, int(population_size/2))]

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover
            child = parent1.copy()
            if np.random.rand() > 0.5:
                child = parent2

            # Mutation
            mutation_rate = 0.3 / (gen + 1)**0.5
            for i in range(len(child)):
                if np.random.rand() < mutation_rate:
                    if i % 3 == 0 or i % 3 == 1:  # x or y coordinate
                        child[i] += np.random.normal(0, 0.1)
                    elif i % 3 == 2:  # angle
                        child[i] += np.random.normal(0, 5)

            new_population.append(child)

        population = new_population[:population_size]

    return best_individual, best_fitness, best_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Define bounds for optimization
    bounds = []
    for i in range(11):  # 11 hexagons
        bounds.extend([
            (-10, 10),  # x coordinate
            (-10, 10),  # y coordinate
            (-180, 180)  # angle degrees
        ])

    # Run evolutionary optimization
    try:
        best_config, best_fitness, best_radius = evolutionary_search(
            n_generations=30,
            population_size=15,
            n_hexagons=11,
            bounds=bounds
        )

        # Convert back to proper format
        inner_hex_data = best_config.reshape(-1, 3)

        # Final validation with refined parameters
        final_valid = validate_configuration(inner_hex_data, np.array([0., 0., 0.]), best_radius)

        if not final_valid:
            # Fallback to simple configuration if validation fails
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
                [3.75, -2.17, 0]
            ])
            outer_radius = 8.0
        else:
            outer_radius = best_radius

    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}, using fallback")
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
            [3.75, -2.17, 0]
        ])
        outer_radius = 8.0

    # Final cleanup and validation
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    # Ensure we have a valid result within time limits
    end_time = time.time()
    if end_time - start_time > 175:  # Leave some buffer
        warnings.warn("Time limit approaching, returning best available result")

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END