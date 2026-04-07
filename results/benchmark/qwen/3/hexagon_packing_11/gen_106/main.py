# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution, minimize
from joblib import Parallel, delayed
import warnings
from numba import jit, prange

@jit(nopython=True)
def generate_hexagon_vertices_numba(center_x, center_y, side_length, rotation_rad):
    """Fast JIT version of hexagon vertex generation using Numba."""
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices_x = np.empty(6)
    vertices_y = np.empty(6)
    for i in range(6):
        vertices_x[i] = center_x + side_length * np.cos(angles[i])
        vertices_y[i] = center_y + side_length * np.sin(angles[i])
    return vertices_x, vertices_y

@jit(nopython=True)
def point_in_hexagon_numba(px, py, hex_center_x, hex_center_y, side_length, rotation_rad):
    """Fast point-in-hexagon check using analytical geometry."""
    # Transform point to hexagon reference frame
    dx = px - hex_center_x
    dy = py - hex_center_y

    # Rotate back to hexagon's local coordinate system
    cos_r = np.cos(-rotation_rad)
    sin_r = np.sin(-rotation_rad)
    local_x = dx * cos_r - dy * sin_r
    local_y = dx * sin_r + dy * cos_r

    # For a regular hexagon with side length s, the distance from center to corner is s
    # and distance from center to edge is s * cos(π/6) = s * √3/2
    edge_distance = side_length * np.sqrt(3) / 2

    # Distance from center
    dist_to_center = np.sqrt(local_x**2 + local_y**2)

    # If too far, definitely not in hexagon
    if dist_to_center > side_length:
        return False

    # If close to center, definitely in hexagon
    if dist_to_center < edge_distance:
        return True

    # Check if point is in any of the 6 sectors of the hexagon
    # We'll check if it's on the correct side of each edge
    # Edge vectors of the hexagon in local coordinates
    for i in range(6):
        angle1 = i * np.pi / 3
        angle2 = ((i+1) % 6) * np.pi / 3
        x1 = side_length * np.cos(angle1)
        y1 = side_length * np.sin(angle1)
        x2 = side_length * np.cos(angle2)
        y2 = side_length * np.sin(angle2)

        # Vector from point to first edge point
        v1x = x1 - local_x
        v1y = y1 - local_y
        # Vector along edge
        v2x = x2 - x1
        v2y = y2 - y1

        # Cross product to determine which side of the edge the point is on
        cross_product = v1x * v2y - v1y * v2x

        # For the correct side, cross product should be non-negative (point on the inside of the edge)
        if cross_product < 0:  # Point is on wrong side of this edge
            return False

    return True

@jit(nopython=True)
def hexagon_overlap_numba(hex1_center_x, hex1_center_y, hex1_side, hex1_rot,
                         hex2_center_x, hex2_center_y, hex2_side, hex2_rot):
    """Fast hexagon overlap check."""
    # For simplicity, we'll check if distance between centers is less than sum of radii
    # This gives a quick approximation for overlap detection
    # More precise check would involve checking if any vertices are within the other hexagon
    # But we're optimizing for speed here

    # Distances between centers (this is conservative)
    dx = hex1_center_x - hex2_center_x
    dy = hex1_center_y - hex2_center_y
    distance = np.sqrt(dx*dx + dy*dy)

    # For unit hexagons, the distance between their centers should be at least ~2
    # to avoid overlap (when touching)
    # But for better precision, let's use a more elaborate check

    # The hexagon has a radius of 1 (distance from center to corner)
    # For touching hexagons, distance between centers is 2
    # So if distance < 2, they overlap
    return distance < 2.0

def generate_hexagon_vertices(center_x, center_y, side_length=1, rotation_deg=0):
    """Generate vertices of a regular hexagon given center, side length, and rotation."""
    rotation_rad = np.radians(rotation_deg)
    angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
    vertices = np.column_stack([
        center_x + side_length * np.cos(angles),
        center_y + side_length * np.sin(angles)
    ])
    return vertices[:-1]  # Remove duplicate last vertex

def check_containment_fast(hexagon_vertices, outer_hex_vertices):
    """Fast containment check using Numba JIT."""
    # Get outer hexagon center and radius (approximate)
    outer_center_x = np.mean(outer_hex_vertices[:, 0])
    outer_center_y = np.mean(outer_hex_vertices[:, 1])

    # Simple fast check: if the center of the inner hex is too far from outer hex center,
    # we can do a quicker check
    inner_center_x = np.mean(hexagon_vertices[:, 0])
    inner_center_y = np.mean(hexagon_vertices[:, 1])

    # Quick distance check from centers
    dx = inner_center_x - outer_center_x
    dy = inner_center_y - outer_center_y
    distance_from_center = np.sqrt(dx*dx + dy*dy)

    # If center is too far away, definitely not contained
    # Let's estimate outer radius based on first few vertices
    outer_vertices = np.array(outer_hex_vertices)
    max_outer_dist = 0
    for vertex in outer_vertices:
        dist = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
        max_outer_dist = max(max_outer_dist, dist)

    # If inner center is further than max outer distance plus hex radius, not contained
    if distance_from_center > max_outer_dist + 1.0:
        return False

    # More precise check with Numba
    outer_center_x_numba = float(outer_center_x)
    outer_center_y_numba = float(outer_center_y)

    # Check each vertex manually with numba for speed
    for vertex in hexagon_vertices:
        px, py = vertex[0], vertex[1]
        # Since we don't have a numba version of point in polygon for our case,
        # we'll rely on the fact that if we've passed the center check,
        # and the overall arrangement is valid, this should be mostly accurate
        # For now, we'll keep using the original slower but reliable approach
        pass

    # Fall back to original approach for reliability
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False

    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using Numba."""
    # Check if two hexagons overlap
    # For now, let's stick with the original reliable approach
    # But we can optimize the distance check for performance

    # Compute centroids
    centroid1 = np.mean(hex1_vertices, axis=0)
    centroid2 = np.mean(hex2_vertices, axis=0)

    # Quick distance check
    dx = centroid1[0] - centroid2[0]
    dy = centroid1[1] - centroid2[1]
    distance = np.sqrt(dx*dx + dy*dy)

    # If distance is greater than sum of circumradii, no overlap
    # For unit hexagons, circumradius is 1
    if distance >= 2.0:
        return False

    # Use Shapely for final determination (more reliable)
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
        if not check_containment_fast(inner_vertices, outer_vertices):
            return False

        # Check overlaps with all other hexagons
        for j in range(i+1, len(inner_hex_data)):
            center_x2, center_y2, angle2 = inner_hex_data[j]
            inner_vertices2 = generate_hexagon_vertices(center_x2, center_y2, 1, angle2)

            if check_overlap_fast(inner_vertices, inner_vertices2):
                return False

    return True

def compute_outer_hexagon_radius(inner_hex_data, outer_hex_data):
    """Compute minimum radius needed to contain all inner hexagons."""
    # Generate outer hexagon vertices - use a large radius for this computation
    outer_vertices = generate_hexagon_vertices(
        outer_hex_data[0], outer_hex_data[1], 1000, outer_hex_data[2]
    )

    # Find max distance from outer center to any inner hexagon vertex
    max_dist = 0
    outer_center_x = np.mean(outer_vertices[:, 0])
    outer_center_y = np.mean(outer_vertices[:, 1])

    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_vertices = generate_hexagon_vertices(center_x, center_y, 1, angle)
        distances = cdist([[outer_center_x, outer_center_y]], inner_vertices)[0]
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

def generate_initial_config_v2(n_hexagons=11):
    """Generate an improved initial configuration with better hexagonal tiling."""
    # Create a more strategic arrangement based on hexagonal tiling principles
    # Central hexagon surrounded by first ring, then second ring
    
    # Start with a better configuration based on mathematical principles
    # Central hexagon
    positions = [[0.0, 0.0, 0.0]]
    
    # First ring (6 hexagons) - arranged in a perfect hexagonal pattern
    for i in range(6):
        angle = i * np.pi / 3
        # Distance between centers of touching hexagons is 2
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        positions.append([x, y, 0.0])
    
    # Second ring (4 hexagons) - strategically placed to reduce space
    # Following the pattern of hexagonal close packing
    second_ring_positions = [
        [-1.0, -1.732, 0.0],   # Bottom left
        [1.0, -1.732, 0.0],    # Bottom right  
        [-1.0, 1.732, 0.0],    # Top left
        [1.0, 1.732, 0.0],     # Top right
    ]
    
    positions.extend(second_ring_positions)
    
    # Take only the first 11 positions
    initial_positions = np.array(positions[:11])
    
    # Add subtle randomized perturbations to avoid getting stuck in local optima
    np.random.seed(42)
    for i in range(len(initial_positions)):
        # Small random perturbations
        initial_positions[i, 0] += np.random.uniform(-0.1, 0.1)
        initial_positions[i, 1] += np.random.uniform(-0.1, 0.1)
        initial_positions[i, 2] += np.random.uniform(-5, 5)
    
    return initial_positions.flatten()

def local_optimization_step(config, bounds, outer_center=(0, 0), outer_angle=0):
    """Perform enhanced local optimization with better convergence."""
    def objective(x):
        # Reshape to proper format
        test_config = x.reshape(-1, 3)
        fitness = evaluate_fitness(test_config, outer_center, outer_angle)
        return -fitness  # Negative because we want to maximize

    # Use L-BFGS-B for local refinement with better parameters
    try:
        result = minimize(objective, config, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        return result.x
    except:
        # If optimization fails, return original config
        return config

def evolutionary_search_v2(n_generations=40, population_size=25, n_hexagons=11, bounds=None):
    """Enhanced evolutionary algorithm for hexagon packing optimization."""
    if bounds is None:
        # Create bounds for each parameter
        bounds_list = []
        for _ in range(n_hexagons):
            bounds_list.extend([(-10, 10), (-10, 10), (-180, 180)])  # x, y, angle
        bounds = bounds_list

    # Initialize population with improved starting points
    population = []
    for i in range(population_size):
        individual = generate_initial_config_v2(n_hexagons)
        population.append(individual)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None
    best_radius = float('inf')
    fitness_history = []
    stagnation_count = 0
    max_stagnation = 15  # Early stopping if no improvement

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
            stagnation_count = 0  # Reset stagnation count
        else:
            stagnation_count += 1

        fitness_history.append(best_fitness)
        
        # Early stopping if no improvement in recent generations
        if stagnation_count >= max_stagnation:
            break

        # Adaptive mutation rate that decreases over time and increases with diversity
        generation_factor = gen / n_generations
        base_mutation_rate = 0.2  # Starting high for exploration
        mutation_decay = 0.95  # Slower decay for sustained exploration
        mutation_rate = max(0.05, base_mutation_rate * (mutation_decay ** gen))
        
        # Create new population through selection and mutation
        # Tournament selection with larger tournaments for more pressure
        new_population = []
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order

        # Elitism: keep top 15%
        elite_count = max(1, int(0.15 * population_size))
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]])

        # Generate rest by crossover and mutation
        while len(new_population) < population_size:
            # Select parents with tournament selection
            tournament_size = 5  # Larger tournament for better selection pressure
            parent1_idx = sorted_indices[np.random.randint(0, tournament_size)]
            parent2_idx = sorted_indices[np.random.randint(0, tournament_size)]

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover: blend between parents
            alpha = 0.7
            child = parent1 * alpha + parent2 * (1 - alpha)

            # Mutation with adaptive rate
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

    # Run enhanced evolutionary optimization
    try:
        best_config, best_fitness, best_radius = evolutionary_search_v2(
            n_generations=40,
            population_size=25,
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