# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import cKDTree
import time
from numba import jit
from joblib import Parallel, delayed
import random
from copy import deepcopy

@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(hex_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hex_poly) or outer_hex_poly.covers(hex_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def calculate_hexagon_distance(h1_center, h2_center, h1_angle, h2_angle):
    """Calculate minimum distance between two hexagons."""
    # Simplified distance calculation for early rejection
    dx = h1_center[0] - h2_center[0]
    dy = h1_center[1] - h2_center[1]
    return np.sqrt(dx*dx + dy*dy)

def get_hexagon_bounds(hex_poly):
    """Get bounding box of hexagon."""
    bounds = hex_poly.bounds
    return np.array([bounds[0], bounds[1], bounds[2], bounds[3]])

def fast_overlapping_check(hex1_poly, hex2_poly, tree=None, hex_indices=None):
    """Fast overlap checking using bounding boxes and spatial indexing."""
    if tree is not None and hex_indices is not None:
        # Use spatial indexing for faster preliminary filtering
        bounds1 = get_hexagon_bounds(hex1_poly)
        bounds2 = get_hexagon_bounds(hex2_poly)

        # Quick bounds check
        if (bounds1[2] < bounds2[0] or bounds1[0] > bounds2[2] or
            bounds1[3] < bounds2[1] or bounds1[1] > bounds2[3]):
            return False

    # Final precise overlap check
    return check_overlap(hex1_poly, hex2_poly)

def evaluate_individual(individual, outer_radius):
    """Evaluate fitness of an individual configuration with spatial indexing."""
    # Decode individual into 12 hexagon parameters
    hex_data = individual.reshape(12, 3)

    # Create outer hexagon (centered at origin)
    outer_hex_poly = compute_hexagon_polygon(0, 0, 0, outer_radius)

    # Initialize variables
    valid = True
    total_penalty = 0.0

    # Create list of hexagon polygons
    hex_polygons = []
    hex_centers = []

    # Process each hexagon
    for i in range(12):
        x, y, angle = hex_data[i]
        hex_poly = compute_hexagon_polygon(x, y, angle)
        hex_polygons.append(hex_poly)
        hex_centers.append([x, y])

        # Check containment
        if not check_containment(hex_poly, outer_hex_poly):
            valid = False
            # Calculate penetration
            try:
                diff = outer_hex_poly.difference(hex_poly)
                if hasattr(diff, 'area'):
                    total_penalty += diff.area
            except:
                total_penalty += 10000

    if not valid:
        return 1e10 + total_penalty * 10000

    # Build spatial index for efficient overlap checking
    centers_array = np.array(hex_centers)
    tree = cKDTree(centers_array)

    # Check pairwise overlaps using spatial indexing
    # Only check neighbors within a certain distance
    max_radius = 2.0  # Maximum distance for potential overlaps (approximately 2 hexagon radii)
    pairs = tree.query_pairs(max_radius, output_type='ndarray')

    # Check overlap for each pair found by spatial index
    for i, j in pairs:
        if i < j:  # Avoid duplicate checks
            if check_overlap(hex_polygons[i], hex_polygons[j]):
                valid = False
                try:
                    overlap = hex_polygons[i].intersection(hex_polygons[j])
                    if hasattr(overlap, 'area'):
                        total_penalty += overlap.area
                except:
                    total_penalty += 10000

    # Also check any remaining pairs that might not have been caught by spatial indexing
    # This ensures correctness while maintaining efficiency
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in zip(pairs[:, 0], pairs[:, 1]):  # Skip if already checked
                if check_overlap(hex_polygons[i], hex_polygons[j]):
                    valid = False
                    try:
                        overlap = hex_polygons[i].intersection(hex_polygons[j])
                        if hasattr(overlap, 'area'):
                            total_penalty += overlap.area
                    except:
                        total_penalty += 10000

    if not valid:
        return 1e10 + total_penalty * 10000

    # Return negative inverse outer radius (we want to maximize 1/R)
    return -1.0 / outer_radius

def generate_initial_population(pop_size, bounds):
    """Generate initial population with symmetry awareness."""
    population = []

    # Generate a few good starting configurations with symmetry
    for _ in range(pop_size):
        individual = np.zeros((12, 3))

        # Start with symmetric pattern
        # Central hexagon
        individual[0] = [0, 0, 0]

        # First ring around center
        ring_positions = []
        angles = [0, 60, 120, 180, 240, 300]
        for i, angle in enumerate(angles):
            rad_angle = np.radians(angle)
            x = 1.75 * np.cos(rad_angle)
            y = 1.75 * np.sin(rad_angle)
            individual[i+1] = [x, y, 0]

        # Second ring
        angles2 = [30, 90, 150, 210, 270, 330]
        for i, angle in enumerate(angles2):
            rad_angle = np.radians(angle)
            x = 3.0 * np.cos(rad_angle)
            y = 3.0 * np.sin(rad_angle)
            individual[i+7] = [x, y, 0]

        # Add randomness
        for i in range(12):
            individual[i][0] += np.random.normal(0, 0.2)
            individual[i][1] += np.random.normal(0, 0.2)
            individual[i][2] += np.random.uniform(-15, 15)

        population.append(individual.flatten())

    return population

def crossover(parent1, parent2, crossover_rate=0.8):
    """Custom crossover operator for hexagon packing."""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Uniform crossover with symmetry preservation
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Split into three groups for more meaningful crossover
    group_size = 4  # Each group has 4 hexagons
    for group_idx in range(3):  # 3 groups of 4 hexagons each
        if random.random() > 0.5:
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size

            # Swap groups between parents
            child1[start_idx:end_idx] = parent2[start_idx:end_idx]
            child2[start_idx:end_idx] = parent1[start_idx:end_idx]

    return child1, child2

def mutate(individual, mutation_rate=0.1, bounds=None):
    """Custom mutation operator with symmetry-aware perturbations."""
    mutated = individual.copy()

    # Determine number of mutations
    num_mutations = int(len(individual) * mutation_rate)
    indices_to_mutate = random.sample(range(len(individual)), num_mutations)

    for idx in indices_to_mutate:
        # Apply different mutation strategies based on index type
        if idx % 3 == 0:  # x coordinate
            mutated[idx] += np.random.normal(0, 0.3)
        elif idx % 3 == 1:  # y coordinate
            mutated[idx] += np.random.normal(0, 0.3)
        else:  # angle
            mutated[idx] += np.random.normal(0, 5)
            # Keep angle in [0, 360]
            mutated[idx] %= 360

    return mutated

def evolve_hexagon_packing():
    """Evolutionary algorithm for hexagon packing."""
    # Parameters
    pop_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_size = 5

    # Bounds for variables
    bounds = [(-10, 10), (-10, 10), (0, 360)] * 12  # x, y, angle for 12 hexagons

    # Initialize population
    population = generate_initial_population(pop_size, bounds)

    # Track best solution
    best_fitness = float('inf')
    best_individual = None
    best_outer_radius = 5.0

    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness of entire population
        fitness_scores = []
        for ind in population:
            # For now, use a fixed outer radius estimate
            estimated_radius = 4.0 + gen * 0.01  # Gradually increase
            fitness = evaluate_individual(ind.reshape(12, 3), estimated_radius)
            fitness_scores.append((fitness, ind))

        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0])

        # Update best solution
        current_best_fitness, current_best_ind = fitness_scores[0]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_ind.copy()
            best_outer_radius = 4.0 + gen * 0.01  # Adjust based on generation

        # Elitism: keep best individuals
        elite = [ind for _, ind in fitness_scores[:elite_size]]

        # Generate new population
        new_population = elite[:]

        # Fill rest of population through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(fitness_scores, 3)
            parent2 = tournament_selection(fitness_scores, 3)

            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            # Mutation
            child1 = mutate(child1, mutation_rate, bounds)
            child2 = mutate(child2, mutation_rate, bounds)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

    # Final refinement of best solution
    if best_individual is not None:
        final_individual = best_individual.reshape(12, 3)

        # Use binary search to find optimal outer radius
        low = 3.0
        high = 8.0
        tolerance = 0.001

        for _ in range(20):  # Binary search iterations
            mid = (low + high) / 2
            fitness = evaluate_individual(final_individual, mid)
            if fitness < 0:  # Valid solution
                high = mid
            else:
                low = mid

        optimal_radius = (low + high) / 2
        return final_individual, optimal_radius

    # Fallback to initial configuration
    return np.array([
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
        [0, -4, 0],
    ]), 8.0

def tournament_selection(fitness_scores, k):
    """Select individual using tournament selection."""
    selected = random.sample(fitness_scores, k)
    selected.sort(key=lambda x: x[0])
    return selected[0][1]

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Track execution time
    start_time = time.time()

    try:
        # Evolve solution
        inner_hex_data, outer_hex_side_length = evolve_hexagon_packing()

        # Create outer hexagon data (centered at origin, no rotation)
        outer_hex_data = np.array([0, 0, 0])

        # Ensure we don't exceed time limits
        end_time = time.time()
        eval_time = end_time - start_time

        return inner_hex_data, outer_hex_data, outer_hex_side_length

    except Exception as e:
        # Fallback to original configuration if optimization fails
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
            [0, -4, 0],
        ])

        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8  # Large enough to contain all inner hexagons

        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END