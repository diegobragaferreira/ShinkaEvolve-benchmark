# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from collections import defaultdict

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin
    base_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = base_vertices @ rotation_matrix.T
    translated_vertices = rotated_vertices + np.array([center_x, center_y])

    return translated_vertices

def check_containment(hexagon_poly, outer_hex_poly):
    """Check if hexagon is fully contained within outer hexagon"""
    return outer_hex_poly.contains(hexagon_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap"""
    return hex1_poly.intersects(hex2_poly)

def get_bounding_box(vertices):
    """Get axis-aligned bounding box for a set of vertices"""
    if len(vertices) == 0:
        return (0, 0, 0, 0)
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    return (min_x, min_y, max_x, max_y)

def boxes_overlap(box1, box2):
    """Check if two axis-aligned boxes overlap"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

def create_spatial_index(hex_polygons, hex_vertices_list):
    """Create a spatial index for fast collision detection"""
    # Create bounding boxes for all hexagons
    boxes = []
    for i, vertices in enumerate(hex_vertices_list):
        box = get_bounding_box(vertices)
        boxes.append((box, i))

    # Group boxes into grid cells for faster lookup
    cell_size = 2.0  # Based on hexagon diameter
    spatial_index = defaultdict(list)

    for box, idx in boxes:
        # Grid cell coordinates
        min_x, min_y, max_x, max_y = box
        cell_min_x = int(min_x // cell_size)
        cell_min_y = int(min_y // cell_size)
        cell_max_x = int(max_x // cell_size)
        cell_max_y = int(max_y // cell_size)

        # Add to all relevant grid cells
        for cx in range(cell_min_x, cell_max_x + 1):
            for cy in range(cell_min_y, cell_max_y + 1):
                spatial_index[(cx, cy)].append(idx)

    return spatial_index, boxes

def calculate_outer_hex_side_length(inner_hex_data):
    """Calculate minimum side length of outer hexagon needed to contain all inner hexagons"""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = generate_hexagon_vertices(center_x, center_y, angle)
        all_vertices.extend(vertices)

    all_vertices = np.array(all_vertices)

    # Find bounding box
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])

    # Calculate approximate side length (simplified approach)
    # A hexagon with side length s has width 2*s and height sqrt(3)*s
    width = max_x - min_x
    height = max_y - min_y

    # Estimate side length from dimensions
    side_len_width = width / 2.0
    side_len_height = height / (np.sqrt(3))

    # Take maximum to ensure containment
    estimated_side_length = max(side_len_width, side_len_height) * 1.1  # Add small buffer

    return estimated_side_length

def evaluate_solution(inner_hex_data):
    """Evaluate fitness of solution - maximize 1/outer_hex_side_length"""
    # Check for overlaps and containment
    try:
        # Create polygons for all inner hexagons
        hex_polygons = []
        hex_vertices_list = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle = inner_hex_data[i]
            vertices = generate_hexagon_vertices(center_x, center_y, angle)
            hex_vertices_list.append(vertices)
            hex_polygons.append(Polygon(vertices))

        # Check containment and overlap
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
        outer_vertices = generate_hexagon_vertices(0, 0, 0, outer_side_length)
        outer_polygon = Polygon(outer_vertices)

        # Check containment
        for poly in hex_polygons:
            if not check_containment(poly, outer_polygon):
                return 0.0  # Invalid - not fully contained

        # Check overlaps using spatial indexing for efficiency
        spatial_index, boxes = create_spatial_index(hex_polygons, hex_vertices_list)

        # Only check actual collisions for potentially overlapping hexagons
        for i in range(len(hex_polygons)):
            box_i = boxes[i]
            # Get nearby candidates from spatial index
            candidates = set()
            min_x, min_y, max_x, max_y = box_i[0]
            cell_size = 2.0
            cell_min_x = int(min_x // cell_size)
            cell_min_y = int(min_y // cell_size)
            cell_max_x = int(max_x // cell_size)
            cell_max_y = int(max_y // cell_size)

            for cx in range(cell_min_x, cell_max_x + 1):
                for cy in range(cell_min_y, cell_max_y + 1):
                    candidates.update(spatial_index[(cx, cy)])

            # Check actual overlaps with candidates
            for j in candidates:
                if i >= j:  # Avoid duplicate checks and self-checks
                    continue
                if boxes_overlap(box_i[0], boxes[j][0]):  # Only check if bounding boxes overlap
                    if check_overlap(hex_polygons[i], hex_polygons[j]):
                        return 0.0  # Invalid - overlaps

        # Return 1/outer_side_length as fitness
        return 1.0 / outer_side_length if outer_side_length > 0 else 0.0

    except Exception:
        return 0.0

def create_initial_population(pop_size, n_inner_hex):
    """Create initial population with diverse arrangements"""
    population = []
    # Generate multiple good starting solutions
    for _ in range(pop_size):
        # Random placement with some clustering around central region
        individual = []
        for i in range(n_inner_hex):
            # Center hexagons more tightly clustered
            if i == 0:  # Center hexagon
                center_x, center_y = 0.0, 0.0
                angle = random.uniform(0, 360)
            elif i <= 6:  # Around center with regular spacing
                distance = random.uniform(1.0, 2.5)
                angle = random.uniform(0, 360)
                center_x = distance * np.cos(np.radians(angle))
                center_y = distance * np.sin(np.radians(angle))
                angle = random.uniform(0, 360)
            else:  # Outer ring
                distance = random.uniform(3.0, 5.0)
                angle = random.uniform(0, 360)
                center_x = distance * np.cos(np.radians(angle))
                center_y = distance * np.sin(np.radians(angle))
                angle = random.uniform(0, 360)

            individual.append([center_x, center_y, angle])

        population.append(np.array(individual))
    return population

def mutate_individual(individual, mutation_rate=0.1, max_disp=0.5):
    """Mutate an individual by slightly changing positions and angles"""
    mutated = individual.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position
            mutated[i][0] += random.uniform(-max_disp, max_disp)  # x
            mutated[i][1] += random.uniform(-max_disp, max_disp)  # y
            # Mutate angle
            mutated[i][2] += random.uniform(-30, 30)  # angle in degrees
            mutated[i][2] %= 360  # Keep angle in [0,360)

    return mutated

def local_refinement(individual, max_iter=50):
    """Apply local refinement to improve individual quality"""
    # Simple gradient-free local search using coordinate descent
    best_individual = individual.copy()
    best_fitness = evaluate_solution(best_individual)

    for _ in range(max_iter):
        improved = False
        # Try small perturbations to each parameter
        for i in range(len(best_individual)):
            for j in range(3):  # x, y, angle
                original_value = best_individual[i][j]
                # Try small positive and negative steps
                steps = [-0.05, 0.05]
                for step in steps:
                    test_individual = best_individual.copy()
                    if j < 2:  # x or y
                        test_individual[i][j] = original_value + step
                    else:  # angle
                        test_individual[i][j] = (original_value + step) % 360

                    fitness = evaluate_solution(test_individual)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = test_individual
                        improved = True

        if not improved:
            break

    return best_individual

def crossover_parents(parent1, parent2, crossover_rate=0.8):
    """Crossover parents to produce offspring"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Single point crossover
    crossover_point = random.randint(1, len(parent1) - 1)

    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return offspring1, offspring2

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    start_time = time.time()
    max_time = 175  # Leave some margin for cleanup

    n = 11
    pop_size = 50
    generations = 200
    elite_size = 5

    # Create initial population
    population = create_initial_population(pop_size, n)

    best_fitness = 0.0
    best_individual = None

    # Evolution loop
    for gen in range(generations):
        if time.time() - start_time > max_time:
            break

        # Evaluate fitness
        fitness_scores = []
        for ind in population:
            fitness = evaluate_solution(ind)
            fitness_scores.append(fitness)

        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Create new generation
        new_population = []
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 3
            parent1_idx = random.choices(range(elite_size*2), k=tournament_size)[0]
            parent2_idx = random.choices(range(elite_size*2), k=tournament_size)[0]

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            child1, child2 = crossover_parents(parent1, parent2)

            # Apply mutations
            child1 = mutate_individual(child1, mutation_rate=0.15)
            child2 = mutate_individual(child2, mutation_rate=0.15)

            # Apply local refinement to offspring
            child1 = local_refinement(child1, max_iter=25)
            child2 = local_refinement(child2, max_iter=25)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

        # Adaptive mutation rate based on diversity
        if gen > 50 and gen % 20 == 0:
            diversity = np.std([evaluate_solution(ind) for ind in population])
            if diversity < 0.001:
                # Reduce mutation rate if not diverging
                pass  # Keep it consistent for this version

    # Final evaluation of best solution
    if best_individual is None:
        # Fallback to initial solution if optimization failed
        best_individual = np.array([
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

    outer_side_length = 1.0 / best_fitness if best_fitness > 0 else 8.0

    # Ensure valid outer hexagon side length
    if outer_side_length > 100:
        outer_side_length = 10.0

    # Center the outer hexagon at origin
    outer_hex_data = np.array([0, 0, 0])

    return best_individual, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END