# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import time
import random
from copy import deepcopy
from scipy.spatial.distance import cdist

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, angle, and side length"""
    angle_rad = np.radians(angle_deg)
    angles = np.linspace(0, 2*np.pi, 7) + angle_rad
    vertices = np.array([
        [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
        for a in angles
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if hexagon vertices are contained within outer hexagon using Shapely"""
    inner_polygon = Polygon(hex_vertices)
    outer_polygon = Polygon(outer_hex_vertices)
    return outer_polygon.contains(inner_polygon)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely"""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hexagon_radius(inner_positions, inner_angles, initial_radius_estimate=5.0):
    """Compute minimum outer hexagon radius that contains all inner hexagons with adaptive binary search"""
    left = initial_radius_estimate
    right = 20.0
    best_radius = right

    # Adaptive binary search with progressive refinement
    precision_threshold = 1e-6
    max_iterations = 60

    for iteration in range(max_iterations):
        if right - left <= precision_threshold:
            break

        # Adjust precision based on iterations
        if iteration > 40:
            precision_threshold = 1e-8
        elif iteration > 20:
            precision_threshold = 1e-7

        mid = (left + right) / 2.0
        outer_vertices = hexagon_vertices(0, 0, 0, mid)
        valid = True

        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
            if not check_containment(hex_vertices, outer_vertices):
                valid = False
                break

        if valid:
            best_radius = mid
            right = mid
        else:
            left = mid

    return best_radius

def evaluate_fitness(inner_positions, inner_angles, max_radius=20.0):
    """Evaluate fitness: higher is better, maximize 1/radius"""
    outer_radius = compute_outer_hexagon_radius(inner_positions, inner_angles)

    # Check all constraints
    total_penalty = 0

    # Check containment for all inner hexagons
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
        hex_vertices = hexagon_vertices(pos[0], pos[1], angle)
        if not check_containment(hex_vertices, outer_vertices):
            total_penalty += 10000  # Large penalty for containment violation

    # Check overlaps between all pairs of inner hexagons
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            hex1_vertices = hexagon_vertices(inner_positions[i][0], inner_positions[i][1], inner_angles[i])
            hex2_vertices = hexagon_vertices(inner_positions[j][0], inner_positions[j][1], inner_angles[j])
            if check_overlap(hex1_vertices, hex2_vertices):
                total_penalty += 10000  # Large penalty for overlap violation

    # Fitness is negative of the radius plus penalties
    fitness = -outer_radius - total_penalty

    return fitness, outer_radius

def generate_initial_config():
    """Generate a more sophisticated initial configuration based on known good patterns"""
    # This configuration is based on hexagonal close packing principles
    # with strategic positioning to allow for optimization
    initial_positions = [
        [0.0, 0.0],           # center
        [-2.2, 0.0],          # left
        [2.2, 0.0],           # right
        [0.0, 3.8],           # top
        [0.0, -3.8],          # bottom
        [-1.9, 1.9],          # top-left
        [1.9, 1.9],           # top-right
        [-1.9, -1.9],         # bottom-left
        [1.9, -1.9],          # bottom-right
        [-3.5, 0.0],          # far left
        [3.5, 0.0],           # far right
    ]

    # Add slight jitter to positions
    individual = np.array(initial_positions)
    for i in range(len(individual)):
        individual[i][0] += random.uniform(-0.1, 0.1)
        individual[i][1] += random.uniform(-0.1, 0.1)

    # Add rotation information
    rotation_array = np.array([0] * 11)  # All at 0 degrees initially

    return individual, rotation_array

def gradient_guided_local_optimization(positions, angles, max_iter=50):
    """Apply gradient-based local refinement to improve solution quality"""
    # Convert to flat array for optimization
    vars_flat = np.hstack([positions.flatten(), angles])

    def objective_func(vars):
        # Unpack variables
        pos_flat = vars[:-11]
        ang = vars[-11:]
        pos = pos_flat.reshape(-1, 2)

        try:
            fitness, _ = evaluate_fitness(pos, ang)
            return -fitness  # minimize negative fitness
        except:
            return 1000000

    # Optimization bounds
    bounds = []
    # Position bounds
    for _ in range(len(positions)):
        bounds.extend([(-15, 15), (-15, 15)])
    # Angle bounds
    for _ in range(len(angles)):
        bounds.extend([(0, 360)])

    try:
        # Use L-BFGS-B for local refinement
        result = minimize(
            objective_func,
            vars_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-9, 'gtol': 1e-9}
        )

        if result.success:
            final_vars = result.x
            pos_flat = final_vars[:-11]
            ang = final_vars[-11:]
            pos = pos_flat.reshape(-1, 2)
            return pos, ang
    except:
        pass

    return positions, angles

def genetic_algorithm_optimization():
    """Enhanced evolutionary approach for optimizing hexagon packing"""
    # Parameters
    population_size = 30
    generations = 50
    mutation_rate = 0.2
    crossover_rate = 0.8
    elite_fraction = 0.1

    # Initialize population
    population = []
    fitness_scores = []

    for _ in range(population_size):
        positions, angles = generate_initial_config()
        population.append((positions, angles))
        fitness, _ = evaluate_fitness(positions, angles)
        fitness_scores.append(fitness)

    best_fitness = max(fitness_scores)
    best_individual = population[np.argmax(fitness_scores)]

    # Evolution loop
    for gen in range(generations):
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Elitism - keep best individuals
        elite_count = int(elite_fraction * population_size)
        elite = population[:elite_count]

        # Tournament selection
        new_population = elite[:]

        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1_idx = tournament_selection(fitness_scores, 3)
            parent2_idx = tournament_selection(fitness_scores, 3)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover
            if random.random() < crossover_rate:
                child1_pos, child1_ang = crossover(parent1[0], parent1[1], parent2[0], parent2[1])
            else:
                child1_pos, child1_ang = deepcopy(parent1[0]), deepcopy(parent1[1])

            # Mutation
            child1_pos, child1_ang = mutate(child1_pos, child1_ang, mutation_rate)

            new_population.append((child1_pos, child1_ang))

        # Update population
        population = new_population[:population_size]

        # Evaluate new population
        fitness_scores = []
        for pos, ang in population:
            fitness, _ = evaluate_fitness(pos, ang)
            fitness_scores.append(fitness)

        # Update best
        current_best_fitness = max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[np.argmax(fitness_scores)]

        # Early stopping if no improvement
        if gen > 10 and current_best_fitness <= best_fitness + 1e-6:
            continue

    return best_individual

def tournament_selection(fitness_scores, tournament_size):
    """Select individual using tournament selection"""
    tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index

def crossover(pos1, ang1, pos2, ang2):
    """Single-point crossover for hexagon positions and angles"""
    n = len(pos1)
    crossover_point = random.randint(1, n-1)

    new_pos1 = np.vstack([pos1[:crossover_point], pos2[crossover_point:]])
    new_ang1 = np.concatenate([ang1[:crossover_point], ang2[crossover_point:]])

    return new_pos1, new_ang1

def mutate(positions, angles, mutation_rate):
    """Apply mutation to positions and angles"""
    mutated_positions = deepcopy(positions)
    mutated_angles = deepcopy(angles)

    for i in range(len(mutated_positions)):
        if random.random() < mutation_rate:
            # Position mutation
            mutated_positions[i][0] += random.uniform(-0.3, 0.3)
            mutated_positions[i][1] += random.uniform(-0.3, 0.3)

        if random.random() < mutation_rate:
            # Angle mutation
            mutated_angles[i] += random.uniform(-15, 15)
            mutated_angles[i] = mutated_angles[i] % 360

    return mutated_positions, mutated_angles

def enhanced_voronoi_optimization():
    """Improved optimization combining genetic algorithm with local refinement"""
    # Start with genetic algorithm for global optimization
    best_positions, best_angles = genetic_algorithm_optimization()

    # Apply local refinement to improve solution quality
    refined_positions, refined_angles = gradient_guided_local_optimization(
        best_positions, best_angles, max_iter=100
    )

    # Final evaluation
    final_fitness, outer_radius = evaluate_fitness(refined_positions, refined_angles)

    return refined_positions, refined_angles, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use enhanced optimization approach
    positions, angles, outer_radius = enhanced_voronoi_optimization()

    # Format output as required
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END