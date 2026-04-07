# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
import time
from copy import deepcopy

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
    """Compute minimum outer hexagon radius that contains all inner hexagons with adaptive precision"""
    # Binary search for tightest fit with adaptive precision
    left = initial_radius_estimate
    right = 20.0
    best_radius = right

    # Use a more efficient adaptive approach with early termination
    precision_threshold = 1e-6
    max_iterations = 100

    iterations = 0
    while right - left > precision_threshold and iterations < max_iterations:
        mid = (left + right) / 2.0
        outer_vertices = hexagon_vertices(0, 0, 0, mid)
        valid = True

        # Check all inner hexagons
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
        iterations += 1

    return best_radius

def evaluate_fitness(inner_positions, inner_angles, max_radius=20.0):
    """Evaluate fitness: higher is better, maximize 1/radius"""
    # Create outer hexagon vertices
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
    # We want to minimize radius, so fitness = -radius
    fitness = -outer_radius - total_penalty

    return fitness, outer_radius

def mutate_individual(individual, mutation_rate=0.1, max_displacement=0.5):
    """Mutate individual with position and rotation changes"""
    mutated = deepcopy(individual)
    n = len(mutated)

    for i in range(n):
        # Mutate position
        if random.random() < mutation_rate:
            mutated[i][0] += random.uniform(-max_displacement, max_displacement)
            mutated[i][1] += random.uniform(-max_displacement, max_displacement)

        # Mutate rotation
        if random.random() < mutation_rate:
            mutated[i][2] += random.uniform(-30, 30)
            mutated[i][2] = mutated[i][2] % 360

    return mutated

def crossover(parent1, parent2, crossover_rate=0.8):
    """Single-point crossover for hexagon packing"""
    if random.random() > crossover_rate:
        return deepcopy(parent1), deepcopy(parent2)

    # Create offspring by combining parent genes
    n = len(parent1)
    crossover_point = random.randint(1, n-1)

    child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2

def create_tiled_initial_individual():
    """Create an initial individual using a hexagonal tiling pattern for better packing density"""
    # Create a pattern that mimics a hexagonal close packing arrangement
    # Central hexagon
    individual = [[0, 0, 0]]

    # First ring around center
    # Hexagonal positions at distance 2.17 (approx sqrt(3) for touching hexagons)
    angles = [0, 60, 120, 180, 240, 300]
    ring1_positions = []
    for angle in angles:
        rad_angle = np.radians(angle)
        x = 2.17 * np.cos(rad_angle)
        y = 2.17 * np.sin(rad_angle)
        ring1_positions.append([x, y, 0])

    individual.extend(ring1_positions)

    # Second ring around first ring
    # Place hexagons between first ring elements to fill gaps
    ring2_positions = []
    # Add positions in between first ring hexagons
    for i in range(6):
        prev_angle = angles[i]
        next_angle = angles[(i + 1) % 6]

        # Midpoint angle
        midpoint_angle = (prev_angle + next_angle) / 2
        rad_angle = np.radians(midpoint_angle)
        x = 2.17 * 1.5 * np.cos(rad_angle)
        y = 2.17 * 1.5 * np.sin(rad_angle)
        ring2_positions.append([x, y, 0])

    individual.extend(ring2_positions)

    # Convert to numpy array
    individual = np.array(individual)

    # Add small random perturbations to avoid getting stuck in local minima
    for i in range(len(individual)):
        individual[i][0] += random.uniform(-0.2, 0.2)
        individual[i][1] += random.uniform(-0.2, 0.2)
        individual[i][2] += random.uniform(-15, 15)
        individual[i][2] = individual[i][2] % 360

    return individual

def initialize_population(population_size):
    """Initialize population with diverse individuals using tiled patterns"""
    population = []

    # Use 70% tiled pattern, 30% random for diversity
    tiled_count = int(population_size * 0.7)
    random_count = population_size - tiled_count

    for i in range(tiled_count):
        individual = create_tiled_initial_individual()
        population.append(individual)

    for i in range(random_count):
        individual = create_random_individual()
        population.append(individual)

    return population

def initialize_population(population_size):
    """Initialize population with diverse individuals"""
    population = []
    for i in range(population_size):
        individual = create_random_individual()
        population.append(individual)
    return population

def select_parents(population, fitnesses, tournament_size=3):
    """Tournament selection"""
    selected = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(deepcopy(population[winner_idx]))
    return selected

def local_optimization_step(individual, max_iter=100):
    """Perform enhanced local optimization with multiple refinement stages"""
    best_individual = deepcopy(individual)
    best_fitness, _ = evaluate_fitness(best_individual[:, :2], best_individual[:, 2])

    # Stage 1: Fine position optimization
    for _ in range(max_iter // 2):
        mutated = deepcopy(best_individual)
        idx = random.randint(0, len(mutated)-1)
        mutated[idx][0] += random.uniform(-0.05, 0.05)
        mutated[idx][1] += random.uniform(-0.05, 0.05)
        mutated[idx][2] += random.uniform(-1, 1)
        mutated[idx][2] = mutated[idx][2] % 360

        mutated_fitness, _ = evaluate_fitness(mutated[:, :2], mutated[:, 2])
        if mutated_fitness > best_fitness:
            best_individual = mutated
            best_fitness = mutated_fitness

    # Stage 2: Rotation optimization only
    for _ in range(max_iter // 4):
        mutated = deepcopy(best_individual)
        idx = random.randint(0, len(mutated)-1)
        mutated[idx][2] += random.uniform(-5, 5)
        mutated[idx][2] = mutated[idx][2] % 360

        mutated_fitness, _ = evaluate_fitness(mutated[:, :2], mutated[:, 2])
        if mutated_fitness > best_fitness:
            best_individual = mutated
            best_fitness = mutated_fitness

    # Stage 3: Small position adjustments
    for _ in range(max_iter // 4):
        mutated = deepcopy(best_individual)
        idx = random.randint(0, len(mutated)-1)
        mutated[idx][0] += random.uniform(-0.01, 0.01)
        mutated[idx][1] += random.uniform(-0.01, 0.01)

        mutated_fitness, _ = evaluate_fitness(mutated[:, :2], mutated[:, 2])
        if mutated_fitness > best_fitness:
            best_individual = mutated
            best_fitness = mutated_fitness

    return best_individual

def evolutionary_hexagon_packing():
    """Evolutionary algorithm for hexagon packing optimization with improvements"""
    # Parameters
    population_size = 50
    generations = 100
    initial_mutation_rate = 0.15
    crossover_rate = 0.8
    elitism_rate = 0.1
    max_time_seconds = 170

    start_time = time.time()

    # Initialize population
    population = initialize_population(population_size)

    best_fitness_history = []
    last_improvement_gen = 0

    for gen in range(generations):
        if time.time() - start_time > max_time_seconds:
            break

        # Adaptive mutation rate that decreases over generations
        mutation_rate = max(0.05, initial_mutation_rate * (1 - gen / generations))

        # Evaluate fitness for all individuals
        fitnesses = []
        for individual in population:
            fitness, _ = evaluate_fitness(individual[:, :2], individual[:, 2])
            fitnesses.append(fitness)

        # Track best
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_fitness_history.append(best_fitness)

        # Early stopping if no improvement in recent generations
        if gen > 0 and best_fitness <= best_fitness_history[-2]:
            last_improvement_gen += 1
        else:
            last_improvement_gen = 0

        if last_improvement_gen > 20:  # Stop if no improvement for 20 generations
            break

        # Local optimization on best individual periodically
        if gen % 5 == 0:
            population[best_idx] = local_optimization_step(population[best_idx])

        # Elitism - keep best individuals
        elite_count = int(elitism_rate * population_size)
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        elites = [deepcopy(population[i]) for i in elite_indices]

        # Selection
        parents = select_parents(population, fitnesses)

        # Crossover and mutation
        new_population = elites.copy()

        while len(new_population) < population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child1, child2 = crossover(parent1, parent2, crossover_rate)

            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:population_size]

    # Final evaluation
    final_fitnesses = []
    for individual in population:
        fitness, _ = evaluate_fitness(individual[:, :2], individual[:, 2])
        final_fitnesses.append(fitness)

    best_idx = np.argmax(final_fitnesses)
    best_individual = population[best_idx]

    # Final optimization with more iterations
    best_individual = local_optimization_step(best_individual, max_iter=150)

    # Get final results
    final_fitness, outer_radius = evaluate_fitness(best_individual[:, :2], best_individual[:, 2])

    return best_individual, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run evolutionary optimization
    inner_hex_data, outer_hex_side_length = evolutionary_hexagon_packing()

    # Format output as required
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END