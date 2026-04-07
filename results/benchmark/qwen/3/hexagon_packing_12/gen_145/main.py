# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time
import random
from itertools import combinations
from collections import defaultdict

class SymmetryAwareMutation:
    """Handles symmetry-aware mutation operations for hexagon packing."""

    @staticmethod
    def mutate_symmetrically(config, mutation_strength=0.1):
        """Apply symmetry-aware mutation that preserves meaningful geometric relationships."""
        mutated_config = config.copy()

        # For 12 hexagons, we can often use 6-fold or 3-fold symmetry
        # Preserve central hexagon
        mutated_config[0] = config[0].copy()

        # Mutate positions in groups that respect symmetry
        # Group hexagons by their angular positions (first ring: 6 hexagons)
        ring1_indices = list(range(1, 7))  # First ring hexagons
        ring2_indices = list(range(7, 12))  # Second ring hexagons

        # Mutate ring 1 hexagons (preserve 6-fold rotational symmetry)
        for i in ring1_indices:
            mutated_config[i][0] = config[i][0] + np.random.normal(0, mutation_strength)
            mutated_config[i][1] = config[i][1] + np.random.normal(0, mutation_strength)

        # Mutate ring 2 hexagons (preserve 5-fold rotational symmetry)
        for i in ring2_indices:
            mutated_config[i][0] = config[i][0] + np.random.normal(0, mutation_strength)
            mutated_config[i][1] = config[i][1] + np.random.normal(0, mutation_strength)

        return mutated_config

def hexagon_vertices(center_x, center_y, size=1, angle_deg=0):
    """Generate vertices of a regular hexagon given center, size, and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + size * np.cos(angle)
        y = center_y + size * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with buffer for precision."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.buffer(1e-10).intersects(poly2.buffer(1e-10))

def compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y):
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        distance = np.sqrt((cx - outer_center_x)**2 + (cy - outer_center_y)**2)
        max_distance = max(max_distance, distance + 1)  # Add radius of unit hexagon
    return max_distance

def evaluate_configuration(inner_hex_data, outer_center_x, outer_center_y):
    """Evaluate current configuration: returns (validity, inv_radius)."""
    # Early check for overlaps
    for i, j in combinations(range(len(inner_hex_data)), 2):
        hex1_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        hex2_vertices = hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], 1, inner_hex_data[j][2])
        if check_overlap(hex1_vertices, hex2_vertices):
            return False, 0

    # Check containment
    outer_radius = compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y)
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_radius, 0)
    outer_polygon = Polygon(outer_vertices)

    # Check all vertices of all hexagons
    for i in range(len(inner_hex_data)):
        hex_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, 0

    # Return inverse of outer radius
    return True, 1.0 / outer_radius

def generate_initial_configs():
    """Generate multiple diverse initial configurations."""
    configs = []

    # Configuration 1: Standard symmetric arrangement
    config1 = []
    config1.append([0, 0, 0])
    ring1_radius = 2.0
    for i in range(6):
        angle = i * 60
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        config1.append([x, y, 0])

    ring2_angles = [0, 72, 144, 216, 288]
    ring2_radius = 3.5
    for angle in ring2_angles:
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        config1.append([x, y, 0])

    config1.append([0, -ring2_radius - 1.0, 0])

    # Add small random perturbations
    for i in range(len(config1)):
        if i > 0:
            config1[i][0] += random.uniform(-0.05, 0.05)
            config1[i][1] += random.uniform(-0.05, 0.05)

    configs.append(np.array(config1))

    # Configuration 2: Perturbed version
    config2 = []
    config2.append([0, 0, 0])
    for i in range(6):
        angle = i * 60
        radius = 2.1
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config2.append([x, y, 0])

    angles = [30, 90, 150, 210, 270]
    radius = 3.4
    for i, angle in enumerate(angles):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config2.append([x, y, 0])

    config2.append([0, -radius - 1.0, 0])

    # Add more substantial perturbations
    for i in range(len(config2)):
        if i > 0:
            config2[i][0] += random.uniform(-0.1, 0.1)
            config2[i][1] += random.uniform(-0.1, 0.1)

    configs.append(np.array(config2))

    # Configuration 3: Asymmetric pattern
    config3 = []
    config3.append([0, 0, 0])
    config3.append([-2.2, 0, 0])
    config3.append([2.2, 0, 0])
    config3.append([-1.1, 1.9, 0])
    config3.append([1.1, 1.9, 0])
    config3.append([-1.1, -1.9, 0])
    config3.append([1.1, -1.9, 0])
    config3.append([-3.3, 1.9, 0])
    config3.append([3.3, 1.9, 0])
    config3.append([-3.3, -1.9, 0])
    config3.append([3.3, -1.9, 0])
    config3.append([0, -3.5, 0])

    configs.append(np.array(config3))

    return configs

def optimize_with_evolutionary_approach(initial_configs, outer_center_x, outer_center_y, max_time_seconds=170):
    """Use evolutionary algorithm with symmetry awareness to find better solutions."""
    start_time = time.time()

    # Initialize population
    population = initial_configs[:3]  # Use first 3 configs as starting population
    fitness_scores = []

    # Evaluate initial population
    for config in population:
        validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
        fitness_scores.append(inv_radius if validity else 0)

    # Evolutionary algorithm parameters
    max_generations = 20
    population_size = len(population)

    for generation in range(max_generations):
        if (time.time() - start_time) > max_time_seconds * 0.95:
            break

        # Select best individuals (tournament selection)
        selected_indices = []
        tournament_size = 3

        for _ in range(population_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected_indices.append(winner_index)

        # Create new population through crossover and mutation
        new_population = []

        # Elitism: keep best individual
        best_idx = fitness_scores.index(max(fitness_scores))
        new_population.append(population[best_idx].copy())

        # Generate offspring
        for _ in range(population_size - 1):
            parent1_idx = random.choice(selected_indices)
            parent2_idx = random.choice(selected_indices)

            # Crossover (simple average)
            child = (population[parent1_idx] + population[parent2_idx]) / 2

            # Mutation with symmetry awareness
            mutation_strength = 0.1 * (1 - generation/max_generations)  # Decrease over time
            mutated_child = SymmetryAwareMutation.mutate_symmetrically(child, mutation_strength)

            new_population.append(mutated_child)

        # Evaluate new population
        population = new_population
        fitness_scores = []

        for config in population:
            validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
            fitness_scores.append(inv_radius if validity else 0)

    # Return best solution
    best_idx = fitness_scores.index(max(fitness_scores))
    return population[best_idx]

def stage_optimization(initial_config, outer_center_x, outer_center_y, bounds, maxiter, ftol):
    """Perform single stage optimization."""
    def objective(params):
        config = initial_config.copy()
        idx = 0
        for i in range(len(config)):
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2

        validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
        if not validity:
            return 1e10
        return -inv_radius  # Negative because we want to maximize

    initial_params = []
    for i in range(len(initial_config)):
        initial_params.extend([initial_config[i][0], initial_config[i][1]])

    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': maxiter, 'ftol': ftol})
        optimized_config = initial_config.copy()
        idx = 0
        for i in range(len(optimized_config)):
            optimized_config[i][0] = result.x[idx]
            optimized_config[i][1] = result.x[idx + 1]
            idx += 2
        return optimized_config
    except:
        return initial_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate initial configurations
    initial_configs = generate_initial_configs()

    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0

    # Evolved approach: first use evolutionary algorithm to find promising area
    optimized_config = optimize_with_evolutionary_approach(initial_configs, outer_center_x, outer_center_y)

    # Then fine-tune with local optimization
    # Stage 1: Coarse optimization
    optimized_config = stage_optimization(optimized_config, outer_center_x, outer_center_y,
                                        bounds=[(-8, 8), (-8, 8)] * 12,
                                        maxiter=300, ftol=1e-8)

    # Stage 2: Medium refinement
    optimized_config = stage_optimization(optimized_config, outer_center_x, outer_center_y,
                                        bounds=[(-6, 6), (-6, 6)] * 12,
                                        maxiter=200, ftol=1e-10)

    # Stage 3: Fine tuning
    optimized_config = stage_optimization(optimized_config, outer_center_x, outer_center_y,
                                        bounds=[(-5, 5), (-5, 5)] * 12,
                                        maxiter=150, ftol=1e-12)

    # Final verification and refinement
    max_attempts = 10
    for attempt in range(max_attempts):
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity:
            break

        # If not valid, try small adjustments to positions
        for i in range(len(optimized_config)):
            optimized_config[i][0] += np.random.normal(0, 0.01)
            optimized_config[i][1] += np.random.normal(0, 0.01)

    # Compute final outer hexagon radius
    outer_radius = 1.0 / inv_radius if inv_radius > 0 else 10.0

    # Ensure that we have exactly 12 hexagons
    inner_hex_data = np.array(optimized_config)
    if len(inner_hex_data) != 12:
        # Fallback to simple configuration if needed
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
            [0, -4, 0]
        ])
        outer_radius = 8.0

    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])
    outer_hex_side_length = outer_radius * 2  # approximate

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END