# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
import math
import random
import time

def generate_hexagon_vertices(center_x, center_y, angle_deg, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = math.radians(angle_deg)
    # Vertices of a regular hexagon with side_length=1 centered at origin
    base_vertices = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = math.cos(theta)
        y = math.sin(theta)
        base_vertices.append((x, y))

    # Scale and translate
    vertices = [(center_x + side_length * vx, center_y + side_length * vy) for vx, vy in base_vertices]
    return vertices

def check_containment(hexagon_vertices, outer_hexagon_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    inner_poly = Polygon(hexagon_vertices)
    outer_poly = Polygon(outer_hexagon_vertices)
    return outer_poly.contains(inner_poly)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_voronoi_hex_centers(n_points=11, bounds=(-5, 5)):
    """Generate initial hexagon centers using Voronoi diagram."""
    # Generate random points
    np.random.seed(42)
    points = np.random.uniform(bounds[0], bounds[1], size=(n_points, 2))

    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Take the centroids of the finite Voronoi cells
    centroids = []
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            points_in_region = [vor.vertices[i] for i in region if i >= 0]
            if points_in_region:
                centroid = np.mean(points_in_region, axis=0)
                # Only keep centroids within bounds
                if bounds[0] <= centroid[0] <= bounds[1] and bounds[0] <= centroid[1] <= bounds[1]:
                    centroids.append(centroid)

    # If we don't have enough centroids, add some random ones
    while len(centroids) < n_points:
        centroids.append(np.random.uniform(bounds[0], bounds[1], size=2))

    # Take first n_points
    return np.array(centroids[:n_points])

def initialize_population(pop_size, n_hexagons=11, bounds=(-5, 5)):
    """Initialize population with Voronoi-based configurations."""
    population = []
    for _ in range(pop_size):
        # Get Voronoi-based centers
        centers = compute_voronoi_hex_centers(n_hexagons, bounds)

        # Add some randomness
        individual = []
        for i in range(n_hexagons):
            x, y = centers[i]
            # Add some noise to positions
            x += np.random.normal(0, 0.3)
            y += np.random.normal(0, 0.3)
            # Random rotation
            angle = np.random.uniform(-180, 180)
            individual.extend([x, y, angle])

        # Add outer hexagon parameters (center, angle, side_length)
        individual.extend([0.0, 0.0, 0.0, 4.0])  # reasonable starting side length

        population.append(individual)

    return population

def evaluate_individual(params, return_penalties=False):
    """Evaluate a single individual (solution)."""
    # Extract inner hexagon positions and rotations
    inner_params = params[:-4]
    outer_center_x, outer_center_y, outer_angle, outer_side_length = params[-4:]

    # Create inner hexagons
    inner_hexagons = []
    for i in range(11):
        x, y, theta = inner_params[3*i:3*i+3]
        vertices = generate_hexagon_vertices(x, y, theta, 1.0)
        inner_hexagons.append(vertices)

    # Create outer hexagon
    outer_vertices = generate_hexagon_vertices(outer_center_x, outer_center_y, outer_angle, outer_side_length)

    # Check constraints
    penalty = 0
    penalties = []

    # Check containment
    for vertices in inner_hexagons:
        if not check_containment(vertices, outer_vertices):
            penalty += 1000000
            penalties.append("containment")

    # Check overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                penalty += 1000000
                penalties.append("overlap")

    # Return negative of inverse side length plus penalties
    if penalty > 0:
        if return_penalties:
            return penalty + 1.0 / outer_side_length, penalties
        return penalty + 1.0 / outer_side_length
    else:
        if return_penalties:
            return -1.0 / outer_side_length, []
        return -1.0 / outer_side_length


def adaptive_mutation_rate(gen, max_gens, initial_rate=0.8, final_rate=0.1):
    """Adaptive mutation rate that decreases over generations."""
    # Linear decay from initial_rate to final_rate
    if max_gens <= 1:
        return final_rate
    return initial_rate - (initial_rate - final_rate) * (gen / (max_gens - 1))

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmin(tournament_fitnesses)]
    return population[winner_index]

def crossover(parent1, parent2):
    """Crossover operation for hexagon packing."""
    # Uniform crossover for positions and rotations
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover for inner hexagons
    for i in range(11):
        if random.random() < 0.5:
            child1[3*i:3*i+3] = parent2[3*i:3*i+3]
            child2[3*i:3*i+3] = parent1[3*i:3*i+3]

    # Crossover for outer hexagon parameters
    if random.random() < 0.5:
        child1[-4:] = parent2[-4:]
        child2[-4:] = parent1[-4:]

    return child1, child2

def mutate(individual, mutation_rate=0.1, bounds=(-10, 10), adaptive_mutation=False, generation=None, max_generations=None):
    """Mutation operation for hexagon packing with adaptive mutation rate."""
    mutated = individual.copy()

    # Use adaptive mutation rate if specified
    if adaptive_mutation and generation is not None and max_generations is not None:
        effective_mutation_rate = adaptive_mutation_rate(generation, max_generations)
    else:
        effective_mutation_rate = mutation_rate

    # Mutate inner hexagons
    for i in range(11):
        for j in range(3):  # x, y, angle
            if random.random() < effective_mutation_rate:
                if j < 2:  # x or y
                    mutated[3*i+j] += np.random.normal(0, 0.5)
                    # Keep within bounds
                    mutated[3*i+j] = np.clip(mutated[3*i+j], bounds[0], bounds[1])
                else:  # angle
                    mutated[3*i+j] += np.random.normal(0, 30)
                    # Normalize angle to [-180, 180]
                    mutated[3*i+j] = ((mutated[3*i+j] + 180) % 360) - 180

    # Mutate outer hexagon
    if random.random() < effective_mutation_rate:
        mutated[-1] += np.random.normal(0, 0.5)  # mutate side_length
        mutated[-1] = max(1.0, mutated[-1])  # ensure positive side_length

    return mutated

def adaptive_genetic_algorithm_optimization():
    """Perform adaptive genetic algorithm optimization with dynamic parameters."""
    pop_size = 30
    max_generations = 50
    initial_mutation_rate = 0.8  # Start with higher mutation rate for exploration

    # Initialize population
    population = initialize_population(pop_size)

    best_fitness_history = []

    for gen in range(max_generations):
        # Evaluate fitness
        fitnesses = [evaluate_individual(ind) for ind in population]

        # Find best individual
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_fitness_history.append(best_fitness)

        # Print progress
        if gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {-best_fitness}")

        # Early stopping if no improvement for last 10 generations
        if len(best_fitness_history) > 10:
            recent_improvement = best_fitness_history[-10] - best_fitness_history[-1]  # Should be positive if improving
            if recent_improvement < 1e-8 and gen > 20:  # Very small improvement, stop early
                print(f"Early stopping at generation {gen}")
                break

        # Create new population
        new_population = []

        # Elitism: keep best individual
        new_population.append(population[best_idx].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, initial_mutation_rate, adaptive_mutation=True, generation=gen, max_generations=max_generations)
            child2 = mutate(child2, initial_mutation_rate, adaptive_mutation=True, generation=gen, max_generations=max_generations)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:pop_size]

    # Return best individual
    fitnesses = [evaluate_individual(ind) for ind in population]
    best_idx = np.argmin(fitnesses)
    return population[best_idx]

def local_refinement_improved(params, max_evaluations=500):
    """Improved local refinement using multiple optimization techniques."""
    # Try differential evolution first (global optimization)
    try:
        # Create a wrapper for the objective function that takes the right parameters
        def wrapped_de_objective(x):
            # Reshape x to match the full parameters (add back outer hexagon)
            full_params = list(x) + [params[-4], params[-3], params[-2], params[-1]]
            return evaluate_individual(full_params)

        # Bounds for inner hexagons only (33 parameters)
        bounds = [(-10, 10), (-10, 10), (-180, 180)] * 11

        # Run differential evolution on inner parameters
        result_de = differential_evolution(
            wrapped_de_objective,
            bounds,
            maxiter=20,
            popsize=5,
            seed=42,
            disp=False
        )

        if result_de.success:
            # Update params with DE result
            refined_inner_params = result_de.x
            refined_params = list(refined_inner_params) + [params[-4], params[-3], params[-2], params[-1]]
            return refined_params
    except Exception as e:
        print(f"Differential evolution failed: {e}")

    # Fall back to Nelder-Mead if DE fails
    try:
        def wrapped_nm_objective(x):
            full_params = list(x) + [params[-4], params[-3], params[-2], params[-1]]
            return evaluate_individual(full_params)

        # Optimize just the inner hexagons using Nelder-Mead
        x0 = params[:-4]  # Remove outer hexagon parameters

        result_nm = minimize(
            wrapped_nm_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'disp': False}
        )

        if result_nm.success:
            refined_inner_params = result_nm.x
            refined_params = list(refined_inner_params) + [params[-4], params[-3], params[-2], params[-1]]
            return refined_params
    except Exception as e:
        print(f"Nelder-Mead failed: {e}")

    # If all fail, return original params
    return params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses adaptive hybrid genetic algorithm with Voronoi initialization and local refinement.
    """
    start_time = time.time()
    max_time_seconds = 175  # Leave some margin for cleanup

    # First run adaptive genetic algorithm for coarse optimization
    try:
        best_params = adaptive_genetic_algorithm_optimization()
    except Exception as e:
        print(f"Genetic algorithm failed: {e}")
        # Fallback to a good initial configuration
        best_params = [
            0.0, 0.0, 0.0,      # center hexagon
            -2.0, 0.0, 0.0,     # left
            2.0, 0.0, 0.0,      # right
            0.0, 2.0, 0.0,      # top
            0.0, -2.0, 0.0,     # bottom
            -1.0, 1.0, 0.0,     # top-left
            1.0, 1.0, 0.0,      # top-right
            -1.0, -1.0, 0.0,    # bottom-left
            1.0, -1.0, 0.0,     # bottom-right
            -2.0, 1.5, 0.0,     # far top-left
            2.0, 1.5, 0.0,      # far top-right
            0.0, 0.0, 0.0, 4.0  # outer hexagon parameters
        ]

    # Local refinement with adaptive optimization
    if time.time() - start_time < max_time_seconds - 5:
        try:
            best_params = local_refinement_improved(best_params)
        except Exception as e:
            print(f"Local refinement failed: {e}")

    # Extract final results
    inner_params = best_params[:-4]
    outer_center_x, outer_center_y, outer_angle, outer_side_length = best_params[-4:]

    # Format inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]]

    outer_hex_data = np.array([outer_center_x, outer_center_y, outer_angle])

    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END