# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon centered at center with given rotation."""
    angle = rotation * np.pi / 180
    # Vertices of a unit hexagon centered at origin
    hex_vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_vertices.append((x + center[0], y + center[1]))
    return Polygon(hex_vertices)

def check_containment(inner_hex, outer_hex):
    """Check if inner_hex is completely contained within outer_hex."""
    return outer_hex.contains(inner_hex)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2) and not hex1.touches(hex2)

def evaluate_fitness(individual, outer_radius=None):
    """Evaluate the fitness of an individual (packing configuration)."""
    # Reshape individual into (11, 3) array: [x, y, angle] for each hexagon
    positions_angles = individual.reshape(-1, 3)

    # If outer radius not specified, calculate minimum required
    if outer_radius is None:
        # Find bounding box of all inner hexagons
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for i in range(11):
            pos = positions_angles[i][:2]
            angle = positions_angles[i][2]
            hexagon = create_unit_hexagon(pos, angle)

            # Get bounds of hexagon
            bounds = hexagon.bounds
            min_x = min(min_x, bounds[0])
            max_x = max(max_x, bounds[2])
            min_y = min(min_y, bounds[1])
            max_y = max(max_y, bounds[3])

        # Calculate required outer hexagon radius to contain all inner hexagons
        width = max_x - min_x
        height = max_y - min_y
        outer_radius = max(width, height) / 2 + 1  # Add buffer

    # Create outer hexagon
    outer_hex = create_unit_hexagon((0, 0), 0)
    # Scale appropriately (we use unit hexagons, so scale to get desired outer radius)
    # We'll scale the positions appropriately to fit within a hexagon of the given radius

    # Check all pairwise intersections
    total_penalty = 0
    outer_hex_penalties = 0

    # Create all inner hexagons
    inner_hexagons = []
    for i in range(11):
        pos = positions_angles[i][:2]
        angle = positions_angles[i][2]
        hexagon = create_unit_hexagon(pos, angle)
        inner_hexagons.append(hexagon)

    # Check containment
    for i in range(11):
        # Check if this hexagon is fully within the outer hexagon
        # Simple containment check: check if center is within outer hexagon
        center_point = Point(inner_hexagons[i].centroid.x, inner_hexagons[i].centroid.y)
        if not outer_hex.contains(center_point):
            outer_hex_penalties += 1000  # Heavy penalty

    # Check overlaps between hexagons
    for i in range(11):
        for j in range(i+1, 11):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                # Add penalty based on overlap area or simply count overlaps
                total_penalty += 1000  # Heavy penalty for overlaps

    # Return negative of penalty since we want to minimize it
    # Fitness is inverse of outer hexagon radius (higher is better)
    fitness_value = 1.0 / outer_radius - total_penalty - outer_hex_penalties
    return fitness_value

def create_random_individual():
    """Create a random valid individual."""
    # Generate 11 positions for hexagons in a reasonable range
    # We'll place them in a circular pattern initially then randomize

    # Start with a good initial layout
    individual = np.zeros((33,))

    # Place first hexagon at center
    individual[0:3] = [0, 0, 0]

    # Place others in a ring formation, with some randomness
    base_radius = 2.5
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)

    for i in range(1, 11):
        # Random angle and distance
        angle = angles[i-1] + np.random.normal(0, 0.2)
        r = base_radius + np.random.normal(0, 0.5)
        if r < 0: r = 0.1

        # Convert to Cartesian coordinates
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        angle_deg = np.random.uniform(0, 360)

        individual[3*i:3*i+3] = [x, y, angle_deg]

    return individual

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    # Single point crossover
    crossover_point = np.random.randint(1, 33)

    child1 = np.copy(parent1)
    child2 = np.copy(parent2)

    child1[crossover_point:] = parent2[crossover_point:]
    child2[crossover_point:] = parent1[crossover_point:]

    return child1, child2

def mutate_individual(individual, mutation_rate=0.1, mutation_strength=1.0):
    """Mutate an individual."""
    mutated = np.copy(individual)

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate position or angle
            if i % 3 < 2:  # Position coordinate
                mutated[i] += np.random.normal(0, mutation_strength * 0.5)
            else:  # Angle
                mutated[i] += np.random.normal(0, mutation_strength * 10)
                # Keep angle in [0, 360)
                mutated[i] = mutated[i] % 360

    return mutated

def optimize_hexagon_packing():
    """Main optimization routine using genetic algorithm."""

    # Parameters
    population_size = 50
    num_generations = 100
    elite_size = 5
    mutation_rate = 0.15
    mutation_strength = 0.8

    # Initialize population
    population = []
    for _ in range(population_size):
        population.append(create_random_individual())

    best_individual = None
    best_fitness = float('-inf')

    for generation in range(num_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for ind in population:
            fitness = evaluate_fitness(ind)
            fitness_scores.append(fitness)

        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = np.copy(population[max_fitness_idx])

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores.sort(reverse=True)

        # Print progress
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

        # Create new population (elitism + crossover + mutation)
        new_population = population[:elite_size]

        # Tournament selection and reproduction
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 5
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]

            # Select two parents
            parent1 = population[winner_index]
            parent2 = population[np.random.choice(len(population))]

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate_individual(child1, mutation_rate, mutation_strength)
            child2 = mutate_individual(child2, mutation_rate, mutation_strength)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Final refinement using local optimization
    final_individual = local_optimization_step(best_individual)

    return final_individual

def local_optimization_step(individual):
    """Perform local optimization on the best individual."""
    # Perform gradient-free local search
    current_individual = np.copy(individual)

    # Try small perturbations to improve fitness
    for iter_num in range(50):  # Limited iterations
        # Store current fitness
        current_fitness = evaluate_fitness(current_individual)

        # Try small changes to each parameter
        best_change_individual = np.copy(current_individual)
        best_change_fitness = current_fitness

        for i in range(len(current_individual)):
            # Try small positive and negative perturbations
            test_individual = np.copy(current_individual)
            if i % 3 < 2:  # Position coordinate
                test_individual[i] += np.random.uniform(-0.2, 0.2)
            else:  # Angle
                test_individual[i] += np.random.uniform(-5, 5)
                test_individual[i] = test_individual[i] % 360

            new_fitness = evaluate_fitness(test_individual)
            if new_fitness > best_change_fitness:
                best_change_fitness = new_fitness
                best_change_individual = np.copy(test_individual)

        # Accept improvement
        if best_change_fitness > current_fitness:
            current_individual = best_change_individual
        else:
            # Random restart occasionally to avoid local minima
            if np.random.random() < 0.1:
                current_individual = create_random_individual()

    return current_individual

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Run optimization
    start_time = time.time()
    optimized_individual = optimize_hexagon_packing()
    end_time = time.time()

    # Extract result
    positions_angles = optimized_individual.reshape(-1, 3)

    # Convert to required output format
    inner_hex_data = positions_angles.copy()
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    # Estimate the minimal outer hexagon radius needed to contain all inner hexagons
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # Calculate outer hexagon dimensions
    for i in range(11):
        pos = positions_angles[i][:2]
        angle = positions_angles[i][2]
        hexagon = create_unit_hexagon(pos, angle)

        # Get bounds of hexagon
        bounds = hexagon.bounds
        min_x = min(min_x, bounds[0])
        max_x = max(max_x, bounds[2])
        min_y = min(min_y, bounds[1])
        max_y = max(max_y, bounds[3])

    # Calculate required outer hexagon side length
    width = max_x - min_x
    height = max_y - min_y
    outer_hex_side_length = max(width, height) / 2 + 1  # Add buffer

    # Update parameters to ensure we have a valid configuration
    # Recalculate fitness to make sure it's valid
    final_fitness = evaluate_fitness(optimized_individual, outer_hex_side_length)

    print(f"Final fitness: {final_fitness}")
    print(f"Outer hex side length: {outer_hex_side_length}")
    print(f"Combined score (1/outer_hex_side_length): {1.0/outer_hex_side_length}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END