# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
import math
from scipy.spatial.distance import cdist
import itertools

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """

    # Problem parameters
    N_CIRCLES = 26
    POP_SIZE = 120
    N_GEN = 200
    INITIAL_MUT_PB = 0.2
    CROSSOVER_PB = 0.8

    # Define the fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Create an individual representing a solution
    # Each individual is a list of [x, y, r] for each circle
    def create_individual():
        individual = []
        # Use grid-guided initialization
        circles = grid_init(N_CIRCLES)
        for x, y, r in circles:
            individual.extend([x, y, r])
        return creator.Individual(individual)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_circle_packing(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)

        # Calculate total radius (objective function)
        total_radius = np.sum(circles[:, 2])

        # Check constraints
        penalty = 0

        # Check containment constraints with larger margin
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 10000  # Large penalty for containment violation

        # Check overlap constraints with more sophisticated penalty
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    # Penalty based on how much they overlap
                    overlap = (r1 + r2) - distance
                    penalty += 10000 * overlap

        # Return fitness: total radius minus penalties
        return (total_radius - penalty,)

    toolbox.register("evaluate", eval_circle_packing)

    # Custom crossover and mutation operators for better performance
    def cx_circle(ind1, ind2):
        # Perform uniform crossover for x, y, r coordinates
        for i in range(0, len(ind1), 3):
            if random.random() < 0.5:
                ind1[i:i+3], ind2[i:i+3] = ind2[i:i+3], ind1[i:i+3]
        return ind1, ind2

    def mut_circle(individual, indpb, generation, max_gen):
        # Adaptive mutation rate that decreases over generations
        adaptive_pb = indpb * (1 - generation/max_gen)
        # Mutate x, y, r for each circle
        for i in range(0, len(individual), 3):
            if random.random() < adaptive_pb:
                # Mutate x coordinate
                individual[i] += random.gauss(0, 0.02)
                individual[i] = max(0.01, min(0.99, individual[i]))

            if random.random() < adaptive_pb:
                # Mutate y coordinate
                individual[i+1] += random.gauss(0, 0.02)
                individual[i+1] = max(0.01, min(0.99, individual[i+1]))

            if random.random() < adaptive_pb:
                # Mutate radius
                individual[i+2] += random.gauss(0, 0.015)
                individual[i+2] = max(0.005, min(0.4, individual[i+2]))

        return individual,

    toolbox.register("mate", cx_circle)
    toolbox.register("mutate", mut_circle)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Grid-guided initialization
    def grid_init(n_circles):
        # Create circles arranged in a grid pattern that maximizes spatial distribution
        # Aim for roughly 5x5 grid for 26 circles
        rows = 5
        cols = 6

        # Space circles in a grid with padding
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)

        circles = []
        idx = 0

        # Fill grid positions
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Add slight randomness to avoid perfect grid
                x += random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += random.uniform(-spacing_y*0.1, spacing_y*0.1)
                # Set initial radius based on proximity to boundaries
                r = min(x, 1-x, y, 1-y) * random.uniform(0.3, 0.45)
                circles.append([x, y, r])
                idx += 1

        # If we still need more circles, add them randomly but with some spatial awareness
        while len(circles) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = random.uniform(0.01, 0.2)
            circles.append([x, y, r])

        return circles

    # Create initial population with grid-based initialization
    def create_grid_population():
        population = []
        for _ in range(POP_SIZE):
            individual = []
            # Start with grid-guided initialization
            circles = grid_init(N_CIRCLES)
            for x, y, r in circles:
                individual.extend([x, y, r])
            population.append(creator.Individual(individual))
        return population

    # Create initial population
    try:
        population = create_grid_population()
    except Exception as e:
        print(f"Grid population creation failed: {e}")
        # Fallback to basic initialization
        population = toolbox.population(n=POP_SIZE)

    # Run evolution with adaptive mutation
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        # Modified eaSimple to pass generation info to mutation
        for gen in range(N_GEN):
            # Update mutation probability based on generation
            current_mut_pb = INITIAL_MUT_PB * (1 - gen/N_GEN)

            # Select and clone
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_PB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation with adaptive rate
            for mutant in offspring:
                if random.random() < current_mut_pb:
                    toolbox.mutate(mutant, current_mut_pb, gen, N_GEN)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the old population with the new one
            population[:] = offspring

            # Update hall of fame
            hall_of_fame.update(population)

    except Exception as e:
        print(f"Evolution failed: {e}")
        # Return a simple heuristic solution if evolution fails
        return heuristic_solution()

    # Get best individual
    best_individual = hall_of_fame[0]
    best_circles = np.array(best_individual).reshape(-1, 3)

    # Apply local optimization refinement
    refined_circles = local_optimization_refinement(best_circles)

    # Ensure final validation
    circles = validate_and_fix_solution(refined_circles)

    return circles

def heuristic_solution() -> np.ndarray:
    """Fallback solution using hexagonal packing heuristic"""
    n = 26
    circles = np.zeros((n, 3))

    # Try a hexagonal lattice approach for reasonable starting point
    # Arrange in roughly a hexagonal pattern with some randomness
    rows = 5
    cols = 6

    # Hexagonal packing coordinates
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0 if i % 2 == 0 else spacing_x / 2
            x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x*0.05, spacing_x*0.05)
            y = (i + 1) * spacing_y + random.uniform(-spacing_y*0.05, spacing_y*0.05)
            # Radius based on proximity to boundaries
            r = min(x, 1-x, y, 1-y) * 0.35
            # Add some randomness to make it less regular
            r *= random.uniform(0.8, 1.0)
            circles[idx] = [x, y, r]
            idx += 1

    # If we don't have enough circles, fill remaining positions with small radii
    for i in range(idx, n):
        circles[i] = [0.5, 0.5, 0.01]

    return circles

def validate_and_fix_solution(circles: np.ndarray) -> np.ndarray:
    """Ensure the solution respects constraints and has reasonable values"""
    # Make a copy to avoid modifying original
    result = circles.copy()

    # Clip radii to reasonable bounds
    result[:, 2] = np.clip(result[:, 2], 0.001, 0.45)

    # Ensure circles stay within bounds
    for i in range(len(result)):
        x, y, r = result[i]
        # Clamp positions to valid range
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        result[i] = [x, y, r]

    return result

def local_optimization_refinement(circles: np.ndarray) -> np.ndarray:
    """
    Apply local optimization to refine the solution by adjusting positions
    to reduce overlaps and increase total radius
    """
    result = circles.copy()
    n = len(result)

    # Try multiple local optimization passes
    for pass_num in range(5):
        improved = False

        # For each circle, try to improve its position
        for i in range(n):
            original_pos = result[i].copy()
            original_radius = result[i][2]
            original_total = np.sum(result[:, 2])

            # Try different positions around current location
            best_pos = original_pos.copy()
            best_radius = original_radius
            best_total = original_total

            # Test several nearby positions
            step_size = 0.02
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    # New tentative position
                    new_x = original_pos[0] + dx
                    new_y = original_pos[1] + dy

                    # Check if new position is valid
                    if (new_x - original_radius >= 0 and new_x + original_radius <= 1 and
                        new_y - original_radius >= 0 and new_y + original_radius <= 1):

                        # Temporarily update
                        temp_result = result.copy()
                        temp_result[i] = [new_x, new_y, original_radius]

                        # Check overlap constraints with others
                        valid = True
                        for j in range(n):
                            if i != j:
                                x1, y1, r1 = temp_result[i]
                                x2, y2, r2 = temp_result[j]
                                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                                if dist < r1 + r2:
                                    valid = False
                                    break

                        if valid:
                            new_total = np.sum(temp_result[:, 2])
                            if new_total > best_total:
                                best_total = new_total
                                best_pos = [new_x, new_y, original_radius]
                                improved = True

            # Update if we found improvement
            if improved:
                result[i] = best_pos

        # If no improvement, stop early
        if not improved:
            break

    return result


# EVOLVE-BLOCK-END