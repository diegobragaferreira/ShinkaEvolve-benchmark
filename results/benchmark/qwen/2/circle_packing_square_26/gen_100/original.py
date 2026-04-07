# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
import random
import time
from typing import Tuple, List

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26

    # Multi-scale grid initialization with adaptive perturbation
    def generate_grid_initialization(n_circles):
        # Create a grid pattern for initial placement with multiple scales
        grid_sizes = [int(np.ceil(np.sqrt(n_circles))),
                     max(1, int(np.ceil(np.sqrt(n_circles)) * 0.7)),
                     max(1, int(np.ceil(np.sqrt(n_circles)) * 1.3))]

        positions = []
        for grid_size in grid_sizes:
            if len(positions) >= n_circles:
                break
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) >= n_circles:
                        break
                    # Add jitter to avoid perfect grid alignment issues
                    x = (i + 0.5 + np.random.uniform(-0.15, 0.15)) / grid_size
                    y = (j + 0.5 + np.random.uniform(-0.15, 0.15)) / grid_size
                    # Clip to valid range to ensure bounds
                    x = np.clip(x, 0.01, 0.99)
                    y = np.clip(y, 0.01, 0.99)
                    positions.append([x, y])

        # Ensure we have exactly n_circles
        result = np.array(positions[:n_circles])
        return result

    # Optimized overlap checking using KDTree
    def check_overlap_fast(circles_arr, idx):
        # Only check against existing circles, not including self
        if len(circles_arr) <= 1:
            return False

        # Build tree excluding the circle we're checking
        other_circles = np.delete(circles_arr, idx, axis=0)
        if len(other_circles) == 0:
            return False

        tree = cKDTree(other_circles[:, :2])  # Only positions for spatial indexing

        # Query neighbors within distance 2*(r1+r2) to find potential overlaps
        query_radius = 2 * (circles_arr[idx, 2] + 0.01)  # Add buffer for numerical stability
        neighbors = tree.query_ball_point(circles_arr[idx, :2], query_radius)

        # Check actual overlap for neighbors
        for neighbor_idx in neighbors:
            if neighbor_idx >= idx:  # Adjust index since we deleted elements
                neighbor_idx += 1
            if neighbor_idx >= len(circles_arr):
                continue

            neighbor_pos = circles_arr[neighbor_idx, :2]
            neighbor_radius = circles_arr[neighbor_idx, 2]

            # Calculate squared distance for efficiency
            dx = circles_arr[idx, 0] - neighbor_pos[0]
            dy = circles_arr[idx, 1] - neighbor_pos[1]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (circles_arr[idx, 2] + neighbor_radius)**2

            if dist_sq < min_dist_sq:
                return True
        return False

    # Check if a circle is within bounds
    def is_valid_circle(pos, r):
        return (r <= pos[0] <= 1 - r) and (r <= pos[1] <= 1 - r)

    # Check if a configuration is valid with optimized overlap detection
    def is_valid_config(config):
        # Check bounds first (vectorized)
        valid_bounds = (
            (config[:, 2] <= config[:, 0]) &
            (config[:, 0] <= 1 - config[:, 2]) &
            (config[:, 2] <= config[:, 1]) &
            (config[:, 1] <= 1 - config[:, 2])
        )

        if not np.all(valid_bounds):
            return False

        # Check overlaps efficiently using KDTree
        for i in range(len(config)):
            if check_overlap_fast(config, i):
                return False
        return True

    # Calculate objective function (negative because we want to maximize sum of radii)
    def evaluate(individual):
        config = individual.reshape(n, 3)
        if not is_valid_config(config):
            return (-1e6,)  # Penalize invalid configurations heavily
        total_radius = np.sum(config[:, 2])
        return (total_radius,)

    # Enhanced local optimization refinement function with geometric improvements
    def refine_configuration(config):
        # More sophisticated refinement using geometric optimization techniques
        improved = True
        max_iterations = 100

        for iteration in range(max_iterations):
            if not improved:
                break
            improved = False

            # Try to optimize each circle individually
            for i in range(n):
                # Save original values
                orig_pos = config[i, :2].copy()
                orig_r = config[i, 2]

                # Try to increase radius while respecting constraints
                max_possible_r = min(
                    orig_pos[0],
                    1 - orig_pos[0],
                    orig_pos[1],
                    1 - orig_pos[1]
                )

                # Check if we can increase radius
                test_r = min(orig_r + 0.005, max_possible_r)

                # Validate new configuration
                valid = True
                if test_r > orig_r:
                    # Check overlap constraints with all other circles
                    for j in range(n):
                        if i != j:
                            dx = orig_pos[0] - config[j, 0]
                            dy = orig_pos[1] - config[j, 1]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (test_r + config[j, 2])**2

                            if dist_sq < min_dist_sq:
                                valid = False
                                break

                # If valid, update radius
                if valid and test_r > orig_r:
                    config[i, 2] = test_r
                    improved = True
                    continue

                # If we can't improve radius, try to slightly move the circle to make room
                # Try strategic movements with larger steps for more effective optimization
                if not valid and orig_r > 0.001:
                    step_sizes = [0.002, 0.005, 0.01]
                    best_pos = orig_pos.copy()
                    best_r = orig_r
                    best_valid = False

                    # Try moves in different directions with various step sizes
                    for step_size in step_sizes:
                        for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                                      (step_size, step_size), (-step_size, -step_size),
                                      (step_size, -step_size), (-step_size, step_size)]:
                            new_x = orig_pos[0] + dx
                            new_y = orig_pos[1] + dy

                            # Check bounds
                            if 0.001 <= new_x <= 0.999 and 0.001 <= new_y <= 0.999:
                                # Check if this movement resolves overlaps
                                valid_move = True
                                for j in range(n):
                                    if i != j:
                                        dx_j = new_x - config[j, 0]
                                        dy_j = new_y - config[j, 1]
                                        dist_sq = dx_j*dx_j + dy_j*dy_j
                                        min_dist_sq = (orig_r + config[j, 2])**2

                                        if dist_sq < min_dist_sq:
                                            valid_move = False
                                            break

                                if valid_move:
                                    best_pos = np.array([new_x, new_y])
                                    best_valid = True
                                    break
                        if best_valid:
                            break

                    if best_valid:
                        config[i, 0] = best_pos[0]
                        config[i, 1] = best_pos[1]
                        improved = True

        return config

    # Create toolboxes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initialize individuals with better starting configuration
    def init_individual():
        # Generate initial grid
        positions = generate_grid_initialization(n)

        # Add more structured randomness to positions and set initial radii
        individual = np.zeros((n, 3))
        for i in range(n):
            # Apply positional perturbations but keep within reasonable bounds
            individual[i, 0] = np.clip(positions[i, 0] + np.random.normal(0, 0.03), 0.01, 0.99)
            individual[i, 1] = np.clip(positions[i, 1] + np.random.normal(0, 0.03), 0.01, 0.99)
            # Start with slightly larger initial radii but ensure they fit
            max_radius = min(
                individual[i, 0],
                1 - individual[i, 0],
                individual[i, 1],
                1 - individual[i, 1]
            ) * 0.95
            individual[i, 2] = np.clip(np.random.uniform(0.03, max_radius * 0.8), 0.001, 0.49)

        return creator.Individual(individual.flatten())

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators with improved parameters
    def mutate_individual(individual, indpb=0.15):
        individual = individual.reshape(n, 3)
        mut_count = 0

        for i in range(n):
            if random.random() < indpb:
                mut_count += 1
                # Mutate one of x, y, or r with different strengths
                choice = random.randint(0, 2)
                if choice == 0:  # x coordinate
                    individual[i, 0] += np.random.normal(0, 0.04)
                    individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
                elif choice == 1:  # y coordinate
                    individual[i, 1] += np.random.normal(0, 0.04)
                    individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)
                else:  # radius
                    # Smaller mutations for radius to preserve feasibility
                    individual[i, 2] += np.random.normal(0, 0.015)
                    individual[i, 2] = np.clip(individual[i, 2], 0.001, 0.49)

        return (individual.flatten(),)

    def cx_uniform(ind1, ind2):
        # Uniform crossover with better probability
        ind1 = ind1.reshape(n, 3)
        ind2 = ind2.reshape(n, 3)

        for i in range(n):
            if random.random() < 0.6:  # Increased crossover probability
                ind1[i] = ind2[i].copy()

        return (ind1.flatten(), ind2.flatten())

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_uniform)
    toolbox.register("mutate", mutate_individual)

    # Tournament selection with adaptive sizing based on diversity
    def sel_tournament_varsize(individuals, k, tournsize=None):
        if tournsize is None:
            # Calculate tournament size based on fitness diversity
            fitnesses = [ind.fitness.values[0] for ind in individuals]
            if len(fitnesses) > 1:
                fitness_std = np.std(fitnesses)
                # Normalize tournament size - higher diversity means more aggressive selection
                tournsize = max(3, min(len(individuals), int(5 + fitness_std * 20)))
            else:
                tournsize = 3

        return tools.selTournament(individuals, k, tournsize)

    toolbox.register("select", sel_tournament_varsize)

    # Run evolution
    pop = toolbox.population(n=60)  # Slightly larger population
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the evolutionary algorithm
    try:
        # Initial evaluation
        for individual in pop:
            individual.fitness.values = toolbox.evaluate(individual)

        # Record best solution before evolution
        best_individual = tools.selBest(pop, 1)[0]
        best_solution = best_individual.reshape(n, 3).copy()

        # Evolution parameters - reduced number of generations but increased efficiency
        n_generations = 40
        # Adaptive mutation rate with exponential decay
        mutation_rates = [0.15 * (0.9 ** i) for i in range(n_generations)]

        # Main evolutionary loop
        for gen in range(n_generations):
            # Current mutation rate
            current_mutation_rate = mutation_rates[gen]

            # Select
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < current_mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = toolbox.evaluate(ind)

            # Replace the current population with the offspring
            pop[:] = offspring

            # Update hall of fame
            hof.update(pop)

            # Keep track of best solution found so far
            current_best = tools.selBest(pop, 1)[0]
            if current_best.fitness.values > best_solution.fitness.values:
                best_solution = current_best.reshape(n, 3).copy()

        # Final refinement of best solution
        refined_solution = refine_configuration(best_solution.copy())
        return refined_solution

    except Exception as e:
        # Fallback to simple grid-based solution if evolutionary fails
        print(f"Evolution failed: {e}")
        positions = generate_grid_initialization(n)
        final_solution = np.zeros((n, 3))
        for i in range(n):
            final_solution[i, 0] = positions[i, 0]
            final_solution[i, 1] = positions[i, 1]
            final_solution[i, 2] = 0.05  # Set small but equal radii
        return final_solution

# EVOLVE-BLOCK-END