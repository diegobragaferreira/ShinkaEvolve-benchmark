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

    # Enhanced local optimization with adaptive strategy based on overlap severity
    def refine_configuration(config):
        # Count overlap violations to classify optimization strategy
        def count_overlap_violations(config):
            violations = 0
            for i in range(n):
                for j in range(i+1, n):
                    dx = config[i, 0] - config[j, 0]
                    dy = config[i, 1] - config[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (config[i, 2] + config[j, 2])**2
                    if dist_sq < min_dist_sq:
                        violations += 1
            return violations

        # Classify the optimization problem based on overlap severity
        def classify_problem(config):
            violation_count = count_overlap_violations(config)
            total_radius = np.sum(config[:, 2])

            if violation_count > n // 2:
                return "high_overlap"
            elif violation_count > 0:
                return "medium_overlap"
            elif total_radius < n * 0.1:  # Very small radii
                return "low_radius"
            else:
                return "balanced"

        # Strategy 1: High overlap - aggressive overlap resolution
        def strategy_high_overlap(config):
            improved = False
            max_iter = 20  # Limit iterations for aggressive approach

            for iteration in range(max_iter):
                improved_in_iteration = False

                # Try to resolve major overlaps first
                for i in range(n):
                    # Check if this circle is causing overlaps
                    violations = []
                    for j in range(n):
                        if i != j:
                            dx = config[i, 0] - config[j, 0]
                            dy = config[i, 1] - config[j, 1]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (config[i, 2] + config[j, 2])**2

                            if dist_sq < min_dist_sq:
                                violations.append(j)

                    # If this circle has serious overlaps, try to move it
                    if len(violations) > 2:
                        # Try to move to center of mass of surrounding circles
                        if len(violations) > 0:
                            center_x = 0.0
                            center_y = 0.0
                            count = 0
                            for j in violations:
                                center_x += config[j, 0]
                                center_y += config[j, 1]
                                count += 1

                            if count > 0:
                                center_x /= count
                                center_y /= count

                                # Move away from center to reduce overlap
                                dx = config[i, 0] - center_x
                                dy = config[i, 1] - center_y
                                dist = np.sqrt(dx*dx + dy*dy)

                                if dist > 0.001:
                                    # Move in opposite direction with larger step
                                    move_x = dx * 0.02 / dist
                                    move_y = dy * 0.02 / dist

                                    new_x = config[i, 0] + move_x
                                    new_y = config[i, 1] + move_y

                                    # Check bounds
                                    new_x = np.clip(new_x, config[i, 2], 1 - config[i, 2])
                                    new_y = np.clip(new_y, config[i, 2], 1 - config[i, 2])

                                    # Check if this is valid
                                    valid = True
                                    for k in range(n):
                                        if i != k:
                                            dx_k = new_x - config[k, 0]
                                            dy_k = new_y - config[k, 1]
                                            dist_sq_k = dx_k*dx_k + dy_k*dy_k
                                            min_dist_sq_k = (config[i, 2] + config[k, 2])**2

                                            if dist_sq_k < min_dist_sq_k:
                                                valid = False
                                                break

                                    if valid:
                                        config[i, 0] = new_x
                                        config[i, 1] = new_y
                                        improved = True
                                        improved_in_iteration = True

                if not improved_in_iteration:
                    break

            return config, improved

        # Strategy 2: Medium overlap - focused refinement
        def strategy_medium_overlap(config):
            improved = False
            max_iter = 30

            for iteration in range(max_iter):
                improved_in_iteration = False

                # Optimize each circle with careful radius adjustment
                for i in range(n):
                    orig_pos = config[i, :2].copy()
                    orig_r = config[i, 2]

                    # Calculate maximum possible radius
                    max_possible_r = min(
                        orig_pos[0],
                        1 - orig_pos[0],
                        orig_pos[1],
                        1 - orig_pos[1]
                    )

                    # Try to increase radius in small steps
                    steps = [0.002, 0.001]
                    for step in steps:
                        test_r = min(orig_r + step, max_possible_r)

                        if test_r > orig_r + 1e-6:
                            # Check all constraints
                            valid = True
                            for j in range(n):
                                if i != j:
                                    dx = orig_pos[0] - config[j, 0]
                                    dy = orig_pos[1] - config[j, 1]
                                    dist_sq = dx*dx + dy*dy
                                    min_dist_sq = (test_r + config[j, 2])**2

                                    if dist_sq < min_dist_sq:
                                        valid = False
                                        break

                            if valid:
                                config[i, 2] = test_r
                                improved = True
                                improved_in_iteration = True
                                break

                if not improved_in_iteration:
                    break

            return config, improved

        # Strategy 3: Balanced/low radius - conservative optimization
        def strategy_balanced(config):
            improved = False
            max_iter = 50

            for iteration in range(max_iter):
                improved_in_iteration = False

                # Try to make small improvements to each circle
                for i in range(n):
                    orig_pos = config[i, :2].copy()
                    orig_r = config[i, 2]

                    # Try to increase radius with very small steps
                    step = 0.0005
                    test_r = min(orig_r + step, 0.5)  # Cap at reasonable value

                    if test_r > orig_r + 1e-6:
                        # Check all constraints
                        valid = True
                        for j in range(n):
                            if i != j:
                                dx = orig_pos[0] - config[j, 0]
                                dy = orig_pos[1] - config[j, 1]
                                dist_sq = dx*dx + dy*dy
                                min_dist_sq = (test_r + config[j, 2])**2

                                if dist_sq < min_dist_sq:
                                    valid = False
                                    break

                        if valid:
                            config[i, 2] = test_r
                            improved = True
                            improved_in_iteration = True

                # Also try position refinement for each circle
                if improved_in_iteration:
                    continue

                for i in range(n):
                    # Try small position adjustments to free up space
                    orig_pos = config[i, :2].copy()
                    orig_r = config[i, 2]

                    # Simple repulsion from nearby circles
                    move_x, move_y = 0.0, 0.0
                    for j in range(n):
                        if i != j:
                            dx = config[i, 0] - config[j, 0]
                            dy = config[i, 1] - config[j, 1]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (orig_r + config[j, 2])**2

                            if dist_sq < min_dist_sq * 1.1:  # Slightly less than constraint for better movement
                                dist = np.sqrt(dist_sq)
                                if dist > 0.001:
                                    force = (min_dist_sq - dist_sq) / (dist * dist_sq + 1e-8)
                                    move_x += force * dx / dist
                                    move_y += force * dy / dist

                    # Apply small movement if there's any force
                    if abs(move_x) > 0.0001 or abs(move_y) > 0.0001:
                        new_x = orig_pos[0] + move_x * 0.005
                        new_y = orig_pos[1] + move_y * 0.005

                        # Check bounds
                        new_x = np.clip(new_x, orig_r, 1 - orig_r)
                        new_y = np.clip(new_y, orig_r, 1 - orig_r)

                        # Check validity
                        valid = True
                        for j in range(n):
                            if i != j:
                                dx = new_x - config[j, 0]
                                dy = new_y - config[j, 1]
                                dist_sq = dx*dx + dy*dy
                                min_dist_sq = (orig_r + config[j, 2])**2

                                if dist_sq < min_dist_sq:
                                    valid = False
                                    break

                        if valid:
                            config[i, 0] = new_x
                            config[i, 1] = new_y
                            improved = True
                            improved_in_iteration = True

                if not improved_in_iteration:
                    break

            return config, improved

        # Main refinement loop with adaptive strategy selection
        improved = True
        iteration = 0
        max_iterations = 100

        while improved and iteration < max_iterations:
            iteration += 1
            improved = False

            # Classify the current problem
            problem_type = classify_problem(config)

            # Apply appropriate strategy
            if problem_type == "high_overlap":
                config, strategy_improved = strategy_high_overlap(config)
            elif problem_type == "medium_overlap":
                config, strategy_improved = strategy_medium_overlap(config)
            else:  # "balanced" or "low_radius"
                config, strategy_improved = strategy_balanced(config)

            improved = strategy_improved

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