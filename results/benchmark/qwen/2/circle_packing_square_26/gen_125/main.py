# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
import random
import time

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

    # Improved spiral-based initialization for better distribution
    def generate_spiral_initialization(n_circles):
        # Use Fibonacci spiral for better circle distribution
        positions = []
        golden_ratio = (1 + np.sqrt(5)) / 2.0

        for i in range(n_circles):
            # Fibonacci spiral placement
            theta = i * 2 * np.pi / golden_ratio
            radius = np.sqrt(i / n_circles)

            # Convert to Cartesian coordinates and normalize to [0,1] range
            x = 0.5 + radius * np.cos(theta) * 0.4
            y = 0.5 + radius * np.sin(theta) * 0.4

            # Add jitter to avoid symmetry issues
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)

            # Clip to valid range
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)

            positions.append([x, y])

        return np.array(positions)

    # Optimized overlap checking using vectorized operations
    def check_overlaps_vectorized(config):
        """Vectorized overlap checking for better performance"""
        if len(config) < 2:
            return np.zeros(len(config), dtype=bool)

        # Build KDTree for efficient neighbor search
        points = config[:, :2]
        tree = cKDTree(points)

        # Query all points for neighbors within 2*(r1+r2) distance
        # This creates a sparse matrix of overlaps
        overlaps = np.zeros(len(config), dtype=bool)

        for i in range(len(config)):
            x, y, r = config[i]

            # Find nearby circles (within reasonable distance)
            nearby = tree.query_ball_point([x, y], 2 * (r + 0.01), p=np.inf)

            # Check actual overlap with neighbors
            for j in nearby:
                if i != j:
                    x2, y2, r2 = config[j]
                    dx = x - x2
                    dy = y - y2
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (r + r2)**2

                    if dist_sq < min_dist_sq:
                        overlaps[i] = True
                        break

        return overlaps

    # Check if a configuration is valid with better error handling
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

        # Check overlaps
        overlaps = check_overlaps_vectorized(config)
        return not np.any(overlaps)

    # Calculate objective function with proper constraints
    def evaluate(individual):
        config = individual.reshape(n, 3)
        if not is_valid_config(config):
            return (-1e8,)  # Heavy penalty for invalid configurations
        total_radius = np.sum(config[:, 2])
        return (total_radius,)

    # Enhanced local optimization with physics-inspired repulsion
    def local_optimization_physics(config, max_iter=50):
        """Physics-inspired local optimization with repulsion forces"""
        # Copy input to avoid modifying original
        optimized = config.copy()
        prev_total = np.sum(optimized[:, 2])

        for iter_num in range(max_iter):
            improved = False

            # For each circle, compute forces from neighbors
            for i in range(len(optimized)):
                x, y, r = optimized[i]
                force_x, force_y = 0.0, 0.0

                # Compute repulsion forces from all other circles
                for j in range(len(optimized)):
                    if i != j:
                        x2, y2, r2 = optimized[j]
                        dx = x - x2
                        dy = y - y2
                        dist = np.sqrt(dx*dx + dy*dy)

                        # Only consider if circles are close enough to overlap
                        if dist < (r + r2 + 1e-6):  # Add small epsilon
                            # Repulsion force (inverse square law)
                            if dist > 1e-8:  # Avoid division by zero
                                force_mag = 0.01 * (r + r2 - dist) / (dist * dist)
                                force_x += force_mag * dx / dist
                                force_y += force_mag * dy / dist

                # Apply forces to move circle away from others
                if abs(force_x) > 1e-8 or abs(force_y) > 1e-8:
                    # Move with bounded velocity
                    new_x = x + np.clip(force_x, -0.005, 0.005)
                    new_y = y + np.clip(force_y, -0.005, 0.005)

                    # Keep within bounds
                    new_x = np.clip(new_x, r, 1 - r)
                    new_y = np.clip(new_y, r, 1 - r)

                    # Try to increase radius if possible
                    max_radius = min(new_x, 1-new_x, new_y, 1-new_y)
                    new_r = min(max_radius, r + 0.005)

                    # Check if this change is valid
                    temp_config = optimized.copy()
                    temp_config[i] = [new_x, new_y, new_r]

                    if is_valid_config(temp_config):
                        optimized[i] = [new_x, new_y, new_r]
                        improved = True

            # Early stopping if no improvement
            current_total = np.sum(optimized[:, 2])
            if current_total <= prev_total + 1e-6:
                break
            prev_total = current_total

        return optimized

    # Advanced refinement with multiple optimization stages
    def advanced_refinement(config):
        """Multi-stage refinement approach"""
        # Stage 1: Basic local optimization
        refined = local_optimization_physics(config, max_iter=30)

        # Stage 2: Radius maximization
        for _ in range(20):
            improved = False
            for i in range(len(refined)):
                orig_r = refined[i, 2]
                # Try to increase radius
                max_possible = min(
                    refined[i, 0], 1 - refined[i, 0],
                    refined[i, 1], 1 - refined[i, 1]
                )

                # Try incremental increases
                for incr in [0.005, 0.002, 0.001]:
                    test_r = min(orig_r + incr, max_possible * 0.99)
                    if test_r > orig_r + 1e-6:
                        # Test if this works
                        temp_config = refined.copy()
                        temp_config[i, 2] = test_r

                        if is_valid_config(temp_config):
                            refined[i, 2] = test_r
                            improved = True
                            break

            if not improved:
                break

        # Stage 3: Final physics-based optimization
        refined = local_optimization_physics(refined, max_iter=20)

        return refined

    # Create toolboxes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initialize individuals with improved starting configuration
    def init_individual():
        # Generate spiral-based initial positions
        positions = generate_spiral_initialization(n)

        # Set initial radii and positions
        individual = np.zeros((n, 3))
        for i in range(n):
            individual[i, 0] = positions[i, 0]
            individual[i, 1] = positions[i, 1]
            # Start with more reasonable initial radii
            individual[i, 2] = 0.02 + np.random.random() * 0.03

        # Ensure all are within bounds
        for i in range(n):
            max_radius = min(
                individual[i, 0],
                1 - individual[i, 0],
                individual[i, 1],
                1 - individual[i, 1]
            )
            individual[i, 2] = min(individual[i, 2], max_radius * 0.95)

        return creator.Individual(individual.flatten())

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Enhanced genetic operators with better parameters
    def mutate_individual(individual, indpb=0.2):
        individual = individual.reshape(n, 3)
        mut_count = 0

        for i in range(n):
            if random.random() < indpb:
                mut_count += 1
                # Choice of mutation type with weighted probabilities
                choice = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]

                if choice == 0:  # x coordinate - larger mutation
                    individual[i, 0] += np.random.normal(0, 0.03)
                    individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
                elif choice == 1:  # y coordinate - larger mutation
                    individual[i, 1] += np.random.normal(0, 0.03)
                    individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)
                else:  # radius - smaller mutation
                    individual[i, 2] += np.random.normal(0, 0.01)
                    individual[i, 2] = np.clip(individual[i, 2], 0.001, 0.49)

        return (individual.flatten(),)

    def cx_uniform(ind1, ind2):
        # Uniform crossover with higher probability for better mixing
        ind1 = ind1.reshape(n, 3)
        ind2 = ind2.reshape(n, 3)

        # More thorough crossover with better probability
        for i in range(n):
            if random.random() < 0.7:  # Increased crossover probability
                ind1[i] = ind2[i].copy()

        return (ind1.flatten(), ind2.flatten())

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_uniform)
    toolbox.register("mutate", mutate_individual)

    # Improved tournament selection
    def sel_tournament_adaptive(individuals, k, tournsize=None):
        if tournsize is None:
            # Adaptive tournament size based on population diversity
            fitnesses = [ind.fitness.values[0] for ind in individuals]
            if len(fitnesses) > 1:
                var_fitness = np.var(fitnesses)
                # Larger tournaments for diverse populations, smaller for homogeneous
                tournsize = max(2, min(len(individuals), int(3 + var_fitness * 15)))
            else:
                tournsize = 3

        return tools.selTournament(individuals, k, tournsize)

    toolbox.register("select", sel_tournament_adaptive)

    # Run evolution
    pop = toolbox.population(n=80)  # Larger population
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

        # Evolution parameters with better progression
        n_generations = 60
        # More aggressive initial mutation, then decay
        mutation_rates = [0.2 * (0.92 ** i) for i in range(n_generations)]

        # Main evolutionary loop
        for gen in range(n_generations):
            # Current mutation rate
            current_mutation_rate = mutation_rates[gen]

            # Select
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.85:  # Higher crossover rate
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

        # Final advanced refinement
        refined_solution = advanced_refinement(best_solution.copy())
        return refined_solution

    except Exception as e:
        # Fallback to simple grid-based solution if evolutionary fails
        print(f"Evolution failed: {e}")
        positions = generate_spiral_initialization(n)
        final_solution = np.zeros((n, 3))
        for i in range(n):
            final_solution[i, 0] = positions[i, 0]
            final_solution[i, 1] = positions[i, 1]
            final_solution[i, 2] = 0.05  # Set small but equal radii
        return final_solution

# EVOLVE-BLOCK-END