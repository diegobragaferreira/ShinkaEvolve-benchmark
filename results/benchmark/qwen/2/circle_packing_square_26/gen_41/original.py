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

    # Precompute grid positions for better initialization
    def generate_grid_initialization(n_circles):
        # Create a grid pattern for initial placement
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n_circles:
                    break
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                positions.append([x, y])
        return np.array(positions[:n_circles])

    # Check if two circles overlap
    def circles_overlap(pos1, r1, pos2, r2):
        dist_sq = (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
        return dist_sq < (r1 + r2)**2

    # Check if a circle is within bounds
    def is_valid_circle(pos, r):
        return (r <= pos[0] <= 1 - r) and (r <= pos[1] <= 1 - r)

    # Check if a configuration is valid
    def is_valid_config(config):
        # Check bounds
        for i in range(n):
            if not is_valid_circle(config[i, :2], config[i, 2]):
                return False

        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                if circles_overlap(config[i, :2], config[i, 2], config[j, :2], config[j, 2]):
                    return False
        return True

    # Calculate objective function (negative because we want to maximize sum of radii)
    def evaluate(individual):
        config = individual.reshape(n, 3)
        if not is_valid_config(config):
            return (-1e6,)  # Penalize invalid configurations heavily
        total_radius = np.sum(config[:, 2])
        return (total_radius,)

    # Local optimization refinement function
    def refine_configuration(config):
        # Simple gradient ascent approach for refinement
        old_total = np.sum(config[:, 2])
        improved = True
        max_iterations = 100

        for _ in range(max_iterations):
            if not improved:
                break
            improved = False

            # Try to increase each radius slightly while maintaining constraints
            for i in range(n):
                orig_r = config[i, 2]
                test_r = min(orig_r + 0.001, 0.5)

                # Check if we can increase this radius
                valid = True
                for j in range(n):
                    if i != j:
                        dist_sq = (config[i, 0] - config[j, 0])**2 + (config[i, 1] - config[j, 1])**2
                        if dist_sq < (test_r + config[j, 2])**2:
                            valid = False
                            break

                # Check bounds
                if valid and test_r <= config[i, 0] and test_r <= config[i, 1] and test_r <= 1 - config[i, 0] and test_r <= 1 - config[i, 1]:
                    config[i, 2] = test_r
                    improved = True

            if not improved:
                break

        return config

    # Create toolboxes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initialize individuals
    def init_individual():
        # Generate initial grid
        positions = generate_grid_initialization(n)

        # Add some randomness and set initial radii
        individual = np.zeros((n, 3))
        for i in range(n):
            individual[i, 0] = positions[i, 0] + np.random.normal(0, 0.05)
            individual[i, 1] = positions[i, 1] + np.random.normal(0, 0.05)
            individual[i, 2] = 0.01 + np.random.random() * 0.05  # Small initial radii

            # Ensure within bounds
            individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
            individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)
            individual[i, 2] = np.clip(individual[i, 2], 0.001, 0.49)

        return creator.Individual(individual.flatten())

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators
    def mutate_individual(individual, indpb=0.1):
        individual = individual.reshape(n, 3)
        mut_count = 0

        for i in range(n):
            if random.random() < indpb:
                mut_count += 1
                # Mutate one of x, y, or r
                choice = random.randint(0, 2)
                if choice == 0:  # x coordinate
                    individual[i, 0] += np.random.normal(0, 0.05)
                    individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
                elif choice == 1:  # y coordinate
                    individual[i, 1] += np.random.normal(0, 0.05)
                    individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)
                else:  # radius
                    individual[i, 2] += np.random.normal(0, 0.02)
                    individual[i, 2] = np.clip(individual[i, 2], 0.001, 0.49)

        return (individual.flatten(),)

    def cx_uniform(ind1, ind2):
        # Uniform crossover
        ind1 = ind1.reshape(n, 3)
        ind2 = ind2.reshape(n, 3)

        for i in range(n):
            if random.random() < 0.5:
                ind1[i] = ind2[i].copy()

        return (ind1.flatten(), ind2.flatten())

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_uniform)
    toolbox.register("mutate", mutate_individual)

    # Tournament selection with fitness variance based sizing
    def sel_tournament_varsize(individuals, k, tournsize=None):
        if tournsize is None:
            # Calculate tournament size based on fitness variance
            fitnesses = [ind.fitness.values[0] for ind in individuals]
            var_fitness = np.var(fitnesses)
            # Normalize tournament size - larger variance means more aggressive selection
            tournsize = max(2, min(len(individuals), int(3 + var_fitness * 10)))

        return tools.selTournament(individuals, k, tournsize)

    toolbox.register("select", sel_tournament_varsize)

    # Run evolution
    pop = toolbox.population(n=50)
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

        # Evolution parameters
        n_generations = 50
        mutation_rate_decay = 0.1  # Rate of decay for mutation probability

        # Main evolutionary loop
        for gen in range(n_generations):
            # Update mutation rate
            current_mutation_rate = 0.1 * (1 - gen / n_generations) * (1 - mutation_rate_decay)

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