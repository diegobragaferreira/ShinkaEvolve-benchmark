# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
import math
from scipy.optimize import minimize
import warnings

# Global constants - optimized parameters
POP_SIZE = 180
NGEN = 120
MUTPB = 0.12
CXPB = 0.6
BOUND_LOW = 0.0
BOUND_UP = 1.0
ELITISM_COUNT = 8

# Define the fitness and individual classes for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def check_containment(circles):
    """Check containment constraints efficiently with vectorized operations"""
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]

    # Vectorized containment check for all circles
    containment_violations = (
        (x_coords - radii < BOUND_LOW) |
        (x_coords + radii > BOUND_UP) |
        (y_coords - radii < BOUND_LOW) |
        (y_coords + radii > BOUND_UP)
    )
    
    return np.sum(containment_violations)

def calculate_overlap_penalty(circles):
    """Calculate overlap penalty using efficient spatial indexing"""
    if len(circles) <= 1:
        return 0.0

    # Build KDTree for efficient neighbor search
    tree = cKDTree(circles[:, :2])

    penalty = 0.0
    
    # For each circle, find neighbors within sum of radii + buffer
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        
        # Query neighbors within a reasonable range with buffer
        neighbors = tree.query_ball_point([x1, y1], 2 * (r1 + 0.1))
        
        # Check overlaps with neighbors
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    penalty += 1000 * (r1 + r2 - distance)

    return penalty

def eval_circles(individual):
    """Evaluate the fitness of an individual (set of circles)"""
    # Convert individual to circles array
    circles = np.array(individual).reshape(-1, 3)

    # Calculate sum of radii
    total_radius = np.sum(circles[:, 2])

    # Check constraints
    containment_violations = check_containment(circles)
    overlap_penalty = calculate_overlap_penalty(circles)

    # Combine penalties with weighted scheme
    total_penalty = 10000 * containment_violations + overlap_penalty

    # Return fitness (higher is better)
    return (total_radius - total_penalty,)

def generate_improved_initialization():
    """Generate improved initial circle positions with strategic placement"""
    n = 26
    circles = []

    # Strategic positions for better initial distribution - more carefully chosen
    strategic_positions = [
        (0.5, 0.5),  # center
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),  # corners
        (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5),  # midpoints
        (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),  # near corners
        (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85), # inner corners
    ]

    placed_positions = set()
    count = 0

    # Place circles in strategic positions first
    for pos in strategic_positions:
        if count >= n:
            break
        x, y = pos
        # Ensure we don't go out of bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))

        # Skip if already placed
        if (round(x, 3), round(y, 3)) not in placed_positions:
            # Calculate max possible radius based on proximity to boundaries
            min_dist_to_bound = min(x, 1-x, y, 1-y)
            r = min(0.12, min_dist_to_bound/2)

            # Add some variance
            r *= random.uniform(0.85, 1.15)
            r = max(0.005, min(0.15, r))

            circles.extend([x, y, r])
            placed_positions.add((round(x, 3), round(y, 3)))
            count += 1

    # Fill remaining spots with grid-based placement
    rows_cols = int(math.ceil(math.sqrt(n - len(placed_positions))))
    if rows_cols < 1:
        rows_cols = 1

    spacing_x = 1.0 / (rows_cols + 1)
    spacing_y = 1.0 / (rows_cols + 1)

    for i in range(rows_cols):
        if count >= n:
            break
        for j in range(rows_cols):
            if count >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y

            # Ensure boundary constraints
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))

            # Check if position already occupied
            if (round(x, 3), round(y, 3)) not in placed_positions:
                min_dist_to_bound = min(x, 1-x, y, 1-y)
                r = min(0.08, min_dist_to_bound/2)

                # Add variance
                r *= random.uniform(0.75, 1.2)
                r = max(0.005, min(0.15, r))

                circles.extend([x, y, r])
                placed_positions.add((round(x, 3), round(y, 3)))
                count += 1

    # Fill any remaining positions with random placement
    while len(circles) < n * 3:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        r = random.uniform(0.005, 0.12)
        circles.extend([x, y, r])

    return circles[:n*3]

def init_individual():
    """Initialize an individual with better initialization"""
    individual = generate_improved_initialization()

    # Add more variance via mutation-like perturbations
    for i in range(len(individual)):
        if i % 3 == 2:  # This is a radius
            # Mutate radius with bounded adjustment
            individual[i] = max(0.001, min(0.4, individual[i] * random.uniform(0.9, 1.1)))
        else:  # This is x or y coordinate
            # Mutate position with bounded adjustment
            individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, 0.015)))

    return individual

def mutate_individual(individual):
    """Mutate an individual with adaptive mutation rates"""
    # Exponential decay for mutation rate - more aggressive decay
    gen_rate = MUTPB * (0.1 ** (1.0 * NGEN / NGEN))  # Adjusted for consistent decay

    for i in range(len(individual)):
        if random.random() < gen_rate:
            if i % 3 == 2:  # This is a radius
                # Mutate radius with bounded adjustment
                individual[i] = max(0.001, min(0.4, individual[i] * random.uniform(0.85, 1.15)))
            else:  # This is x or y coordinate
                # Mutate position with bounded adjustment
                individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, 0.025)))

    return individual,

def constraint_aware_crossover(ind1, ind2):
    """Perform crossover that respects constraints"""
    # Perform uniform crossover
    tools.cxUniform(ind1, ind2, indpb=0.1)

    # Convert to numpy arrays for easier manipulation
    arr1 = np.array(ind1).reshape(-1, 3)
    arr2 = np.array(ind2).reshape(-1, 3)

    # Check if offspring violate constraints and repair if needed
    def repair_if_needed(circles):
        # Repair containment violations - more aggressive bounding
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Bound the circle to stay within unit square with margin
            x = max(r + 0.005, min(1-r - 0.005, x))
            y = max(r + 0.005, min(1-r - 0.005, y))
            circles[i] = [x, y, r]

        # Repair overlap violations through iterative adjustment
        changed = True
        iterations = 0
        while changed and iterations < 25:
            changed = False
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if distance < r1 + r2:
                            # Adjust positions to resolve overlap
                            overlap = (r1 + r2) - distance
                            # Move circles apart along the line connecting their centers
                            dx = x2 - x1
                            dy = y2 - y1
                            if dx == 0 and dy == 0:
                                # Random movement if they're at the same point
                                angle = random.uniform(0, 2*math.pi)
                                dx = math.cos(angle)
                                dy = math.sin(angle)

                            # Normalize
                            norm = math.sqrt(dx*dx + dy*dy)
                            dx /= norm
                            dy /= norm

                            # Move both circles apart (smaller move amount to prevent overshooting)
                            move_amount = overlap / 4.0
                            circles[i][0] -= dx * move_amount
                            circles[i][1] -= dy * move_amount
                            circles[j][0] += dx * move_amount
                            circles[j][1] += dy * move_amount

                            changed = True

            iterations += 1

        return circles

    # Repair both offspring
    repaired_ind1 = repair_if_needed(arr1.copy())
    repaired_ind2 = repair_if_needed(arr2.copy())

    # Convert back to individual format
    ind1[:] = repaired_ind1.flatten().tolist()
    ind2[:] = repaired_ind2.flatten().tolist()

    return ind1, ind2

def local_refinement_with_improved_strategy(circles_array, max_iter=80):
    """Apply improved local optimization strategy to refine final solution"""
    def objective(params):
        circles = params.reshape(-1, 3)
        # Calculate sum of radii (negative because we want to maximize)
        sum_radii = -np.sum(circles[:, 2])

        # Penalty for constraint violations
        penalty = 0

        # Boundary constraints
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        # Check containment violations
        containment_violations = (
            (x_coords - radii < BOUND_LOW) |
            (x_coords + radii > BOUND_UP) |
            (y_coords - radii < BOUND_LOW) |
            (y_coords + radii > BOUND_UP)
        )
        penalty += 10000 * np.sum(containment_violations)

        # Overlap penalties
        if len(circles) > 1:
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        penalty += 1000 * (r1 + r2 - distance)

        return sum_radii + penalty

    def constraint_handling(circles):
        """Handle boundary and overlap constraints manually"""
        circles_copy = circles.copy()
        for i in range(len(circles_copy)):
            x, y, r = circles_copy[i]
            # Fix containment - ensure circle is within bounds with margin
            x = max(r + 0.005, min(1-r - 0.005, x))
            y = max(r + 0.005, min(1-r - 0.005, y))
            circles_copy[i] = [x, y, r]
        return circles_copy

    # Start with constraint handling
    circles_array = constraint_handling(circles_array)

    # Flatten to optimize
    flat_params = circles_array.flatten()

    # Use L-BFGS-B optimization which is fast and effective for this problem
    try:
        result = minimize(
            objective,
            flat_params,
            method='L-BFGS-B',
            bounds=[(0.005, 0.995) if i%3 != 2 else (0.001, 0.4) for i in range(len(flat_params))],
            options={'maxiter': max_iter//2, 'ftol': 1e-6},
            tol=1e-6
        )

        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Apply constraint fixes to results to ensure feasibility
            optimized_circles = constraint_handling(optimized_circles)
            return optimized_circles
    except Exception:
        pass

    # Fallback to original
    return circles_array

def adaptive_tournament_selection(population, k, diversity_threshold=0.1):
    """Adaptive tournament selection based on population diversity"""
    # Calculate diversity metric
    if len(population) < 2:
        return tools.selTournament(population, k, tournsize=3)

    # Compute diversity based on average distance between individuals
    distances = []
    for i in range(min(20, len(population))):
        for j in range(i+1, min(20, len(population))):
            dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
            distances.append(dist)

    avg_diversity = np.mean(distances) if distances else 0

    # Adjust tournament size based on diversity - smoother adjustments
    if avg_diversity > diversity_threshold:
        tournsize = max(3, min(8, int(5 + avg_diversity * 12)))  # Slightly higher diversity = larger tournaments
    else:
        tournsize = max(3, min(8, int(3 + avg_diversity * 25)))   # Lower diversity = smaller tournaments

    return tools.selTournament(population, k, tournsize=tournsize)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_circles)
    toolbox.register("mate", constraint_aware_crossover)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", adaptive_tournament_selection)

    # Create population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop with enhanced adaptive parameters
    for gen in range(NGEN):
        # Adjust mutation rate using exponential decay
        current_mutation_rate = MUTPB * (0.1 ** (1.0 * gen / NGEN))

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < current_mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Elitism: keep best individuals
        if ELITISM_COUNT > 0:
            best_individuals = tools.selBest(pop, ELITISM_COUNT)
            offspring[:ELITISM_COUNT] = best_individuals

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    # Find the best individual
    best_ind = tools.selBest(pop, 1)[0]
    circles = np.array(best_ind).reshape(-1, 3)

    # Apply enhanced local refinement to improve the final solution
    circles = local_refinement_with_improved_strategy(circles)

    return circles

# EVOLVE-BLOCK-END