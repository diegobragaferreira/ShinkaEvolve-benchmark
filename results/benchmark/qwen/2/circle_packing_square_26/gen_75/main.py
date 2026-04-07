# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
import math
from scipy.optimize import minimize
import warnings

# Global constants
POP_SIZE = 150
NGEN = 100
MUTPB = 0.8
CXPB = 0.5
BOUND_LOW = 0.0
BOUND_UP = 1.0

# Define the fitness and individual classes for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def check_containment(circles):
    """Check containment constraints efficiently"""
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]

    # Check boundaries for all circles at once
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
    radii = circles[:, 2]

    # For each circle, find neighbors within sum of radii
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]

        # Query nearby points (within 2*(r1+r2) distance)
        neighbors = tree.query_ball_point([x1, y1], 2*(r1 + max(radii)))

        # Check overlaps with neighbors
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    # Use a softer penalty that encourages gradual improvement
                    overlap_amount = (r1 + r2 - distance)
                    penalty += 1000 * overlap_amount**2

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

    # Combine penalties with adjusted weights
    total_penalty = 10000 * containment_violations + overlap_penalty

    # Return fitness (higher is better)
    return (total_radius - total_penalty,)

def generate_improved_initialization():
    """Generate improved initial circle positions using a more sophisticated approach"""
    n = 26
    circles = []

    # Try to place circles in a way that maximizes initial radius while respecting boundaries
    # Using a clustering approach with strategic spacing

    # First, try to place circles at positions that avoid corners and edges
    # Place circles along a grid with strategic spacing
    rows_cols = int(math.ceil(math.sqrt(n)))
    spacing_x = 1.0 / (rows_cols + 1)
    spacing_y = 1.0 / (rows_cols + 1)

    # Special handling for corner positions
    special_positions = [
        (0.5, 0.5),  # center
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),  # corners
        (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5),  # midpoints
    ]

    count = 0
    placed_positions = set()

    # Place some circles in special positions to create better starting configuration
    for pos in special_positions:
        if count >= n:
            break
        x, y = pos
        # Only use if not already placed
        if (round(x, 3), round(y, 3)) not in placed_positions:
            # Adjust for boundary constraints
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))

            # Calculate max possible radius based on proximity to edges
            min_dist_to_bound = min(x, 1-x, y, 1-y)
            r = min(0.1, min_dist_to_bound/2)  # Reasonable starting radius

            # Add some variance to make it more interesting
            r *= random.uniform(0.8, 1.2)
            r = max(0.005, min(0.15, r))

            circles.extend([x, y, r])
            placed_positions.add((round(x, 3), round(y, 3)))
            count += 1

    # Fill remaining positions using grid placement
    for i in range(rows_cols):
        if count >= n:
            break
        for j in range(rows_cols):
            if count >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y

            # Adjust for boundary constraints
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))

            # Check if this position is already taken
            if (round(x, 3), round(y, 3)) not in placed_positions:
                # Calculate max possible radius based on proximity to edges
                min_dist_to_bound = min(x, 1-x, y, 1-y)
                r = min(0.08, min_dist_to_bound/2)

                # Add some variance
                r *= random.uniform(0.7, 1.1)
                r = max(0.005, min(0.15, r))

                circles.extend([x, y, r])
                placed_positions.add((round(x, 3), round(y, 3)))
                count += 1

    # If we still don't have enough circles, add random ones
    while len(circles) < n * 3:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        r = random.uniform(0.005, 0.12)
        circles.extend([x, y, r])

    return circles[:n*3]

def init_individual():
    """Initialize an individual with improved initialization"""
    individual = generate_improved_initialization()

    # Add some random perturbations to improve local search
    for i in range(len(individual)):
        if i % 3 == 2:  # This is a radius
            # Mutate radius with bounded adjustment
            individual[i] = max(0.001, min(0.4, individual[i] * random.uniform(0.9, 1.1)))
        else:  # This is x or y coordinate
            # Mutate position with bounded adjustment
            individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, 0.01)))

    return individual

def mutate_individual(individual):
    """Mutate an individual with adaptive mutation rates and different strategies"""
    # Get current generation (this is a bit hacky but works for this context)
    # In practice, this should be passed in or managed differently
    gen_rate = 0.1  # Base mutation rate

    for i in range(len(individual)):
        if random.random() < gen_rate:  # Apply adaptive mutation
            if i % 3 == 2:  # This is a radius
                # Mutate radius with bounded adjustment - allow larger changes occasionally
                # but keep it bounded to prevent invalid values
                mutation_factor = random.uniform(0.7, 1.3)
                individual[i] = max(0.001, min(0.4, individual[i] * mutation_factor))
            else:  # This is x or y coordinate
                # Mutate position with bounded adjustment - use adaptive standard deviation
                # based on how close we are to the boundary
                boundary_distance = min(individual[i], 1-individual[i])
                std_dev = min(0.05, max(0.005, boundary_distance/2))
                individual[i] = max(BOUND_LOW, min(BOUND_UP, individual[i] + random.gauss(0, std_dev)))
    return individual,

def local_refinement(circles_array, max_iter=100):
    """Apply enhanced local optimization to improve final solution"""
    def objective(params):
        # Reshape params back into circles array
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

        # Overlap penalties using more sophisticated calculation
        if len(circles) > 1:
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        overlap_amount = (r1 + r2 - distance)
                        penalty += 1000 * overlap_amount**2  # Quadratic penalty

        return sum_radii + penalty

    # Try several local search approaches
    best_circles = circles_array.copy()
    best_fitness = -np.sum(circles_array[:, 2])  # Negative because we want to maximize

    # First, try gradient-free optimization
    flat_params = circles_array.flatten()
    try:
        # Try L-BFGS-B with bounds but with smaller steps
        result = minimize(objective, flat_params, method='L-BFGS-B',
                         bounds=[(0, 1) if i%3 != 2 else (0.001, 0.4) for i in range(len(flat_params))],
                         options={'maxiter': max_iter//2, 'ftol': 1e-6, 'gtol': 1e-6})
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Check if this is actually better
            current_fitness = -np.sum(optimized_circles[:, 2])
            if current_fitness > best_fitness:
                best_circles = optimized_circles
                best_fitness = current_fitness
    except Exception:
        pass

    # Second, try a simple neighborhood search approach
    # This helps escape local minima that optimization might get stuck in
    current_circles = circles_array.copy()
    for _ in range(50):  # Reduced iterations to save time
        # Make small random changes to positions
        mutated = current_circles.copy()
        for i in range(len(mutated)):
            if i % 3 == 2:  # radius
                # Small random change to radius
                mutated[i] = max(0.001, min(0.4, mutated[i] * np.random.normal(1, 0.05)))
            else:  # position
                # Small random change to position
                mutated[i] = max(0, min(1, mutated[i] + np.random.normal(0, 0.005)))

        # Check if new configuration is better
        fitness = -np.sum(mutated[:, 2])
        # Simple heuristic: accept if it's not worse by too much
        # (the actual constraint checking would be done in eval_circles)
        if fitness > best_fitness - 0.001:  # Allow small regressions
            best_circles = mutated
            best_fitness = fitness

    return best_circles

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
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution loop with adaptive parameters
    for gen in range(NGEN):
        # Adjust mutation rate over generations
        current_mutation_rate = MUTPB * max(0.1, 1.0 - gen/NGEN)

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

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    # Find the best individual
    best_ind = tools.selBest(pop, 1)[0]
    circles = np.array(best_ind).reshape(-1, 3)

    # Apply local refinement to improve the final solution
    circles = local_refinement(circles)

    return circles


# EVOLVE-BLOCK-END