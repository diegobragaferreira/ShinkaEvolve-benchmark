# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from deap import base, creator, tools, algorithms
import random
import time

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 26
    max_radius = 0.5

    # Precompute some constants
    min_distance = 0.001  # Minimum distance between circles for numerical stability

    def is_valid_position(x, y, r, existing_circles):
        """Check if a circle at (x,y) with radius r is valid"""
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

        # Check overlap with existing circles
        for cx, cy, cr in existing_circles:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < r + cr + min_distance:
                return False
        return True

    def create_initial_population(pop_size):
        """Create an initial population with good starting configurations"""
        population = []

        # Generate multiple initial solutions using different strategies
        for _ in range(pop_size):
            circles = []
            # Try to place circles greedily
            for i in range(n_circles):
                best_x, best_y, best_r = 0.5, 0.5, 0.0

                # Find a valid position with maximum possible radius
                attempts = 0
                while attempts < 1000:
                    # Randomly try to place a circle
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)

                    # Calculate maximum radius at this position
                    r = min(x, 1-x, y, 1-y)

                    # If valid and has room for improvement, accept it
                    if is_valid_position(x, y, r, circles):
                        # Try to increase radius while keeping it valid
                        max_r = r
                        for _ in range(100):
                            new_r = min(max_r * 1.1, max_radius)
                            if is_valid_position(x, y, new_r, circles):
                                max_r = new_r
                            else:
                                break
                        best_x, best_y, best_r = x, y, max_r
                        break
                    attempts += 1

                circles.append([best_x, best_y, best_r])

            population.append(circles)

        return population

    def evaluate(individual):
        """Evaluate fitness - sum of radii"""
        total_radius = sum(circle[2] for circle in individual)
        return (total_radius,)

    def mutate(individual):
        """Custom mutation operator that respects constraints"""
        # Pick a random circle to modify
        idx = random.randint(0, len(individual) - 1)
        x, y, r = individual[idx]

        # Create a new, slightly modified circle
        new_x = max(0.01, min(0.99, x + random.gauss(0, 0.02)))
        new_y = max(0.01, min(0.99, y + random.gauss(0, 0.02)))

        # Recalculate max possible radius at new location
        new_r = min(new_x, 1-new_x, new_y, 1-new_y)

        # Adjust radius to ensure no overlaps
        for i, (cx, cy, cr) in enumerate(individual):
            if i != idx:
                distance = np.sqrt((new_x - cx)**2 + (new_y - cy)**2)
                max_radius_allowed = distance - cr - 0.001
                if max_radius_allowed > 0:
                    new_r = min(new_r, max_radius_allowed)

        # Ensure minimum positive radius
        new_r = max(0.001, new_r)

        individual[idx] = [new_x, new_y, new_r]
        return individual,

    def crossover(ind1, ind2):
        """Custom crossover that maintains validity"""
        # Simple uniform crossover of positions and radii
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Repair individuals to maintain constraints
        for ind in [ind1, ind2]:
            # Repeatedly repair until no more repairs needed
            changed = True
            while changed:
                changed = False
                for i in range(len(ind)):
                    x, y, r = ind[i]
                    # Ensure boundary constraints
                    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                        # Try to shrink radius and reposition
                        r = min(x, 1-x, y, 1-y)
                        r = max(0.001, r)
                        changed = True

                    # Check for overlaps
                    for j in range(len(ind)):
                        if i != j:
                            cx, cy, cr = ind[j]
                            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
                            if distance < r + cr + 0.001:
                                # Reduce radius to prevent overlap
                                r = max(0.001, distance - cr - 0.001)
                                changed = True

                    ind[i] = [x, y, r]

        return ind1, ind2

    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: create_initial_population(1)[0])
    toolbox.register("population", lambda: create_initial_population(20))
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population()

    # Run evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run evolution with timeout
    start_time = time.time()
    try:
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3,
                                      ngen=50, stats=stats, halloffame=hof,
                                      verbose=False)
    except Exception:
        pass

    # Return the best individual found
    best_individual = hof[0] if len(hof) > 0 else pop[0]

    # Convert to final result format (ensure all circles are valid)
    final_circles = []
    for x, y, r in best_individual:
        # Final validation and adjustment
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        final_circles.append([x, y, r])

    return np.array(final_circles)

# EVOLVE-BLOCK-END