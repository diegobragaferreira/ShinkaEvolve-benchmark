# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Problem parameters
    n_circles = 32
    square_size = 1.0

    # Better initialization using hexagonal packing pattern
    def initialize_hexagonal():
        # Create a hexagonal grid pattern (more efficient than random)
        rows = int(math.ceil(math.sqrt(n_circles)))
        cols = int(math.ceil(n_circles / rows))

        # Hexagon parameters for better packing efficiency
        side_length = 0.12  # Adjusted for better results
        height = side_length * math.sqrt(3) / 2

        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n_circles:
                    break
                # Offset every other row
                x = (j + (i % 2) * 0.5) * side_length * 2
                y = i * height
                # Only add if within bounds
                if x <= 1 and y <= 1:
                    positions.append([x, y])

        # Adjust for edge cases
        if len(positions) < n_circles:
            # Fill remaining positions randomly in valid areas
            for i in range(n_circles - len(positions)):
                x = random.uniform(side_length, 1 - side_length)
                y = random.uniform(side_length, 1 - side_length)
                positions.append([x, y])

        positions = positions[:n_circles]
        return positions

    # Fitness function with improved constraint checking
    def evaluate(individual):
        # Convert individual to circles array (x, y, r)
        circles = np.array(individual).reshape(-1, 3)

        # Check constraints
        total_radius = np.sum(circles[:, 2])

        # Penalty for invalid positions
        penalty = 0

        # Check containment constraints efficiently
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]

        # Vectorized containment check
        containment_ok = (radii <= x_coords) & (x_coords <= 1 - radii) & \
                         (radii <= y_coords) & (y_coords <= 1 - radii)
        if not np.all(containment_ok):
            penalty += 1000000

        # Efficient overlap checking using vectorized operations
        if len(circles) > 1:
            # Use cdist for pairwise distance calculation
            points = circles[:, :2]
            distances = cdist(points, points)

            # Create matrix of sums of radii
            radii_matrix = np.add.outer(radii, radii)

            # Zero out diagonal (same circle comparison)
            np.fill_diagonal(distances, np.inf)

            # Check if any distances are less than sum of radii
            overlaps = distances < radii_matrix
            if np.any(overlaps):
                penalty += 1000000

        # Return fitness (negative because we want to maximize)
        return (-total_radius - penalty,),

    # Create classes for GA
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_circles * 3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.03, indpb=0.2)  # Reduced sigma
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population with better hexagonal packing
    def initialize_population_hexagonal(pop_size):
        population = []
        for _ in range(pop_size):
            # Get initial hexagonal positions
            positions = initialize_hexagonal()

            # Set initial radii based on proximity to others
            circles = []
            for i, (x, y) in enumerate(positions):
                # Start with a reasonable initial radius that can be optimized
                r = min(0.05, x, 1-x, y, 1-y)
                # Make some variations for diversity
                r = max(0.001, r * random.uniform(0.8, 1.2))
                circles.append((x, y, r))

            population.append(creator.Individual([item for circle in circles for item in circle]))
        return population

    # Run evolution
    pop = initialize_population_hexagonal(30)  # Reduced population size for speed
    hof = tools.HallOfFame(1)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evolution parameters - more focused on quality
    CXPB = 0.7
    MUTPB = 0.15
    NGEN = 80  # Fewer generations but higher quality

    # Run evolution with early stopping
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN,
                                          stats=stats, halloffame=hof, verbose=False)
    except Exception as e:
        # Fallback to basic solution if evolution fails
        print(f"Evolution failed with error: {e}")
        # Return simple configuration
        result = np.zeros((n_circles, 3))
        # Place circles along diagonal
        for i in range(n_circles):
            r = 0.02  # Small radius for initial placement
            x = 0.05 + i * 0.03
            y = 0.05 + i * 0.03
            result[i] = [x, y, r]
        return result

    # Extract best solution
    best_individual = hof[0]
    circles_array = np.array(best_individual).reshape(-1, 3)

    # Apply improved local optimization using hybrid approach
    def hybrid_local_optimize(circles):
        # Start with the current best solution
        current_circles = circles.copy()
        current_total_radius = np.sum(current_circles[:, 2])

        # First pass: Local optimization with better move generation
        # This is a more sophisticated approach using coordinate descent with adaptive steps
        for iteration in range(5000):
            # Pick a random circle to perturb
            idx = np.random.randint(len(current_circles))

            # Adaptive perturbation based on current state
            old_x, old_y, old_r = current_circles[idx]

            # Determine step sizes based on current radius and position
            step_size_pos = min(0.02, old_r * 0.5)  # Position step proportional to radius
            step_size_rad = min(0.01, old_r * 0.2)  # Radius step proportional to radius

            # Try various moves
            moves = []
            # Move position
            moves.append(('pos', old_x + random.uniform(-step_size_pos, step_size_pos),
                         old_y + random.uniform(-step_size_pos, step_size_pos), old_r))
            # Change radius
            moves.append(('rad', old_x, old_y, max(old_r + random.uniform(-step_size_rad, step_size_rad), 0.001)))
            # Move position and radius
            moves.append(('both', old_x + random.uniform(-step_size_pos, step_size_pos),
                          old_y + random.uniform(-step_size_pos, step_size_pos),
                          max(old_r + random.uniform(-step_size_rad, step_size_rad), 0.001)))

            # Evaluate all moves and pick the best valid one
            best_move = None
            best_score = -np.inf

            for move_type, new_x, new_y, new_r in moves:
                # Ensure bounds
                new_x = max(new_r, min(1 - new_r, new_x))
                new_y = max(new_r, min(1 - new_r, new_y))

                # Check if this is a valid move
                valid = True
                penalty = 0

                # Check containment
                if new_x < new_r or new_x > 1 - new_r or new_y < new_r or new_y > 1 - new_r:
                    valid = False
                    penalty += 1000000

                # Check overlaps with all other circles
                temp_circles = current_circles.copy()
                temp_circles[idx] = [new_x, new_y, new_r]

                # Check overlaps efficiently
                if len(temp_circles) > 1:
                    points = temp_circles[:, :2]
                    distances = cdist(points, points)
                    radii = temp_circles[:, 2]
                    radii_matrix = np.add.outer(radii, radii)
                    np.fill_diagonal(distances, np.inf)

                    overlaps = distances < radii_matrix
                    if np.any(overlaps):
                        valid = False
                        penalty += 1000000

                if valid:
                    # Compute new score
                    new_total_radius = np.sum(temp_circles[:, 2]) - penalty/1000000
                    if new_total_radius > best_score:
                        best_score = new_total_radius
                        best_move = (new_x, new_y, new_r)

            # If we found a better move, apply it
            if best_move:
                current_circles[idx] = best_move
                current_total_radius = best_score

        return current_circles

    # Perform final hybrid optimization
    optimized_circles = hybrid_local_optimize(circles_array)

    return optimized_circles

# EVOLVE-BLOCK-END