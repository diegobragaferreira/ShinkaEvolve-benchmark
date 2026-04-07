# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms
import random
import time

# Fixed seed for reproducibility
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
    min_radius = 0.001

    # Helper function to check if a circle is valid
    def is_valid_circle(x, y, r):
        return (r >= min_radius and
                r <= x <= 1 - r and
                r <= y <= 1 - r)

    # Helper function to compute distance between two circles
    def circle_distance(c1, c2):
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Constraint checking function
    def is_valid_configuration(circles_array):
        # Check containment constraints
        for x, y, r in circles_array:
            if not is_valid_circle(x, y, r):
                return False

        # Check overlap constraints using KDTree for efficiency
        tree = cKDTree(circles_array[:, :2])
        pairs = tree.query_pairs(2 * max_radius)  # Only check nearby pairs

        for i, j in pairs:
            c1 = circles_array[i]
            c2 = circles_array[j]
            dist = circle_distance(c1, c2)
            if dist < c1[2] + c2[2]:  # Overlapping
                return False

        return True

    # Voronoi-based initialization for better starting configuration
    def initialize_voronoi_population(pop_size):
        population = []
        for _ in range(pop_size):
            # Generate seed points for Voronoi diagram
            np.random.seed(int(time.time() * 1000) % 2**32)  # Different seed each time

            # Create grid of seed points with jitter
            grid_size = 6  # 6x6 grid = 36 points
            spacing = 1.0 / (grid_size + 1)
            points = []

            for i in range(grid_size):
                for j in range(grid_size):
                    x = (i + 1) * spacing + np.random.uniform(-spacing/6, spacing/6)
                    y = (j + 1) * spacing + np.random.uniform(-spacing/6, spacing/6)
                    points.append([x, y])

            points = np.array(points[:n_circles])

            # Generate Voronoi diagram
            vor = Voronoi(points)

            # Create circles based on Voronoi cells
            circles = []
            for i in range(n_circles):
                if i < len(vor.points):
                    x, y = vor.points[i]

                    # Estimate safe radius based on Voronoi cell geometry
                    # Find distances to neighbors
                    min_dist = float('inf')
                    for j in range(n_circles):
                        if i != j:
                            dist = np.sqrt((vor.points[i, 0] - vor.points[j, 0])**2 +
                                         (vor.points[i, 1] - vor.points[j, 1])**2)
                            min_dist = min(min_dist, dist)

                    # Safe radius is half the minimum neighbor distance or 0.2, whichever is smaller
                    estimated_radius = min(0.2, min_dist / 4.0) if min_dist < float('inf') else 0.1

                    # Ensure reasonable bounds
                    radius = max(min_radius, min(max_radius, estimated_radius))

                    # Clamp to unit square
                    x = max(radius, min(1-radius, x))
                    y = max(radius, min(1-radius, y))

                    circles.append([x, y, radius])
                else:
                    # Fall back to simple random if Voronoi fails
                    x = np.random.uniform(min_radius, 1 - min_radius)
                    y = np.random.uniform(min_radius, 1 - min_radius)
                    r = np.random.uniform(min_radius, max_radius)
                    circles.append([x, y, r])

            population.append(np.array(circles))
        return population

    # Constraint-aware mutation operator
    def mut_constraint_aware(individual, mu=0, sigma=0.01, indpb=0.3):
        """Mutation that preserves constraints and repairs violations"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # Mutate position (x, y) and radius (r)
                individual[i][0] += np.random.normal(mu, sigma)
                individual[i][1] += np.random.normal(mu, sigma)
                individual[i][2] += np.random.normal(mu, sigma/2)

                # Ensure position stays within unit square with radius
                individual[i][0] = np.clip(individual[i][0], individual[i][2], 1 - individual[i][2])
                individual[i][1] = np.clip(individual[i][1], individual[i][2], 1 - individual[i][2])
                individual[i][2] = np.clip(individual[i][2], min_radius, max_radius)

        # Repair any constraint violations after mutation
        repair_individual(individual)
        return individual,

    # Individual repair function to fix constraint violations
    def repair_individual(individual):
        """Repair an individual to ensure no overlaps or boundary violations"""
        # First pass: enforce boundary constraints
        for i in range(len(individual)):
            x, y, r = individual[i]
            individual[i][0] = np.clip(x, r, 1 - r)
            individual[i][1] = np.clip(y, r, 1 - r)
            individual[i][2] = np.clip(r, min_radius, max_radius)

        # Second pass: resolve overlaps
        for i in range(len(individual)):
            x1, y1, r1 = individual[i]

            # Check for overlaps with all other circles
            for j in range(len(individual)):
                if i != j:
                    x2, y2, r2 = individual[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    if dist < r1 + r2:
                        # Push circles apart
                        dx = x2 - x1
                        dy = y2 - y1
                        length = np.sqrt(dx*dx + dy*dy)

                        if length > 0:
                            # Normalize and move in opposite directions
                            dx /= length
                            dy /= length

                            # Separate by a small amount
                            separation = (r1 + r2) - dist
                            individual[i][0] -= dx * separation * 0.5
                            individual[i][1] -= dy * separation * 0.5

                            # Ensure constraints are met after movement
                            individual[i][0] = np.clip(individual[i][0], individual[i][2], 1 - individual[i][2])
                            individual[i][1] = np.clip(individual[i][1], individual[i][2], 1 - individual[i][2])

    # Fitness function
    def evaluate(individual):
        circles_array = np.array(individual).reshape(-1, 3)
        if not is_valid_configuration(circles_array):
            return (0,)  # Invalid configuration gets zero fitness
        total_radius = np.sum(circles_array[:, 2])
        return (total_radius,)

    # Create types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator with Voronoi initialization
    def create_individual():
        # Create a Voronoi-based initial configuration
        # First generate a few candidates and pick the best one
        candidates = initialize_voronoi_population(5)
        best_candidate = max(candidates, key=lambda c: np.sum(c[:, 2]))

        # Convert to individual representation
        individual = []
        for x, y, r in best_candidate:
            individual.extend([x, y, r])
        return creator.Individual(individual)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mut_constraint_aware)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initial population generation with better initialization
    population = toolbox.population(n=30)  # Reduced population size for faster execution

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run evolution
    try:
        hof = tools.ParetoFront()
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.7, mutpb=0.3, ngen=70,  # Slightly more generations
            stats=stats, halloffame=hof, verbose=False
        )

        # Get best individual
        best_individual = hof[0] if hof else population[0]
        final_circles = np.array(best_individual).reshape(-1, 3)

        # Final validation
        if is_valid_configuration(final_circles):
            # Refine if needed
            refined_circles = refine_solution(final_circles.copy())
            return refined_circles
        else:
            # Fall back to a greedy solution if evolution failed
            return generate_greedy_solution()

    except Exception as e:
        # Fallback to greedy approach if anything goes wrong
        return generate_greedy_solution()

def generate_greedy_solution():
    """Generate a simple greedy solution as fallback"""
    n_circles = 26
    circles = np.zeros((n_circles, 3))

    # Simple grid-based approach first
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)

    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n_circles:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Set radius to be proportional to available space
            r = min(spacing_x, spacing_y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1

    # If we don't have enough circles, fill with random ones
    for i in range(idx, n_circles):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, r]

    return circles

def refine_solution(circles):
    """Apply local refinement to improve solution quality"""
    # Simple refinement: try to increase radii slightly if possible
    for _ in range(30):  # More iterations for better refinement
        improved = False
        for i in range(len(circles)):
            # Try to increase radius slightly
            old_r = circles[i][2]
            new_r = min(old_r * 1.03, 0.5)  # Smaller increment for stability

            # Check if we can increase radius without violating constraints
            temp_circles = circles.copy()
            temp_circles[i][2] = new_r

            # Basic containment check
            x, y, r = temp_circles[i]
            if r <= x <= 1 - r and r <= y <= 1 - r:
                # Check overlap with others more efficiently
                valid = True
                # Only check nearby circles for efficiency
                tree = cKDTree(temp_circles[:, :2])
                neighbors = tree.query_ball_point([x, y], 2 * max_radius)

                for j in neighbors:
                    if i != j:
                        x1, y1, r1 = temp_circles[i]
                        x2, y2, r2 = temp_circles[j]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < r1 + r2:
                            valid = False
                            break

                if valid:
                    circles[i][2] = new_r
                    improved = True

        if not improved:
            break

    return circles

# EVOLVE-BLOCK-END