# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Problem parameters
    N_CIRCLES = 26

    def initialize_population(pop_size):
        """Initialize population with improved Voronoi-based starting points"""
        population = []
        for _ in range(pop_size):
            circles = np.zeros((N_CIRCLES, 3))

            # Generate initial points using a more sophisticated approach
            # Start with a grid for good coverage
            grid_size = int(np.ceil(np.sqrt(N_CIRCLES * 1.2)))  # Extra space to ensure coverage
            points = []

            # Create grid points
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < N_CIRCLES:
                        x = (j + 0.5) / grid_size
                        y = (i + 0.5) / grid_size
                        points.append([x, y])

            # Add boundary points to improve edge coverage
            boundary_points = []
            for _ in range(20):  # Add more boundary points
                side = np.random.randint(0, 4)
                if side == 0:  # Top
                    boundary_points.append([np.random.rand(), 1.0])
                elif side == 1:  # Bottom
                    boundary_points.append([np.random.rand(), 0.0])
                elif side == 2:  # Left
                    boundary_points.append([0.0, np.random.rand()])
                else:  # Right
                    boundary_points.append([1.0, np.random.rand()])

            # Combine and add some jitter to avoid regular patterns
            points = np.array(points)
            points += np.random.uniform(-0.02, 0.02, points.shape)

            # Add boundary points
            boundary_points = np.array(boundary_points)
            points = np.vstack([points, boundary_points])

            # Trim to exact number needed
            if len(points) > N_CIRCLES:
                points = points[:N_CIRCLES]
            elif len(points) < N_CIRCLES:
                # Fill remaining with random points
                for _ in range(N_CIRCLES - len(points)):
                    points = np.vstack([points, [np.random.rand(), np.random.rand()]])

            # Ensure all points are within bounds
            points = np.clip(points, 0.01, 0.99)

            # Calculate Voronoi cell areas for better radius estimation
            try:
                from scipy.spatial import Voronoi
                vor = Voronoi(points)
                # Use the Voronoi cell centers as circle positions
                centroids = vor.points[vor.point_region[:-1]]

                # Limit to needed number of circles
                if len(centroids) > N_CIRCLES:
                    centroids = centroids[:N_CIRCLES]
                elif len(centroids) < N_CIRCLES:
                    # Extend with additional points if needed
                    remaining = N_CIRCLES - len(centroids)
                    for _ in range(remaining):
                        # Add points near existing ones
                        rand_idx = np.random.randint(0, len(centroids))
                        new_point = centroids[rand_idx] + np.random.normal(0, 0.05, 2)
                        new_point = np.clip(new_point, 0.01, 0.99)
                        centroids = np.vstack([centroids, new_point])
                    centroids = centroids[:N_CIRCLES]

                # Create circles with radius based on Voronoi cell properties and space density
                for i in range(N_CIRCLES):
                    x, y = centroids[i]

                    # Calculate minimum distance to neighbors to determine appropriate radius
                    distances = []
                    for j in range(N_CIRCLES):
                        if i != j:
                            dist = np.sqrt((x - centroids[j][0])**2 + (y - centroids[j][1])**2)
                            distances.append(dist)

                    if distances:
                        min_dist = min(distances)
                        # Use a more sophisticated formula for radius based on density
                        # Area per circle in a hexagonal packing is roughly pi * r^2 = A/N
                        # But we want to be conservative about overlap
                        r = min(0.15, min_dist * 0.3)
                    else:
                        # Fallback if no neighbors found
                        r = np.random.uniform(0.02, 0.08)

                    # Ensure radius respects boundaries
                    boundary_safe = min(x, 1-x, y, 1-y)
                    r = min(r, boundary_safe * 0.9)

                    # Ensure minimum reasonable radius
                    r = max(0.005, min(0.2, r))

                    circles[i] = [x, y, r]

            except Exception:
                # Fallback to simpler approach if Voronoi fails
                # Generate points in hexagonal pattern with better spacing
                grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
                spacing_x = 0.9 / (grid_size + 1)
                spacing_y = 0.9 / (grid_size + 1)

                points = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(points) < N_CIRCLES:
                            offset = (j % 2) * spacing_x / 2
                            x = (j + 1) * spacing_x + offset
                            y = (i + 1) * spacing_y
                            points.append([x, y])

                # Fill remaining positions
                while len(points) < N_CIRCLES:
                    points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

                points = points[:N_CIRCLES]

                # Create circles with radius based on neighbor distances
                for i in range(N_CIRCLES):
                    x, y = points[i]

                    # Calculate minimum distance to neighbors
                    min_dist = float('inf')
                    for j in range(N_CIRCLES):
                        if i != j:
                            dist = np.sqrt((x - points[j][0])**2 + (y - points[j][1])**2)
                            min_dist = min(min_dist, dist)

                    # Determine radius with more sophisticated approach
                    if min_dist > 0:
                        # Conservative estimate: 1/4 of minimum neighbor distance
                        r = min(0.12, min_dist * 0.25)
                    else:
                        r = np.random.uniform(0.01, 0.08)

                    # Ensure boundary constraints
                    r = min(r, x, 1-x, y, 1-y)
                    r = max(0.005, min(0.15, r))

                    circles[i] = [x, y, r]

            population.append(circles)
        return population

    def is_valid_solution(circles):
        """Check if solution satisfies all constraints efficiently"""
        # Check containment - early exit if violated
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

        # Check non-overlap using KDTree for efficiency with early termination
        try:
            points = circles[:, :2]
            tree = KDTree(points)
            pairs = tree.query_pairs(0, return_distance=False)

            # Check pairs with early termination
            for i, j in pairs:
                if i < j:  # Avoid checking same pair twice
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        except Exception:
            # Fallback to brute force if KDTree fails
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False

        return True

    def evaluate_fitness(individual):
        """Evaluate fitness as negative sum of radii (since we want to maximize)"""
        # Only consider valid solutions
        if not is_valid_solution(individual):
            # Apply large penalty for invalid solutions
            return (-np.sum(individual[:, 2]) - 1000,)

        # Return negative sum of radii for maximization problem
        return (-np.sum(individual[:, 2]),)

    def mut_gaussian(individual, mu=0, sigma=0.01, indpb=0.2):
        """Custom mutator that keeps circles within bounds and maintains validity"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # Mutate position
                individual[i][0] += np.random.normal(mu, sigma)
                individual[i][1] += np.random.normal(mu, sigma)
                # Mutate radius
                individual[i][2] += np.random.normal(mu, sigma/3)

                # Ensure boundaries
                individual[i][0] = np.clip(individual[i][0], individual[i][2], 1 - individual[i][2])
                individual[i][1] = np.clip(individual[i][1], individual[i][2], 1 - individual[i][2])
                individual[i][2] = np.clip(individual[i][2], 0.001, 0.5)

        # Attempt to repair collisions if they occur
        repair_individual(individual)
        return individual,

    def repair_individual(individual):
        """Repair an individual to ensure no overlaps and containment"""
        # First fix containment issues
        for i in range(len(individual)):
            x, y, r = individual[i]

            # Ensure containment
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            individual[i][0] = x
            individual[i][1] = y

        # Then resolve overlaps with more aggressive approach
        max_iterations = 5
        for iteration in range(max_iterations):
            overlapped = False
            for i in range(len(individual)):
                x, y, r = individual[i]
                for j in range(len(individual)):
                    if i != j:
                        x2, y2, r2 = individual[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)

                        if dist < r + r2:
                            overlapped = True
                            # Move circles apart with more aggressive separation
                            dx = x2 - x
                            dy = y2 - y
                            if dx == 0 and dy == 0:
                                # If same position, move randomly
                                angle = np.random.uniform(0, 2*np.pi)
                                dx = np.cos(angle)
                                dy = np.sin(angle)

                            # Normalize direction vector
                            length = np.sqrt(dx*dx + dy*dy)
                            if length > 0:
                                dx /= length
                                dy /= length

                            # Adjust positions to separate them with larger separation factor
                            separation = (r + r2) - dist
                            individual[i][0] += dx * separation * 0.7
                            individual[i][1] += dy * separation * 0.7

                            # Clip to bounds
                            individual[i][0] = np.clip(individual[i][0], r, 1 - r)
                            individual[i][1] = np.clip(individual[i][1], r, 1 - r)

            if not overlapped:
                break

    def cx_uniform(individual1, individual2):
        """Uniform crossover for circle positions and radii"""
        size = len(individual1)
        for i in range(size):
            if random.random() < 0.5:
                individual1[i], individual2[i] = individual2[i], individual1[i]

        # Repair offspring
        repair_individual(individual1)
        repair_individual(individual2)
        return individual1, individual2

    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: [np.random.uniform(0.01, 0.99),
                              np.random.uniform(0.01, 0.99),
                              np.random.uniform(0.01, 0.1)],
                     n=N_CIRCLES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", cx_uniform)
    toolbox.register("mutate", mut_gaussian)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population with better starting points (larger population)
    pop = initialize_population(100)  # Increased population size

    # Convert to DEAP individuals
    deap_pop = []
    for p in pop:
        ind = creator.Individual(p.tolist())
        ind.fitness.values = evaluate_fitness(p)
        deap_pop.append(ind)

    # Run evolution with improved parameters
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evolution parameters - optimized for better convergence and efficiency
    n_generations = 150  # Balanced number of generations
    mutation_rate = 0.05  # Lower initial mutation rate for stability
    crossover_rate = 0.9   # Higher crossover rate for genetic diversity

    # Run evolution with adaptive parameters
    for gen in range(n_generations):
        # Adaptive mutation rate with faster decay
        current_mutation_rate = max(0.005, mutation_rate * (1 - gen/n_generations)**1.5)

        # Select parents
        offspring = toolbox.select(deap_pop, len(deap_pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation with adaptive rate
        for mutant in offspring:
            if random.random() < current_mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not hasattr(ind.fitness, 'values') or len(ind.fitness.values) == 0]
        for ind in invalid_ind:
            ind.fitness.values = evaluate_fitness(np.array(ind))

        # Update population
        deap_pop[:] = offspring

        # Update hall of fame
        hof.update(deap_pop)

        # Print progress
        if gen % 20 == 0:
            try:
                best_fit = max([ind.fitness.values[0] for ind in deap_pop])
                print(f"Generation {gen}: Best fitness = {-best_fit}")
            except:
                pass

    # Get best solution
    best_individual = hof[0] if hof else deap_pop[0]
    best_solution = np.array(best_individual)

    # Final validation and repair
    if not is_valid_solution(best_solution):
        repair_individual(best_solution)

    # Additional refinement using local search with binary optimization
    try:
        # Perform local optimizations on the best solution
        refined_circles = best_solution.copy()

        # Iterative local improvement with binary search for optimal radii
        for iteration in range(30):
            improved = False

            # Try to increase radii for each circle
            for i in range(N_CIRCLES):
                orig_x, orig_y, orig_r = refined_circles[i]

                # Find minimum distance to other circles
                min_dist_to_others = float('inf')
                for j in range(N_CIRCLES):
                    if i != j:
                        x2, y2, r2 = refined_circles[j]
                        dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                        min_dist_to_others = min(min_dist_to_others, dist)

                # Calculate maximum possible radius with binary search
                max_new_radius = min_dist_to_others - 0.001 if min_dist_to_others > 0.001 else orig_r

                if max_new_radius > orig_r:
                    # Binary search for optimal radius
                    low, high = orig_r, max_new_radius
                    best_radius = orig_r

                    # Binary search loop
                    for _ in range(10):
                        test_r = (low + high) / 2
                        # Check if this radius is valid
                        temp_circles = refined_circles.copy()
                        temp_circles[i][2] = test_r

                        if is_valid_solution(temp_circles):
                            best_radius = test_r
                            low = test_r
                        else:
                            high = test_r

                    if best_radius > orig_r:
                        refined_circles[i][2] = best_radius
                        improved = True

                # Try slight position adjustments
                for _ in range(3):
                    new_x = orig_x + np.random.uniform(-0.002, 0.002)
                    new_y = orig_y + np.random.uniform(-0.002, 0.002)

                    # Clip to bounds using potential new radius
                    target_r = refined_circles[i][2]
                    new_x = np.clip(new_x, target_r, 1 - target_r)
                    new_y = np.clip(new_y, target_r, 1 - target_r)

                    # Check validity
                    temp_circles = refined_circles.copy()
                    temp_circles[i][0] = new_x
                    temp_circles[i][1] = new_y

                    if is_valid_solution(temp_circles):
                        refined_circles[i][0] = new_x
                        refined_circles[i][1] = new_y
                        improved = True
                        break

            if not improved:
                break

        # Final repair if needed
        if not is_valid_solution(refined_circles):
            repair_individual(refined_circles)

        best_solution = refined_circles

    except Exception as e:
        # If refinement fails, return best solution so far
        pass

    return best_solution

# EVOLVE-BLOCK-END