# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools
import time
from itertools import combinations
from collections import defaultdict

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    N_CIRCLES = 26

    def initialize_population(pop_size):
        """Initialize population using enhanced Voronoi-based approach with better spatial distribution"""
        population = []
        for _ in range(pop_size):
            # Generate initial points using a combination of hexagonal grid and random jittering
            points = []
            grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)

            # Create points in a hexagonal pattern with strategic jittering for better distribution
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < N_CIRCLES:
                        # Use hexagonal offset pattern for better distribution
                        offset = (j % 2) * spacing_x / 2
                        jitter_x = np.random.uniform(-spacing_x/10, spacing_x/10)
                        jitter_y = np.random.uniform(-spacing_y/10, spacing_y/10)
                        x = (i + 1) * spacing_x + offset + jitter_x
                        y = (j + 1) * spacing_y + jitter_y
                        points.append([x, y])

            # Add boundary points to encourage better coverage
            boundary_points = []
            for _ in range(15):  # Add more boundary points
                side = np.random.randint(0, 4)
                if side == 0:  # Top
                    boundary_points.append([np.random.rand(), 1.0])
                elif side == 1:  # Bottom
                    boundary_points.append([np.random.rand(), 0.0])
                elif side == 2:  # Left
                    boundary_points.append([0.0, np.random.rand()])
                else:  # Right
                    boundary_points.append([1.0, np.random.rand()])

            points.extend(boundary_points)

            # Fill remaining positions with strategic placement
            while len(points) < N_CIRCLES:
                max_attempts = 50
                placed = False
                for attempt in range(max_attempts):
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)

                    # Check minimum distance to existing points for better spread
                    min_dist = float('inf')
                    for existing_point in points:
                        dist = np.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                        min_dist = min(min_dist, dist)

                    # Place if sufficiently far from others
                    if min_dist > 0.06:  # Slightly more liberal distance threshold
                        points.append([x, y])
                        placed = True
                        break

                # If couldn't place far enough, just place randomly
                if not placed:
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    points.append([x, y])

            # Create Voronoi diagram for better initial placement
            try:
                points_array = np.array(points)
                vor = Voronoi(points_array)

                # Extract centroids of finite Voronoi cells as initial circle centers
                centroids = []
                for i, (x, y) in enumerate(vor.points):
                    if i < len(vor.point_region) and vor.point_region[i] >= 0:
                        region = vor.regions[vor.point_region[i]]
                        if len(region) > 0 and all(r >= 0 for r in region):
                            vertices = np.array([vor.vertices[r] for r in region])
                            if len(vertices) > 0:
                                centroid = np.mean(vertices, axis=0)
                                centroid[0] = np.clip(centroid[0], 0.01, 0.99)
                                centroid[1] = np.clip(centroid[1], 0.01, 0.99)
                                centroids.append(centroid)

                # If insufficient centroids, use original points
                if len(centroids) < N_CIRCLES:
                    centroids = points_array[:N_CIRCLES].tolist()
                else:
                    centroids = centroids[:N_CIRCLES]

                # Compute radii based on Voronoi cell properties with better calculation
                circles = np.zeros((N_CIRCLES, 3))
                for i, (cx, cy) in enumerate(centroids):
                    # Calculate minimum distance to other points
                    min_dist = float('inf')
                    for j, (px, py) in enumerate(centroids):
                        if i != j:
                            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                            min_dist = min(min_dist, dist)

                    # Set radius based on Voronoi cell size and proximity with more conservative approach
                    if min_dist > 0:
                        r = min(0.18, min_dist/2.0)  # More conservative radius calculation
                    else:
                        r = np.random.uniform(0.01, 0.05)

                    # Enforce hard bounds
                    r = max(0.005, min(0.25, r))
                    circles[i] = [cx, cy, r]

            except Exception:
                # Fall back to simple initialization if Voronoi fails
                circles = np.zeros((N_CIRCLES, 3))
                for i in range(N_CIRCLES):
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    r = np.random.uniform(0.01, 0.1)
                    circles[i] = [x, y, r]

            population.append(circles)
        return population

    def calculate_diversity(population):
        """Calculate population diversity based on Euclidean distances between individuals"""
        if len(population) < 2:
            return 0.0

        total_distances = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i+1, len(population)):
                # Calculate Euclidean distance between individuals (flattened)
                flat_i = population[i].flatten()
                flat_j = population[j].flatten()
                dist = np.linalg.norm(flat_i - flat_j)
                total_distances += dist
                count += 1

        return total_distances / count if count > 0 else 0.0

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

    def mut_adaptive_gaussian(individual, mu=0, sigma=0.01, indpb=0.2, generation=None, diversity=None):
        """Enhanced adaptive mutator with diversity-aware parameters"""
        # Adjust mutation strength based on generation and diversity
        if generation is not None and diversity is not None:
            # More aggressive mutation in early generations with high diversity
            adaptive_sigma = sigma * (1.0 + max(0, 0.5 * (diversity - 0.05)) * (1.0 - generation/250.0))
        else:
            adaptive_sigma = sigma

        for i in range(len(individual)):
            if random.random() < indpb:
                # Mutate position with adaptive sigma
                individual[i][0] += np.random.normal(mu, adaptive_sigma)
                individual[i][1] += np.random.normal(mu, adaptive_sigma)

                # Mutate radius with smaller changes
                individual[i][2] += np.random.normal(mu, adaptive_sigma/3)

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
                            # More aggressive separation with weighted motion
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
                            # Weighted separation based on radius ratio
                            weight = min(1.0, r/(r+r2) * 2.0) if r+r2 > 0 else 0.5
                            individual[i][0] += dx * separation * weight * 0.8
                            individual[i][1] += dy * separation * weight * 0.8

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

    def speciate_population(population, threshold=0.1):
        """Group similar individuals into species based on geometric similarity"""
        species = defaultdict(list)
        if not population:
            return species

        # Assign first individual to first species
        species[0].append(population[0])

        # Assign remaining individuals to species
        for ind in population[1:]:
            assigned = False
            # Find closest species center
            for sid, members in species.items():
                # Calculate average distance from this individual to all members in species
                avg_dist = 0
                for member in members:
                    avg_dist += np.linalg.norm(ind.flatten() - member.flatten())
                avg_dist /= len(members)

                if avg_dist < threshold:
                    species[sid].append(ind)
                    assigned = True
                    break

            if not assigned:
                # Create new species
                new_sid = max(species.keys()) + 1 if species else 0
                species[new_sid].append(ind)

        return species

    def select_from_species(species_dict, k):
        """Select individuals from each species proportional to species size"""
        selected = []
        for sid, members in species_dict.items():
            # Select proportionally to species size (with minimum of 1)
            num_to_select = max(1, len(members) * k // len(species_dict))
            # Randomly select from this species
            selected.extend(random.sample(members, min(num_to_select, len(members))))

        return selected[:k]

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
    toolbox.register("mutate", mut_adaptive_gaussian)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population with Voronoi-based starting points
    pop = initialize_population(150)  # Increased population size

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
    n_generations = 250  # Increased generations
    mutation_rate = 0.05  # Lower initial mutation rate for stability
    crossover_rate = 0.9   # Higher crossover rate for genetic diversity

    # Run evolution with adaptive parameters and diversity management
    for gen in range(n_generations):
        # Adaptive mutation rate with faster decay
        current_mutation_rate = max(0.005, mutation_rate * (1 - gen/n_generations)**1.5)

        # Calculate population diversity
        diversity = calculate_diversity(pop)

        # Select parents with diversity consideration
        offspring = toolbox.select(deap_pop, len(deap_pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation with adaptive rate and diversity consideration
        for i, mutant in enumerate(offspring):
            if random.random() < current_mutation_rate:
                # Pass generation info to mutation function
                toolbox.mutate(mutant, generation=gen, diversity=diversity)
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

    # Additional refinement using improved local search with constraint-aware optimization
    try:
        # Perform local optimizations on the best solution with enhanced approach
        refined_circles = best_solution.copy()

        # Iterative local improvement with more sophisticated binary search and multi-pass approach
        for iteration in range(100):  # Increased iterations for better convergence
            improved = False

            # Try to increase radii for each circle with better precision
            for i in range(N_CIRCLES):
                orig_x, orig_y, orig_r = refined_circles[i]

                # Find minimum distance to other circles
                min_dist_to_others = float('inf')
                for j in range(N_CIRCLES):
                    if i != j:
                        x2, y2, r2 = refined_circles[j]
                        dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                        min_dist_to_others = min(min_dist_to_others, dist)

                # Calculate maximum possible radius with binary search and improved bounds
                max_new_radius = min(min_dist_to_others - 0.001 if min_dist_to_others > 0.001 else orig_r, 0.5)

                if max_new_radius > orig_r:
                    # Binary search for optimal radius with higher precision
                    low, high = orig_r, max_new_radius
                    best_radius = orig_r

                    # Binary search loop - more precise than before
                    for _ in range(15):  # More iterations for accuracy
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

                # Try slight position adjustments with more diverse movements
                for _ in range(5):  # More attempts per circle
                    # Try multiple random movements
                    dx = np.random.uniform(-0.005, 0.005)
                    dy = np.random.uniform(-0.005, 0.005)
                    new_x = orig_x + dx
                    new_y = orig_y + dy

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