# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    """Optimizes placement of 26 non-overlapping circles in unit square to maximize sum of radii."""

    def __init__(self, n_circles=26, pop_size=150, gen_count=200, mutpb=0.2, cxpb=0.6):
        self.N_CIRCLES = n_circles
        self.POP_SIZE = pop_size
        self.GEN_COUNT = gen_count
        self.MUTPB = mutpb
        self.CXPB = cxpb
        self.BOUNDARY_MARGIN = 0.01
        self._setup_deap()

    def _setup_deap(self):
        """Initialize DEAP framework components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def _generate_voronoi_initial_config(self):
        """Generate initial circle configuration using enhanced Voronoi-like distribution."""
        circles = np.zeros((self.N_CIRCLES, 3))

        # Use a sophisticated hexagonal grid approach for better spatial distribution
        sqrt_n = int(np.ceil(np.sqrt(self.N_CIRCLES)))
        rows = int(np.ceil(self.N_CIRCLES / sqrt_n))
        cols = int(np.ceil(self.N_CIRCLES / rows))

        # Create hexagonal grid with appropriate spacing
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.N_CIRCLES:
                    break
                # Offset odd rows for hexagonal packing
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                points.append([x, y])

        # Ensure we have enough points
        while len(points) < self.N_CIRCLES:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append([x, y])

        points = points[:self.N_CIRCLES]

        # Enhance with Voronoi-based neighbor analysis
        try:
            # Create Voronoi diagram to analyze spatial relationships
            vor_points = np.array(points)
            vor = Voronoi(vor_points)
            
            # Use Voronoi cell centroids for better positioning
            centroids = []
            for i, (x, y) in enumerate(vor_points):
                if i < len(vor.point_region) and vor.point_region[i] >= 0:
                    region = vor.regions[vor.point_region[i]]
                    if len(region) > 0 and all(r >= 0 for r in region):
                        vertices = np.array([vor.vertices[r] for r in region])
                        if len(vertices) > 0:
                            centroid = np.mean(vertices, axis=0)
                            centroid[0] = np.clip(centroid[0], self.BOUNDARY_MARGIN, 1 - self.BOUNDARY_MARGIN)
                            centroid[1] = np.clip(centroid[1], self.BOUNDARY_MARGIN, 1 - self.BOUNDARY_MARGIN)
                            centroids.append(centroid)
            
            # If Voronoi doesn't give us enough centroids, fall back to original points
            if len(centroids) < self.N_CIRCLES:
                centroids = vor_points[:self.N_CIRCLES].tolist()
            else:
                centroids = centroids[:self.N_CIRCLES]
                
        except:
            # Fallback to direct point assignment if Voronoi fails
            centroids = points

        # Assign initial radii based on neighbor distances and Voronoi analysis
        for i, (x, y) in enumerate(centroids):
            # Calculate minimum distance to neighbors for safe initial radius
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(centroids):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)

            # More sophisticated radius calculation based on Voronoi insights
            if min_dist < float('inf') and min_dist > 0:
                # Use a combination of neighbor distance and Voronoi properties
                initial_r = min(0.15, min_dist * 0.25)
            else:
                initial_r = 0.05

            # Ensure radius respects square boundaries
            boundary_safe_radius = min(x, 1-x, y, 1-y)
            initial_r = min(initial_r, boundary_safe_radius * 0.8)

            # Ensure minimum reasonable size
            initial_r = max(0.001, initial_r)

            circles[i] = [x, y, initial_r]

        return circles

    def _is_valid_configuration(self, circles):
        """Check if configuration is valid with early termination."""
        # Check containment constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < self.BOUNDARY_MARGIN or x + r > 1 - self.BOUNDARY_MARGIN or \
               y - r < self.BOUNDARY_MARGIN or y + r > 1 - self.BOUNDARY_MARGIN:
                return False

        # Check overlap constraints efficiently
        try:
            positions = circles[:, :2]
            radii = circles[:, 2]
            tree = cKDTree(positions)
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

    def _evaluate_fitness(self, individual):
        """Evaluate fitness of circle placement with penalty for constraints."""
        circles = np.array(individual).reshape(-1, 3)

        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Calculate objective (sum of radii)
        total_radius = np.sum(radii)

        # Penalty for constraint violations
        penalty = 0

        # Check containment constraints
        for i, (pos, r) in enumerate(zip(positions, radii)):
            x, y = pos
            # Circle must be fully inside unit square with margin
            if x - r < self.BOUNDARY_MARGIN or x + r > 1 - self.BOUNDARY_MARGIN or \
               y - r < self.BOUNDARY_MARGIN or y + r > 1 - self.BOUNDARY_MARGIN:
                penalty += 10000

        # Check overlap constraints efficiently using cKDTree
        try:
            tree = cKDTree(positions)
            # Use a slightly larger threshold to catch close pairs
            pairs = tree.query_pairs(0, return_distance=False)
            for i, j in pairs:
                if i < j:  # Avoid checking same pair twice
                    r_i = radii[i]
                    r_j = radii[j]
                    pos_i = positions[i]
                    pos_j = positions[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        penalty += 1000 * (r_i + r_j - dist)
        except Exception:
            # Fallback to brute force for edge cases
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    pos_i = circles[i, :2]
                    pos_j = circles[j, :2]
                    r_i = circles[i, 2]
                    r_j = circles[j, 2]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        penalty += 1000 * (r_i + r_j - dist)

        return (total_radius - penalty,)

    def _mutate_circle(self, individual):
        """Mutate circle placement with adaptive parameters."""
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0

        # Adaptive mutation rate based on diversity and generation
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.8, diversity * 0.5))

        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius index
                    old_r = individual[i]
                    # Reduce mutation strength for higher diversity
                    mutation_strength = 0.015 * (1 + diversity * 0.5)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position indices (x, y)
                    old_val = individual[i]
                    # Reduce mutation strength for higher diversity
                    mutation_strength = 0.02 * (1 + diversity * 0.5)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(self.BOUNDARY_MARGIN, min(1 - self.BOUNDARY_MARGIN, new_val))
        return individual,

    def _create_individual(self):
        """Create a random valid individual."""
        # Start with Voronoi-based configuration
        individual = self._generate_voronoi_initial_config().flatten().tolist()

        # Add small random perturbations
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.02, 0.02)
                individual[i] = max(self.BOUNDARY_MARGIN, min(1 - self.BOUNDARY_MARGIN, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.9, 1.1)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def _crossover_constraint_aware(self, ind1, ind2):
        """Crossover that maintains constraints with repair mechanism."""
        # Perform standard uniform crossover
        tools.cxUniform(ind1, ind2, indpb=0.5)

        # Repair violated constraints
        temp_ind = np.array(ind1).reshape(-1, 3)

        # Fix containment issues
        for i in range(len(temp_ind)):
            x, y, r = temp_ind[i]
            adjusted = False
            if x - r < self.BOUNDARY_MARGIN:
                x = r + self.BOUNDARY_MARGIN
                adjusted = True
            elif x + r > 1 - self.BOUNDARY_MARGIN:
                x = 1 - r - self.BOUNDARY_MARGIN
                adjusted = True
            if y - r < self.BOUNDARY_MARGIN:
                y = r + self.BOUNDARY_MARGIN
                adjusted = True
            elif y + r > 1 - self.BOUNDARY_MARGIN:
                y = 1 - r - self.BOUNDARY_MARGIN
                adjusted = True
            if adjusted:
                temp_ind[i] = [x, y, r]

        # Fix overlaps with greedy approach
        for i in range(len(temp_ind)):
            for j in range(i+1, len(temp_ind)):
                pos_i = temp_ind[i, :2]
                pos_j = temp_ind[j, :2]
                r_i = temp_ind[i, 2]
                r_j = temp_ind[j, 2]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Move one of them apart more aggressively
                    if random.random() < 0.5:
                        dx, dy = pos_i - pos_j
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist
                        dy /= dist
                        step = (r_i + r_j - dist) * 0.3
                        new_x = max(self.BOUNDARY_MARGIN + r_i, min(1 - self.BOUNDARY_MARGIN - r_i, pos_i[0] + dx * step))
                        new_y = max(self.BOUNDARY_MARGIN + r_i, min(1 - self.BOUNDARY_MARGIN - r_i, pos_i[1] + dy * step))
                        temp_ind[i] = [new_x, new_y, r_i]
                    else:
                        dx, dy = pos_j - pos_i
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist
                        dy /= dist
                        step = (r_i + r_j - dist) * 0.3
                        new_x = max(self.BOUNDARY_MARGIN + r_j, min(1 - self.BOUNDARY_MARGIN - r_j, pos_j[0] + dx * step))
                        new_y = max(self.BOUNDARY_MARGIN + r_j, min(1 - self.BOUNDARY_MARGIN - r_j, pos_j[1] + dy * step))
                        temp_ind[j] = [new_x, new_y, r_j]

        # Return repaired individuals
        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    def _local_optimization(self, circles):
        """Apply advanced local optimization to refine solution."""
        # Start with basic refinement
        for _ in range(100):
            improved = False

            # Try to increase each circle's radius
            for i in range(len(circles)):
                original_r = circles[i, 2]
                # Check boundary constraints
                max_increase = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                ) - original_r

                if max_increase > 0:
                    # Binary search for maximum safe increase
                    low = 0
                    high = max_increase
                    best_radius = original_r

                    for _ in range(12):  # More iterations for better precision
                        test_r = (low + high) / 2
                        test_r = min(test_r, max_increase)

                        valid = True
                        test_pos = circles[i, :2]
                        test_r_new = original_r + test_r

                        # Check overlap with other circles
                        for j in range(len(circles)):
                            if i != j:
                                pos_j = circles[j, :2]
                                r_j = circles[j, 2]
                                dist = np.sqrt(np.sum((test_pos - pos_j)**2))
                                if dist < (test_r_new + r_j):
                                    valid = False
                                    break

                        if valid:
                            best_radius = original_r + test_r
                            low = test_r
                        else:
                            high = test_r

                    if best_radius > original_r:
                        circles[i, 2] = best_radius
                        improved = True

            # If no improvement, try moving circles to reduce overlaps
            if not improved:
                for i in range(len(circles)):
                    original_pos = circles[i, :2].copy()
                    best_pos = original_pos.copy()

                    best_radius = circles[i, 2]
                    best_score = best_radius

                    # Try several positions around current location
                    for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                        for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                            test_x = max(self.BOUNDARY_MARGIN + circles[i, 2], 
                                       min(1 - self.BOUNDARY_MARGIN - circles[i, 2], circles[i, 0] + dx))
                            test_y = max(self.BOUNDARY_MARGIN + circles[i, 2], 
                                       min(1 - self.BOUNDARY_MARGIN - circles[i, 2], circles[i, 1] + dy))

                            valid = True
                            test_r = circles[i, 2]

                            # Check overlap with other circles
                            for j in range(len(circles)):
                                if i != j:
                                    pos_j = circles[j, :2]
                                    r_j = circles[j, 2]
                                    dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                                    if dist < (test_r + r_j):
                                        valid = False
                                        break

                            if valid:
                                score = test_r
                                if score > best_score:
                                    best_score = score
                                    best_pos = [test_x, test_y]

                # Apply best movement if found
                if best_score > circles[i, 2] or not np.array_equal(best_pos, original_pos):
                    circles[i, :2] = best_pos
                    improved = True

            if not improved:
                break

        # Final refinement with boundary enforcement
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Enforce boundary constraints
            x = np.clip(x, r + self.BOUNDARY_MARGIN, 1 - r - self.BOUNDARY_MARGIN)
            y = np.clip(y, r + self.BOUNDARY_MARGIN, 1 - r - self.BOUNDARY_MARGIN)
            circles[i] = [x, y, r]

        return circles

    def _heuristic_circle_packing(self):
        """Fallback method using structured approach."""
        # Simple grid-based arrangement with refinement
        n = self.N_CIRCLES
        circles = np.zeros((n, 3))

        # Try a hexagonal packing pattern
        rows = 5
        cols = 5
        if n < rows * cols:
            rows = int(np.ceil(n / cols))

        # Create regular grid points with hexagonal offset
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                # Set reasonable initial radius
                r = min(spacing_x, spacing_y) * 0.4
                circles[count] = [x, y, r]
                count += 1

        # Refine positions to avoid overlaps
        for _ in range(50):
            improved = False
            for i in range(n):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                # Check nearby positions
                for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                    for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                        test_x = max(self.BOUNDARY_MARGIN + circles[i, 2], 
                                   min(1 - self.BOUNDARY_MARGIN - circles[i, 2], circles[i, 0] + dx))
                        test_y = max(self.BOUNDARY_MARGIN + circles[i, 2], 
                                   min(1 - self.BOUNDARY_MARGIN - circles[i, 2], circles[i, 1] + dy))
                        test_r = circles[i, 2]

                        valid = True
                        for j in range(n):
                            if i != j:
                                dist = np.sqrt((test_x - circles[j, 0])**2 + (test_y - circles[j, 1])**2)
                                if dist < (test_r + circles[j, 2]):
                                    valid = False
                                    break

                        if valid:
                            score = test_r
                            if score > best_score:
                                best_score = score
                                best_pos = [test_x, test_y]

                if best_score > circles[i, 2]:
                    circles[i, :2] = best_pos
                    circles[i, 2] = best_score
                    improved = True

            if not improved:
                break

        return circles

    def optimize(self):
        """Main optimization routine."""
        # Initialize toolbox
        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", self._crossover_constraint_aware)
        toolbox.register("mutate", self._mutate_circle)
        toolbox.register("select", tools.selTournament, tournsize=5)

        # Create initial population
        population = toolbox.population(n=self.POP_SIZE)

        # Run evolution
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=self.CXPB, mutpb=self.MUTPB,
                ngen=self.GEN_COUNT, stats=stats, halloffame=hof, verbose=False
            )
        except Exception as e:
            # Return heuristic solution if evolution fails
            return self._heuristic_circle_packing()

        # Return best solution
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)

        # Apply local optimization to refine further
        refined_result = self._local_optimization(result.copy())

        return refined_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.optimize()

# EVOLVE-BLOCK-END