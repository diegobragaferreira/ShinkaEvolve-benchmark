# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingProblem:
    """Encapsulates the circle packing optimization problem with unified interface."""

    def __init__(self, n_circles=26, pop_size=80, gen_count=60, mutpb=0.25, cxpb=0.4):
        self.N_CIRCLES = n_circles
        self.POP_SIZE = pop_size
        self.GEN_COUNT = gen_count
        self.MUTPB = mutpb
        self.CXPB = cxpb
        self._setup_deap()

    def _setup_deap(self):
        """Initialize DEAP framework components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def create_initial_population(self):
        """Generate initial population using Voronoi-inspired initialization."""
        population = []
        for _ in range(self.POP_SIZE):
            individual = self._create_individual()
            population.append(individual)
        return population

    def _create_individual(self):
        """Create a single individual with Voronoi initialization and perturbation."""
        circles = self._voronoi_like_initialization(self.N_CIRCLES)

        # Add small random perturbations
        individual = circles.flatten().tolist()
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.02, 0.02)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.9, 1.1)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def _voronoi_like_initialization(self, n_circles):
        """Generate initial configuration using a more effective grid-based approach."""
        # Use a simple but effective grid-based approach with randomized perturbations
        circles = np.zeros((n_circles, 3))

        # Create a more even spacing arrangement
        sqrt_n = int(np.ceil(np.sqrt(n_circles)))
        rows = sqrt_n
        cols = sqrt_n

        # Ensure we have enough cells for all circles
        if rows * cols < n_circles:
            cols += 1
            if rows * cols < n_circles:
                rows += 1

        # Calculate spacing
        spacing_x = 0.95 / (cols + 1)
        spacing_y = 0.95 / (rows + 1)

        # Create grid points
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n_circles:
                    break
                # Add slight jitter to avoid regular patterns
                x = (j + 1) * spacing_x + random.uniform(-spacing_x * 0.1, spacing_x * 0.1)
                y = (i + 1) * spacing_y + random.uniform(-spacing_y * 0.1, spacing_y * 0.1)
                # Keep within bounds
                x = max(0.02, min(0.98, x))
                y = max(0.02, min(0.98, y))

                # Set initial radius based on proximity to others and boundaries
                min_dist = float('inf')
                for k in range(count):
                    existing_x, existing_y = circles[k, 0], circles[k, 1]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    min_dist = min(min_dist, dist)

                # Base radius based on minimum distance and boundary constraints
                if min_dist < float('inf'):
                    base_radius = min(0.1, min_dist * 0.2)
                else:
                    base_radius = 0.08

                # Respect boundaries
                boundary_radius = min(x, 1-x, y, 1-y)
                initial_r = min(base_radius, boundary_radius * 0.9)
                initial_r = max(0.005, min(0.15, initial_r))

                circles[count] = [x, y, initial_r]
                count += 1

        return circles

    def evaluate_fitness(self, individual):
        """Evaluate fitness with penalty for constraint violations."""
        circles = np.array(individual).reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]

        total_radius = np.sum(radii)
        penalty = 0

        # Check containment constraints
        if not self._check_containment(positions, radii):
            penalty += 10000

        # Check overlap constraints
        penalty += self._check_overlaps_kdtree(positions, radii)

        return (total_radius - penalty,)

    def _check_containment(self, positions, radii):
        """Check if all circles are fully contained in unit square."""
        for i, (pos, r) in enumerate(zip(positions, radii)):
            x, y = pos
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True

    def _check_overlaps_kdtree(self, positions, radii, penalty_factor=1000):
        """Check overlaps efficiently using KDTree."""
        try:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            penalty = 0
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    penalty += penalty_factor * (r_i + r_j - dist)
            return penalty
        except:
            # Fallback to brute force
            return self._brute_force_overlap_check(positions, radii, penalty_factor)

    def _brute_force_overlap_check(self, positions, radii, penalty_factor):
        """Brute force overlap checking for edge cases."""
        penalty = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                r_i = radii[i]
                r_j = radii[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    penalty += penalty_factor * (r_i + r_j - dist)
        return penalty

    def mutate_individual(self, individual):
        """Mutate individual with adaptive parameters based on population diversity."""
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0

        # Adaptive mutation rate
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.8, diversity))

        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius mutation
                    old_r = individual[i]
                    mutation_strength = 0.015 * (1 + diversity)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position mutation
                    old_val = individual[i]
                    mutation_strength = 0.02 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    def crossover_constraint_aware(self, ind1, ind2):
        """Crossover that maintains constraints with repair mechanism."""
        # Standard uniform crossover
        tools.cxUniform(ind1, ind2, indpb=0.5)

        # Repair constraints
        temp_ind = np.array(ind1).reshape(-1, 3)

        # Fix containment issues
        for i in range(len(temp_ind)):
            x, y, r = temp_ind[i]
            adjusted = False
            if x - r < 0:
                x = r
                adjusted = True
            elif x + r > 1:
                x = 1 - r
                adjusted = True
            if y - r < 0:
                y = r
                adjusted = True
            elif y + r > 1:
                y = 1 - r
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
                    # Move one of them apart
                    if random.random() < 0.5:
                        dx, dy = pos_i - pos_j
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist
                        dy /= dist
                        step = (r_i + r_j - dist) * 0.2
                        new_x = max(0.01, min(0.99, pos_i[0] + dx * step))
                        new_y = max(0.01, min(0.99, pos_i[1] + dy * step))
                        temp_ind[i] = [new_x, new_y, r_i]
                    else:
                        dx, dy = pos_j - pos_i
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist
                        dy /= dist
                        step = (r_i + r_j - dist) * 0.2
                        new_x = max(0.01, min(0.99, pos_j[0] + dx * step))
                        new_y = max(0.01, min(0.99, pos_j[1] + dy * step))
                        temp_ind[j] = [new_x, new_y, r_j]

        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    def local_optimization(self, circles):
        """Apply local refinement to improve final solution."""
        # Multi-stage optimization approach
        for stage in range(3):
            improved = False

            # Stage 1: Increase radii where possible
            for i in range(len(circles)):
                original_r = circles[i, 2]
                max_increase = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                ) - original_r

                if max_increase > 0:
                    # Binary search for maximum safe increase
                    low, high = 0, max_increase
                    best_radius = original_r

                    for _ in range(10):
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

            if not improved:
                # Stage 2: Position refinement
                for i in range(len(circles)):
                    original_pos = circles[i, :2].copy()
                    best_pos = original_pos.copy()
                    best_radius = circles[i, 2]
                    best_score = best_radius

                    # Try several positions around current location
                    for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                        for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                            test_x = max(0.01, min(0.99, circles[i, 0] + dx))
                            test_y = max(0.01, min(0.99, circles[i, 1] + dy))

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

        return circles

    def heuristic_circle_packing(self):
        """Fallback method using structured approach."""
        n = self.N_CIRCLES
        circles = np.zeros((n, 3))

        # Grid-based arrangement
        rows = 5
        cols = 5
        if n < rows * cols:
            rows = int(np.ceil(n / cols))

        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                r = min(spacing_x, spacing_y) * 0.4
                circles[count] = [x, y, r]
                count += 1

        # Refine positions
        for _ in range(100):
            improved = False
            for i in range(n):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        test_x = max(0.01, min(0.99, circles[i, 0] + dx))
                        test_y = max(0.01, min(0.99, circles[i, 1] + dy))
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
        toolbox.register("evaluate", self.evaluate_fitness)
        toolbox.register("mate", self.crossover_constraint_aware)
        toolbox.register("mutate", self.mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)

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
            return self.heuristic_circle_packing()

        # Return best solution
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)

        # Apply local optimization to refine further
        refined_result = self.local_optimization(result.copy())

        return refined_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    problem = CirclePackingProblem()
    return problem.optimize()

# EVOLVE-BLOCK-END