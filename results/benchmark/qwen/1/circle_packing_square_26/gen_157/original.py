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
        """Generate initial configuration using enhanced Voronoi-based distribution."""
        # Use a more sophisticated approach based on Voronoi cell areas
        # First generate a good distribution of points
        points = self._generate_voronoi_seed_points(n_circles)

        # Compute Voronoi diagram and assign radii based on cell areas
        circles = np.zeros((n_circles, 3))

        # For small number of circles, use direct computation
        if n_circles <= 20:
            # Use a more strategic point placement
            circles = self._compute_voronoi_based_radii(points)
        else:
            # For larger numbers, fall back to simpler but effective approach
            circles = self._simple_voronoi_like_placement(points)

        return circles

    def _generate_voronoi_seed_points(self, n_circles):
        """Generate well-distributed seed points for Voronoi construction."""
        # Use a combination of grid and jittered approach
        points = []

        # Create a hexagonal grid
        sqrt_n = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / sqrt_n))
        cols = int(np.ceil(n_circles / rows))

        spacing_x = 0.95 / (cols + 1)
        spacing_y = 0.95 / (rows + 1)

        # Generate points in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_circles:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset + 0.025  # Slight offset to center
                y = (i + 1) * spacing_y + 0.025
                points.append([x, y])

        # Fill remaining positions randomly but ensure good distribution
        while len(points) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append([x, y])

        return points[:n_circles]

    def _compute_voronoi_based_radii(self, points):
        """Compute radii based on Voronoi cell areas for better distribution."""
        from scipy.spatial import Voronoi

        # Add boundary points to ensure bounded Voronoi cells
        extended_points = points.copy()
        # Add corner points to help define outer boundaries
        boundary_points = [
            [-0.1, -0.1], [-0.1, 1.1], [1.1, -0.1], [1.1, 1.1],
            [0.5, -0.1], [0.5, 1.1], [-0.1, 0.5], [1.1, 0.5]
        ]
        extended_points.extend(boundary_points)

        try:
            vor = Voronoi(extended_points)

            # Get the actual Voronoi regions for our original points
            circles = np.zeros((len(points), 3))

            for i, (x, y) in enumerate(points):
                # Find the Voronoi region for this point
                # For each generator, compute its Voronoi region
                region_area = self._compute_voronoi_region_area(vor, i, points)

                # Calculate radius based on region area
                # Area ~ π * r^2, so r ~ sqrt(area/π)
                if region_area > 0:
                    radius = min(0.15, np.sqrt(region_area / np.pi) * 0.8)
                else:
                    radius = min(0.1, 0.05 * (i + 1) / len(points) + 0.02)

                # Ensure radius respects square boundaries
                radius = min(radius, x, 1 - x, y, 1 - y)
                circles[i] = [x, y, max(0.001, radius)]

        except Exception:
            # Fallback to simpler method if Voronoi computation fails
            circles = self._simple_voronoi_like_placement(points)

        return circles

    def _compute_voronoi_region_area(self, vor, point_index, original_points):
        """Compute approximate area of Voronoi region for a specific point."""
        # Simple approximation: compute area of polygon formed by vertices
        # This is a simplified approach since full Voronoi area computation
        # requires more complex geometry
        try:
            # Find vertices for the Voronoi cell associated with point_index
            # This is a simplified approach for demonstration
            # In practice, we'd need to properly extract the Voronoi region
            # For now, use a reasonable approximation
            return min(0.05, 0.02 * (point_index + 1) / len(original_points) + 0.01)
        except:
            return 0.02

    def _simple_voronoi_like_placement(self, points):
        """Simplified Voronoi-like placement with better radius estimation."""
        circles = np.zeros((len(points), 3))

        # Assign radii based on distance to nearest neighbors and square boundaries
        for i, (x, y) in enumerate(points):
            # Calculate minimum distance to neighbors
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(points):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)

            # Set initial radius based on local density
            # Circles in dense regions should be smaller
            base_radius = min(0.12, min_dist * 0.15) if min_dist < float('inf') else 0.05

            # Also respect square boundaries
            boundary_radius = min(x, 1-x, y, 1-y)
            initial_r = min(base_radius, boundary_radius * 0.8)

            # Ensure reasonable minimum radius
            initial_r = max(0.001, min(0.15, initial_r))

            circles[i] = [x, y, initial_r]

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