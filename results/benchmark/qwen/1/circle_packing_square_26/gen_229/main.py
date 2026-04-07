# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
import time
from numba import jit
from collections import defaultdict

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class AdaptiveVoronoiEvolutionImproved:
    """Improved circle packing optimizer with adaptive parameters and enhanced initialization."""

    def __init__(self, n_circles=26, pop_size=150, gen_count=80, mutpb=0.15, cxpb=0.7):
        self.N_CIRCLES = n_circles
        self.POP_SIZE = pop_size
        self.GEN_COUNT = gen_count
        self.MUTPB = mutpb
        self.CXPB = cxpb
        self.BENCHMARK = 2.6358627564136983
        self._setup_deap()

    def _setup_deap(self):
        """Initialize DEAP framework components."""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    @staticmethod
    @jit(nopython=True)
    def check_validity_jit(circles_np):
        """Fast validity check using Numba"""
        n = len(circles_np)
        for i in range(n):
            x, y, r = circles_np[i]
            # Check containment
            if r > x or r > y or r > 1-x or r > 1-y:
                return False
            # Check overlap with all previous circles
            for j in range(i):
                x2, y2, r2 = circles_np[j]
                dx = x - x2
                dy = y - y2
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r+r2)*(r+r2)
                if dist_sq < min_dist_sq:
                    return False
        return True

    def _generate_voronoi_seed_points(self, n_circles):
        """Generate well-distributed seed points using Lloyd relaxation and Voronoi-based approach."""
        # Start with a good initial distribution
        points = []

        # Create a more sophisticated initial grid
        sqrt_n = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / sqrt_n))
        cols = int(np.ceil(n_circles / rows))

        # Use a more refined spacing
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        # Generate points in hexagonal pattern with better spacing
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_circles:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x/10, spacing_x/10)
                y = (i + 1) * spacing_y + random.uniform(-spacing_y/10, spacing_y/10)
                points.append([x, y])

        # Fill remaining positions with random points
        while len(points) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append([x, y])

        points = points[:n_circles]

        # Apply Lloyd relaxation to improve distribution
        # This iteratively improves point positions
        for _ in range(10):  # Limited iterations to keep it fast
            if len(points) < 2:
                break

            try:
                # Use scipy's Voronoi to compute centroids
                vor = Voronoi(points)
                new_points = []

                for i in range(len(points)):
                    # Find Voronoi region for this point
                    region_idx = vor.point_region[i] if i < len(vor.point_region) else -1

                    if region_idx != -1 and region_idx < len(vor.regions):
                        region = vor.regions[region_idx]
                        if len(region) > 0 and -1 not in region:
                            # Get the vertices of this region
                            vertices = np.array(vor.vertices)[region]

                            # Compute centroid of the region (bounded by unit square)
                            if len(vertices) > 2:
                                # Clip vertices to unit square
                                clipped_vertices = []
                                for v in vertices:
                                    clipped_v = [
                                        max(0.01, min(0.99, v[0])),
                                        max(0.01, min(0.99, v[1]))
                                    ]
                                    clipped_vertices.append(clipped_v)

                                if len(clipped_vertices) >= 3:
                                    # Compute centroid using shoelace formula
                                    x_coords = [v[0] for v in clipped_vertices]
                                    y_coords = [v[1] for v in clipped_vertices]

                                    # Simple centroid calculation
                                    centroid_x = np.mean(x_coords)
                                    centroid_y = np.mean(y_coords)
                                    new_points.append([centroid_x, centroid_y])
                                else:
                                    new_points.append(points[i])
                            else:
                                new_points.append(points[i])
                        else:
                            new_points.append(points[i])
                    else:
                        new_points.append(points[i])

                # If we have fewer new points, just keep original
                if len(new_points) == len(points):
                    points = new_points
                else:
                    break

            except:
                # Fallback to simple averaging for robustness
                break

        return points

    def _voronoi_like_initialization(self, n_circles):
        """Generate initial configuration using enhanced Voronoi-based distribution."""
        points = self._generate_voronoi_seed_points(n_circles)
        circles = np.zeros((n_circles, 3))

        # Compute radii based on Voronoi cell analysis for better distribution
        for i, (x, y) in enumerate(points):
            # Calculate minimum distance to neighbors for local density estimation
            min_dist = float('inf')
            distances = []
            for j, (other_x, other_y) in enumerate(points):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)
                    distances.append(dist)

            # Estimate Voronoi cell area using the average neighbor distance
            if distances:
                avg_dist = np.mean(distances)
                # Approximate Voronoi cell area (assuming roughly hexagonal cells)
                approx_area = avg_dist * avg_dist * 0.866  # sqrt(3)/2 for hexagon
                # Convert area to radius (assuming circular circles)
                base_radius = np.sqrt(approx_area / np.pi) * 0.7  # Scale factor for safety
            else:
                base_radius = 0.05

            # Improve radius calculation based on boundary constraints
            boundary_dist = min(x, 1-x, y, 1-y)
            boundary_radius = boundary_dist * 0.9  # Leave some margin

            # Use the smaller of the two constraints (neighbor density vs boundaries)
            initial_r = min(base_radius, boundary_radius, 0.15)  # Cap maximum radius

            # Ensure reasonable minimum radius
            initial_r = max(0.005, initial_r)

            circles[i] = [x, y, initial_r]

        return circles

    def _create_individual(self):
        """Create a single individual with Voronoi initialization and perturbation."""
        circles = self._voronoi_like_initialization(self.N_CIRCLES)

        # Add small random perturbations but with smarter distribution
        individual = circles.flatten().tolist()
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                # Add adaptive perturbation based on position
                if individual[i] < 0.2 or individual[i] > 0.8:
                    individual[i] += random.uniform(-0.015, 0.015)
                else:
                    individual[i] += random.uniform(-0.02, 0.02)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                # Add more controlled perturbation for radii
                individual[i] *= random.uniform(0.85, 1.15)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def _evaluate_fitness(self, individual):
        """Evaluate fitness with penalty for constraint violations."""
        circles = np.array(individual).reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]

        total_radius = np.sum(radii)
        penalty = 0

        # Check containment constraints using the fast Numba method
        if not self.check_validity_jit(circles):
            penalty += 10000

        # Check overlap constraints efficiently using KDTree
        try:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    penalty += 1000 * (r_i + r_j - dist)
        except:
            # Fallback to brute force if KDTree fails
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

    def _mutate_individual(self, individual):
        """Mutate individual with adaptive parameters based on population diversity."""
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0

        # Adaptive mutation rate with generation-based decay
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.8, diversity))

        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius mutation
                    old_r = individual[i]
                    # Larger mutation for smaller radii, smaller for larger
                    mutation_strength = 0.015 * (1 + diversity) * (1.0 / (old_r + 0.01))
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position mutation
                    old_val = individual[i]
                    mutation_strength = 0.02 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    def _crossover_constraint_aware(self, ind1, ind2):
        """Improved crossover with better constraint maintenance and adaptive parameters."""
        # Adaptive crossover probability based on generation
        adaptive_cx_prob = self.CXPB * (1 - min(0.5, self.current_generation / 50.0))

        # Use adaptive uniform crossover with different probabilities for genes
        # Higher probability for positions, lower for radii
        for i in range(len(ind1)):
            gene_type = i % 3
            if random.random() < adaptive_cx_prob:
                if gene_type == 2:  # radius - use a more conservative crossover
                    # Interpolate radii
                    val1 = ind1[i]
                    val2 = ind2[i]
                    # Average with some randomness
                    new_val = (val1 + val2) / 2 + random.uniform(-0.005, 0.005)
                    ind1[i] = max(0.001, min(0.5, new_val))
                    ind2[i] = max(0.001, min(0.5, new_val))
                else:  # position - full crossover
                    # Swap positions with some smoothing
                    val1 = ind1[i]
                    val2 = ind2[i]
                    # Blend values
                    blend_factor = random.uniform(0.3, 0.7)
                    new_val1 = val1 * (1 - blend_factor) + val2 * blend_factor
                    new_val2 = val2 * (1 - blend_factor) + val1 * blend_factor

                    # Apply bounds
                    new_val1 = max(0, min(1, new_val1))
                    new_val2 = max(0, min(1, new_val2))

                    ind1[i] = new_val1
                    ind2[i] = new_val2

        # Repair constraints with more thorough checking
        temp_ind = np.array(ind1).reshape(-1, 3)

        # Fix containment issues carefully
        for i in range(len(temp_ind)):
            x, y, r = temp_ind[i]
            # Adjust position to stay within bounds with margin
            x = np.clip(x, r + 0.005, 1 - r - 0.005)
            y = np.clip(y, r + 0.005, 1 - r - 0.005)
            temp_ind[i] = [x, y, r]

        # Fix overlaps with improved algorithm that tries multiple solutions
        for i in range(len(temp_ind)):
            for j in range(i+1, len(temp_ind)):
                pos_i = temp_ind[i, :2]
                pos_j = temp_ind[j, :2]
                r_i = temp_ind[i, 2]
                r_j = temp_ind[j, 2]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Move circles apart along the line connecting centers
                    dx, dy = pos_i - pos_j
                    dist_total = np.sqrt(dx*dx + dy*dy) + 1e-8
                    dx /= dist_total
                    dy /= dist_total

                    # Calculate how much to move them apart
                    separation_needed = (r_i + r_j) - dist
                    # Move each circle by half the separation amount but with a small margin
                    move_amount = separation_needed * 0.6

                    # Apply movement, but ensure they stay within bounds
                    new_x_i = max(r_i + 0.005, min(1 - r_i - 0.005, pos_i[0] + dx * move_amount))
                    new_y_i = max(r_i + 0.005, min(1 - r_i - 0.005, pos_i[1] + dy * move_amount))
                    new_x_j = max(r_j + 0.005, min(1 - r_j - 0.005, pos_j[0] - dx * move_amount))
                    new_y_j = max(r_j + 0.005, min(1 - r_j - 0.005, pos_j[1] - dy * move_amount))

                    temp_ind[i] = [new_x_i, new_y_i, r_i]
                    temp_ind[j] = [new_x_j, new_y_j, r_j]

        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    def _local_optimization_advanced(self, circles):
        """Advanced local optimization with multi-phase approach."""
        # Phase 1: Maximize radii using binary search
        improved = True
        phase1_iterations = 0

        while improved and phase1_iterations < 100:
            improved = False
            phase1_iterations += 1

            for i in range(len(circles)):
                original_r = circles[i, 2]
                # Calculate maximum possible increase
                max_increase = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                ) - original_r

                if max_increase > 0:
                    # Binary search for maximum safe increase
                    low, high = 0, max_increase
                    best_radius = original_r

                    for _ in range(12):  # More iterations for precision
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

        # Phase 2: Position refinement
        phase2_iterations = 0
        while phase2_iterations < 50:
            phase2_iterations += 1
            improved = False

            for i in range(len(circles)):
                original_pos = circles[i, :2].copy()
                best_pos = original_pos.copy()
                best_radius = circles[i, 2]
                best_score = best_radius

                # Try a grid of positions around current location
                step = 0.01
                for dx in [-step*2, -step, 0, step, step*2]:
                    for dy in [-step*2, -step, 0, step, step*2]:
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
                            score = test_r  # Maximizing radius
                            if score > best_score:
                                best_score = score
                                best_pos = [test_x, test_y]

                if best_score > circles[i, 2] or not np.array_equal(best_pos, original_pos):
                    circles[i, :2] = best_pos
                    circles[i, 2] = best_score
                    improved = True

            if not improved:
                break

        return circles

    def _heuristic_fallback(self):
        """Fallback method using a more structured approach."""
        # Simple grid-based arrangement with optimization
        n = self.N_CIRCLES
        circles = np.zeros((n, 3))

        # Try a more optimized hexagonal packing pattern
        rows = 5
        cols = 5
        if n < rows * cols:
            rows = int(np.ceil(n / cols))

        # Use more consistent spacing
        spacing_x = 0.95 / (cols + 1)
        spacing_y = 0.95 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Create hexagonal grid pattern
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y

                # Slight randomization for better distribution
                x += random.uniform(-spacing_x/8, spacing_x/8)
                y += random.uniform(-spacing_y/8, spacing_y/8)

                # Set reasonable initial radius
                r = min(spacing_x, spacing_y) * 0.35
                circles[count] = [x, y, r]
                count += 1

        # Apply refinement to avoid overlaps
        for _ in range(150):  # More iterations
            improved = False
            for i in range(n):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                # Try nearby positions
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
        """Main optimization routine with improved parameters and termination."""
        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", self._crossover_constraint_aware)
        toolbox.register("mutate", self._mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Create initial population with better diversity
        population = toolbox.population(n=self.POP_SIZE)

        # Run evolution with performance monitoring
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
            print(f"GA failed with error: {e}")
            return self._heuristic_fallback()

        # Return best solution
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)

        # Apply advanced local optimization
        refined_result = self._local_optimization_advanced(result.copy())

        return refined_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    try:
        optimizer = AdaptiveVoronoiEvolutionImproved()
        result = optimizer.optimize()

        # Ensure correct shape and valid radii
        if len(result) < 26:
            while len(result) < 26:
                result = np.vstack([result, [0.5, 0.5, 0.01]])
        elif len(result) > 26:
            result = result[:26]

        result[:, 2] = np.maximum(0.001, result[:, 2])

        print(f"Total evaluation time: {time.time() - start_time:.2f}s")
        print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
        print(f"Benchmark ratio: {np.sum(result[:, 2]) / 2.6358627564136983:.6f}")

        return result

    except Exception as e:
        print(f"Unexpected error in circle_packing26: {e}")
        # Return a basic fallback
        return AdaptiveVoronoiEvolutionImproved()._heuristic_fallback()

# EVOLVE-BLOCK-END