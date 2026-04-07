# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    """Optimizes placement of 26 non-overlapping circles in unit square to maximize sum of radii."""

    def __init__(self, n_circles=26, pop_size=100, gen_count=80, mutpb=0.3, cxpb=0.5):
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

    def _generate_voronoi_initial_config(self):
        """Generate initial circle configuration using enhanced Voronoi-based distribution."""
        from scipy.spatial import Voronoi
        import matplotlib.pyplot as plt
        import math

        circles = np.zeros((self.N_CIRCLES, 3))

        # Generate initial points using a more sophisticated approach
        # Start with hexagonal grid pattern for good spatial distribution
        sqrt_n = int(np.ceil(np.sqrt(self.N_CIRCLES)))
        rows = int(np.ceil(self.N_CIRCLES / sqrt_n))
        cols = int(np.ceil(self.N_CIRCLES / rows))

        # Create points with hexagonal offset
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.N_CIRCLES:
                    break
                x_offset = 0 if i % 2 == 0 else 0.5
                x = (j + 0.5 + x_offset) / cols * 0.9 + 0.05  # Keep away from edges
                y = (i + 0.5) / rows * 0.9 + 0.05
                points.append([x, y])

        # Add extra boundary points to improve edge coverage
        boundary_points = []
        for _ in range(20):  # Add some boundary points
            side = random.randint(0, 3)
            if side == 0:  # Top
                boundary_points.append([random.uniform(0.05, 0.95), 0.95])
            elif side == 1:  # Bottom
                boundary_points.append([random.uniform(0.05, 0.95), 0.05])
            elif side == 2:  # Left
                boundary_points.append([0.05, random.uniform(0.05, 0.95)])
            else:  # Right
                boundary_points.append([0.95, random.uniform(0.05, 0.95)])

        points.extend(boundary_points)
        points = points[:self.N_CIRCLES]

        # Ensure we have the right number of points
        while len(points) < self.N_CIRCLES:
            points.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])

        points = points[:self.N_CIRCLES]

        # Use Voronoi to get cell structure and calculate better cell areas
        try:
            vor = Voronoi(points)

            # Better Voronoi cell area estimation using more precise method
            cell_areas = []
            for i in range(len(points)):
                # Use the Voronoi ridge approach to get more accurate cell estimates
                if i < len(vor.points):
                    # Use a more reliable method for estimating cell areas
                    # For Voronoi cells, we can estimate area by computing the mean distance to neighbors
                    # squared and scaling appropriately for circular approximation
                    neighbors = []
                    for j in range(len(points)):
                        if i != j:
                            dist = np.sqrt(sum((vor.points[i] - vor.points[j])**2))
                            neighbors.append((dist, j))

                    if neighbors:
                        # Sort neighbors by distance
                        neighbors.sort(key=lambda x: x[0])

                        # Take first 4 neighbors and compute average distance
                        avg_dist = np.mean([n[0] for n in neighbors[:min(4, len(neighbors))]])
                        # Approximate cell area for a Voronoi cell (assuming roughly circular)
                        estimated_area = avg_dist * avg_dist * math.pi * 0.25
                        cell_areas.append(estimated_area)
                    else:
                        cell_areas.append(1.0)
                else:
                    cell_areas.append(1.0)

            # Normalize cell areas for priority ordering (larger areas = higher priority)
            if len(cell_areas) > 0:
                max_area = max(cell_areas)
                normalized_areas = [area/max_area for area in cell_areas] if max_area > 0 else [1.0]*len(cell_areas)
            else:
                normalized_areas = [1.0]*len(cell_areas)

            # Sort points by priority (larger cell areas first)
            priority_order = sorted(range(len(normalized_areas)), key=lambda i: normalized_areas[i], reverse=True)

            # Now assign radii based on Voronoi geometry with better algorithm
            for idx, original_idx in enumerate(priority_order):
                x, y = points[original_idx]

                # Calculate minimum distance to neighbors
                min_dist = float('inf')
                for j, (other_x, other_y) in enumerate(points):
                    if j != original_idx:
                        dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                        min_dist = min(min_dist, dist)

                # Determine initial radius based on Voronoi characteristics
                if min_dist < float('inf'):
                    # Use Voronoi cell area priority to influence radius sizing
                    area_priority = normalized_areas[original_idx]
                    # Improve the radius estimation using Voronoi insights
                    base_radius = min(0.18, min_dist * 0.3 * area_priority)
                else:
                    base_radius = 0.08

                # Constrain to boundary limits with better margin
                boundary_radius = min(x, 1-x, y, 1-y)
                initial_r = min(base_radius, boundary_radius * 0.9)
                initial_r = max(0.005, min(0.2, initial_r))

                circles[idx] = [x, y, initial_r]

        except Exception as e:
            # Fallback to simpler approach if Voronoi fails
            print(f"Voronoi failed: {e}")
            # Resort to the simpler initialization approach with better parameters
            for i, (x, y) in enumerate(points):
                # Calculate minimum distance to neighbors
                min_dist = float('inf')
                for j, (other_x, other_y) in enumerate(points):
                    if i != j:
                        dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                        min_dist = min(min_dist, dist)

                # Improved initial radius determination
                if min_dist < float('inf'):
                    # Use more aggressive radius setting based on spacing
                    initial_r = min(0.18, min_dist * 0.25)
                else:
                    initial_r = 0.08

                # Constrain to boundary limits with better margin
                boundary_radius = min(x, 1-x, y, 1-y)
                initial_r = min(initial_r, boundary_radius * 0.9)
                initial_r = max(0.005, min(0.2, initial_r))

                circles[i] = [x, y, initial_r]

        return circles

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

        # Check containment constraints efficiently
        # Vectorized check for all circles at once
        contain_penalty = np.where(
            (positions[:, 0] - radii < 0) |
            (positions[:, 0] + radii > 1) |
            (positions[:, 1] - radii < 0) |
            (positions[:, 1] + radii > 1)
        )[0]

        if len(contain_penalty) > 0:
            penalty += 10000 * len(contain_penalty)

        # Check overlap constraints using cKDTree (more efficient than nested loops)
        try:
            tree = cKDTree(positions)
            # Query pairs within sum of radii plus small buffer
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            overlap_penalty = 0
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    overlap_penalty += 1000 * (r_i + r_j - dist)
            penalty += overlap_penalty
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

        # Adaptive mutation rate based on diversity
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.8, diversity))

        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius index
                    old_r = individual[i]
                    # Larger mutations for diverse populations
                    mutation_strength = 0.02 * (1 + diversity)
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position indices (x, y)
                    old_val = individual[i]
                    # Mutate position with diversity scaling
                    mutation_strength = 0.03 * (1 + diversity)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    def _create_individual(self):
        """Create a random valid individual."""
        # Start with Voronoi-based configuration
        individual = self._generate_voronoi_initial_config().flatten().tolist()

        # Add small random perturbations to encourage exploration
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.015, 0.015)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.92, 1.08)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def _crossover_constraint_aware(self, ind1, ind2):
        """Crossover that maintains constraints with repair mechanism."""
        # Perform standard uniform crossover
        tools.cxUniform(ind1, ind2, indpb=0.5)

        # Repair violated constraints in both children
        for ind in [ind1, ind2]:
            temp_ind = np.array(ind).reshape(-1, 3)

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

            # Fix overlaps with smarter repair
            for i in range(len(temp_ind)):
                for j in range(i+1, len(temp_ind)):
                    pos_i = temp_ind[i, :2]
                    pos_j = temp_ind[j, :2]
                    r_i = temp_ind[i, 2]
                    r_j = temp_ind[j, 2]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        # Move one circle away from the other
                        # Prefer to adjust the circle that's more constrained
                        dx, dy = pos_i - pos_j
                        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                        dx /= dist
                        dy /= dist
                        step = (r_i + r_j - dist) * 0.3

                        # Only move if it stays within bounds
                        new_x = max(0.005, min(0.995, pos_i[0] + dx * step))
                        new_y = max(0.005, min(0.995, pos_i[1] + dy * step))
                        temp_ind[i] = [new_x, new_y, r_i]

            ind[:] = temp_ind.flatten()
        return ind1, ind2

    def _local_optimization(self, circles):
        """Apply local optimization to refine solution."""
        # Multiple refinement passes
        for pass_num in range(3):
            improved = False

            # Pass 1: Increase radii where possible
            for i in range(len(circles)):
                original_r = circles[i, 2]
                max_increase = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                ) - original_r

                if max_increase > 0.001:
                    # Binary search for maximum safe increase
                    low = 0
                    high = max_increase
                    best_radius = original_r

                    # Binary search iterations
                    for _ in range(8):
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

                    if best_radius > original_r + 0.001:
                        circles[i, 2] = best_radius
                        improved = True

            if not improved:
                # Pass 2: Position refinement
                for i in range(len(circles)):
                    original_pos = circles[i, :2].copy()
                    best_pos = original_pos.copy()
                    best_radius = circles[i, 2]
                    best_score = best_radius

                    # Try several positions around current location
                    for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                        for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                            test_x = max(0.005, min(0.995, circles[i, 0] + dx))
                            test_y = max(0.005, min(0.995, circles[i, 1] + dy))

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

    def _heuristic_circle_packing(self):
        """Fallback method using structured approach."""
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
                r = min(spacing_x, spacing_y) * 0.35
                circles[count] = [x, y, r]
                count += 1

        # Refine positions to avoid overlaps
        for _ in range(100):
            improved = False
            for i in range(n):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                # Check nearby positions
                for dx in [-0.015, -0.01, 0, 0.01, 0.015]:
                    for dy in [-0.015, -0.01, 0, 0.01, 0.015]:
                        test_x = max(0.005, min(0.995, circles[i, 0] + dx))
                        test_y = max(0.005, min(0.995, circles[i, 1] + dy))
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