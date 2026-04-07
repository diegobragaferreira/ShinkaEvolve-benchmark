# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
import time

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class AdaptiveVoronoiEvolution:
    """Enhanced circle packing optimizer with adaptive parameters, robust initialization, and advanced optimization."""
    
    def __init__(self, n_circles=26, pop_size=100, gen_count=100, mutpb=0.15, cxpb=0.7):
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
        
    def _generate_distributed_seed_points(self, n_circles):
        """Generate well-distributed seed points using a hybrid approach."""
        points = []
        
        # Create a hexagonal grid pattern with better distribution
        sqrt_n = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / sqrt_n))
        cols = int(np.ceil(n_circles / rows))

        spacing_x = 0.95 / (cols + 1)
        spacing_y = 0.95 / (rows + 1)

        # Generate points in hexagonal pattern with better spacing
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_circles:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                # Add strategic variation to avoid perfect grid patterns
                x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x/10, spacing_x/10)
                y = (i + 1) * spacing_y + random.uniform(-spacing_y/10, spacing_y/10)
                points.append([x, y])

        # Fill remaining positions with strategic randomization
        while len(points) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append([x, y])

        return points[:n_circles]

    def _initialization_with_density_analysis(self, n_circles):
        """Generate initial configuration using density-aware Voronoi-based distribution."""
        points = self._generate_distributed_seed_points(n_circles)
        circles = np.zeros((n_circles, 3))

        # Compute radii based on local density analysis
        for i, (x, y) in enumerate(points):
            # Calculate minimum distance to neighbors
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(points):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)

            # Set initial radius with better density consideration
            if min_dist < float('inf'):
                # Use a more sophisticated radius calculation
                base_radius = min(0.12, min_dist * 0.15)
            else:
                base_radius = 0.05

            # Respect square boundaries
            boundary_radius = min(x, 1-x, y, 1-y)
            initial_r = min(base_radius, boundary_radius * 0.8)

            # Ensure reasonable minimum radius and upper bound
            initial_r = max(0.001, min(0.15, initial_r))

            circles[i] = [x, y, initial_r]

        return circles

    def _create_individual(self):
        """Create a single individual with improved initialization and perturbation."""
        circles = self._initialization_with_density_analysis(self.N_CIRCLES)

        # Add strategic random perturbations with better diversity
        individual = circles.flatten().tolist()
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.01, 0.01)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                # Apply multiplicative perturbation for better parameter exploration
                individual[i] *= random.uniform(0.9, 1.1)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def _evaluate_fitness(self, individual):
        """Evaluate fitness with optimized constraint handling."""
        circles = np.array(individual).reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]

        total_radius = np.sum(radii)
        penalty = 0

        # Check containment constraints thoroughly
        if not self._check_containment(positions, radii):
            penalty += 100000  # High penalty for containment violations

        # Check overlap constraints efficiently
        penalty += self._check_overlaps_efficient(positions, radii)

        # Additional penalty for very small radii to encourage meaningful circles
        small_radius_penalty = np.sum(radii[radii < 0.005]) * 1000
        penalty += small_radius_penalty

        return (total_radius - penalty,)

    def _check_containment(self, positions, radii):
        """Efficiently check if all circles are fully contained in unit square."""
        # Vectorized containment check
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Check if circles are fully within bounds
        return np.all((x_coords - radii >= 0) & 
                     (x_coords + radii <= 1) & 
                     (y_coords - radii >= 0) & 
                     (y_coords + radii <= 1))

    def _check_overlaps_efficient(self, positions, radii, penalty_factor=1000):
        """Efficient overlap checking using optimized KDTree queries."""
        try:
            # Use KDTree for efficient neighbor searches
            tree = cKDTree(positions)
            # Query pairs with a safe margin
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            penalty = 0
            
            # Process pairs with precise distance calculations
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Use penalty function that increases with overlap amount
                    overlap_amount = (r_i + r_j - dist) 
                    penalty += penalty_factor * overlap_amount * (1 + overlap_amount * 0.1)
            return penalty
        except:
            # Fallback to brute force for edge cases
            return self._brute_force_overlap_check(positions, radii, 1000)

    def _brute_force_overlap_check(self, positions, radii, penalty_factor):
        """Brute force overlap checking for robustness."""
        penalty = 0
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                pos_i = positions[i]
                pos_j = positions[j]
                r_i = radii[i]
                r_j = radii[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    overlap_amount = (r_i + r_j - dist)
                    penalty += penalty_factor * overlap_amount * (1 + overlap_amount * 0.1)
        return penalty

    def _mutate_individual(self, individual):
        """Enhanced mutation operator with adaptive parameters."""
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0

        # Adaptive mutation rate based on diversity
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.8, diversity * 0.4))
        
        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius mutation
                    old_r = individual[i]
                    # Dynamic mutation strength based on current value and diversity
                    mutation_strength = 0.015 * (1 + diversity) * (1.0 / (old_r + 0.02))
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position mutation
                    old_val = individual[i]
                    # Position mutation with diversity-aware scaling
                    mutation_strength = 0.02 * (1 + diversity * 0.2)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    def _crossover_constraint_aware(self, ind1, ind2):
        """Improved crossover with better constraint maintenance."""
        # Use uniform crossover with higher recombination chance
        tools.cxUniform(ind1, ind2, indpb=0.7)
        
        # Repair constraints efficiently
        temp_ind = np.array(ind1).reshape(-1, 3)
        
        # Fix containment issues with vectorized operations
        x_coords = temp_ind[:, 0]
        y_coords = temp_ind[:, 1]
        radii = temp_ind[:, 2]
        
        # Clamp positions to valid ranges
        x_coords = np.clip(x_coords, radii, 1 - radii)
        y_coords = np.clip(y_coords, radii, 1 - radii)
        
        temp_ind[:, 0] = x_coords
        temp_ind[:, 1] = y_coords
        
        # Fix overlaps with intelligent movement
        for i in range(len(temp_ind)):
            for j in range(i+1, len(temp_ind)):
                pos_i = temp_ind[i, :2]
                pos_j = temp_ind[j, :2]
                r_i = temp_ind[i, 2]
                r_j = temp_ind[j, 2]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Move circles apart with physics-based movement
                    dx, dy = pos_i - pos_j
                    dist_total = np.sqrt(dx*dx + dy*dy) + 1e-8
                    dx /= dist_total
                    dy /= dist_total
                    
                    # Calculate separation with dynamic factor
                    separation_needed = (r_i + r_j) - dist
                    move_amount = separation_needed * 0.3
                    
                    # Apply movement with bounds
                    new_x_i = max(r_i, min(1 - r_i, pos_i[0] + dx * move_amount))
                    new_y_i = max(r_i, min(1 - r_i, pos_i[1] + dy * move_amount))
                    new_x_j = max(r_j, min(1 - r_j, pos_j[0] - dx * move_amount))
                    new_y_j = max(r_j, min(1 - r_j, pos_j[1] - dy * move_amount))
                    
                    temp_ind[i] = [new_x_i, new_y_i, r_i]
                    temp_ind[j] = [new_x_j, new_y_j, r_j]
        
        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    def _advanced_local_optimization(self, circles):
        """Comprehensive local optimization with two-phase refinement."""
        # Phase 1: Radii maximization with precision
        improved = True
        phase1_iterations = 0
        
        while improved and phase1_iterations < 100:
            improved = False
            phase1_iterations += 1
            
            for i in range(len(circles)):
                original_r = circles[i, 2]
                # Calculate maximum increase respecting boundaries
                max_increase = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                ) - original_r
                
                if max_increase > 0:
                    # Binary search for maximum safe increase
                    low, high = 0, max_increase
                    best_radius = original_r
                    
                    for _ in range(12):  # More precise search
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
        
        # Phase 2: Position refinement with comprehensive search
        phase2_iterations = 0
        while phase2_iterations < 50:
            phase2_iterations += 1
            improved = False
            
            for i in range(len(circles)):
                original_pos = circles[i, :2].copy()
                best_pos = original_pos.copy()
                best_radius = circles[i, 2]
                best_score = best_radius
                
                # Try a more extensive neighborhood search
                step_sizes = [-0.02, -0.01, 0, 0.01, 0.02]
                for dx in step_sizes:
                    for dy in step_sizes:
                        if dx == 0 and dy == 0:
                            continue
                            
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
                
                if best_score > circles[i, 2] or not np.array_equal(best_pos, original_pos):
                    circles[i, :2] = best_pos
                    circles[i, 2] = best_score
                    improved = True
                    
            if not improved:
                break
                
        return circles

    def _diversity_preservation_strategy(self, population):
        """Maintain population diversity through selective retention."""
        if len(population) < 10:
            return population
            
        # Sort by fitness and preserve top individuals
        fitness_values = [self._evaluate_fitness(ind)[0] for ind in population]
        sorted_indices = np.argsort(fitness_values)[::-1]  # Descending order
        
        # Keep top performing individuals
        top_count = max(1, len(population) // 2)
        selected = [population[i] for i in sorted_indices[:top_count]]
        
        # Add some diversity by including randomly selected individuals
        remaining_count = len(population) - top_count
        if remaining_count > 0:
            random_sample = random.sample([population[i] for i in sorted_indices[top_count:]], 
                                        min(remaining_count, len(population) // 4))
            selected.extend(random_sample)
            
        return selected

    def _hybrid_fallback_method(self):
        """Sophisticated fallback method with better starting configuration."""
        # Create a more refined grid-based arrangement
        n = self.N_CIRCLES
        circles = np.zeros((n, 3))

        # Improved hexagonal grid approach
        rows = 5
        cols = 5
        if n < rows * cols:
            rows = int(np.ceil(n / cols))

        # Better spacing with strategic padding
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Hexagonal offset with better positioning
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x/12, spacing_x/12)
                y = (i + 1) * spacing_y + random.uniform(-spacing_y/12, spacing_y/12)
                
                # Set reasonable initial radius with better distribution
                r = min(spacing_x, spacing_y) * 0.35
                circles[count] = [x, y, r]
                count += 1

        # Apply advanced refinement to reduce overlaps
        for _ in range(150):  # More iterations for better convergence
            improved = False
            for i in range(n):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                # Comprehensive neighborhood search
                step_sizes = [-0.025, -0.015, -0.005, 0, 0.005, 0.015, 0.025]
                for dx in step_sizes:
                    for dy in step_sizes:
                        if dx == 0 and dy == 0:
                            continue
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
        """Main optimization routine with enhanced strategies."""
        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_fitness)
        toolbox.register("mate", self._crossover_constraint_aware)
        toolbox.register("mutate", self._mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        population = toolbox.population(n=self.POP_SIZE)

        # Run evolution with comprehensive statistics
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
            return self._hybrid_fallback_method()
            
        # Apply diversity preservation
        population = self._diversity_preservation_strategy(population)
        
        # Return best solution
        best_individual = hof[0]
        result = np.array(best_individual).reshape(-1, 3)

        # Apply advanced local optimization
        refined_result = self._advanced_local_optimization(result.copy())
        
        return refined_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = AdaptiveVoronoiEvolution()
    return optimizer.optimize()

# EVOLVE-BLOCK-END