# EVOLVE-BLOCK-START
import numpy as np
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
import time
from collections import defaultdict

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class ImprovedAdaptiveVoronoiEvolution:
    """Enhanced circle packing optimizer with adaptive parameters, robust initialization, and hybrid optimization."""
    
    def __init__(self, n_circles=26, pop_size=150, gen_count=80, mutpb=0.12, cxpb=0.7):
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
        
    def _generate_advanced_voronoi_seed_points(self, n_circles):
        """Generate well-distributed seed points using advanced Voronoi principles."""
        points = []
        
        # Create a sophisticated hexagonal grid pattern
        sqrt_n = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / sqrt_n))
        cols = int(np.ceil(n_circles / rows))

        spacing_x = 0.92 / (cols + 1)
        spacing_y = 0.92 / (rows + 1)

        # Generate points in hexagonal pattern with enhanced distribution
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_circles:
                    break
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x/8, spacing_x/8)
                y = (i + 1) * spacing_y + random.uniform(-spacing_y/8, spacing_y/8)
                points.append([x, y])

        # Fill remaining positions with strategic randomization
        while len(points) < n_circles:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            points.append([x, y])

        return points[:n_circles]

    def _voronoi_like_initialization(self, n_circles):
        """Generate initial configuration using enhanced Voronoi-based distribution."""
        points = self._generate_advanced_voronoi_seed_points(n_circles)
        circles = np.zeros((n_circles, 3))

        # Compute radii based on local density analysis for better distribution
        for i, (x, y) in enumerate(points):
            # Calculate minimum distance to neighbors
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(points):
                if i != j:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)

            # Set initial radius based on local density and boundary considerations
            base_radius = min(0.12, (min_dist * 0.18) if min_dist < float('inf') else 0.05)
            
            # Also respect square boundaries
            boundary_radius = min(x, 1-x, y, 1-y)
            initial_r = min(base_radius, boundary_radius * 0.75)

            # Ensure reasonable minimum radius and upper bound
            initial_r = max(0.001, min(0.15, initial_r))

            circles[i] = [x, y, initial_r]

        return circles

    def _create_individual(self):
        """Create a single individual with Voronoi initialization and perturbation."""
        circles = self._voronoi_like_initialization(self.N_CIRCLES)

        # Add strategic random perturbations with enhanced diversity
        individual = circles.flatten().tolist()
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.015, 0.015)
                individual[i] = max(0, min(1, individual[i]))
            else:  # radius
                # Apply multiplicative perturbation for better parameter exploration
                individual[i] *= random.uniform(0.85, 1.15)
                individual[i] = max(0.001, min(0.5, individual[i]))
        return creator.Individual(individual)

    def _evaluate_fitness(self, individual):
        """Evaluate fitness with comprehensive constraint violation penalties."""
        circles = np.array(individual).reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]

        total_radius = np.sum(radii)
        penalty = 0

        # Check containment constraints thoroughly
        if not self._check_containment(positions, radii):
            penalty += 50000  # Higher penalty for containment violations

        # Check overlap constraints with optimized performance
        penalty += self._check_overlaps_optimized(positions, radii)

        # Additional penalty for very small radii to encourage meaningful circles
        small_radius_penalty = np.sum(radii[radii < 0.01]) * 500
        penalty += small_radius_penalty

        return (total_radius - penalty,)

    def _check_containment(self, positions, radii):
        """Efficiently check if all circles are fully contained in unit square."""
        for i, (pos, r) in enumerate(zip(positions, radii)):
            x, y = pos
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True

    def _check_overlaps_optimized(self, positions, radii, penalty_factor=500):
        """Optimized overlap checking using advanced filtering strategies."""
        try:
            # Use KDTree for efficient neighbor searches
            tree = cKDTree(positions)
            # Query pairs with a safe margin
            pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
            penalty = 0
            
            # Process pairs with more accurate distance calculations
            for i, j in pairs:
                r_i = radii[i]
                r_j = radii[j]
                pos_i = positions[i]
                pos_j = positions[j]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Use a more sophisticated penalty function
                    overlap_amount = (r_i + r_j - dist)
                    penalty += penalty_factor * overlap_amount * (1 + overlap_amount)
            return penalty
        except:
            # Fallback to brute force for edge cases
            return self._brute_force_overlap_check(positions, radii, 500)

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
                    penalty += penalty_factor * overlap_amount * (1 + overlap_amount)
        return penalty

    def _mutate_individual(self, individual):
        """Enhanced mutation operator with adaptive parameters and improved strategies."""
        individual_array = np.array(individual).reshape(-1, 3)
        radii = individual_array[:, 2]
        diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0

        # Adaptive mutation rate with exponential decay
        adaptive_mutation_rate = self.MUTPB * (1 - min(0.85, diversity * 0.5))
        
        for i in range(len(individual)):
            if random.random() < adaptive_mutation_rate:
                idx = i % 3
                if idx == 2:  # radius mutation
                    old_r = individual[i]
                    # Smart mutation strength based on current value and diversity
                    mutation_strength = 0.012 * (1 + diversity) * (1.0 / (old_r + 0.02))
                    new_r = old_r + random.gauss(0, mutation_strength)
                    individual[i] = max(0.001, min(0.5, new_r))
                else:  # position mutation
                    old_val = individual[i]
                    # Position mutation with diversity-aware scaling
                    mutation_strength = 0.015 * (1 + diversity * 0.3)
                    new_val = old_val + random.gauss(0, mutation_strength)
                    individual[i] = max(0, min(1, new_val))
        return individual,

    def _crossover_constraint_aware(self, ind1, ind2):
        """Advanced crossover with superior constraint maintenance."""
        # Use uniform crossover with higher recombination chance
        tools.cxUniform(ind1, ind2, indpb=0.7)
        
        # Repair constraints with enhanced precision
        temp_ind = np.array(ind1).reshape(-1, 3)
        
        # Fix containment issues carefully while preserving solution structure
        for i in range(len(temp_ind)):
            x, y, r = temp_ind[i]
            # Clamp position to valid range ensuring circle stays fully within bounds
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            temp_ind[i] = [x, y, r]
        
        # Fix overlaps with improved physics-based movement
        for i in range(len(temp_ind)):
            for j in range(i+1, len(temp_ind)):
                pos_i = temp_ind[i, :2]
                pos_j = temp_ind[j, :2]
                r_i = temp_ind[i, 2]
                r_j = temp_ind[j, 2]
                dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                if dist < (r_i + r_j):
                    # Move circles apart along the line connecting centers with physics-inspired calculation
                    dx, dy = pos_i - pos_j
                    dist_total = np.sqrt(dx*dx + dy*dy) + 1e-8
                    dx /= dist_total
                    dy /= dist_total
                    
                    # Calculate separation with dynamic factor
                    separation_needed = (r_i + r_j) - dist
                    move_amount = separation_needed * 0.3
                    
                    # Apply movement with bounded adjustments
                    new_x_i = max(r_i, min(1 - r_i, pos_i[0] + dx * move_amount))
                    new_y_i = max(r_i, min(1 - r_i, pos_i[1] + dy * move_amount))
                    new_x_j = max(r_j, min(1 - r_j, pos_j[0] - dx * move_amount))
                    new_y_j = max(r_j, min(1 - r_j, pos_j[1] - dy * move_amount))
                    
                    temp_ind[i] = [new_x_i, new_y_i, r_i]
                    temp_ind[j] = [new_x_j, new_y_j, r_j]
        
        ind1[:] = temp_ind.flatten()
        return ind1, ind2

    def _advanced_local_optimization(self, circles):
        """Comprehensive local optimization with enhanced refinement phases."""
        # Phase 1: Radii maximization with careful validation
        improved = True
        phase1_iterations = 0
        
        while improved and phase1_iterations < 150:
            improved = False
            phase1_iterations += 1
            
            for i in range(len(circles)):
                original_r = circles[i, 2]
                # Calculate maximum possible increase respecting boundaries
                max_increase = min(
                    circles[i, 0], 1 - circles[i, 0],
                    circles[i, 1], 1 - circles[i, 1]
                ) - original_r
                
                if max_increase > 0:
                    # Binary search for maximum safe increase with more iterations
                    low, high = 0, max_increase
                    best_radius = original_r
                    
                    for _ in range(15):  # Increased precision iterations
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
        
        # Phase 2: Position refinement with enhanced neighborhood search
        phase2_iterations = 0
        while phase2_iterations < 100:
            phase2_iterations += 1
            improved = False
            
            for i in range(len(circles)):
                original_pos = circles[i, :2].copy()
                best_pos = original_pos.copy()
                best_radius = circles[i, 2]
                best_score = best_radius
                
                # Try a more comprehensive grid of positions around current location
                steps = [-0.025, -0.015, -0.005, 0, 0.005, 0.015, 0.025]
                for dx in steps:
                    for dy in steps:
                        # Skip center point to avoid redundant checks
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
                            score = test_r  # Focus on maximizing radius
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

    def _diversity_preserving_speciation(self, population):
        """Preserve diversity through smart speciation of similar individuals."""
        if len(population) < 5:
            return population
            
        # Simple but effective diversity preservation
        # Retain top performers and add some diversity through random selection
        if len(population) > 0:
            # Sort by fitness
            sorted_pop = sorted(population, key=lambda ind: self._evaluate_fitness(ind)[0], reverse=True)
            
            # Keep top 70%
            keep_count = int(len(sorted_pop) * 0.7)
            selected = sorted_pop[:keep_count]
            
            # Add some random individuals for diversity
            remaining_count = len(population) - keep_count
            if remaining_count > 0:
                random.shuffle(sorted_pop[keep_count:])
                selected.extend(sorted_pop[keep_count:keep_count+remaining_count])
            
            return selected
        return population

    def _hybrid_heuristic_fallback(self):
        """Enhanced fallback method using sophisticated grid-based arrangement."""
        # Advanced hexagonal packing pattern
        n = self.N_CIRCLES
        circles = np.zeros((n, 3))

        # Create optimized hexagonal pattern
        rows = 5
        cols = 5
        if n < rows * cols:
            rows = int(np.ceil(n / cols))

        # Better spacing with padding
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Hexagonal offset pattern
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x/10, spacing_x/10)
                y = (i + 1) * spacing_y + random.uniform(-spacing_y/10, spacing_y/10)
                
                # Set reasonable initial radius
                r = min(spacing_x, spacing_y) * 0.3
                circles[count] = [x, y, r]
                count += 1

        # Apply sophisticated refinement to avoid overlaps
        for _ in range(200):  # More iterations for better convergence
            improved = False
            for i in range(n):
                best_pos = circles[i, :2].copy()
                best_rad = circles[i, 2]
                best_score = -1000

                # Try a more complete neighborhood search
                steps = [-0.03, -0.015, 0, 0.015, 0.03]
                for dx in steps:
                    for dy in steps:
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
        """Main optimization routine with enhanced strategies and improved termination."""
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
            return self._hybrid_heuristic_fallback()
            
        # Apply diversity preservation
        population = self._diversity_preserving_speciation(population)
        
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
    optimizer = ImprovedAdaptiveVoronoiEvolution()
    return optimizer.optimize()

# EVOLVE-BLOCK-END