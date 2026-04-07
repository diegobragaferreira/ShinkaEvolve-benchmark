# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from typing import Tuple, List, Optional
from numba import jit
import time

@jit(nopython=True)
def fast_pairwise_distances(points):
    """Fast computation of pairwise distances using numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio with enhanced strategies."""

    def __init__(self, num_points: int = 16, dimension: int = 2):
        self.num_points = num_points
        self.dimension = dimension
        self.bounds = [(0.001, 0.999) for _ in range(num_points * dimension)]
        self.max_evaluations = 10000

    def calculate_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Calculate min/max distance ratio along with actual values."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        # Use fast numba-based distance calculation
        distances = fast_pairwise_distances(points)
        distances_flat = distances[np.triu_indices_from(distances, k=1)]
        
        if len(distances_flat) == 0:
            return 0.0, 0.0, 0.0

        min_dist = np.min(distances_flat)
        max_dist = np.max(distances_flat)

        if max_dist == 0:
            return 0.0, min_dist, max_dist

        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist

    def objective_function(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, self.dimension)
        ratio, _, _ = self.calculate_ratio(points)
        return -ratio

    def create_better_hexagonal_initialization(self) -> np.ndarray:
        """Create a more sophisticated hexagonal-like arrangement of points."""
        points = np.zeros((16, 2))

        # Create a more regular hexagonal arrangement with better spacing
        row_positions = [0, 1, 2, 3]
        col_positions = [0, 1, 2, 3]
        spacing_x = 1.0 / 4.0
        spacing_y = spacing_x * np.sqrt(3) / 2.0

        idx = 0
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                if idx < 16:
                    # Offset every other row for proper hexagonal packing
                    x = (col + 0.5 * (row % 2)) * spacing_x
                    y = row * spacing_y
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_concentric_ring_initialization(self) -> np.ndarray:
        """Create a concentric ring-like arrangement."""
        points = np.zeros((16, 2))

        # Place points in concentric rings
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.8, 4)  # Four layers
        layer_points = [4, 4, 4, 4]  # 4 points per layer

        idx = 0
        for i, radius in enumerate(radii):
            num_points_in_layer = layer_points[i]
            layer_angles = np.linspace(0, 2*np.pi, num_points_in_layer, endpoint=False)
            for angle in layer_angles:
                if idx < 16:
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_fibonacci_sphere_like_initialization(self) -> np.ndarray:
        """Create a Fibonacci-like arrangement for better point distribution."""
        points = np.zeros((16, 2))

        # Use Fibonacci-inspired pattern in 2D
        golden_ratio = (1 + np.sqrt(5)) / 2.0
        for i in range(16):
            theta = 2 * np.pi * i / golden_ratio
            r = np.sqrt(i / 15.0)  # Normalize to [0,1]
            x = 0.5 + r * np.cos(theta) * 0.8
            y = 0.5 + r * np.sin(theta) * 0.8
            points[i] = [x, y]

        return points

    def create_grid_initialization(self) -> np.ndarray:
        """Create a regular grid initialization."""
        points = np.zeros((16, 2))
        idx = 0

        # Create 4x4 grid with better spacing
        for i in range(4):
            for j in range(4):
                if idx < 16:
                    # Use more uniform spacing across the unit square
                    x = j / 3.0 if j > 0 else 0.0
                    y = i / 3.0 if i > 0 else 0.0
                    # Ensure proper bounds
                    x = min(1.0, max(0.0, x))
                    y = min(1.0, max(0.0, y))
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_structured_grid_initialization(self) -> np.ndarray:
        """Create a more structured grid initialization designed for good dispersion."""
        points = np.zeros((16, 2))

        # Create a 4x4 grid with strategic spacing
        # We'll place points at positions that ensure good minimum distance properties
        positions = []

        # Generate points in a structured way that promotes even distribution
        for i in range(4):
            for j in range(4):
                # Create points that are evenly spaced but not perfectly aligned
                x = (j + 0.5) / 4.0
                y = (i + 0.5) / 4.0
                positions.append([x, y])

        # Convert to numpy array
        points = np.array(positions)

        # Add small structured perturbations to break any perfect symmetry
        # Use a pseudo-random pattern based on positions
        np.random.seed(42)
        for i in range(16):
            # Add small perturbations based on position indices
            perturbation_x = (i % 4) * 0.005 - 0.01  # Small variation in x
            perturbation_y = (i // 4) * 0.005 - 0.01  # Small variation in y
            points[i, 0] += perturbation_x
            points[i, 1] += perturbation_y

        # Ensure all points are within bounds
        points = np.clip(points, 0.0, 1.0)

        return points

    def create_perturbed_initialization(self, base_points, perturbation_magnitude=0.015):
        """Create a perturbed version of base initialization."""
        perturbed = base_points.copy()
        # Add random perturbation
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        # Ensure points stay within bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def create_adaptive_perturbed_initialization(self, base_points, initial_ratio, iteration=0):
        """Create an adaptive perturbed initialization based on current optimization state."""
        # Base perturbation magnitude
        base_perturbation = 0.015

        # Adaptive scaling based on initial quality and optimization iteration
        if initial_ratio < 0.1:
            # Poor initial configuration - use larger perturbations to explore more
            perturbation_magnitude = base_perturbation * (1.0 + (0.1 - initial_ratio) * 5.0)
        elif initial_ratio > 0.25:
            # Good initial configuration - use smaller perturbations to refine
            perturbation_magnitude = base_perturbation * max(0.1, 1.0 - (initial_ratio - 0.25) * 2.0)
        else:
            # Medium quality - use moderate perturbations
            perturbation_magnitude = base_perturbation

        # Additional adjustment based on iteration (decrease over time)
        if iteration > 0:
            perturbation_magnitude *= max(0.1, 1.0 - iteration * 0.02)

        # Add more sophisticated adaptive scaling based on distance distribution
        distances = pdist(base_points)
        if len(distances) > 0:
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            # If distribution is very uniform, increase perturbation to break symmetry
            if std_dist / avg_dist < 0.1:
                perturbation_magnitude *= 2.0
            # If distribution is very uneven, decrease perturbation to preserve good structure
            elif std_dist / avg_dist > 0.5:
                perturbation_magnitude *= 0.7

        # Add another layer of adaptivity based on the ratio of min to max distance
        # This helps focus the perturbations on areas that need more attention
        if initial_ratio < 0.15:
            # Very low ratio - need significant changes
            perturbation_magnitude *= 1.5
        elif initial_ratio < 0.2:
            # Low ratio - somewhat more changes needed
            perturbation_magnitude *= 1.2

        perturbed = base_points.copy()
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def create_intelligent_perturbed_initialization(self, base_points, current_ratio):
        """Create a more intelligent perturbed initialization based on current solution analysis."""
        # Analyze the distance distribution
        distances = pdist(base_points)
        if len(distances) == 0:
            return base_points

        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Determine perturbation magnitude based on several factors:
        # 1. Overall quality of solution (lower ratio = more aggressive perturbations)
        # 2. Distance distribution uniformity (uniform = more aggressive)
        # 3. Relative minimum distance (very small = more aggressive)

        base_perturbation = 0.02

        # Factor 1: Solution quality
        quality_factor = 1.0
        if current_ratio < 0.1:
            quality_factor = 2.0  # Very poor - aggressive perturbations
        elif current_ratio < 0.15:
            quality_factor = 1.5  # Poor - moderately aggressive
        elif current_ratio < 0.2:
            quality_factor = 1.2  # Fair - light perturbations

        # Factor 2: Uniformity of distances
        uniformity_factor = 1.0
        if std_dist / avg_dist < 0.1:
            uniformity_factor = 2.0  # Very uniform - break symmetry
        elif std_dist / avg_dist < 0.2:
            uniformity_factor = 1.5  # Somewhat uniform - moderate perturbations

        # Factor 3: Minimum distance consideration
        min_dist_factor = 1.0
        if min_dist < 0.05:
            min_dist_factor = 2.0  # Very small min distance - aggressive changes
        elif min_dist < 0.1:
            min_dist_factor = 1.5  # Small min distance - moderate changes

        # Combine all factors
        perturbation_magnitude = base_perturbation * quality_factor * uniformity_factor * min_dist_factor

        # Ensure perturbation is not too extreme
        perturbation_magnitude = min(perturbation_magnitude, 0.1)

        # Apply perturbations
        perturbed = base_points.copy()
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def optimize_with_local_refinement(self, initial_points, max_iter=500, method='L-BFGS-B'):
        """Perform local optimization refinement on initial configuration."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]

        # Get current ratio to adapt optimization parameters
        current_ratio = self.calculate_ratio(initial_points)[0]

        # Adaptive optimization parameters based on solution quality
        if current_ratio < 0.1:
            # Very poor solution - aggressive optimization
            max_iter_adjusted = max_iter * 2
            ftol_adjusted = 1e-10
            gtol_adjusted = 1e-10
        elif current_ratio < 0.2:
            # Poor solution - moderately aggressive
            max_iter_adjusted = max_iter * 1.5
            ftol_adjusted = 1e-11
            gtol_adjusted = 1e-11
        elif current_ratio > 0.25:
            # Good solution - precise optimization
            max_iter_adjusted = max_iter
            ftol_adjusted = 1e-12
            gtol_adjusted = 1e-12
        else:
            # Medium solution - balanced
            max_iter_adjusted = max_iter
            ftol_adjusted = 1e-11
            gtol_adjusted = 1e-11

        # Try multiple optimization methods with varying tolerances
        methods_and_tolerances = [
            ('L-BFGS-B', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted}),
            ('SLSQP', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted//2}),
            ('TNC', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted//2})
        ]

        best_points = initial_points.copy()
        best_ratio = self.calculate_ratio(initial_points)[0]

        for method_name, options in methods_and_tolerances:
            try:
                result = minimize(
                    lambda flat_points: self.objective_function(flat_points.reshape(-1, 2)),
                    initial_flat,
                    method=method_name,
                    bounds=bounds,
                    options=options,
                    callback=None
                )

                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = self.calculate_ratio(optimized_points)[0]

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points

            except Exception as e:
                continue

        return best_points

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float], tournament_size: int = 3) -> np.ndarray:
        """Select an individual from population using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents."""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        # Create offspring through crossover
        mask = np.random.random(parent1.shape) > 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        return child1, child2

    def mutate(self, individual: np.ndarray, mutation_strength: float = 0.02):
        """Apply Gaussian mutation to an individual."""
        mutated = individual.copy()
        noise = np.random.normal(0, mutation_strength, individual.shape)
        mutated += noise
        # Ensure bounds
        mutated = np.clip(mutated, 0, 1)
        return mutated

    def sphere_packing_evolution(self, max_generations: int = 100, population_size: int = 30) -> np.ndarray:
        """Evolutionary algorithm focused on sphere packing optimization."""
        # Create initial population with diverse strategies
        population = []
        fitness_scores = []

        # Generate multiple initial configurations with increased diversity using improved methods
        strategies = [
            self.create_better_hexagonal_initialization(),
            self.create_concentric_ring_initialization(),
            self.create_fibonacci_sphere_like_initialization(),
            self.create_grid_initialization(),
            self.create_structured_grid_initialization()  # Add our new structured grid
        ]

        # Add a few more diverse strategies
        np.random.seed(42)
        # Random configurations with different characteristics
        for _ in range(3):
            random_points = np.random.rand(16, 2)
            strategies.append(random_points)

        # Add variants of each strategy with intelligent perturbations
        for strategy in strategies:
            # Original strategy
            population.append(strategy)
            # Perturbed variants with intelligent magnitudes
            current_ratio = self.calculate_ratio(strategy)[0]
            for i in range(3):  # Create 3 variants with different perturbations
                # Use intelligent perturbations based on current solution quality
                if i == 0:
                    # Base perturbation
                    perturbed = self.create_intelligent_perturbed_initialization(strategy, current_ratio)
                elif i == 1:
                    # Larger perturbation
                    perturbed = self.create_intelligent_perturbed_initialization(strategy, current_ratio)
                    # Apply larger perturbation by manually increasing it
                    perturbed += np.random.normal(0, 0.03, strategy.shape)
                    perturbed = np.clip(perturbed, 0, 1)
                else:
                    # Smaller perturbation
                    perturbed = self.create_intelligent_perturbed_initialization(strategy, current_ratio)
                    # Apply smaller perturbation
                    perturbed += np.random.normal(0, 0.01, strategy.shape)
                    perturbed = np.clip(perturbed, 0, 1)
                population.append(perturbed)

        # Ensure we have enough population members
        while len(population) < population_size:
            random_points = np.random.rand(16, 2)
            population.append(random_points)

        # Limit population to exact size
        population = population[:population_size]

        # Evaluate initial population
        for individual in population:
            fitness = self.calculate_ratio(individual)[0]
            fitness_scores.append(fitness)

        # Evolution loop
        best_individual = population[np.argmax(fitness_scores)].copy()
        best_fitness = max(fitness_scores)

        for gen in range(max_generations):
            # Create new population through selection, crossover, and mutation
            new_population = []

            # Elitism: keep the best individuals
            elite_count = max(2, population_size // 6)  # Keep at least 2 elites
            sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
            for i in sorted_indices:
                new_population.append(population[i].copy())

            # Generate rest through evolution
            while len(new_population) < population_size:
                # Tournament selection to get two parents
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Specialized crossover - preserve some characteristics of better parent
                if self.calculate_ratio(parent1)[0] > self.calculate_ratio(parent2)[0]:
                    better_parent = parent1
                    worse_parent = parent2
                else:
                    better_parent = parent2
                    worse_parent = parent1

                # Use crossover that favors better parent characteristics
                crossover_rate = 0.7 if gen < max_generations // 2 else 0.5
                child1, child2 = self.crossover(parent1, parent2, crossover_rate)

                # Mutation with adaptive rate based on generation and population diversity
                # Calculate population diversity
                if len(population) > 1:
                    diversity = np.std([self.calculate_ratio(ind)[0] for ind in population])
                    # Adapt mutation based on diversity - higher diversity = lower mutation rate
                    mutation_rate_factor = max(0.3, 1.0 - diversity) if diversity > 0 else 1.0
                else:
                    mutation_rate_factor = 1.0

                # Dynamic mutation strength based on generation and diversity
                base_mutation_strength = 0.02
                adaptive_mutation = base_mutation_strength * (1.0 - gen/max_generations) * mutation_rate_factor

                # Further adapt based on fitness quality
                current_avg_fitness = np.mean(fitness_scores)
                if best_fitness > current_avg_fitness * 1.2:  # If we're doing really well
                    adaptive_mutation *= 0.7
                elif best_fitness < current_avg_fitness * 0.8:  # If we're struggling
                    adaptive_mutation *= 1.3

                mutation_strength = max(0.005, adaptive_mutation)
                child1 = self.mutate(child1, mutation_strength)
                child2 = self.mutate(child2, mutation_strength)

                new_population.extend([child1, child2])

            # Trim to exact population size
            new_population = new_population[:population_size]

            # Evaluate new population
            new_fitness_scores = []
            for individual in new_population:
                fitness = self.calculate_ratio(individual)[0]
                new_fitness_scores.append(fitness)

            # Update best solution
            current_best_idx = np.argmax(new_fitness_scores)
            if new_fitness_scores[current_best_idx] > best_fitness:
                best_fitness = new_fitness_scores[current_best_idx]
                best_individual = new_population[current_best_idx].copy()

            # Replace old population
            population = new_population
            fitness_scores = new_fitness_scores

        # Post-evolution refinement of the best individual
        refined_best = self.optimize_with_local_refinement(best_individual, max_iter=300)
        refined_fitness = self.calculate_ratio(refined_best)[0]

        if refined_fitness > best_fitness:
            return refined_best

        return best_individual.copy()

    def generate_initial_configurations(self) -> List[np.ndarray]:
        """Generate multiple diverse initial configurations."""
        configs = []

        # Generate different base configurations
        configs.append(self.create_better_hexagonal_initialization())
        configs.append(self.create_fibonacci_sphere_like_initialization())
        configs.append(self.create_grid_initialization())
        configs.append(self.create_concentric_ring_initialization())
        configs.append(self.create_structured_grid_initialization())

        # Add perturbed versions with different magnitudes
        np.random.seed(42)
        perturbed_configs = []
        for config in configs:
            # Three levels of perturbations
            for perturbation_magnitude in [0.01, 0.02, 0.03]:
                perturbed = self.create_perturbed_initialization(config, perturbation_magnitude)
                perturbed_configs.append(perturbed)

        # Add some specialized patterns
        # Spider web pattern
        spider_web = np.zeros((16, 2))
        center = [0.5, 0.5]
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.8, 16)
        for i in range(16):
            spider_web[i] = [center[0] + radii[i]*np.cos(angles[i]),
                           center[1] + radii[i]*np.sin(angles[i])]
        # Clip to bounds
        spider_web = np.clip(spider_web, 0, 1)
        perturbed_configs.append(spider_web)

        return perturbed_configs

    def multi_strategy_optimization(self) -> np.ndarray:
        """Main optimization routine using hybrid approach."""
        np.random.seed(42)
        start_time = time.time()

        # Store all candidate solutions for comparison
        candidates = []
        candidate_ratios = []

        # Phase 1: Evolutionary optimization with multiple runs
        print("Starting evolutionary optimization...")
        
        # Run several evolutionary optimization attempts with different parameters
        for i in range(3):
            try:
                if time.time() - start_time > 160:  # Leave some buffer time
                    break
                    
                evolved_solution = self.sphere_packing_evolution(max_generations=120 + i*10, population_size=30 + i*5)
                ratio = self.calculate_ratio(evolved_solution)[0]
                candidates.append(evolved_solution)
                candidate_ratios.append(ratio)
            except Exception as e:
                continue

        # Phase 2: Enhanced local optimizations from best evolutionary results
        if candidates and len(candidate_ratios) > 0:
            # Get the best evolutionary result
            best_evolutionary_idx = np.argmax(candidate_ratios)
            best_evolutionary = candidates[best_evolutionary_idx]

            # Multiple local refinement approaches from the best evolutionary result
            refined_candidates = []

            # Direct refinement with adaptive iteration count
            direct_refined = self.optimize_with_local_refinement(best_evolutionary, max_iter=400)
            refined_candidates.append(direct_refined)

            # Refinement with different methods
            for method in ['SLSQP', 'TNC']:
                try:
                    if time.time() - start_time > 160:  # Leave some buffer time
                        break
                    method_refined = self.optimize_with_local_refinement(best_evolutionary, max_iter=300, method=method)
                    refined_candidates.append(method_refined)
                except Exception as e:
                    continue

            # Add all refined candidates
            for candidate in refined_candidates:
                ratio = self.calculate_ratio(candidate)[0]
                candidates.append(candidate)
                candidate_ratios.append(ratio)

        # Phase 3: Compare with diverse baseline strategies (with adaptive perturbations)
        baseline_strategies = [
            ("hex", self.create_better_hexagonal_initialization()),
            ("ring", self.create_concentric_ring_initialization()),
            ("fibonacci", self.create_fibonacci_sphere_like_initialization()),
            ("grid", self.create_grid_initialization()),
            ("random", np.random.rand(16, 2))  # Add truly random
        ]

        for name, strategy in baseline_strategies:
            try:
                if time.time() - start_time > 160:  # Leave some buffer time
                    break
                    
                # Perturb the base strategy with adaptive magnitude
                current_ratio = self.calculate_ratio(strategy)[0]
                # Use adaptive perturbation that considers current quality
                adapted_perturbation = self.create_adaptive_perturbed_initialization(strategy, current_ratio, 0)
                local_optimized = self.optimize_with_local_refinement(adapted_perturbation, max_iter=200)
                ratio = self.calculate_ratio(local_optimized)[0]

                candidates.append(local_optimized)
                candidate_ratios.append(ratio)

                # Also try a more aggressive refinement with adaptive parameters
                aggressive_refined = self.optimize_with_local_refinement(local_optimized, max_iter=300)
                ratio_aggressive = self.calculate_ratio(aggressive_refined)[0]

                if ratio_aggressive > ratio:
                    candidates.append(aggressive_refined)
                    candidate_ratios.append(ratio_aggressive)
            except Exception as e:
                continue

        # Phase 4: Multi-start differential evolution approach
        if time.time() - start_time < 160:  # Leave some buffer time
            # Try different initialization strategies with DE optimization
            try:
                initial_configs = []

                # Strategy 1: Better hexagonal initialization
                hex_initial = self.create_better_hexagonal_initialization()
                initial_configs.append(('hex', self.create_perturbed_initialization(hex_initial, 0.015)))

                # Strategy 2: Concentric ring initialization
                ring_initial = self.create_concentric_ring_initialization()
                initial_configs.append(('ring', self.create_perturbed_initialization(ring_initial, 0.02)))

                # Strategy 3: Fibonacci-like arrangement
                fib_initial = self.create_fibonacci_sphere_like_initialization()
                initial_configs.append(('fibonacci', self.create_perturbed_initialization(fib_initial, 0.01)))

                # Strategy 4: Grid initialization
                grid_initial = self.create_grid_initialization()
                initial_configs.append(('grid', self.create_perturbed_initialization(grid_initial, 0.02)))

                # Strategy 5: Pure random with better seed control
                np.random.seed(42)
                random_initial = np.random.rand(16, 2)
                initial_configs.append(('random', random_initial))

                # Try each initialization with DE optimization
                for i, (config_type, initial_config) in enumerate(initial_configs):
                    try:
                        if time.time() - start_time > 160:  # Leave some buffer time
                            break
                            
                        # Compute adaptive scaling factor based on initial quality
                        initial_ratio = self.calculate_ratio(initial_config)[0]

                        # First perform global optimization with DE
                        bounds = [(0, 1)] * 32

                        # Use differential evolution for global search with better parameters
                        de_result = differential_evolution(
                            lambda x: self.objective_function(x.reshape(-1, 2)),
                            bounds,
                            seed=42 + i,
                            maxiter=300,  # Reduced iterations for efficiency
                            popsize=25,    # Increased population size for better exploration
                            mutation=(0.5, 1),
                            recombination=0.7,
                            tol=1e-8,      # Tighter tolerance
                            disp=False
                        )

                        # Refine with local optimization
                        refined_points = self.optimize_with_local_refinement(de_result.x.reshape(-1, 2), 400)
                        ratio = self.calculate_ratio(refined_points)[0]

                        candidates.append(refined_points)
                        candidate_ratios.append(ratio)

                    except Exception as e:
                        # If optimization fails, continue with next configuration
                        continue

            except Exception as e:
                pass  # Continue with other methods if DE fails

        # Phase 5: Final comprehensive optimization
        if candidates and len(candidate_ratios) > 0:
            # Select the best candidate from all efforts
            best_idx = np.argmax(candidate_ratios)
            final_candidate = candidates[best_idx]

            # Do one final comprehensive refinement with adaptive parameters
            final_refinement = self.optimize_with_local_refinement(final_candidate, max_iter=500)
            final_ratio = self.calculate_ratio(final_refinement)[0]

            # If better, return the final refinement; otherwise return the best existing candidate
            if final_ratio > candidate_ratios[best_idx]:
                return final_refinement
            else:
                return final_candidate
        else:
            # Fallback to basic optimization with adaptive perturbations
            fallback_points = self.create_better_hexagonal_initialization()
            adapted_fallback = self.create_adaptive_perturbed_initialization(fallback_points, 0.1, 0)
            return self.optimize_with_local_refinement(adapted_fallback, max_iter=500)

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize optimizer
    optimizer = PointDispersionOptimizer(16, 2)

    # Find best solution with comprehensive optimization
    best_points = optimizer.multi_strategy_optimization()

    return best_points

# EVOLVE-BLOCK-END