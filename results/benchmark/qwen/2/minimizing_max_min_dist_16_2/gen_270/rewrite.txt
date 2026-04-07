# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0

        # Calculate pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0

        return dmin / dmax

    def create_better_hexagonal_initialization():
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

    def create_concentric_ring_initialization():
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

    def create_fibonacci_sphere_like_initialization():
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

    def create_grid_initialization():
        """Create a regular grid initialization."""
        points = np.zeros((16, 2))
        idx = 0

        # Create 4x4 grid
        for i in range(4):
            for j in range(4):
                if idx < 16:
                    x = j / 3.0 if j > 0 else 0.0
                    y = i / 3.0 if i > 0 else 0.0
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_perturbed_initialization(base_points, perturbation_magnitude=0.015):
        """Create a perturbed version of base initialization."""
        perturbed = base_points.copy()
        # Add random perturbation
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        # Ensure points stay within bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def create_adaptive_perturbed_initialization(base_points, initial_ratio, iteration=0):
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

        perturbed = base_points.copy()
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def compute_gradient(points):
        """Compute gradient of the min-max ratio objective function using finite differences."""
        epsilon = 1e-8
        n_points = points.shape[0]
        grad = np.zeros_like(points)

        # For each point, compute finite difference gradient
        for i in range(n_points):
            for j in range(2):  # x and y coordinates
                # Perturb point
                points_plus = points.copy()
                points_minus = points.copy()
                points_plus[i, j] += epsilon
                points_minus[i, j] -= epsilon

                # Ensure bounds
                points_plus = np.clip(points_plus, 0, 1)
                points_minus = np.clip(points_minus, 0, 1)

                # Evaluate function
                ratio_plus = calculate_min_max_ratio(points_plus)
                ratio_minus = calculate_min_max_ratio(points_minus)

                # Finite difference gradient
                grad[i, j] = (ratio_plus - ratio_minus) / (2 * epsilon)

        return grad.flatten()

    def optimize_with_local_refinement(initial_points, max_iter=500, method='L-BFGS-B'):
        """Perform local optimization refinement on initial configuration."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]

        # Get current ratio to adapt optimization parameters
        current_ratio = calculate_min_max_ratio(initial_points)

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

        # Try gradient-based method first if available (L-BFGS-B supports it)
        best_points = initial_points.copy()
        best_ratio = current_ratio

        try:
            # For L-BFGS-B, we can provide a gradient function
            def objective_with_grad(flat_points):
                points = flat_points.reshape(-1, 2)
                ratio = calculate_min_max_ratio(points)
                # Return negative ratio (since we want to maximize) and its gradient
                return -ratio, -compute_gradient(points)

            result = minimize(
                lambda flat_points: -calculate_min_max_ratio(flat_points.reshape(-1, 2)),
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                jac=compute_gradient,  # Provide gradient
                options={'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted},
                callback=None
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = calculate_min_max_ratio(optimized_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points

        except Exception as e:
            # Fall back to standard methods if gradient computation fails
            pass

        # Try remaining methods
        methods_and_tolerances = [
            ('SLSQP', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted//2}),
            ('TNC', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted//2})
        ]

        for method_name, options in methods_and_tolerances:
            try:
                result = minimize(
                    lambda flat_points: -calculate_min_max_ratio(flat_points.reshape(-1, 2)),
                    initial_flat,
                    method=method_name,
                    bounds=bounds,
                    options=options,
                    callback=None
                )

                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = calculate_min_max_ratio(optimized_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points

            except Exception as e:
                continue

        return best_points

    def tournament_selection(population: List[np.ndarray], fitnesses: List[float], tournament_size: int = 3) -> np.ndarray:
        """Select an individual from population using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    def crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents."""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()

        # Create offspring through crossover
        mask = np.random.random(parent1.shape) > 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        return child1, child2

    def mutate(individual: np.ndarray, mutation_strength: float = 0.02):
        """Apply Gaussian mutation to an individual."""
        mutated = individual.copy()
        noise = np.random.normal(0, mutation_strength, individual.shape)
        mutated += noise
        # Ensure bounds
        mutated = np.clip(mutated, 0, 1)
        return mutated

    def sphere_packing_evolution(max_generations: int = 100, population_size: int = 30) -> np.ndarray:
        """Evolutionary algorithm focused on sphere packing optimization."""
        # Create initial population with diverse strategies
        population = []
        fitness_scores = []

        # Generate multiple initial configurations with increased diversity
        strategies = [
            create_better_hexagonal_initialization(),
            create_concentric_ring_initialization(),
            create_fibonacci_sphere_like_initialization(),
            create_grid_initialization()
        ]

        # Add a few more diverse strategies
        np.random.seed(42)
        # Random configurations with different characteristics
        for _ in range(3):
            random_points = np.random.rand(16, 2)
            strategies.append(random_points)

        # Add variants of each strategy with adaptive perturbations
        for strategy in strategies:
            # Original strategy
            population.append(strategy)
            # Perturbed variants with adaptive magnitudes
            for i, perturbation_magnitude in enumerate([0.01, 0.02, 0.03]):
                # Use adaptive perturbations based on current solution quality
                adapted_perturbed = create_adaptive_perturbed_initialization(strategy, 0.15, i)
                population.append(adapted_perturbed)

        # Ensure we have enough population members
        while len(population) < population_size:
            random_points = np.random.rand(16, 2)
            population.append(random_points)

        # Limit population to exact size
        population = population[:population_size]

        # Evaluate initial population
        for individual in population:
            fitness = calculate_min_max_ratio(individual)
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
                parent1 = tournament_selection(population, fitness_scores)
                parent2 = tournament_selection(population, fitness_scores)

                # Specialized crossover - preserve some characteristics of better parent
                if calculate_min_max_ratio(parent1) > calculate_min_max_ratio(parent2):
                    better_parent = parent1
                    worse_parent = parent2
                else:
                    better_parent = parent2
                    worse_parent = parent1

                # Use crossover that favors better parent characteristics
                crossover_rate = 0.7 if gen < max_generations // 2 else 0.5
                child1, child2 = crossover(parent1, parent2, crossover_rate)

                # Mutation with adaptive rate based on generation and population diversity
                # Calculate population diversity
                if len(population) > 1:
                    diversity = np.std([calculate_min_max_ratio(ind) for ind in population])
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
                child1 = mutate(child1, mutation_strength)
                child2 = mutate(child2, mutation_strength)

                new_population.extend([child1, child2])

            # Trim to exact population size
            new_population = new_population[:population_size]

            # Evaluate new population
            new_fitness_scores = []
            for individual in new_population:
                fitness = calculate_min_max_ratio(individual)
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
        refined_best = optimize_with_local_refinement(best_individual, max_iter=300)
        refined_fitness = calculate_min_max_ratio(refined_best)

        if refined_fitness > best_fitness:
            return refined_best

        return best_individual.copy()

    def multi_strategy_optimization():
        """Main optimization routine using hybrid approach."""
        np.random.seed(42)

        # Store all candidate solutions for comparison
        candidates = []
        candidate_ratios = []

        # Phase 1: Evolutionary optimization with multiple runs
        print("Starting evolutionary optimization...")
        
        # Run several evolutionary optimization attempts with different parameters
        for i in range(3):
            try:
                evolved_solution = sphere_packing_evolution(max_generations=120 + i*10, population_size=30 + i*5)
                ratio = calculate_min_max_ratio(evolved_solution)
                candidates.append(evolved_solution)
                candidate_ratios.append(ratio)
            except Exception as e:
                continue

        # Phase 2: Enhanced local optimizations from best evolutionary results
        if candidates:
            # Get the best evolutionary result
            best_evolutionary_idx = np.argmax(candidate_ratios)
            best_evolutionary = candidates[best_evolutionary_idx]

            # Multiple local refinement approaches from the best evolutionary result
            refined_candidates = []

            # Direct refinement with adaptive iteration count
            direct_refined = optimize_with_local_refinement(best_evolutionary, max_iter=400)
            refined_candidates.append(direct_refined)

            # Refinement with different methods
            for method in ['SLSQP', 'TNC']:
                try:
                    method_refined = optimize_with_local_refinement(best_evolutionary, max_iter=300, method=method)
                    refined_candidates.append(method_refined)
                except Exception as e:
                    continue

            # Add all refined candidates
            for candidate in refined_candidates:
                ratio = calculate_min_max_ratio(candidate)
                candidates.append(candidate)
                candidate_ratios.append(ratio)

        # Phase 3: Compare with diverse baseline strategies (with adaptive perturbations)
        baseline_strategies = [
            ("hex", create_better_hexagonal_initialization()),
            ("ring", create_concentric_ring_initialization()),
            ("fibonacci", create_fibonacci_sphere_like_initialization()),
            ("grid", create_grid_initialization()),
            ("random", np.random.rand(16, 2))  # Add truly random
        ]

        for name, strategy in baseline_strategies:
            try:
                # Perturb the base strategy with adaptive magnitude
                current_ratio = calculate_min_max_ratio(strategy)
                # Use adaptive perturbation that considers current quality
                adapted_perturbation = create_adaptive_perturbed_initialization(strategy, current_ratio, 0)
                local_optimized = optimize_with_local_refinement(adapted_perturbation, max_iter=200)
                ratio = calculate_min_max_ratio(local_optimized)

                candidates.append(local_optimized)
                candidate_ratios.append(ratio)

                # Also try a more aggressive refinement with adaptive parameters
                aggressive_refined = optimize_with_local_refinement(local_optimized, max_iter=300)
                ratio_aggressive = calculate_min_max_ratio(aggressive_refined)

                if ratio_aggressive > ratio:
                    candidates.append(aggressive_refined)
                    candidate_ratios.append(ratio_aggressive)
            except Exception as e:
                continue

        # Phase 4: Multi-start differential evolution approach
        # Try different initialization strategies with DE optimization
        try:
            initial_configs = []
            
            # Strategy 1: Better hexagonal initialization
            hex_initial = create_better_hexagonal_initialization()
            initial_configs.append(('hex', create_perturbed_initialization(hex_initial, 0.015)))

            # Strategy 2: Concentric ring initialization
            ring_initial = create_concentric_ring_initialization()
            initial_configs.append(('ring', create_perturbed_initialization(ring_initial, 0.02)))

            # Strategy 3: Fibonacci-like arrangement
            fib_initial = create_fibonacci_sphere_like_initialization()
            initial_configs.append(('fibonacci', create_perturbed_initialization(fib_initial, 0.01)))

            # Strategy 4: Grid initialization
            grid_initial = create_grid_initialization()
            initial_configs.append(('grid', create_perturbed_initialization(grid_initial, 0.02)))

            # Strategy 5: Pure random with better seed control
            np.random.seed(42)
            random_initial = np.random.rand(16, 2)
            initial_configs.append(('random', random_initial))

            # Strategy 6: Spider web pattern
            spider_web = np.zeros((16, 2))
            center = [0.5, 0.5]
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            radii = np.linspace(0.1, 0.8, 16)
            for i in range(16):
                spider_web[i] = [center[0] + radii[i]*np.cos(angles[i]),
                               center[1] + radii[i]*np.sin(angles[i])]
            # Clip to bounds
            spider_web = np.clip(spider_web, 0, 1)
            initial_configs.append(('spider', spider_web))

            # Try each initialization with DE optimization
            for i, (config_type, initial_config) in enumerate(initial_configs):
                try:
                    # Compute adaptive scaling factor based on initial quality
                    initial_ratio = calculate_min_max_ratio(initial_config)
                    
                    # First perform global optimization with DE
                    bounds = [(0, 1)] * 32

                    # Use differential evolution for global search with better parameters
                    de_result = differential_evolution(
                        lambda x: -calculate_min_max_ratio(x.reshape(-1, 2)),
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
                    refined_points = optimize_with_local_refinement(de_result.x.reshape(-1, 2), 400)
                    ratio = calculate_min_max_ratio(refined_points)

                    candidates.append(refined_points)
                    candidate_ratios.append(ratio)

                except Exception as e:
                    # If optimization fails, continue with next configuration
                    continue
                    
        except Exception as e:
            pass  # Continue with other methods if DE fails

        # Phase 5: Final comprehensive optimization
        if candidates:
            # Select the best candidate from all efforts
            best_idx = np.argmax(candidate_ratios)
            final_candidate = candidates[best_idx]

            # Do one final comprehensive refinement with adaptive parameters
            final_refinement = optimize_with_local_refinement(final_candidate, max_iter=500)
            final_ratio = calculate_min_max_ratio(final_refinement)

            # If better, return the final refinement; otherwise return the best existing candidate
            if final_ratio > candidate_ratios[best_idx]:
                return final_refinement
            else:
                return final_candidate
        else:
            # Fallback to basic optimization with adaptive perturbations
            fallback_points = create_better_hexagonal_initialization()
            adapted_fallback = create_adaptive_perturbed_initialization(fallback_points, 0.1, 0)
            return optimize_with_local_refinement(adapted_fallback, max_iter=500)

    # Execute the main optimization
    final_points = multi_strategy_optimization()

    return final_points

# EVOLVE-BLOCK-END