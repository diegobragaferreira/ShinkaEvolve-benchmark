# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
from typing import Tuple, List
import random
from functools import partial

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

    def calculate_sphere_packing_fitness(points):
        """Calculate a fitness metric based on sphere packing density."""
        if len(points) < 2:
            return 0

        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0

        # Fitness based on minimum distance relative to maximum distance
        # Higher values indicate better packing (larger minimum distances)
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
        """Compute gradient of the min-max ratio objective function."""
        # This implementation computes finite differences for simplicity
        # In practice, one could implement analytical gradients for better accuracy
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

        # Try multiple optimization methods with varying tolerances
        methods_and_tolerances = [
            ('L-BFGS-B', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted}),
            ('SLSQP', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted//2}),
            ('TNC', {'ftol': ftol_adjusted, 'gtol': gtol_adjusted, 'maxiter': max_iter_adjusted//2})
        ]

        best_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(initial_points)

        # Try gradient-based method first if available (L-BFGS-B supports it)
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
        for method_name, options in methods_and_tolerances:
            # Skip L-BFGS-B since we already tried it with gradient if possible
            if method_name == 'L-BFGS-B':
                continue

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
            fitness = calculate_sphere_packing_fitness(individual)
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
                if calculate_sphere_packing_fitness(parent1) > calculate_sphere_packing_fitness(parent2):
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
                    diversity = np.std([calculate_sphere_packing_fitness(ind) for ind in population])
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
                fitness = calculate_sphere_packing_fitness(individual)
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
        """Main optimization routine using hierarchical multi-start optimization."""
        np.random.seed(42)

        # Store all candidate solutions for comparison
        candidates = []
        candidate_ratios = []

        # Level 1: Basic deterministic strategies
        print("Level 1: Testing basic deterministic strategies...")
        basic_strategies = [
            ("hex", create_better_hexagonal_initialization()),
            ("grid", create_grid_initialization()),
            ("ring", create_concentric_ring_initialization()),
            ("fibonacci", create_fibonacci_sphere_like_initialization())
        ]

        for name, strategy in basic_strategies:
            try:
                # Local optimization with different methods
                local_optimized = optimize_with_local_refinement(strategy, max_iter=300)
                ratio = calculate_min_max_ratio(local_optimized)
                candidates.append(local_optimized)
                candidate_ratios.append(ratio)
            except Exception as e:
                continue

        # Level 1a: Add gradient-enhanced refinement for top candidates from level 1
        print("Level 1a: Gradient-enhanced refinement of top candidates...")
        if candidates and len(candidate_ratios) > 0:
            # Sort by ratio to get top performers
            sorted_indices = np.argsort(candidate_ratios)[::-1][:3]  # Top 3
            for idx in sorted_indices:
                try:
                    # Apply gradient-based refinement specifically for the top candidates
                    gradient_enhanced = optimize_with_local_refinement(candidates[idx], max_iter=200)
                    ratio = calculate_min_max_ratio(gradient_enhanced)
                    candidates.append(gradient_enhanced)
                    candidate_ratios.append(ratio)
                except Exception as e:
                    continue

        # Level 2: Perturbed variants with adaptive scaling
        print("Level 2: Testing perturbed variants...")
        for name, strategy in basic_strategies:
            try:
                # Multiple perturbed versions
                for perturb_mag in [0.01, 0.02, 0.03]:
                    perturbed = create_perturbed_initialization(strategy, perturb_mag)
                    local_optimized = optimize_with_local_refinement(perturbed, max_iter=250)
                    ratio = calculate_min_max_ratio(local_optimized)
                    candidates.append(local_optimized)
                    candidate_ratios.append(ratio)
            except Exception as e:
                continue

        # Level 3: Enhanced local optimization from best candidates
        print("Level 3: Enhanced local optimization...")
        if candidates:
            # Select top candidates for further refinement
            top_indices = np.argsort(candidate_ratios)[-5:]  # Top 5 candidates
            for idx in top_indices:
                try:
                    # More aggressive refinement for top candidates
                    refined = optimize_with_local_refinement(candidates[idx], max_iter=400)
                    ratio = calculate_min_max_ratio(refined)
                    candidates.append(refined)
                    candidate_ratios.append(ratio)
                except Exception as e:
                    continue

        # Level 4: Hybrid evolutionary approach for global exploration
        print("Level 4: Hybrid evolutionary approach...")
        try:
            # Run evolutionary optimization with adjusted parameters
            evolved_solution = sphere_packing_evolution(max_generations=100, population_size=25)
            ratio = calculate_min_max_ratio(evolved_solution)
            candidates.append(evolved_solution)
            candidate_ratios.append(ratio)
        except Exception as e:
            pass

        # Level 5: Random initialization with careful refinement
        print("Level 5: Random initialization with refinement...")
        for i in range(5):  # Try 5 random configurations
            try:
                random_points = np.random.rand(16, 2)
                # Light refinement first, then heavy refinement if promising
                light_refined = optimize_with_local_refinement(random_points, max_iter=150)
                ratio_light = calculate_min_max_ratio(light_refined)

                if ratio_light > 0.15:  # Only do intensive refinement if initial quality is decent
                    heavy_refined = optimize_with_local_refinement(light_refined, max_iter=300)
                    ratio_heavy = calculate_min_max_ratio(heavy_refined)
                    if ratio_heavy > ratio_light:
                        candidates.append(heavy_refined)
                        candidate_ratios.append(ratio_heavy)
                    else:
                        candidates.append(light_refined)
                        candidate_ratios.append(ratio_light)
                else:
                    candidates.append(light_refined)
                    candidate_ratios.append(ratio_light)
            except Exception as e:
                continue

        # Final selection: Choose best among all candidates
        if candidates:
            best_idx = np.argmax(candidate_ratios)
            best_candidate = candidates[best_idx]

            # Final validation and possible enhancement
            final_validation = optimize_with_local_refinement(best_candidate, max_iter=300)
            final_ratio = calculate_min_max_ratio(final_validation)

            if final_ratio > candidate_ratios[best_idx]:
                return final_validation
            else:
                return best_candidate
        else:
            # Fallback to the best basic strategy
            fallback_points = create_better_hexagonal_initialization()
            return optimize_with_local_refinement(fallback_points, max_iter=500)

    # Execute the main optimization
    final_points = multi_strategy_optimization()

    return final_points

# EVOLVE-BLOCK-END