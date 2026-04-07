# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Improved Voronoi-based evolutionary approach with adaptive mechanisms and enhanced convergence.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def initialize_points_on_sphere(n):
        """Initialize points on a unit sphere using fibonacci spiral method."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2 * (i / (n - 1)))
            phi = np.arctan2(np.sin(i * 2 * np.pi / golden_ratio), np.cos(i * 2 * np.pi / golden_ratio))
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def voronoi_entropy_score(points):
        """
        Calculate entropy-based score of Voronoi cell distribution.
        High entropy indicates more uniform cell distribution.
        """
        try:
            sv = SphericalVoronoi(points)
            areas = sv.calculate_areas()
            # Normalize areas
            areas = areas / np.sum(areas)
            # Entropy calculation
            entropy = -np.sum(areas * np.log(areas + 1e-10))
            return entropy
        except:
            return 0.0

    def adaptive_fitness(points, ratio_weight=1.0, uniformity_weight=0.1):
        """Adaptive fitness function with better balance control."""
        ratio = calculate_min_max_ratio(points)
        uniformity = voronoi_entropy_score(points)
        # Dynamic weight adjustment based on problem characteristics
        dynamic_weight = ratio_weight * (1.0 + 0.1 * uniformity)
        return ratio * dynamic_weight

    def adaptive_neighbor_generation(current_points, perturbation_strength=0.05,
                                   generation=0, max_generations=2000,
                                   history_buffer=None, diversity_score=0.0):
        """Enhanced neighbor generation with comprehensive adaptive scaling."""
        neighbor_points = current_points.copy()

        # Comprehensive perturbation scaling based on multiple factors
        distances = pdist(current_points)
        min_dist = np.min(distances) if len(distances) > 0 else 1.0
        avg_dist = np.mean(distances) if len(distances) > 0 else 1.0

        # Cluster-based scaling
        if min_dist < 0.15:
            cluster_factor = 3.0
        elif min_dist < 0.3:
            cluster_factor = 2.0
        elif min_dist < 0.5:
            cluster_factor = 1.0
        else:
            cluster_factor = 0.5

        # Generation-based scaling (more exploration early, more exploitation later)
        generation_factor = 1.0 - 0.5 * (generation / max_generations)

        # Diversity-based scaling
        diversity_factor = 1.0 + 0.5 * (1.0 - min(diversity_score, 1.0))

        # Adaptive perturbation magnitude
        base_perturbation = perturbation_strength * cluster_factor * generation_factor * diversity_factor

        # Cap the perturbation magnitude to prevent instability
        effective_perturbation = min(0.15, max(0.005, base_perturbation))

        # Select multiple points to perturb for better exploration
        num_modify = max(2, min(len(current_points) // 3, 5))
        indices_to_modify = np.random.choice(len(current_points), num_modify, replace=False)

        for idx in indices_to_modify:
            # Generate perturbation that preserves spherical nature using tangent plane projection
            random_vec = np.random.randn(3)
            normal_vec = current_points[idx]
            tangent_vec = random_vec - np.dot(random_vec, normal_vec) * normal_vec
            # Normalize tangent vector
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 1e-10:
                tangent_vec = tangent_vec / tangent_norm
            # Apply perturbation
            perturbation = tangent_vec * np.random.normal(0, effective_perturbation)
            neighbor_points[idx] += perturbation
            # Project back to sphere ensuring numerical stability
            norm = np.linalg.norm(neighbor_points[idx])
            if norm > 1e-10:
                neighbor_points[idx] = neighbor_points[idx] / norm

        return neighbor_points

    def adaptive_evolutionary_optimize(initial_points, max_generations=2000, population_size=20):
        """
        Enhanced evolutionary optimization with adaptive control mechanisms.
        """
        # Track optimization history for adaptive control and diversity management
        history = []
        diversity_history = []
        elite_history = []

        # Initialize population
        population = [initial_points.copy()]
        for i in range(population_size - 1):
            individual = adaptive_neighbor_generation(initial_points, 0.1 + 0.05 * i, 0, max_generations)
            population.append(individual)

        best_individual = None
        best_fitness = -np.inf
        best_generation = 0

        # Precompute reference values for normalization
        fitness_reference = [adaptive_fitness(ind) for ind in population[:5]] if len(population) >= 5 else [0.0]
        reference_mean = np.mean(fitness_reference) if len(fitness_reference) > 0 else 1.0
        reference_std = np.std(fitness_reference) if len(fitness_reference) > 1 else 1.0

        for generation in range(max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = adaptive_fitness(individual)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    best_generation = generation

            # Track history
            history.append(best_fitness)
            if len(history) > 10:
                history.pop(0)

            # Calculate diversity score
            diversity_score = 0
            if len(population) >= 2:
                sample_pop = population[:min(5, len(population))]
                pairwise_distances = []
                for i in range(len(sample_pop)):
                    for j in range(i+1, len(sample_pop)):
                        dist = np.linalg.norm(sample_pop[i].flatten() - sample_pop[j].flatten())
                        pairwise_distances.append(dist)
                if pairwise_distances:
                    diversity_score = np.mean(pairwise_distances)
            diversity_history.append(diversity_score)
            if len(diversity_history) > 10:
                diversity_history.pop(0)

            # Improved selection with adaptive pressure
            normalized_fitness = np.array(fitness_scores)
            if reference_std > 1e-10:
                normalized_fitness = (normalized_fitness - reference_mean) / reference_std
            normalized_fitness = np.maximum(normalized_fitness, 0.0)  # Ensure non-negative

            # Adaptive selection pressure based on convergence
            if len(history) >= 2:
                recent_improvement = history[-1] - history[0]
                if recent_improvement < 1e-6 and generation > max_generations * 0.3:
                    # Slow progress - increase selection pressure
                    selection_pressure = 2.0
                else:
                    selection_pressure = 1.0
            else:
                selection_pressure = 1.0

            # Apply selection pressure
            if not np.allclose(normalized_fitness, 0):
                probabilities = normalized_fitness ** selection_pressure
                probabilities = probabilities / np.sum(probabilities)
            else:
                probabilities = np.ones(len(population)) / len(population)

            # Tournament selection with adaptive size
            selected_population = []
            tournament_size = max(2, min(5, len(population) // 4))

            # Keep elites
            elite_indices = np.argsort(fitness_scores)[-3:]
            for elite_idx in elite_indices:
                selected_population.append(population[elite_idx].copy())

            # Fill remaining slots
            remaining_slots = population_size - len(elite_indices)
            for _ in range(remaining_slots):
                tournament_indices = np.random.choice(len(population), tournament_size, p=probabilities)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                selected_population.append(population[winner_index].copy())

            # Enhanced crossover and mutation
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[min(i+1, len(selected_population)-1)]

                # Adaptively weighted crossover
                if len(selected_population) > 1:
                    parent_fitness_ratio = fitness_scores[i] / (fitness_scores[i] + fitness_scores[min(i+1, len(selected_population)-1)] + 1e-10)
                    alpha = 0.2 + 0.6 * parent_fitness_ratio  # Bias towards better parent
                else:
                    alpha = 0.5

                child1 = parent1 * alpha + parent2 * (1 - alpha)
                child2 = parent2 * alpha + parent1 * (1 - alpha)

                # Project children back to sphere
                for j in range(len(child1)):
                    norm = np.linalg.norm(child1[j])
                    if norm > 1e-10:
                        child1[j] = child1[j] / norm
                    norm = np.linalg.norm(child2[j])
                    if norm > 1e-10:
                        child2[j] = child2[j] / norm

                # Adaptive mutation based on generation and performance
                generation_ratio = generation / max_generations
                mutation_strength = 0.05 * (1 - generation_ratio) * (1 + 0.5 * diversity_score)

                # More aggressive mutation in early stages or when stagnating
                if generation < max_generations * 0.2 or (len(history) >= 3 and history[-1] <= history[-3]):
                    mutation_strength *= 2.0

                # Apply mutation
                child1 = adaptive_neighbor_generation(child1, mutation_strength, generation, max_generations)
                child2 = adaptive_neighbor_generation(child2, mutation_strength, generation, max_generations)

                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:population_size]

            # Advanced diversity maintenance
            if generation % 100 == 0 and generation > 0:
                # Monitor diversity trend
                if len(diversity_history) >= 2:
                    diversity_change = diversity_history[-1] - diversity_history[0]
                    if diversity_change < 0.01:  # Declining diversity
                        # Introduce diversity with new configurations
                        for i in range(3):
                            if len(population) < population_size:
                                random_individual = adaptive_neighbor_generation(initial_points, 0.2)
                                population.append(random_individual)
                            else:
                                # Replace worst individuals
                                worst_indices = np.argsort(fitness_scores)[:3]
                                for idx in worst_indices:
                                    if idx < len(population):
                                        population[idx] = adaptive_neighbor_generation(initial_points, 0.25)

                # Occasionally add completely random individuals
                if generation % 200 == 0:
                    for i in range(2):
                        if len(population) < population_size:
                            random_individual = adaptive_neighbor_generation(initial_points, 0.3)
                            population.append(random_individual)
                        else:
                            idx = np.random.randint(len(population))
                            population[idx] = adaptive_neighbor_generation(initial_points, 0.3)

            # Early termination criteria
            if generation - best_generation > 500 and generation > max_generations * 0.5:
                # Stagnant for a while - terminate early
                break

        return best_individual, best_fitness

    def project_to_unit_cube(points):
        """Project points to unit cube [0,1]^3"""
        # Find min/max along each axis
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # Handle case where there's no variation
        ranges = max_coords - min_coords
        if np.any(ranges == 0):
            # If any dimension has no variation, return points centered at 0.5
            return np.full_like(points, 0.5)

        # Scale to [0,1] range
        normalized = (points - min_coords) / ranges

        # Ensure they're clipped to [0,1]
        return np.clip(normalized, 0, 1)

    def enhanced_gradient_refinement(points):
        """Robust gradient-based refinement with multiple stages."""
        def objective(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit sphere constraint
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 1e-10:
                    points_local[i] = points_local[i] / norm
            return -calculate_min_max_ratio(points_local)

        # Multi-stage refinement approach
        refined_points = points.copy()

        # Stage 1: Preliminary coarse refinement
        try:
            result1 = minimize(objective, refined_points.flatten(), method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-5, 'gtol': 1e-5})
            refined_points = result1.x.reshape(-1, 3)

            # Project back to sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 1e-10:
                    refined_points[i] = refined_points[i] / norm

        except Exception:
            pass  # Fall back to iterative method

        # Stage 2: Iterative refinement with better sphere constraint handling
        try:
            current_points = refined_points.copy()
            best_ratio = calculate_min_max_ratio(current_points)
            best_points = current_points.copy()

            # More thorough iterative improvement
            for iteration in range(500):
                neighbor_points = current_points.copy()
                point_idx = np.random.randint(len(neighbor_points))

                # Perturbation with adaptive scaling
                perturbation_magnitude = 0.001 * (1.0 - iteration/500.0)  # Gradually decrease
                perturbation = np.random.normal(0, perturbation_magnitude, 3)
                neighbor_points[point_idx] += perturbation

                # Project back to sphere
                norm = np.linalg.norm(neighbor_points[point_idx])
                if norm > 1e-10:
                    neighbor_points[point_idx] = neighbor_points[point_idx] / norm

                new_ratio = calculate_min_max_ratio(neighbor_points)

                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = neighbor_points.copy()
                    current_points = neighbor_points.copy()

            refined_points = best_points.copy()
        except Exception:
            pass  # Final fallback

        return refined_points, calculate_min_max_ratio(refined_points)

    def initialize_fibonacci_sphere(n):
        """Better fibonacci-based sphere initialization"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def initialize_regular_polyhedron():
        """Initialize points based on regular icosahedron vertices"""
        # Regular icosahedron vertices (normalized)
        phi = (1 + np.sqrt(5)) / 2
        vertices = [
            (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
            (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
            (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
        ]
        vertices = np.array(vertices)
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        # Add more points by taking edge midpoints for better distribution
        edges = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if abs(dist - 2) < 0.1:  # approximately the edge length of our icosahedron
                    edges.append((i, j))

        # Add midpoints of edges
        additional_points = []
        for i, j in edges[:2]:  # Take first 2 edges for simplicity
            midpoint = (vertices[i] + vertices[j]) / 2
            midpoint = midpoint / np.linalg.norm(midpoint)  # normalize
            additional_points.append(midpoint)

        # Combine and ensure we have proper number of points
        all_points = np.vstack([vertices, additional_points])
        if len(all_points) > 14:
            # Select 14 points that are well spread
            return all_points[:14]
        elif len(all_points) < 14:
            # Fill with fibonacci points
            fib_points = initialize_fibonacci_sphere(14 - len(all_points))
            return np.vstack([all_points, fib_points])
        else:
            return all_points

    def multi_start_optimization(initial_points_list, max_generations=1500):
        """Run optimization from multiple starting points and return the best result"""
        best_points = None
        best_fitness = -np.inf

        for i, initial_points in enumerate(initial_points_list):
            # Run adaptive evolutionary optimization from this starting point
            optimized_points, fitness = adaptive_evolutionary_optimize(initial_points, max_generations=max_generations)

            if fitness > best_fitness:
                best_fitness = fitness
                best_points = optimized_points.copy()

        return best_points, best_fitness

    # Main execution flow
    np.random.seed(42)

    # Try multiple initialization strategies
    initial_strategies = []

    # Strategy 1: Basic Fibonacci sphere
    init1 = initialize_points_on_sphere(14)
    initial_strategies.append(("fibonacci_basic", init1))

    # Strategy 2: Improved Fibonacci sphere with different parameters
    init2 = initialize_fibonacci_sphere(14)
    initial_strategies.append(("fibonacci_improved", init2))

    # Strategy 3: Icosahedron-based initialization
    try:
        init3 = initialize_regular_polyhedron()
        initial_strategies.append(("icosahedron", init3))
    except:
        pass

    # Strategy 4: Random initialization with different seeds
    for seed in [123, 456, 789]:
        np.random.seed(seed)
        init4 = np.random.uniform(-1, 1, (14, 3))
        # Normalize to unit sphere
        for i in range(len(init4)):
            norm = np.linalg.norm(init4[i])
            if norm > 0:
                init4[i] = init4[i] / norm
        initial_strategies.append(("random_seed_" + str(seed), init4))

    # Strategy 5: Perturbed version of the basic initialization
    np.random.seed(999)
    init5 = initialize_points_on_sphere(14)
    # Apply small random perturbations
    for i in range(len(init5)):
        perturbation = np.random.normal(0, 0.05, 3)
        init5[i] += perturbation
        # Project back to sphere
        norm = np.linalg.norm(init5[i])
        if norm > 0:
            init5[i] = init5[i] / norm
    initial_strategies.append(("perturbed_fibonacci", init5))

    # Run multi-start optimization on all strategies
    best_points, _ = multi_start_optimization(
        [strategy[1] for strategy in initial_strategies],
        max_generations=1500
    )

    # Final refinement with enhanced gradient-based approach
    final_points, _ = enhanced_gradient_refinement(best_points)

    # Normalize to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(final_points)

    return points_in_cube

# EVOLVE-BLOCK-END