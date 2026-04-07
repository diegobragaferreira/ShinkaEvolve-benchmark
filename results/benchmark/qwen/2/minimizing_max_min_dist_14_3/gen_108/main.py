# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Uses a spherical Voronoi-based evolutionary approach for improved convergence.
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
        sv = SphericalVoronoi(points)
        areas = sv.calculate_areas()
        # Normalize areas
        areas = areas / np.sum(areas)
        # Entropy calculation
        entropy = -np.sum(areas * np.log(areas + 1e-10))
        return entropy

    def spherical_voronoi_fitness(points):
        """Fitness function based on Voronoi structure and distance ratio."""
        ratio = calculate_min_max_ratio(points)
        entropy = voronoi_entropy_score(points)
        # Combine fitness: prioritize both uniformity and distance ratio
        return ratio * (1.0 + 0.1 * entropy)

    def generate_neighbor_voronoi_config(current_points, perturbation_strength=0.05):
        """Generate neighbor configuration by modifying Voronoi structure."""
        # Start with current points
        neighbor_points = current_points.copy()

        # Select random points to modify
        num_modify = max(1, len(current_points) // 4)
        indices_to_modify = np.random.choice(len(current_points), num_modify, replace=False)

        for idx in indices_to_modify:
            # Generate perturbation that preserves spherical nature
            # Create random vector and project onto tangent plane, then back to sphere
            random_vec = np.random.randn(3)
            # Project onto sphere surface normal
            normal_vec = current_points[idx]
            # Tangent vector (orthogonal to normal)
            tangent_vec = random_vec - np.dot(random_vec, normal_vec) * normal_vec
            # Normalize tangent vector
            tangent_norm = np.linalg.norm(tangent_vec)
            if tangent_norm > 1e-10:
                tangent_vec = tangent_vec / tangent_norm
            # Apply perturbation
            perturbation = tangent_vec * np.random.normal(0, perturbation_strength)
            neighbor_points[idx] += perturbation
            # Project back to sphere
            norm = np.linalg.norm(neighbor_points[idx])
            if norm > 0:
                neighbor_points[idx] = neighbor_points[idx] / norm

        return neighbor_points

    def evolutionary_voronoi_optimize(initial_points, max_generations=2000, population_size=20):
        """
        Evolutionary optimization based on Voronoi structures.
        """
        # Initialize population
        population = [initial_points.copy()]
        for i in range(population_size - 1):
            # Generate diverse initial individuals
            individual = generate_neighbor_voronoi_config(initial_points, 0.1)
            population.append(individual)

        best_individual = None
        best_fitness = -np.inf

        for generation in range(max_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = spherical_voronoi_fitness(individual)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            # Selection: tournament selection
            selected_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                selected_population.append(population[winner_index].copy())

            # Crossover and mutation
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected_population[i]
                parent2 = selected_population[min(i+1, population_size-1)]

                # Crossover: blend two parents
                alpha = np.random.random()
                child1 = parent1 * alpha + parent2 * (1 - alpha)
                child2 = parent2 * alpha + parent1 * (1 - alpha)

                # Project children back to sphere
                for j in range(len(child1)):
                    norm = np.linalg.norm(child1[j])
                    if norm > 0:
                        child1[j] = child1[j] / norm
                    norm = np.linalg.norm(child2[j])
                    if norm > 0:
                        child2[j] = child2[j] / norm

                # Mutation
                mutation_strength = max(0.01, 0.1 * (1 - generation / max_generations))
                child1 = generate_neighbor_voronoi_config(child1, mutation_strength)
                child2 = generate_neighbor_voronoi_config(child2, mutation_strength)

                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:population_size]

            # Occasionally introduce diversity
            if generation % 100 == 0 and generation > 0:
                # Add some random individuals to prevent stagnation
                for i in range(2):
                    random_individual = generate_neighbor_voronoi_config(initial_points, 0.2)
                    population.append(random_individual)
                    population.pop(0)  # Remove oldest

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

    def multi_start_optimization(initial_points_list, max_generations=1500):
        """Run optimization from multiple starting points and return the best result"""
        best_points = None
        best_fitness = -np.inf

        for i, initial_points in enumerate(initial_points_list):
            print(f"Starting optimization run {i+1}/{len(initial_points_list)}")
            # Apply evolutionary Voronoi optimization from this starting point
            optimized_points, fitness = evolutionary_voronoi_optimize(initial_points, max_generations=max_generations)

            if fitness > best_fitness:
                best_fitness = fitness
                best_points = optimized_points.copy()

        return best_points, best_fitness

    # Main execution flow
    np.random.seed(42)

    # Create multiple initialization strategies
    initial_strategies = []

    # Strategy 1: Fibonacci sphere initialization
    init1 = initialize_points_on_sphere(14)
    initial_strategies.append(("fibonacci", init1))

    # Strategy 2: Multiple Fibonacci spheres with different seeds
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        init2 = initialize_points_on_sphere(14)
        initial_strategies.append((f"fibonacci_seed_{seed}", init2))

    # Strategy 3: Random initialization on sphere
    np.random.seed(789)
    init3 = np.random.uniform(-1, 1, (14, 3))
    # Normalize to unit sphere
    for i in range(len(init3)):
        norm = np.linalg.norm(init3[i])
        if norm > 0:
            init3[i] = init3[i] / norm
    initial_strategies.append(("random_sphere", init3))

    # Run multi-start optimization on all strategies
    best_points, best_fitness = multi_start_optimization(
        [strategy[1] for strategy in initial_strategies],
        max_generations=1500
    )

    # Final refinement with gradient-based approach
    def refine_with_gradient(points):
        def objective(x_flat):
            points_local = x_flat.reshape(-1, 3)
            # Keep points on unit sphere constraint
            for i in range(len(points_local)):
                norm = np.linalg.norm(points_local[i])
                if norm > 0:
                    points_local[i] = points_local[i] / norm
            return -calculate_min_max_ratio(points_local)

        try:
            # Use L-BFGS-B for fine tuning
            from scipy.optimize import minimize
            result = minimize(objective, points.flatten(), method='L-BFGS-B',
                            options={'maxiter': 500}, tol=1e-6)
            refined_points = result.x.reshape(-1, 3)
            # Project back to sphere
            for i in range(len(refined_points)):
                norm = np.linalg.norm(refined_points[i])
                if norm > 0:
                    refined_points[i] = refined_points[i] / norm
            return refined_points, -result.fun
        except:
            return points, calculate_min_max_ratio(points)

    final_points, _ = refine_with_gradient(best_points)

    # Normalize to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(final_points)

    return points_in_cube

# EVOLVE-BLOCK-END