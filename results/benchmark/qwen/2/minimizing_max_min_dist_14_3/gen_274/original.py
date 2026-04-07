# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import cdist
from numba import jit
import time
import random
from copy import deepcopy

@jit(nopython=True)
def compute_min_max_ratio_numba(points):
    """Optimized distance computation using numba"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0

    for i in range(n):
        for j in range(i+1, n):
            # Compute squared distance to avoid sqrt computation
            dist_sq = (points[i,0]-points[j,0])**2 + (points[i,1]-points[j,1])**2 + (points[i,2]-points[j,2])**2
            dist = np.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist

    return min_dist, max_dist

def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n points distributed as evenly as possible on a unit sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def compute_min_max_ratio(points: np.ndarray) -> tuple:
    """Compute the minimum and maximum distances between all pairs of points, and return their ratio."""
    if len(points) < 2:
        return 0.0, 0.0, 0.0

    # Use numba-optimized version
    min_distance, max_distance = compute_min_max_ratio_numba(points)

    # Avoid division by zero
    if max_distance == 0:
        ratio = 0.0
    else:
        ratio = min_distance / max_distance

    return min_distance, max_distance, ratio

def project_to_unit_sphere(points):
    """Project points to the unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Handle case where norm might be zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def generate_voronoi_fitness(points):
    """Calculate fitness based on Voronoi diagram properties"""
    try:
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points, radius=1.0)

        # Calculate Voronoi cell areas
        cell_areas = sv.calculate_areas()

        # Fitness: penalize large variance in cell areas (more uniform distribution)
        area_variance = np.var(cell_areas)

        # Also consider distance distribution
        min_dist, max_dist, _ = compute_min_max_ratio(points)
        if max_dist == 0:
            distance_ratio = 0.0
        else:
            distance_ratio = min_dist / max_dist

        # Combined fitness: balance uniformity and distance ratio
        # Lower area variance + higher distance ratio = better fitness
        fitness = distance_ratio - 0.1 * area_variance

        return fitness, distance_ratio
    except:
        # Fallback to simple distance ratio if Voronoi fails
        min_dist, max_dist, ratio = compute_min_max_ratio(points)
        return ratio, ratio

def create_individual(num_points=14):
    """Create a random individual (point configuration) on unit sphere"""
    points = np.random.randn(num_points, 3)
    points = project_to_unit_sphere(points)
    return points

def adaptive_crossover(parent1, parent2, crossover_rate=0.8):
    """Adaptive crossover that considers point proximity and distribution"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Analyze point distributions to make smarter crossover decisions
    distances1 = cdist(parent1, parent1)
    distances2 = cdist(parent2, parent2)

    # Compute average distances for each individual
    avg_dist1 = np.mean(distances1[distances1 > 0]) if np.any(distances1 > 0) else 1.0
    avg_dist2 = np.mean(distances2[distances2 > 0]) if np.any(distances2 > 0) else 1.0

    # Choose crossover strategy based on distribution characteristics
    if avg_dist1 < avg_dist2 * 0.8:  # parent1 is more clustered
        # Prefer more spread out parent for first half
        crossover_point = random.randint(int(len(parent1) * 0.3), int(len(parent1) * 0.7))
        child1 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
        child2 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    elif avg_dist2 < avg_dist1 * 0.8:  # parent2 is more clustered
        # Prefer more spread out parent for first half
        crossover_point = random.randint(int(len(parent1) * 0.3), int(len(parent1) * 0.7))
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
    else:
        # Standard crossover
        crossover_point = random.randint(1, len(parent1)-1)
        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

    # Project back to unit sphere
    child1 = project_to_unit_sphere(child1)
    child2 = project_to_unit_sphere(child2)

    return child1, child2

def intelligent_mutation(individual, mutation_rate=0.1, base_strength=0.05):
    """Mutation operator that intelligently targets under-separated regions"""
    mutated = individual.copy()

    # Analyze current distribution
    distances = cdist(mutated, mutated)
    np.fill_diagonal(distances, np.inf)

    # Find points that are too close to others (potential bottlenecks)
    min_distances = np.min(distances, axis=1)
    avg_min_dist = np.mean(min_distances)

    # Identify under-separated points (those with below-average minimum distance)
    under_separated_mask = min_distances < avg_min_dist * 0.7

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Increase mutation strength for under-separated points
            strength = base_strength
            if under_separated_mask[i]:
                strength *= 2.0  # Double strength for clustered points

            # Add small random perturbation
            delta = np.random.normal(0, strength, 3)
            mutated[i] += delta

            # Project back to unit sphere
            mutated[i] = project_to_unit_sphere(mutated[i].reshape(1, 3)).flatten()

    return mutated

def diversity_preservation(population, fitness_scores, population_size):
    """Maintain diversity by checking Voronoi uniformity and reintroducing variety"""
    # Calculate Voronoi uniformity scores for population
    uniformity_scores = []
    for individual in population:
        try:
            sv = SphericalVoronoi(individual)
            areas = sv.calculate_areas()
            # Standard deviation of areas as measure of uniformity
            uniformity = np.std(areas) if len(areas) > 1 else 0.0
            uniformity_scores.append(uniformity)
        except:
            uniformity_scores.append(np.inf)

    # If population has low diversity (high uniformity variance), add fresh individuals
    if len(uniformity_scores) > 1:
        std_uniformity = np.std(uniformity_scores)
        mean_uniformity = np.mean(uniformity_scores)

        # If diversity is low, inject some random individuals
        if std_uniformity < 0.1 * mean_uniformity and mean_uniformity > 0:
            # Replace worst performers with random individuals
            worst_indices = np.argsort(fitness_scores)[:len(population)//4]
            for idx in worst_indices:
                if random.random() < 0.3:  # 30% chance to replace
                    population[idx] = create_individual(14)

    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select individuals using tournament selection"""
    selected_indices = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected_indices.append(winner_index)

    return [population[i] for i in selected_indices]

def spherical_voronoi_evolution() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses evolutionary algorithm with Voronoi-based fitness evaluation.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Evolutionary parameters
    population_size = 40
    generations = 1500
    elite_size = 8
    initial_mutation_rate = 0.15
    initial_crossover_rate = 0.8

    # Adaptive parameters tracking
    convergence_window = 50
    convergence_threshold = 1e-6
    best_fitness_history = []

    # Different initialization strategies
    def initialize_population():
        population = []
        strategies = [
            ("fibonacci", lambda: fibonacci_sphere(14)),
            ("icosahedron", lambda: initialize_icosahedron_points(14)),
            ("random_sphere", lambda: create_individual(14)),
        ]

        # Create population with different initialization strategies
        for i in range(population_size):
            strategy_name, initializer = strategies[i % len(strategies)]
            points = initializer()
            population.append(points)
        return population

    def initialize_icosahedron_points(n):
        """Initialize points based on regular icosahedron vertices"""
        # Regular icosahedron vertices (normalized)
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1, 0, phi], [1, 0, phi], [-1, 0, -phi], [1, 0, -phi],
            [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
            [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]
        ])
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])

        # For 14 points, use 12 vertices plus 2 additional points
        if n == 14:
            # Add two more points for better coverage
            additional_points = []
            additional_points.append([0, 0, 1])  # North pole
            additional_points.append([0, 0, -1])  # South pole
            return np.vstack([vertices, additional_points])
        elif n == 12:
            return vertices
        else:
            return fibonacci_sphere(n)

    # Initialize population
    population = initialize_population()

    best_solution = None
    best_fitness = -np.inf
    best_ratio = 0.0

    # Main evolution loop
    for generation in range(generations):
        # Adaptive parameters based on convergence
        if len(best_fitness_history) >= convergence_window:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-convergence_window]
            if abs(recent_improvement) < convergence_threshold:
                # Slow convergence, increase exploration
                mutation_rate = min(0.3, initial_mutation_rate * 1.2)
                crossover_rate = min(0.9, initial_crossover_rate * 1.1)
            else:
                # Fast convergence, increase exploitation
                mutation_rate = initial_mutation_rate * 0.8
                crossover_rate = initial_crossover_rate * 0.9
        else:
            mutation_rate = initial_mutation_rate
            crossover_rate = initial_crossover_rate

        # Evaluate fitness for all individuals
        fitness_scores = []
        ratios = []

        for individual in population:
            fitness, ratio = generate_voronoi_fitness(individual)
            fitness_scores.append(fitness)
            ratios.append(ratio)

        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_ratio = ratios[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Store fitness history for convergence detection
        best_fitness_history.append(best_fitness)

        # Print progress
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, Ratio = {best_ratio:.6f}")

        # Create new population
        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_indices]

        # Selection with diversity consideration
        # Better selection mechanism
        scores_array = np.array(fitness_scores)
        # Convert to positive values for selection
        normalized_scores = scores_array - np.min(scores_array) + 1e-10
        probabilities = normalized_scores / np.sum(normalized_scores)
        selected_indices = np.random.choice(len(population), size=len(population)-elite_size, p=probabilities)
        selected_parents = [population[i] for i in selected_indices]

        # Generate offspring with adaptive operators
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = adaptive_crossover(parent1, parent2, crossover_rate)

            child1 = intelligent_mutation(child1, mutation_rate, 0.03)
            child2 = intelligent_mutation(child2, mutation_rate, 0.03)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:population_size]

        # Apply diversity preservation
        population = diversity_preservation(population, fitness_scores, population_size)

        # Diversity maintenance - periodically introduce new random individuals
        if generation % 200 == 0 and generation > 0:
            for i in range(2):
                if len(population) < population_size:
                    # Replace worst individuals with random ones
                    worst_idx = np.argmin(fitness_scores)
                    population[worst_idx] = create_individual(14)

    # Final refinement using multiple local optimizations
    if best_solution is not None:
        # Try to improve the best solution further with local search
        try:
            from scipy.optimize import minimize

            def objective(x_flat):
                points_test = x_flat.reshape(-1, 3)
                min_dist, max_dist, _ = compute_min_max_ratio(points_test)
                if max_dist == 0:
                    return 0.0
                return -min_dist / max_dist  # Negative because we minimize

            def constraint_func(x_flat):
                points_test = x_flat.reshape(-1, 3)
                norms = np.linalg.norm(points_test, axis=1)
                return norms - 1.0  # Should equal zero for unit sphere

            # Multiple restarts for better optimization
            best_final_points = best_solution.copy()
            best_final_ratio = best_ratio

            for restart in range(3):
                # Add small random perturbation to starting point
                x0 = best_solution.flatten()
                noise = np.random.normal(0, 0.01, len(x0))
                x0 = x0 + noise

                cons = {'type': 'eq', 'fun': constraint_func}

                result = minimize(objective, x0, method='SLSQP', constraints=cons,
                                 options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9})

                if result.success:
                    refined_points = result.x.reshape(-1, 3)
                    _, _, final_ratio = compute_min_max_ratio(refined_points)
                    if final_ratio > best_final_ratio:
                        best_final_ratio = final_ratio
                        best_final_points = refined_points

            best_solution = best_final_points
        except:
            pass

    return best_solution if best_solution is not None else fibonacci_sphere(14)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Run the evolutionary algorithm
    points = spherical_voronoi_evolution()

    # Ensure normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    points = points / norms

    return points

# EVOLVE-BLOCK-END