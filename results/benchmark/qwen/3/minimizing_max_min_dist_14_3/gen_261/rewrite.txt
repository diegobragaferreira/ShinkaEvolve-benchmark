# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
import warnings
warnings.filterwarnings('ignore')

def fibonacci_sphere(n):
    """Generate n points evenly distributed on a unit sphere."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def normalize_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def project_to_cube(points):
    """Project points from sphere to unit cube [0,1]^3."""
    # Normalize to unit sphere first
    sphere_points = normalize_to_sphere(points)
    # Map to cube [0,1]^3
    return (sphere_points + 1) / 2

def spherical_voronoi_quality(sphere_points):
    """Calculate quality based on Voronoi cell areas on sphere."""
    if len(sphere_points) < 2:
        return 0
    try:
        sv = SphericalVoronoi(sphere_points)
        cell_areas = sv.calculate_areas()
        if len(cell_areas) > 0:
            mean_area = np.mean(cell_areas)
            if mean_area > 0:
                variance = np.var(cell_areas)
                # Return inverse of normalized variance (higher is better)
                return 1.0 / (1.0 + variance / mean_area**2)
    except Exception:
        pass
    return 0

def min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0
    distances = pdist(points)
    d_min = np.min(distances)
    d_max = np.max(distances)
    return d_min / d_max if d_max > 0 else 0

def generate_initial_population(pop_size=20):
    """Generate diverse initial population using multiple strategies."""
    population = []

    # Strategy 1: Fibonacci sphere
    points_fib = fibonacci_sphere(14)
    population.append(points_fib)

    # Strategy 2: Random points on sphere
    np.random.seed(42)
    points_random = np.random.randn(14, 3)
    points_random = normalize_to_sphere(points_random)
    population.append(points_random)

    # Strategy 3: Perturbed Fibonacci
    points_perturbed = points_fib + np.random.normal(0, 0.05, (14, 3))
    points_perturbed = normalize_to_sphere(points_perturbed)
    population.append(points_perturbed)

    # Fill remaining slots
    while len(population) < pop_size:
        np.random.seed(len(population) + 42)
        points = np.random.randn(14, 3)
        points = normalize_to_sphere(points)
        population.append(points)

    return population[:pop_size]

def spherical_evolution_operator(parents, mutation_rate=0.1):
    """Specialized evolution operator working in spherical space."""
    children = []
    for parent in parents:
        # Create offspring through spherical mutation
        child = parent.copy()

        # Apply spherical mutation - perturb along tangent plane then reproject
        for i in range(len(child)):
            if np.random.random() < mutation_rate:
                # Generate random displacement in tangent plane
                tangent_displacement = np.random.normal(0, 0.05, 3)

                # Ensure we maintain sphere constraint
                point = child[i]
                normal = point / np.linalg.norm(point)
                # Remove component parallel to normal (tangent plane)
                tangent_displacement = tangent_displacement - np.dot(tangent_displacement, normal) * normal

                # Apply displacement
                child[i] = point + tangent_displacement

        # Re-project to unit sphere
        child = normalize_to_sphere(child)
        children.append(child)
    return children

def tournament_selection(population, fitnesses, tournament_size=3):
    """Tournament selection for evolutionary algorithm."""
    selected = []
    for _ in range(len(population)):
        # Tournament selection
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_index].copy())
    return selected

def evaluate_individual(individual):
    """Evaluate individual using combined fitness function."""
    # Project to cube for distance calculations
    cube_points = project_to_cube(individual)
    ratio = min_max_ratio(cube_points)
    voronoi_quality = spherical_voronoi_quality(individual)
    # Combined fitness: prioritize min/max ratio but reward good distribution
    return ratio + 0.1 * voronoi_quality

def run_evolutionary_search(max_generations=50):
    """Run evolutionary search with spherical operators."""
    # Initialize population
    population = generate_initial_population()
    best_individual = None
    best_fitness = -np.inf

    # Track convergence for adaptive population sizing
    stagnation_count = 0
    max_stagnation = 10

    for generation in range(max_generations):
        # Evaluate population
        fitnesses = [evaluate_individual(ind) for ind in population]

        # Track best
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            stagnation_count = 0  # Reset stagnation counter on improvement
        else:
            stagnation_count += 1

        # Adaptive population sizing based on convergence
        current_pop_size = len(population)
        if stagnation_count > max_stagnation and current_pop_size < 50:
            # Increase population size to improve exploration
            population = generate_initial_population(int(current_pop_size * 1.5))
            stagnation_count = 0  # Reset stagnation counter

        # Selection
        selected = tournament_selection(population, fitnesses)

        # Create new population through crossover and mutation
        new_population = []

        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)

        # Generate offspring
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < len(selected) else selected[0]

            # Crossover (uniform)
            child1 = parent1.copy()
            child2 = parent2.copy()
            mask = np.random.random(14) < 0.5
            child1[mask] = parent2[mask]
            child2[mask] = parent1[mask]

            # Mutation
            child1 = spherical_evolution_operator([child1])[0]
            child2 = spherical_evolution_operator([child2])[0]

            new_population.extend([child1, child2])

        # Trim to original population size
        population = new_population[:len(population)]

    return best_individual

def local_refinement(points, max_iter=100):
    """Local refinement using L-BFGS-B."""
    try:
        # Convert to flat array for optimization
        x0 = points.flatten()

        def obj(x):
            points_refined = x.reshape((14, 3))
            # Ensure points are on sphere
            points_refined = normalize_to_sphere(points_refined)
            cube_points = project_to_cube(points_refined)
            distances = pdist(cube_points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                return -min_dist / max_dist
            else:
                return 0

        result = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            bounds=[(None, None)] * 42,  # No bounds for internal optimization
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': max_iter},
            tol=1e-12
        )

        refined_points = result.x.reshape((14, 3))
        refined_points = normalize_to_sphere(refined_points)
        return refined_points

    except Exception:
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Start with a good initial configuration using hybrid approach
    np.random.seed(42)
    
    # Try multiple initialization strategies and pick the best
    initial_candidates = []
    
    # Strategy 1: Fibonacci with slight perturbations
    fib_points = fibonacci_sphere(14)
    fib_perturbed = fib_points + np.random.normal(0, 0.03, (14, 3))
    fib_perturbed = normalize_to_sphere(fib_perturbed)
    initial_candidates.append(fib_perturbed)
    
    # Strategy 2: Pure random on sphere
    random_points = np.random.randn(14, 3)
    random_points = normalize_to_sphere(random_points)
    initial_candidates.append(random_points)
    
    # Strategy 3: Perturbed Fibonacci with larger variance
    large_perturbed = fib_points + np.random.normal(0, 0.07, (14, 3))
    large_perturbed = normalize_to_sphere(large_perturbed)
    initial_candidates.append(large_perturbed)
    
    # Evaluate initial candidates
    best_initial = initial_candidates[0]
    best_ratio = min_max_ratio(project_to_cube(initial_candidates[0]))
    
    for candidate in initial_candidates[1:]:
        ratio = min_max_ratio(project_to_cube(candidate))
        if ratio > best_ratio:
            best_ratio = ratio
            best_initial = candidate
    
    # Main evolutionary process
    best_spherical_points = run_evolutionary_search(max_generations=50)

    # Final local refinement
    refined_points = local_refinement(best_spherical_points)

    # Convert to cube coordinates and ensure valid bounds
    final_points = project_to_cube(refined_points)
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END