# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import math
import random
from numba import jit
import time

@jit(nopython=True)
def compute_min_max_ratio_numba(points):
    """Fast computation of min/max distance ratio using numba"""
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
    
    if max_dist == 0:
        return 0.0
    return min_dist / max_dist

def fibonacci_sphere(n: int, seed: int = 42) -> np.ndarray:
    """Generate n points distributed approximately uniformly on a sphere using Fibonacci spiral method."""
    np.random.seed(seed)
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms[:, np.newaxis]

def voronoi_uniformity_score(points):
    """Calculate Voronoi uniformity score based on cell area variance."""
    try:
        sv = SphericalVoronoi(points)
        areas = sv.calculate_areas()
        if len(areas) == 0:
            return 0.0
        # Lower variance = more uniform distribution
        return 1.0 / (1e-10 + np.var(areas))
    except:
        return 0.0

def combined_fitness(points, ratio_weight=1.0, uniformity_weight=0.3):
    """Combined fitness function using both distance ratio and Voronoi uniformity."""
    ratio = compute_min_max_ratio_numba(points)
    uniformity = voronoi_uniformity_score(points)
    return ratio_weight * ratio + uniformity_weight * uniformity

def adaptive_crossover(parent1, parent2, diversity_score):
    """Adaptive crossover that varies based on population diversity."""
    if diversity_score < 0.3:  # Low diversity - more exploratory crossover
        crossover_rate = 0.9
    elif diversity_score < 0.6:  # Medium diversity - balanced crossover
        crossover_rate = 0.7
    else:  # High diversity - more exploitative crossover
        crossover_rate = 0.4
    
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    # Uniform crossover
    num_points = parent1.shape[0]
    crossover_point = random.randint(1, num_points-1)
    
    child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
    
    return child1, child2

def adaptive_mutation(individual, diversity_score, generation):
    """Adaptive mutation that varies based on diversity and generation."""
    if diversity_score < 0.3:  # Low diversity - high mutation rate
        mutation_rate = 0.2
        mutation_strength = 0.08
    elif diversity_score < 0.6:  # Medium diversity - moderate mutation
        mutation_rate = 0.1
        mutation_strength = 0.05
    else:  # High diversity - low mutation rate
        mutation_rate = 0.05
        mutation_strength = 0.02
    
    # Adjust based on generation (decrease over time)
    generation_factor = max(0.1, 1.0 - generation / 1000.0)
    mutation_rate *= generation_factor
    mutation_strength *= generation_factor
    
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random perturbation
            delta = np.random.normal(0, mutation_strength, 3)
            mutated[i] += delta
            
            # Project back to sphere
            mutated[i] = project_to_sphere(mutated[i:i+1])[0]
    
    return mutated

def evaluate_diversity(population):
    """Evaluate population diversity using standard deviation of pairwise distances."""
    if len(population) < 2:
        return 0.0
    
    distances = []
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            dist = np.linalg.norm(population[i] - population[j])
            distances.append(dist)
    
    if not distances:
        return 0.0
    
    return np.std(distances) / (np.mean(distances) + 1e-10)

def multi_resolution_initialization(n: int = 14, seed: int = 42):
    """Generate multiple initial configurations at different resolutions."""
    initial_configs = []
    
    # Different initialization strategies
    strategies = [
        ("fibonacci", fibonacci_sphere(n, seed)),
        ("random", np.random.uniform(-1, 1, (n, 3))),
        ("icosahedron", generate_icosahedron_points(n, seed))
    ]
    
    for name, points in strategies:
        # Apply some perturbation to avoid perfect symmetry
        points = points + 0.01 * np.random.randn(*points.shape)
        points = project_to_sphere(points)
        initial_configs.append(points)
    
    return initial_configs

def generate_icosahedron_points(n: int, seed: int = 42):
    """Generate points resembling an icosahedron."""
    np.random.seed(seed)
    # Icosahedron vertices scaled to unit sphere
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    points = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    # Normalize
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    # Return n points (either originals or interpolated)
    if n <= 12:
        return points[:n]
    else:
        # For more than 12 points, sample and add extra points
        # We'll create additional points by averaging and normalizing
        extra_points = []
        for _ in range(n - 12):
            # Sample two random points and create midpoint
            idx1, idx2 = random.sample(range(12), 2)
            midpoint = (points[idx1] + points[idx2]) / 2
            norm = np.linalg.norm(midpoint)
            if norm > 0:
                midpoint = midpoint / norm
            extra_points.append(midpoint)
        
        return np.vstack([points, extra_points])

def local_geometric_refinement(points, max_iter=50):
    """Refine points using geometric optimization techniques."""
    current_points = points.copy()
    best_points = current_points.copy()
    best_ratio = compute_min_max_ratio_numba(current_points)
    
    for iteration in range(max_iter):
        improved = False
        
        # Try to improve each point individually
        for i in range(len(current_points)):
            old_point = current_points[i].copy()
            old_ratio = compute_min_max_ratio_numba(current_points)
            
            # Estimate gradient for point i using finite differences
            grad = np.zeros(3)
            eps = 1e-5
            
            for j in range(3):
                test_points = current_points.copy()
                test_points[i, j] += eps
                # Project to sphere
                test_points[i] = project_to_sphere(test_points[i:i+1])[0]
                new_ratio = compute_min_max_ratio_numba(test_points)
                grad[j] = (new_ratio - old_ratio) / eps
            
            # Move in gradient direction
            if np.linalg.norm(grad) > 1e-10:
                current_points[i] = current_points[i] + 0.01 * grad
                # Project back to sphere
                current_points[i] = project_to_sphere(current_points[i:i+1])[0]
                improved = True
        
        # If no improvement, stop early
        if not improved:
            break
            
        # Update best solution
        new_ratio = compute_min_max_ratio_numba(current_points)
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_points = current_points.copy()
    
    return best_points, best_ratio

def spherical_voronoi_evolution() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid evolutionary algorithm with adaptive operators and multi-resolution approach.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Multi-resolution initialization
    initial_configs = multi_resolution_initialization(14, 42)
    
    # Best solution tracking
    best_solution = None
    best_ratio = 0.0
    
    # Try all initial configurations
    for i, initial_points in enumerate(initial_configs):
        # Run evolutionary optimization for each initialization
        config_best_sol, config_best_ratio = evolutionary_optimization(initial_points, f"config_{i}")
        
        if config_best_ratio > best_ratio:
            best_ratio = config_best_ratio
            best_solution = config_best_sol.copy()
    
    # Final refinement of best solution
    if best_solution is not None:
        final_points, final_ratio = local_geometric_refinement(best_solution, 100)
        if final_ratio > best_ratio:
            best_solution = final_points
            best_ratio = final_ratio
    
    return best_solution if best_solution is not None else fibonacci_sphere(14, 42)

def evolutionary_optimization(initial_points, config_name="default"):
    """Main evolutionary optimization loop with adaptive operators."""
    # Evolutionary parameters
    population_size = 25
    generations = 1000
    elite_size = 5
    max_time = 360  # seconds
    
    start_time = time.time()
    
    # Initialize population
    population = []
    for i in range(population_size):
        if i == 0:
            # First individual from given initialization
            individual = initial_points.copy()
        else:
            # Create variants with small perturbations
            individual = initial_points.copy()
            individual += 0.01 * np.random.randn(*individual.shape)
            individual = project_to_sphere(individual)
        population.append(individual)
    
    best_solution = None
    best_fitness = -np.inf
    best_ratio = 0.0
    
    # Main evolution loop
    for generation in range(generations):
        if time.time() - start_time > max_time:
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = []
        ratios = []
        
        for individual in population:
            fitness = combined_fitness(individual)
            ratio = compute_min_max_ratio_numba(individual)
            fitness_scores.append(fitness)
            ratios.append(ratio)
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_ratio = ratios[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()
        
        # Print progress periodically
        if generation % 100 == 0:
            print(f"Gen {generation}: Best ratio = {best_ratio:.6f}")
        
        # Calculate population diversity
        diversity = evaluate_diversity(population)
        
        # Create new population
        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_indices]
        
        # Selection, crossover, and mutation
        # Tournament selection
        selected_parents = []
        for _ in range(population_size - elite_size):
            tournament_indices = random.sample(range(len(population)), 3)
            tournament_fitnesses = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected_parents.append(population[winner_index])
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = adaptive_crossover(parent1, parent2, diversity)
            
            child1 = adaptive_mutation(child1, diversity, generation)
            child2 = adaptive_mutation(child2, diversity, generation)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    return best_solution, best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    return spherical_voronoi_evolution()

# EVOLVE-BLOCK-END