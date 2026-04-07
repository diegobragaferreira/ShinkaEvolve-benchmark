# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time
from typing import Tuple
import math

def _fibonacci_sphere_points(n: int) -> np.ndarray:
    """Generate n points on a sphere using Fibonacci method."""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)

def _stereographic_project(points_3d: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using stereographic projection from south pole."""
    points_2d = []
    for x, y, z in points_3d:
        # Stereographic projection from south pole (0,0,-1)
        w = 1 / (1 + z)
        proj_x = x * w
        proj_y = y * w
        points_2d.append([proj_x, proj_y])
    
    return np.array(points_2d)

def _normalize_to_unit_square(points: np.ndarray) -> np.ndarray:
    """Normalize points to fit within [0,1] x [0,1]."""
    if len(points) == 0:
        return points
    
    # Find bounding box
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    
    # Avoid division by zero
    if x_max > x_min and y_max > y_min:
        points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
        points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
    
    return points

def _initialize_population(n_points: int, population_size: int = 20) -> list:
    """Initialize diverse population using spherical geometry."""
    population = []
    
    # Generate points on sphere and project to 2D
    sphere_points = _fibonacci_sphere_points(n_points)
    projected_points = _stereographic_project(sphere_points)
    normalized_points = _normalize_to_unit_square(projected_points)
    population.append(normalized_points.copy())
    
    # Add variations of this base configuration
    for i in range(population_size - 1):
        # Add small random perturbations
        perturbed = normalized_points + np.random.normal(0, 0.01, normalized_points.shape)
        # Clip to bounds
        perturbed = np.clip(perturbed, 0, 1)
        population.append(perturbed)
    
    return population

def _compute_distance_ratio(points: np.ndarray) -> float:
    """Compute the ratio of minimum to maximum distance between all point pairs."""
    if len(points) < 2:
        return 0.0
    
    try:
        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Handle case where all points might be coincident
        if max_dist == 0 or np.isinf(min_dist):
            return 0.0
        
        return min_dist / max_dist
    except Exception:
        return 0.0

def _evaluate_fitness(points: np.ndarray) -> float:
    """Evaluate fitness as negative of distance ratio (since we minimize for maximization)."""
    return -_compute_distance_ratio(points)

def _mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual by adding Gaussian noise to some points."""
    mutated = individual.copy()
    
    # Determine how many points to mutate
    n_points = len(individual)
    n_mutate = max(1, int(n_points * mutation_rate))
    
    # Select random points to mutate
    indices = np.random.choice(n_points, n_mutate, replace=False)
    
    # Apply mutations
    for idx in indices:
        # Add Gaussian noise
        noise = np.random.normal(0, 0.005, 2)
        mutated[idx] = individual[idx] + noise
        
        # Ensure bounds
        mutated[idx] = np.clip(mutated[idx], 0, 1)
    
    return mutated

def _crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Perform uniform crossover between two parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Apply crossover with given rate
    if np.random.random() < crossover_rate:
        # Randomly swap points between parents
        n_points = len(parent1)
        crossover_point = np.random.randint(1, n_points)
        
        child1[crossover_point:] = parent2[crossover_point:]
        child2[crossover_point:] = parent1[crossover_point:]
    
    return child1, child2

def _local_refinement(points: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Perform local refinement using gradient-based method."""
    def objective(x_flat):
        points_candidate = x_flat.reshape(-1, 2)
        return -_compute_distance_ratio(points_candidate)
    
    # Flatten points for optimization
    x0 = points.flatten()
    
    # Define bounds
    bounds = [(0, 1) for _ in range(len(x0))]
    
    try:
        # Use L-BFGS-B for refinement
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            return np.clip(refined_points, 0, 1)
    except:
        pass
    
    return points

def _evolutionary_search(initial_population: list, max_generations: int = 50) -> np.ndarray:
    """Perform evolutionary search on the population."""
    population = initial_population.copy()
    population_size = len(population)
    
    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitness_scores = [_evaluate_fitness(ind) for ind in population]
        
        # Sort by fitness (ascending since we minimize)
        sorted_indices = np.argsort(fitness_scores)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep top 50% (elitism)
        elite_count = population_size // 2
        elite_population = population[:elite_count]
        
        # Create new population through reproduction
        new_population = elite_population.copy()
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.randint(0, elite_count)
            parent2_idx = np.random.randint(0, elite_count)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child1, child2 = _crossover(parent1, parent2)
            
            # Mutation with adaptive rate
            mutation_rate = max(0.05, 0.2 * (1 - generation / max_generations))
            child1 = _mutate_individual(child1, mutation_rate)
            child2 = _mutate_individual(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
        
        # Early stopping condition - if we've converged
        if generation > 10:
            recent_improvement = abs(fitness_scores[0] - fitness_scores[-1])
            if recent_improvement < 1e-8:
                break
    
    # Return best individual
    fitness_scores = [_evaluate_fitness(ind) for ind in population]
    best_idx = np.argmin(fitness_scores)
    return population[best_idx]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    np.random.seed(42)
    
    # Time tracking
    start_time = time.time()
    
    # Initialize population using spherical geometry
    initial_population = _initialize_population(16, population_size=20)
    
    # Perform evolutionary search
    best_solution = _evolutionary_search(initial_population, max_generations=50)
    
    # Final local refinement
    refined_solution = _local_refinement(best_solution, max_iter=200)
    
    # Double-check the result
    final_ratio = _compute_distance_ratio(refined_solution)
    
    # If something went wrong, return the best we have
    if final_ratio <= 0:
        return initial_population[0]
    
    return refined_solution

# EVOLVE-BLOCK-END