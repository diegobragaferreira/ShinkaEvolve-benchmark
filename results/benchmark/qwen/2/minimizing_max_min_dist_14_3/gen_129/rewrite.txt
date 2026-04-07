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

def crossover(parent1, parent2, crossover_rate=0.8):
    """Crossover operator for spherical point sets"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    # Simple crossover: mix points from both parents
    num_points = parent1.shape[0]
    crossover_point = random.randint(1, num_points-1)
    
    child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])
    
    # Project back to unit sphere
    child1 = project_to_unit_sphere(child1)
    child2 = project_to_unit_sphere(child2)
    
    return child1, child2

def mutate(individual, mutation_rate=0.1, strength=0.05):
    """Mutation operator for spherical point sets"""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add small random perturbation
            delta = np.random.normal(0, strength, 3)
            mutated[i] += delta
            
            # Project back to unit sphere
            mutated[i] = project_to_unit_sphere(mutated[i].reshape(1, 3)).flatten()
    
    return mutated

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
    population_size = 30
    generations = 1000
    elite_size = 5
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize population with Fibonacci sphere and random configurations
    population = []
    for i in range(population_size):
        if i % 2 == 0:
            # Fibonacci initialization
            points = fibonacci_sphere(14)
        else:
            # Random initialization
            points = create_individual(14)
        population.append(points)
    
    best_solution = None
    best_fitness = -np.inf
    best_ratio = 0.0
    
    # Main evolution loop
    for generation in range(generations):
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
        
        # Print progress
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, Ratio = {best_ratio:.6f}")
        
        # Create new population
        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_indices]
        
        # Selection, crossover, and mutation
        selected_parents = tournament_selection(population, fitness_scores)
        
        # Generate offspring
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            
            child1 = mutate(child1, mutation_rate, 0.03)
            child2 = mutate(child2, mutation_rate, 0.03)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Final refinement using local optimization
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
            
            x0 = best_solution.flatten()
            cons = {'type': 'eq', 'fun': constraint_func}
            
            result = minimize(objective, x0, method='SLSQP', constraints=cons,
                             options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                _, _, final_ratio = compute_min_max_ratio(refined_points)
                if final_ratio > best_ratio:
                    best_solution = refined_points
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