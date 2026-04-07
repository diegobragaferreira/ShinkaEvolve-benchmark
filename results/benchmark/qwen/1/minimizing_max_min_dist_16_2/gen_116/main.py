# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical packing evolution approach.
    """
    
    np.random.seed(42)
    n_points = 16
    max_time = 180.0
    start_time = time.time()
    
    def spherical_fibonacci_points(n):
        """Generate n points on unit sphere using Fibonacci spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            # Latitude
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Longitude
            theta = phi * i
            
            # Convert to Cartesian coordinates
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def stereographic_project(points_3d):
        """Project 3D points to 2D using stereographic projection from south pole"""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        return np.array(points_2d)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        if len(points) < 2:
            return 0.0
            
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max <= 0:
                return 0.0
                
            return d_min / d_max
        except:
            return 0.0
    
    def mutate_individual(individual, mutation_rate, generation, max_generations):
        """Adaptive mutation that decreases over generations"""
        mutated = individual.copy()
        adaptive_rate = mutation_rate * (1 - generation/max_generations)
        
        for i in range(len(mutated)):
            if np.random.random() < adaptive_rate:
                # Add Gaussian noise with decreasing variance
                noise_magnitude = 0.05 * (1 - generation/max_generations)
                mutated[i] += np.random.normal(0, noise_magnitude)
                
        # Keep within bounds
        mutated = np.clip(mutated, 0, 1)
        return mutated
    
    def crossover_parents(parent1, parent2):
        """Uniform crossover between two parents"""
        child = parent1.copy()
        mask = np.random.random(len(parent1)) > 0.5
        child[mask] = parent2[mask]
        return child
    
    def evaluate_fitness(individual):
        """Evaluate fitness of individual (higher is better)"""
        points = individual.reshape(-1, 2)
        return compute_min_max_ratio(points)
    
    # Generate initial population using spherical approach
    population_size = 50
    population = []
    
    # Create initial individuals using spherical fibonacci points
    for _ in range(population_size):
        # Generate spherical points
        sph_points = spherical_fibonacci_points(n_points)
        
        # Project to 2D
        proj_points = stereographic_project(sph_points)
        
        # Normalize to [0,1] range
        if len(proj_points) > 0:
            x_min, y_min = np.min(proj_points, axis=0)
            x_max, y_max = np.max(proj_points, axis=0)
            
            if x_max > x_min and y_max > y_min:
                proj_points[:, 0] = (proj_points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                proj_points[:, 1] = (proj_points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        
        # Add small random perturbation
        noise = np.random.normal(0, 0.01, proj_points.shape)
        proj_points += noise
        proj_points = np.clip(proj_points, 0, 1)
        
        population.append(proj_points.flatten())
    
    # Evolutionary optimization
    generations = 100
    mutation_rate = 0.1
    elite_size = 5
    
    for generation in range(generations):
        if time.time() - start_time > max_time - 5:
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = population[:elite_size]
        
        # Create new population through selection and reproduction
        new_population = elite.copy()
        
        # Tournament selection and crossover
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            
            # Select another parent
            tournament_indices2 = np.random.choice(len(population), tournament_size)
            tournament_fitness2 = [fitness_scores[i] for i in tournament_indices2]
            winner_index2 = tournament_indices2[np.argmax(tournament_fitness2)]
            
            # Crossover
            child = crossover_parents(population[winner_index], population[winner_index2])
            
            # Mutate
            child = mutate_individual(child, mutation_rate, generation, generations)
            
            new_population.append(child)
        
        population = new_population
    
    # Final evaluation of best solution
    best_fitness = -np.inf
    best_individual = None
    
    for individual in population:
        fitness = evaluate_fitness(individual)
        if fitness > best_fitness:
            best_fitness = fitness
            best_individual = individual
    
    # Convert best individual to points
    if best_individual is not None:
        best_points = best_individual.reshape(-1, 2)
    else:
        # Fallback to random points if nothing found
        best_points = np.random.rand(n_points, 2)
    
    # Ensure all points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points

# EVOLVE-BLOCK-END
