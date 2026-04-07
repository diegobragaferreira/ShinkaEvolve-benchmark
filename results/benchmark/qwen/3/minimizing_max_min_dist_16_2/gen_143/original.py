# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    np.random.seed(42)
    
    # Initialize points using a hexagonal grid pattern for better starting configuration
    def initialize_hexagonal_grid():
        # Create a 4x4 grid with alternating rows offset
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points.append([x, y])
        return np.array(points)
    
    # Compute min/max distance ratio with Voronoi-based quality measure
    def compute_fitness(points):
        if len(points) < 2:
            return 0
        
        # Calculate all pairwise distances
        distances = cdist(points, points)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0
            
        # Additional quality metric based on Voronoi cell areas
        try:
            vor = Voronoi(points)
            # Calculate average Voronoi cell area
            areas = []
            for region in vor.regions:
                if not any(v == -1 for v in region):  # Skip infinite regions
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) >= 3:
                        # Simple area calculation for convex polygons
                        area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] - 
                                           polygon[(i+1)%len(polygon)][0] * polygon[i][1] 
                                           for i in range(len(polygon))))
                        areas.append(area)
            avg_area = np.mean(areas) if areas else 0
        except:
            avg_area = 0
        
        # Combined fitness: min/max ratio weighted with Voronoi uniformity
        ratio = d_min / d_max
        uniformity_factor = avg_area / (1.0/16.0) if avg_area > 0 else 0
        return ratio * (1 + 0.5 * uniformity_factor)
    
    # Evolutionary operators
    def mutate_individual(points, mutation_strength=0.02):
        new_points = points.copy()
        # Select random point to mutate
        idx = np.random.randint(len(points))
        # Apply small random displacement
        new_points[idx] += np.random.normal(0, mutation_strength, 2)
        # Clamp to bounds
        new_points = np.clip(new_points, 0, 1)
        return new_points
    
    def crossover(parent1, parent2, crossover_rate=0.7):
        if np.random.rand() > crossover_rate:
            return parent1.copy()
        
        # Uniform crossover
        mask = np.random.rand(*parent1.shape) > 0.5
        child = np.where(mask, parent1, parent2).copy()
        return child
    
    # Initialize population
    population_size = 50
    population = []
    
    # Create initial population with various starting configurations
    for _ in range(population_size):
        # Start with hexagonal grid
        points = initialize_hexagonal_grid()
        # Add some randomness
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        population.append(points)
    
    # Evolution parameters
    generations = 200
    elite_size = 5
    tournament_size = 3
    
    best_fitness = -np.inf
    best_individual = None
    
    # Evolutionary loop
    for generation in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness = compute_fitness(individual)
            fitness_scores.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(elite_size):
            new_population.append(population[i].copy())
        
        # Generate offspring through tournament selection and crossover
        while len(new_population) < population_size:
            # Tournament selection
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            # Select second parent
            tournament_indices2 = np.random.choice(len(population), tournament_size)
            tournament_fitness2 = [fitness_scores[i] for i in tournament_indices2]
            winner_idx2 = tournament_indices2[np.argmax(tournament_fitness2)]
            
            # Crossover
            child = crossover(population[winner_idx], population[winner_idx2])
            
            # Mutation
            child = mutate_individual(child, 0.01)
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    # Final refinement using local search around best solution
    if best_individual is not None:
        # Use gradient-based refinement for final optimization
        points = best_individual.copy()
        
        # Simple gradient ascent with adaptive step sizes
        for _ in range(1000):
            current_fitness = compute_fitness(points)
            
            # Estimate gradient by finite differences
            gradient = np.zeros_like(points)
            eps = 1e-4
            
            for i in range(len(points)):
                for j in range(2):
                    # Perturb point coordinate
                    points_plus = points.copy()
                    points_plus[i, j] += eps
                    points_plus = np.clip(points_plus, 0, 1)
                    
                    points_minus = points.copy()
                    points_minus[i, j] -= eps
                    points_minus = np.clip(points_minus, 0, 1)
                    
                    fitness_plus = compute_fitness(points_plus)
                    fitness_minus = compute_fitness(points_minus)
                    
                    gradient[i, j] = (fitness_plus - fitness_minus) / (2 * eps)
            
            # Update points
            step_size = 0.01
            points = points + step_size * gradient
            
            # Ensure bounds
            points = np.clip(points, 0, 1)
            
            # Check for convergence
            if np.all(np.abs(gradient) < 1e-6):
                break
        
        # Final evaluation
        final_fitness = compute_fitness(points)
        if final_fitness > best_fitness:
            best_individual = points
    
    return best_individual if best_individual is not None else initialize_hexagonal_grid()

# EVOLVE-BLOCK-END
