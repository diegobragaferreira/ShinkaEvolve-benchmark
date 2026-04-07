# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import time

def compute_min_max_ratio(points):
    """Compute the minimum to maximum distance ratio for given points."""
    if len(points) < 2:
        return 0.0

    # Compute pairwise distances
    distances = cdist(points, points)

    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)

    # Find min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero
    if max_dist == 0:
        return 0.0

    return min_dist / max_dist

def generate_sphere_packing_initial(n_points=16):
    """Initialize points using a sphere-packing inspired approach with circular arrangement."""
    # Start with a circular arrangement of points
    points = []
    
    # Place points in a circle with some randomness to break symmetry
    center = np.array([0.5, 0.5])
    radius = 0.4
    
    # Generate points along a circle with slight perturbations
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        # Add some randomness to avoid perfect circular arrangement
        noise = np.random.normal(0, 0.02)
        x = center[0] + radius * np.cos(angle) + noise
        y = center[1] + radius * np.sin(angle) + noise
        points.append([x, y])
    
    # Ensure points are within bounds [0,1]
    points = np.clip(points, 0, 1)
    
    # Add small random perturbations to break any remaining symmetries
    for i in range(len(points)):
        points[i] += np.random.normal(0, 0.01, 2)
    
    points = np.clip(points, 0, 1)
    
    return points

def mutate_points(points, mutation_strength=0.02):
    """Apply geometric mutations to points."""
    mutated = points.copy()
    
    # Randomly select mutation type
    mutation_type = np.random.choice(['translate', 'scale', 'rotate'])
    
    if mutation_type == 'translate':
        # Translate all points
        delta = np.random.normal(0, mutation_strength, 2)
        mutated += delta
    
    elif mutation_type == 'scale':
        # Scale around center
        center = np.mean(mutated, axis=0)
        scale_factor = 1.0 + np.random.normal(0, mutation_strength * 0.5)
        mutated = center + (mutated - center) * scale_factor
        
    elif mutation_type == 'rotate':
        # Rotate around center
        center = np.mean(mutated, axis=0)
        angle = np.random.normal(0, mutation_strength * 2)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation
        translated = mutated - center
        rotated = translated @ rotation_matrix.T
        mutated = rotated + center
    
    # Ensure all points remain within bounds
    mutated = np.clip(mutated, 0, 1)
    return mutated

def evolve_points_population(initial_points, max_time=180):
    """Use evolutionary approach to optimize point distribution."""
    start_time = time.time()
    
    # Parameters for evolutionary algorithm
    population_size = 20
    generations = 100
    elite_size = 4
    
    # Create initial population
    population = []
    population.append(initial_points)  # Add original
    
    # Generate diverse initial members
    for i in range(population_size - 1):
        mutated = mutate_points(initial_points, 0.03)
        population.append(mutated)
    
    best_solution = None
    best_ratio = -np.inf
    
    # Evolutionary loop
    for gen in range(generations):
        if time.time() - start_time > max_time - 5:
            break
            
        # Evaluate fitness of each individual
        fitness_scores = []
        for individual in population:
            ratio = compute_min_max_ratio(individual)
            fitness_scores.append(ratio)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_solution = individual.copy()
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Create next generation
        new_population = population[:elite_size]  # Keep elites
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.randint(0, elite_size)
            parent2_idx = np.random.randint(0, elite_size)
            
            # Simple crossover: average parents
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Blend crossover
            alpha = np.random.random()
            child = parent1 * alpha + parent2 * (1 - alpha)
            
            # Add mutation
            child = mutate_points(child, 0.01)
            
            new_population.append(child)
            
        population = new_population[:population_size]
    
    return best_solution if best_solution is not None else population[0]

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Generate initial points using sphere packing inspired approach
    initial_points = generate_sphere_packing_initial(16)
    
    # Evolve the solution using genetic algorithm
    evolved_points = evolve_points_population(initial_points, max_time=170)
    
    return evolved_points

# EVOLVE-BLOCK-END