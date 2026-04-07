# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel hybrid approach combining spherical optimization with evolutionary strategies.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def fibonacci_sphere(samples=14):
        """Generate points distributed evenly on a sphere using Fibonacci method"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0
            
        # Calculate pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def project_to_unit_sphere(points):
        """Project points to unit sphere and normalize"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def initialize_population(size=50):
        """Initialize population with diverse strategies"""
        population = []
        
        # Strategy 1: Fibonacci sphere distribution
        fib_points = fibonacci_sphere(14)
        population.append(fib_points.copy())
        
        # Strategy 2: Random points on sphere
        random_points = np.random.randn(14, 3)
        random_points = project_to_unit_sphere(random_points)
        population.append(random_points.copy())
        
        # Strategy 3: Clustered points with some randomness
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        base_points = fibonacci_sphere(14)
        labels = kmeans.fit_predict(base_points)
        clustered_points = base_points.copy()
        for i in range(14):
            if np.random.random() < 0.3:  # 30% chance to jitter
                clustered_points[i] += np.random.normal(0, 0.05, 3)
        clustered_points = project_to_unit_sphere(clustered_points)
        population.append(clustered_points.copy())
        
        # Strategy 4: Perturbed Fibonacci with noise
        noisy_fib = fib_points + np.random.normal(0, 0.02, (14, 3))
        noisy_fib = project_to_unit_sphere(noisy_fib)
        population.append(noisy_fib.copy())
        
        # Fill remaining slots with random sphere points
        for _ in range(size - len(population)):
            rand_points = np.random.randn(14, 3)
            rand_points = project_to_unit_sphere(rand_points)
            population.append(rand_points.copy())
            
        return population
    
    def mutate_individual(individual, strength=0.05):
        """Mutate an individual with spherical constraints"""
        # Create a copy to avoid modifying the original
        mutated = individual.copy()
        
        # Select random indices to modify
        indices = np.random.choice(14, size=max(1, int(14 * 0.3)), replace=False)
        
        for idx in indices:
            # Add Gaussian noise
            noise = np.random.normal(0, strength, 3)
            mutated[idx] += noise
            
            # Project back to sphere
            norm = np.linalg.norm(mutated[idx])
            if norm > 0:
                mutated[idx] = mutated[idx] / norm
                
        return mutated
    
    def crossover(parent1, parent2, crossover_rate=0.7):
        """Perform crossover between two parents"""
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
            
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover
        crossover_point = np.random.randint(1, 14)
        
        child1[crossover_point:] = parent2[crossover_point:]
        child2[crossover_point:] = parent1[crossover_point:]
        
        # Project children back to sphere
        for child in [child1, child2]:
            for i in range(14):
                norm = np.linalg.norm(child[i])
                if norm > 0:
                    child[i] = child[i] / norm
                    
        return child1, child2
    
    def evaluate_population(population):
        """Evaluate fitness of entire population"""
        fitness_scores = []
        for individual in population:
            fitness = calculate_ratio(individual)
            fitness_scores.append(fitness)
        return np.array(fitness_scores)
    
    def select_parents(population, fitness_scores, num_parents=10):
        """Tournament selection for parent selection"""
        parents = []
        tournament_size = 3
        
        for _ in range(num_parents):
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = fitness_scores[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx].copy())
            
        return parents
    
    def optimize_on_sphere():
        """Main evolutionary optimization on unit sphere"""
        # Initialize population
        population = initialize_population(50)
        best_fitness = -np.inf
        best_individual = None
        
        # Evolutionary parameters
        generations = 100
        mutation_strength = 0.08
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = evaluate_population(population)
            
            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            
            # Select parents
            parents = select_parents(population, fitness_scores, 10)
            
            # Generate offspring
            offspring = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-5:]
            for idx in elite_indices:
                offspring.append(population[idx].copy())
            
            # Generate new offspring through crossover and mutation
            while len(offspring) < 50:
                # Select two random parents
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                
                # Crossover
                child1, child2 = crossover(parent1, parent2)
                
                # Mutate
                child1 = mutate_individual(child1, mutation_strength)
                child2 = mutate_individual(child2, mutation_strength)
                
                offspring.extend([child1, child2])
            
            # Trim offspring to exact population size
            population = offspring[:50]
            
            # Adaptive mutation strength
            if generation > 20 and generation % 10 == 0:
                improvement = fitness_scores.max() - best_fitness
                if improvement < 0.001:
                    mutation_strength *= 0.95  # Reduce mutation if stuck
        
        return best_individual, best_fitness
    
    # Main optimization loop with multi-start approach
    best_final_ratio = 0.0
    best_final_points = None
    
    # Try multiple random seeds for better exploration
    seeds = [42, 123, 456, 789]
    
    for seed in seeds:
        np.random.seed(seed)
        
        # Run evolutionary optimization on sphere
        optimized_points, ratio = optimize_on_sphere()
        
        # Refine using local search if we have a good candidate
        if ratio > best_final_ratio:
            best_final_ratio = ratio
            best_final_points = optimized_points.copy()
    
    # Final refinement: Local hill climbing on best solution
    if best_final_points is not None:
        current_points = best_final_points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # Simple hill climbing
        for _ in range(1000):
            # Try small perturbations to each point
            for i in range(14):
                # Save original
                original_point = current_points[i].copy()
                
                # Try small random perturbation
                perturbation = np.random.normal(0, 0.005, 3)
                current_points[i] += perturbation
                
                # Project back to sphere
                norm = np.linalg.norm(current_points[i])
                if norm > 0:
                    current_points[i] = current_points[i] / norm
                
                # Calculate new ratio
                new_ratio = calculate_ratio(current_points)
                
                # Accept only if improvement
                if new_ratio > current_ratio:
                    current_ratio = new_ratio
                else:
                    # Revert
                    current_points[i] = original_point
        
        best_final_points = current_points.copy()
        best_final_ratio = calculate_ratio(best_final_points)
    
    # Final check: If no good solution found, fallback to fibonacci
    if best_final_points is None or best_final_ratio < 0.1:
        fallback_points = fibonacci_sphere(14)
        fallback_points = (fallback_points - np.min(fallback_points, axis=0)) / \
                         (np.max(fallback_points, axis=0) - np.min(fallback_points, axis=0))
        fallback_points = fallback_points * 0.9 + 0.05
        best_final_points = fallback_points
    
    return best_final_points

# EVOLVE-BLOCK-END