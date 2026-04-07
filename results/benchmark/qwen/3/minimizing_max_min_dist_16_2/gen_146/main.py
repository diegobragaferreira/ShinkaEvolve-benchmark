# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from typing import Tuple

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0

        # Compute pairwise distances efficiently
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0

        return min_dist / max_dist
    
    def create_grid_initialization():
        """Create a structured grid-based initialization with enhanced diversity"""
        # Use a 4x4 grid with optimized spacing
        points = []
        
        # Create a more sophisticated grid with non-uniform spacing that promotes better distribution
        # This creates a more natural hexagonal-like structure
        for i in range(4):
            for j in range(4):
                # Create a staggered grid pattern with varying offsets
                x = j * 0.3333 + (i % 2) * 0.1667  # Staggered x positions
                y = i * 0.3333
                
                # Add structured perturbations based on position
                # This creates a more diverse initial population
                pos_factor = (i * 7 + j * 3) % 10
                noise_x = (pos_factor - 5) * 0.005
                noise_y = (pos_factor - 5) * 0.005
                
                x += noise_x + np.random.normal(0, 0.01)
                y += noise_y + np.random.normal(0, 0.01)
                
                points.append([x, y])
        
        points = np.array(points)
        
        # Normalize to [0,1] x [0,1] ensuring proper scaling
        x_range = np.ptp(points[:, 0])
        y_range = np.ptp(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
        
        # Ensure all points are within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)
        
        return points
    
    def create_diverse_initial_populations(n_individuals: int = 50) -> list:
        """Create a diverse initial population for evolutionary optimization"""
        populations = []
        
        # 1. Grid-based initialization (primary)
        grid_pop = create_grid_initialization()
        populations.append(grid_pop)
        
        # 2. Random population with controlled density
        for _ in range(n_individuals - 1):
            # Create random points but ensure they don't cluster too much
            points = np.random.rand(16, 2)
            
            # Apply some structure to avoid extreme clustering
            # Add small perturbations to ensure minimal overlap
            for i in range(16):
                points[i, 0] += np.random.normal(0, 0.005)
                points[i, 1] += np.random.normal(0, 0.005)
                
            # Keep within bounds
            points[:, 0] = np.clip(points[:, 0], 0, 1)
            points[:, 1] = np.clip(points[:, 1], 0, 1)
            
            populations.append(points)
        
        return populations
    
    def fitness_function(population: np.ndarray) -> np.ndarray:
        """Calculate fitness (min/max ratio) for entire population"""
        fitness_scores = []
        for points in population:
            score = calculate_ratio(points)
            fitness_scores.append(score)
        return np.array(fitness_scores)
    
    def crossover(parent1: np.ndarray, parent2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend crossover operation preserving spatial relationships"""
        # Create offspring using arithmetic crossover
        child = alpha * parent1 + (1 - alpha) * parent2
        
        # Apply boundary correction
        child[:, 0] = np.clip(child[:, 0], 0, 1)
        child[:, 1] = np.clip(child[:, 1], 0, 1)
        
        return child
    
    def mutate(individual: np.ndarray, mutation_rate: float = 0.1, sigma: float = 0.02) -> np.ndarray:
        """Apply mutation with spatial awareness"""
        mutated = individual.copy()
        
        # For each point, decide whether to mutate
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Add Gaussian noise with spatial correlation
                noise_x = np.random.normal(0, sigma)
                noise_y = np.random.normal(0, sigma)
                
                # Apply to this point
                mutated[i, 0] += noise_x
                mutated[i, 1] += noise_y
                
                # Boundary handling with reflection
                if mutated[i, 0] < 0:
                    mutated[i, 0] = -mutated[i, 0]
                elif mutated[i, 0] > 1:
                    mutated[i, 0] = 2 - mutated[i, 0]
                    
                if mutated[i, 1] < 0:
                    mutated[i, 1] = -mutated[i, 1]
                elif mutated[i, 1] > 1:
                    mutated[i, 1] = 2 - mutated[i, 1]
        
        return mutated
    
    def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, 
                           tournament_size: int = 3) -> np.ndarray:
        """Select individuals using tournament selection"""
        selected_indices = []
        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitnesses = fitnesses[tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected_indices.append(winner_index)
        
        return population[selected_indices]
    
    def evolutionary_optimization(max_generations: int = 1000, 
                                population_size: int = 50) -> Tuple[np.ndarray, float]:
        """Main evolutionary optimization loop"""
        
        # Initialize population
        population = create_diverse_initial_populations(population_size)
        
        best_individual = None
        best_fitness = -np.inf
        generation_best_fitnesses = []
        
        for generation in range(max_generations):
            # Evaluate fitness
            fitnesses = fitness_function(population)
            
            # Track best solution
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
            
            generation_best_fitnesses.append(current_best_fitness)
            
            # Selection
            selected_population = tournament_selection(population, fitnesses)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best_individual.copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Select two parents
                parent1_idx = np.random.randint(len(selected_population))
                parent2_idx = np.random.randint(len(selected_population))
                
                parent1 = selected_population[parent1_idx]
                parent2 = selected_population[parent2_idx]
                
                # Crossover
                if np.random.random() < 0.8:  # 80% crossover probability
                    child = crossover(parent1, parent2)
                else:
                    # Clone one parent
                    child = parent1.copy()
                
                # Mutation
                child = mutate(child, mutation_rate=0.1, sigma=0.01)
                
                new_population.append(child)
            
            # Trim to exact population size
            population = new_population[:population_size]
            
            # Adapt mutation rate based on generation
            if generation > max_generations // 2:
                # Decrease mutation rate in later generations
                mutation_rate = 0.05
            else:
                mutation_rate = 0.1
        
        return best_individual, best_fitness
    
    # Run evolutionary optimization
    np.random.seed(42)
    
    # Run optimization with reduced iterations since we're using a more efficient approach
    best_solution, best_score = evolutionary_optimization(
        max_generations=800, 
        population_size=40
    )
    
    # Perform final refinement with local search
    def local_refinement(points: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
        """Perform local refinement around the best solution"""
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # Simulated annealing with very low temperature for fine tuning
        T = 0.001
        cooling_rate = 0.9999
        
        for _ in range(max_iterations):
            # Create candidate by perturbing one point
            test_points = current_points.copy()
            idx = np.random.randint(16)
            
            # Small perturbation
            dx = np.random.normal(0, T * 0.01)
            dy = np.random.normal(0, T * 0.01)
            
            test_points[idx, 0] += dx
            test_points[idx, 1] += dy
            
            # Boundary handling
            test_points[:, 0] = np.clip(test_points[:, 0], 0, 1)
            test_points[:, 1] = np.clip(test_points[:, 1], 0, 1)
            
            # Calculate new ratio
            test_ratio = calculate_ratio(test_points)
            
            # Accept or reject
            if test_ratio > current_ratio or np.random.rand() < np.exp((test_ratio - current_ratio) / T):
                current_points = test_points
                current_ratio = test_ratio
            
            T *= cooling_rate
            if T < 1e-8:
                break
        
        return current_points
    
    # Apply local refinement
    refined_solution = local_refinement(best_solution)
    
    # Final check and return
    final_ratio = calculate_ratio(refined_solution)
    if final_ratio > best_score:
        return refined_solution
    else:
        return best_solution

# EVOLVE-BLOCK-END