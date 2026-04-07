# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time

class CirclePackingOptimizer:
    def __init__(self, n_circles: int = 26):
        self.n_circles = n_circles
        self.max_iterations = 1000
        self.population_size = 50
        self.mutation_rate = 0.1
        self.elite_size = 5
        self.seed = 42
        
    def validate_circle(self, x: float, y: float, r: float) -> bool:
        """Check if circle is within bounds"""
        return (r <= x <= 1-r) and (r <= y <= 1-r)
    
    def check_overlap(self, circles: np.ndarray) -> bool:
        """Check if any circles overlap"""
        if len(circles) < 2:
            return False
            
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Check for overlaps
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                distance = distances[i, j]
                min_distance = radii[i] + radii[j]
                if distance < min_distance:
                    return True
        return False
    
    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as sum of radii, penalize overlaps"""
        if len(circles) != self.n_circles:
            return 0.0
            
        # Check if all circles are within bounds
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if not self.validate_circle(x, y, r):
                return 0.0
                
        # Check for overlaps
        if self.check_overlap(circles):
            return 0.0
            
        # Return sum of radii
        return np.sum(circles[:, 2])
    
    def initialize_population(self) -> List[np.ndarray]:
        """Initialize population with diverse configurations"""
        population = []
        np.random.seed(self.seed)
        
        # Generate multiple random initializations
        for _ in range(self.population_size):
            circles = np.zeros((self.n_circles, 3))
            
            # Initialize with some randomness but ensure validity
            for i in range(self.n_circles):
                attempts = 0
                while attempts < 100:
                    # Random position and radius
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                    r = np.random.uniform(0.01, 0.2)
                    
                    # Check if valid
                    if self.validate_circle(x, y, r):
                        circles[i] = [x, y, r]
                        break
                    attempts += 1
                    
                if attempts >= 100:
                    # Fallback to simple grid-like initialization
                    idx = i % 5
                    idy = i // 5
                    x = 0.1 + idx * 0.2
                    y = 0.1 + idy * 0.2
                    r = 0.05
                    circles[i] = [x, y, r]
            
            population.append(circles.copy())
            
        return population
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply mutation to an individual"""
        mutated = individual.copy()
        
        # Mutate some circles
        num_mutations = max(1, int(self.n_circles * self.mutation_rate))
        indices = np.random.choice(self.n_circles, num_mutations, replace=False)
        
        for i in indices:
            # Randomly change position or radius
            if np.random.random() < 0.5:
                # Mutate position
                mutated[i, 0] += np.random.normal(0, 0.01)
                mutated[i, 1] += np.random.normal(0, 0.01)
                
                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:
                # Mutate radius
                mutated[i, 2] += np.random.normal(0, 0.005)
                mutated[i, 2] = np.clip(mutated[i, 2], 0.005, 0.5)
                
        return mutated
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring via uniform crossover"""
        offspring = parent1.copy()
        
        # Uniform crossover for each circle
        for i in range(self.n_circles):
            if np.random.random() < 0.5:
                offspring[i] = parent2[i].copy()
                
        return offspring
    
    def optimize_individual(self, circles: np.ndarray) -> np.ndarray:
        """Refine individual using local optimization"""
        # Convert to flat array for optimization
        def objective(params):
            # Reshape params back to circles
            temp_circles = circles.copy()
            for i in range(self.n_circles):
                temp_circles[i, 0] = params[i*3]
                temp_circles[i, 1] = params[i*3 + 1]
                temp_circles[i, 2] = params[i*3 + 2]
            
            # Return negative because we want to maximize
            return -self.evaluate_fitness(temp_circles)
        
        # Create flat parameters
        flat_params = []
        for i in range(self.n_circles):
            flat_params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        
        # Optimize
        try:
            result = minimize(objective, flat_params, method='L-BFGS-B', 
                            bounds=[(0, 1), (0, 1), (0.001, 0.5)] * self.n_circles,
                            options={'maxiter': 50})
            
            if result.success:
                # Update circles with optimized values
                optimized_circles = circles.copy()
                for i in range(self.n_circles):
                    optimized_circles[i, 0] = result.x[i*3]
                    optimized_circles[i, 1] = result.x[i*3 + 1]
                    optimized_circles[i, 2] = result.x[i*3 + 2]
                return optimized_circles
        except:
            pass
            
        return circles
    
    def evolve(self) -> np.ndarray:
        """Main evolution loop"""
        # Initialize population
        population = self.initialize_population()
        best_fitness = 0.0
        best_individual = None
        convergence_counter = 0
        max_convergence = 50
        
        for generation in range(self.max_iterations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                fitness_scores.append(fitness)
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1] 
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Track best
            if fitness_scores[0] > best_fitness:
                best_fitness = fitness_scores[0]
                best_individual = population[0].copy()
                convergence_counter = 0
            else:
                convergence_counter += 1
                
            # Check convergence
            if convergence_counter >= max_convergence:
                break
                
            # Apply elitism
            new_population = population[:self.elite_size]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                selected_indices = np.random.choice(self.population_size, tournament_size)
                winner_index = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
                
                # Select another parent
                selected_indices = np.random.choice(self.population_size, tournament_size)
                second_winner_index = selected_indices[np.argmax([fitness_scores[i] for i in selected_indices])]
                
                # Crossover
                offspring = self.crossover(population[winner_index], population[second_winner_index])
                
                # Mutation
                offspring = self.mutate(offspring)
                
                # Local optimization
                offspring = self.optimize_individual(offspring)
                
                new_population.append(offspring)
            
            population = new_population
            
            # Adaptive mutation rate
            if generation > 100:
                self.mutation_rate = max(0.01, 0.1 * (1 - generation/1000))
        
        return best_individual

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Run optimization
    optimizer = CirclePackingOptimizer(n_circles=26)
    circles = optimizer.evolve()
    
    # Ensure final validation
    if circles is None or len(circles) == 0:
        # Fallback to default configuration
        circles = np.zeros((26, 3))
        for i in range(26):
            circles[i] = [0.1 + (i % 5) * 0.2, 0.1 + (i // 5) * 0.2, 0.05]
    
    return circles

# EVOLVE-BLOCK-END
