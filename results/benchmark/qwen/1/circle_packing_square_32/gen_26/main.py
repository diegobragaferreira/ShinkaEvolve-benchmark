# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 32
    max_iterations = 1000
    population_size = 50
    
    # Initialize population
    def create_individual():
        # Start with grid-based placement in corners/edges
        circles = np.zeros((n_circles, 3))
        
        # Place some circles in corners with maximum possible radii
        corner_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        placed_in_corners = 0
        
        # Place some circles in corners
        for i in range(min(4, n_circles)):
            x, y = corner_positions[i]
            # Maximum radius that fits in corner
            max_r = min(x, y, 1-x, 1-y)
            circles[i] = [x + max_r/2, y + max_r/2, max_r/2]
            placed_in_corners += 1
            
        # Fill remaining positions randomly but with reasonable constraints
        for i in range(placed_in_corners, n_circles):
            # Try to place circles near edges but not too close to boundaries
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            max_r = min(x, y, 1-x, 1-y)
            r = np.random.uniform(0.01, max_r * 0.5)
            circles[i] = [x, y, r]
            
        return circles
    
    # Create initial population
    population = [create_individual() for _ in range(population_size)]
    
    # Helper function to check constraints
    def is_valid(circles):
        # Check containment and overlap constraints
        for i, (x, y, r) in enumerate(circles):
            # Boundary check
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
            # Overlap check with other circles
            for j in range(i):
                x2, y2, r2 = circles[j]
                distance_sq = (x - x2)**2 + (y - y2)**2
                if distance_sq < (r + r2)**2:
                    return False
        return True
    
    # Fitness function with penalty for constraint violations
    def fitness(circles):
        if not is_valid(circles):
            return -np.inf
        
        sum_radii = np.sum(circles[:, 2])
        return sum_radii
    
    # Mutation operator
    def mutate(individual):
        new_individual = individual.copy()
        # Choose one circle to mutate
        idx = np.random.randint(0, len(individual))
        x, y, r = individual[idx]
        
        # Mutate x, y, r with small changes
        # Mutate x
        new_x = x + np.random.normal(0, 0.01)
        new_x = np.clip(new_x, r, 1-r)
        
        # Mutate y
        new_y = y + np.random.normal(0, 0.01)
        new_y = np.clip(new_y, r, 1-r)
        
        # Mutate r
        new_r = r + np.random.normal(0, 0.01)
        new_r = np.clip(new_r, 0.001, min(new_x, new_y, 1-new_x, 1-new_y))
        
        new_individual[idx] = [new_x, new_y, new_r]
        return new_individual
    
    # Crossover operator
    def crossover(parent1, parent2):
        # Simple average crossover
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        # For each circle, take either parent's values
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
                
        return child1, child2
    
    # Selection operator
    def select_parent(population, fitnesses):
        # Tournament selection
        tournament_size = 5
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx]
    
    # Evolve the population
    for generation in range(max_iterations):
        # Calculate fitness for all individuals
        fitnesses = [fitness(ind) for ind in population]
        
        # Find best individual
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        
        # Print progress every 100 iterations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        # Create new population
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = int(population_size * 0.1)
        sorted_indices = np.argsort(fitnesses)[::-1][:elite_count]
        for idx in sorted_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            # Select parents
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            # Add to new population
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Return the best individual
    final_fitnesses = [fitness(ind) for ind in population]
    best_idx = np.argmax(final_fitnesses)
    
    return population[best_idx]

# EVOLVE-BLOCK-END
