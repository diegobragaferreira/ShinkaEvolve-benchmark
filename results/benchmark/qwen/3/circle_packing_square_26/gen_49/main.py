# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Problem parameters
    n_circles = 26
    population_size = 100
    max_generations = 500
    mutation_rate = 0.15
    tournament_size = 5
    
    def initialize_population(size: int, n_circles: int) -> np.ndarray:
        """Initialize population with diverse circle configurations."""
        population = []
        
        # Generate initial population with Voronoi-inspired distribution
        for _ in range(size):
            circles = np.zeros((n_circles, 3))
            
            # Distribute centers using a grid-like approach with perturbations
            grid_size = int(np.ceil(np.sqrt(n_circles)))
            base_positions = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(base_positions) < n_circles:
                        x = (i + 0.5) / grid_size
                        y = (j + 0.5) / grid_size
                        base_positions.append([x, y])
            
            # Add some randomness to positions and set initial radii
            for k in range(n_circles):
                # Start with small random perturbations
                base_x, base_y = base_positions[k]
                x = max(0.05, min(0.95, base_x + np.random.normal(0, 0.05)))
                y = max(0.05, min(0.95, base_y + np.random.normal(0, 0.05)))
                
                # Initial radius based on available space
                r = min(x, 1-x, y, 1-y) * 0.4
                
                circles[k] = [x, y, r]
            
            population.append(circles)
        
        return np.array(population)
    
    def is_valid(circles: np.ndarray) -> bool:
        """Check if circles are valid (within bounds and non-overlapping)."""
        n = len(circles)
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints using spatial indexing for efficiency
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Find neighbors within 2*r distance (potential overlaps)
        pairs = tree.query_pairs(2.0 * min(circles[:, 2]), output_type='ndarray')
        
        for i, j in pairs:
            if i >= j:
                continue
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2:
                return False
        
        return True
    
    def evaluate_fitness(circles: np.ndarray) -> float:
        """Evaluate fitness of a circle configuration."""
        if not is_valid(circles):
            # Apply penalty for constraint violations
            penalty = 0
            
            # Boundary violations
            for i in range(len(circles)):
                x, y, r = circles[i]
                boundary_violation = 0
                if x - r < 0:
                    boundary_violation += (r - x)**2
                if x + r > 1:
                    boundary_violation += (x + r - 1)**2
                if y - r < 0:
                    boundary_violation += (r - y)**2
                if y + r > 1:
                    boundary_violation += (y + r - 1)**2
                penalty += boundary_violation * 1000
            
            # Overlap violations
            positions = circles[:, :2]
            tree = cKDTree(positions)
            pairs = tree.query_pairs(2.0 * min(circles[:, 2]), output_type='ndarray')
            
            for i, j in pairs:
                if i >= j:
                    continue
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < r1 + r2:
                    overlap = (r1 + r2 - dist)
                    penalty += overlap * 10000
            
            # Return negative penalty (lower fitness for more violations)
            return -penalty
        
        # Valid solution - return sum of radii
        return np.sum(circles[:, 2])
    
    def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, 
                           tournament_size: int) -> np.ndarray:
        """Select parent using tournament selection."""
        selected = []
        n = len(population)
        
        for _ in range(n):
            # Select random individuals for tournament
            indices = np.random.choice(n, tournament_size, replace=False)
            tournament_fitnesses = fitnesses[indices]
            winner_idx = indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx])
        
        return np.array(selected)
    
    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents."""
        n = len(parent1)
        child = np.zeros_like(parent1)
        
        # Single-point crossover for positions and radii
        crossover_point = np.random.randint(0, n)
        
        # Copy first part from parent1, second part from parent2
        child[:crossover_point] = parent1[:crossover_point]
        child[crossover_point:] = parent2[crossover_point:]
        
        # Local refinement to fix constraints
        for i in range(n):
            x, y, r = child[i]
            
            # Fix boundary constraints
            r = min(r, x, 1-x, y, 1-y)
            
            # Ensure minimum radius (avoid degenerate cases)
            r = max(0.001, r)
            
            child[i] = [x, y, r]
        
        return child
    
    def mutate(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Mutate an individual."""
        mutated = individual.copy()
        n = len(mutated)
        
        for i in range(n):
            if np.random.random() < mutation_rate:
                # Randomly choose what to mutate
                choice = np.random.randint(0, 3)
                
                if choice == 0:  # Mutate x position
                    mutated[i, 0] = max(0.05, min(0.95, mutated[i, 0] + np.random.normal(0, 0.05)))
                elif choice == 1:  # Mutate y position
                    mutated[i, 1] = max(0.05, min(0.95, mutated[i, 1] + np.random.normal(0, 0.05)))
                else:  # Mutate radius
                    mutated[i, 2] = max(0.001, min(0.4, mutated[i, 2] * np.random.normal(1, 0.2)))
        
        return mutated
    
    # Main evolutionary algorithm
    population = initialize_population(population_size, n_circles)
    best_fitness_history = []
    
    for generation in range(max_generations):
        # Evaluate fitness of entire population
        fitnesses = np.array([evaluate_fitness(individual) for individual in population])
        
        # Track best fitness
        best_fitness = np.max(fitnesses)
        best_fitness_history.append(best_fitness)
        
        # Adaptive mutation rate (decrease over time)
        current_mutation_rate = mutation_rate * (1 - generation / max_generations)
        
        # Selection
        selected = tournament_selection(population, fitnesses, tournament_size)
        
        # Create new population through crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[min(i+1, population_size-1)]
            
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            
            child1 = mutate(child1, current_mutation_rate)
            child2 = mutate(child2, current_mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = np.array(new_population[:population_size])
    
    # Final evaluation of best solution
    final_fitnesses = [evaluate_fitness(individual) for individual in population]
    best_idx = np.argmax(final_fitnesses)
    best_solution = population[best_idx]
    
    return best_solution

# EVOLVE-BLOCK-END
