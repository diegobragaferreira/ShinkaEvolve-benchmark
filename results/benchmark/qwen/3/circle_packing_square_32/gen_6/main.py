# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def initialize_population(n_circles: int, pop_size: int) -> np.ndarray:
    """Initialize population with greedy heuristic placement."""
    # Start with a rough grid layout
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    positions = []
    
    # Place circles in a grid pattern first
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) >= n_circles:
                break
            x = (i + 0.5) / grid_size
            y = (j + 0.5) / grid_size
            positions.append([x, y])
    
    # Fill remaining positions with random placements
    while len(positions) < n_circles:
        positions.append([random.random(), random.random()])
    
    # Create initial population
    population = []
    for _ in range(pop_size):
        individual = []
        for i in range(n_circles):
            # Add some randomness to initial positions
            x = max(0.01, min(0.99, positions[i][0] + np.random.normal(0, 0.05)))
            y = max(0.01, min(0.99, positions[i][1] + np.random.normal(0, 0.05)))
            r = min(0.1, max(0.01, np.random.uniform(0.01, 0.1)))
            individual.append([x, y, r])
        population.append(individual)
    
    return np.array(population)

def check_overlap(circle1: np.ndarray, circle2: np.ndarray) -> bool:
    """Check if two circles overlap."""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    
    # Calculate distance between centers
    dx = x1 - x2
    dy = y1 - y2
    distance_squared = dx*dx + dy*dy
    
    # Circles overlap if distance < sum of radii
    return distance_squared < (r1 + r2)**2

def is_valid_placement(circles: np.ndarray, index: int) -> bool:
    """Check if circle at given index is valid (within bounds and no overlaps)."""
    x, y, r = circles[index]
    
    # Check containment constraints
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False
    
    # Check overlap with all other circles
    for i in range(len(circles)):
        if i != index and check_overlap(circles[index], circles[i]):
            return False
    
    return True

def fitness_function(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii (maximize this)."""
    total_radius = np.sum(circles[:, 2])
    
    # Penalize invalid configurations heavily
    for i in range(len(circles)):
        if not is_valid_placement(circles, i):
            return 0.0
    
    return total_radius

def mutate_individual(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position and radius
            mutated[i][0] = max(0.01, min(0.99, mutated[i][0] + np.random.normal(0, 0.02)))
            mutated[i][1] = max(0.01, min(0.99, mutated[i][1] + np.random.normal(0, 0.02)))
            mutated[i][2] = max(0.001, min(0.49, mutated[i][2] + np.random.normal(0, 0.01)))
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two parents."""
    child = parent1.copy()
    crossover_point = random.randint(0, len(parent1) - 1)
    
    # Take second half from parent2
    child[crossover_point:] = parent2[crossover_point:]
    
    return child

def select_tournament(population: np.ndarray, fitnesses: np.ndarray, tournament_size: int = 3) -> np.ndarray:
    """Select individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 32
    pop_size = 50
    generations = 300
    tournament_size = 3
    mutation_rate = 0.1
    
    # Initialize population
    population = initialize_population(n_circles, pop_size)
    
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(generations):
        # Evaluate fitness for entire population
        fitnesses = []
        for individual in population:
            fit = fitness_function(individual)
            fitnesses.append(fit)
        
        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(best_individual)
        
        # Generate offspring through selection and crossover
        while len(new_population) < pop_size:
            parent1 = select_tournament(population, fitnesses, tournament_size)
            parent2 = select_tournament(population, fitnesses, tournament_size)
            
            child = crossover(parent1, parent2)
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = np.array(new_population)
    
    # Ensure final result is valid
    if best_individual is not None:
        # Final validation and refinement
        validated_individual = best_individual.copy()
        for i in range(len(validated_individual)):
            # Try to improve each circle's placement
            for _ in range(10):  # Minor local optimization
                x, y, r = validated_individual[i]
                old_x, old_y, old_r = x, y, r
                
                # Try small moves
                candidates = [
                    (max(0.01, min(0.99, x + np.random.normal(0, 0.005))), 
                     max(0.01, min(0.99, y + np.random.normal(0, 0.005))), 
                     max(0.001, min(0.49, r + np.random.normal(0, 0.001)))),
                    (x + np.random.normal(0, 0.001), 
                     y + np.random.normal(0, 0.001), 
                     r + np.random.normal(0, 0.0005))
                ]
                
                for cx, cy, cr in candidates:
                    test_individual = validated_individual.copy()
                    test_individual[i] = [cx, cy, cr]
                    
                    if all(is_valid_placement(test_individual, j) for j in range(len(test_individual))):
                        validated_individual = test_individual
                        break
        
        return validated_individual
    else:
        # Return default if no good solution found
        return np.array([[0.5, 0.5, 0.1]] * 32)

# EVOLVE-BLOCK-END
