# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_hexagonal_grid(n_circles: int) -> np.ndarray:
    """
    Generate initial circle positions using hexagonal grid pattern.
    This provides a much better starting configuration than pure random placement.
    """
    # Determine grid dimensions based on number of circles
    sqrt_n = math.ceil(math.sqrt(n_circles))
    rows = sqrt_n
    cols = math.ceil(n_circles / rows)
    
    # Adjust to fit exactly n_circles
    while rows * cols < n_circles:
        rows += 1
    
    # Calculate spacing to fit within unit square with padding
    padding = 0.05
    cell_width = (1 - 2 * padding) / cols
    cell_height = (1 - 2 * padding) / rows
    
    # Use hexagonal packing where circles are arranged in triangular lattice
    hex_radius = min(cell_width, cell_height) * 0.45  # Slightly smaller than cell size
    
    circles = np.zeros((n_circles, 3))
    
    circle_idx = 0
    for i in range(rows):
        for j in range(cols):
            if circle_idx >= n_circles:
                break
                
            # Offset every other row for hexagonal arrangement
            offset = (i % 2) * (cell_width / 2)
            
            # Calculate position
            x = padding + offset + j * cell_width + cell_width / 2
            y = padding + i * cell_height + cell_height / 2
            r = hex_radius
            
            # Add slight randomness to avoid perfect grid (smaller noise)
            noise_scale = cell_width * 0.05
            x += np.random.normal(0, noise_scale)
            y += np.random.normal(0, noise_scale)
            
            # Clip to unit square bounds with radius consideration
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            circles[circle_idx] = [x, y, r]
            circle_idx += 1
            
    return circles

def get_fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii for valid configurations."""
    if not is_valid_configuration(circles):
        return 0.0
    
    return np.sum(circles[:, 2])

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if circles are within bounds and non-overlapping."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)
    
    # For each circle, check if it overlaps with others
    for i in range(n):
        x, y, r = circles[i]
        # Find nearby points (within 2*r distance)
        nearby_indices = tree.query_ball_point([x, y], 2 * r)
        
        # Check each nearby circle for overlap
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False
    
    return True

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Generate initial population of circle configurations using hexagonal initialization."""
    population = []
    for _ in range(pop_size):
        # Start with hexagonal grid initialization
        circles = generate_hexagonal_grid(n_circles)
        
        # Apply slight random perturbations to improve diversity
        for i in range(n_circles):
            # Perturb position slightly
            circles[i, 0] += np.random.normal(0, 0.01)
            circles[i, 1] += np.random.normal(0, 0.01)
            # Clip to bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
            
        population.append(circles)
    
    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float], 
                         tournament_size: int = 3) -> np.ndarray:
    """Select individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray, 
             crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parent configurations."""
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    
    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Uniform crossover on positions
    crossover_mask = np.random.random(n) < 0.5
    
    # Swap positions where mask is True
    child1[crossover_mask, :2] = parent2[crossover_mask, :2]
    child2[crossover_mask, :2] = parent1[crossover_mask, :2]
    
    # For radii, keep one parent's radii and slightly modify the other
    # Keep parent1's radii for child1, parent2's radii for child2
    # But apply small changes to avoid stagnation
    
    # Slightly perturb child1 radii
    for i in range(n):
        if np.random.random() < 0.3:  # 30% chance to change radius
            child1[i, 2] = np.clip(child1[i, 2] + np.random.normal(0, 0.01), 0.01, 0.2)
    
    # Slightly perturb child2 radii
    for i in range(n):
        if np.random.random() < 0.3:  # 30% chance to change radius
            child2[i, 2] = np.clip(child2[i, 2] + np.random.normal(0, 0.01), 0.01, 0.2)
    
    return child1, child2

def mutate(individual: np.ndarray, generation: int, max_generations: int, 
          base_mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate an individual configuration with adaptive mutation rate."""
    mutated = individual.copy()
    
    # Adaptive mutation rate - decreases over generations
    mutation_rate = base_mutation_rate * (1 - generation / max_generations)
    mutation_rate = max(mutation_rate, 0.02)  # Minimum mutation rate
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Change position (larger perturbation for early generations)
            pos_change_factor = 1.0 if generation < max_generations/2 else 0.5
            mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.03 * pos_change_factor), 0, 1)
            mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.03 * pos_change_factor), 0, 1)
            
            # Change radius with some constraints
            mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.02), 0.01, 0.2)
    
    return mutated

def evaluate_fitness_parallel(population: List[np.ndarray]) -> List[float]:
    """Evaluate fitness of entire population in parallel."""
    fitnesses = Parallel(n_jobs=-1)(
        delayed(get_fitness)(individual) for individual in population
    )
    return fitnesses

def evolve_circles(n_circles: int = 32, pop_size: int = 100, 
                  generations: int = 150, elite_size: int = 10) -> np.ndarray:
    """Main evolutionary algorithm to pack circles optimally."""
    # Initialize population with better hexagonal grid initialization
    population = initialize_population(pop_size, n_circles)
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = evaluate_fitness_parallel(population)
        
        # Get best individuals
        sorted_indices = np.argsort(fitnesses)[::-1]
        best_individuals = [population[i] for i in sorted_indices[:elite_size]]
        
        # Create new population with elitism
        new_population = best_individuals.copy()
        
        # Fill rest of population with offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation with generation-aware rate
            child1 = mutate(child1, generation, generations)
            child2 = mutate(child2, generation, generations)
            
            # Add to new population if valid
            if is_valid_configuration(child1):
                new_population.append(child1)
            if len(new_population) < pop_size and is_valid_configuration(child2):
                new_population.append(child2)
        
        population = new_population[:pop_size]
    
    # Return best solution
    final_fitnesses = evaluate_fitness_parallel(population)
    best_idx = np.argmax(final_fitnesses)
    
    return population[best_idx]

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = evolve_circles(n_circles=32, pop_size=100, generations=150, elite_size=10)
        # Ensure the result is valid
        if not is_valid_configuration(circles):
            # Fallback to basic configuration if something went wrong
            circles = np.zeros((32, 3))
            # Use hexagonal grid for fallback
            hex_circles = generate_hexagonal_grid(32)
            for i in range(32):
                circles[i] = hex_circles[i]
    except Exception as e:
        # On error, fallback to basic configuration
        print(f"Error during evolution: {e}")
        circles = np.zeros((32, 3))
        # Use hexagonal grid for fallback
        hex_circles = generate_hexagonal_grid(32)
        for i in range(32):
            circles[i] = hex_circles[i]
    
    return circles

# EVOLVE-BLOCK-END
