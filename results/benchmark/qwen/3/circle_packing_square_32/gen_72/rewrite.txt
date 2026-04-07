# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_hexagonal_grid(n_circles: int) -> np.ndarray:
    """
    Generate initial circle positions using hexagonal grid pattern.
    This provides a much better starting configuration than pure random placement.
    """
    # Determine grid dimensions
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))

    # Ensure we don't exceed our target count
    actual_cells = rows * cols
    if actual_cells > n_circles:
        # Adjust to fit exactly n_circles
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))

    # Calculate spacing to fit within unit square
    padding = 0.05  # Leave margin around edges
    cell_width = (1 - 2 * padding) / cols
    cell_height = (1 - 2 * padding) / rows

    # Hexagonal packing parameters
    hex_radius = min(cell_width, cell_height) * 0.4  # Slightly smaller than cell size

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

            # Add slight randomness to avoid perfect grid
            noise_scale = cell_width * 0.1
            x += np.random.normal(0, noise_scale)
            y += np.random.normal(0, noise_scale)

            # Clip to unit square bounds
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
    """Generate initial population of circle configurations."""
    population = []
    for _ in range(pop_size):
        # Start with hexagonal grid initialization
        circles = generate_hexagonal_grid(n_circles)
        
        # Apply some random perturbations to make it more diverse
        for i in range(n_circles):
            # Perturb positions slightly
            circles[i, 0] += np.random.normal(0, 0.01)
            circles[i, 1] += np.random.normal(0, 0.01)
            # Clip to keep within bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
            
        # Ensure validity of initial configuration
        if not is_valid_configuration(circles):
            # Fallback to random initialization if needed
            circles = np.zeros((n_circles, 3))
            for i in range(n_circles):
                # Try multiple times to place circle without overlap
                max_attempts = 1000
                attempts = 0
                
                while attempts < max_attempts:
                    # Generate random position and radius
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                    r = np.random.uniform(0.01, 0.15)
                    
                    # Check if this would be valid
                    temp_circles = circles.copy()
                    temp_circles[i] = [x, y, r]
                    
                    # Check validity
                    if is_valid_configuration(temp_circles):
                        circles[i] = [x, y, r]
                        break
                        
                    attempts += 1
                    
                # If could not place, use minimal valid configuration
                if attempts >= max_attempts:
                    circles[i] = [0.1, 0.1, 0.05]
        
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
    
    # Single-point crossover on positions
    crossover_point = np.random.randint(1, n)
    
    # Swap positions after crossover point
    child1[crossover_point:, :2] = parent2[crossover_point:, :2]
    child2[crossover_point:, :2] = parent1[crossover_point:, :2]
    
    # Keep original radii for now, later modify them to be valid
    return child1, child2

def mutate(individual: np.ndarray, mutation_rate: float = 0.1,
          max_radius_change: float = 0.02) -> np.ndarray:
    """Mutate an individual configuration."""
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Randomly change position slightly
            mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02), 0, 1)
            mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02), 0, 1)
            
            # Change radius with some constraints
            mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, max_radius_change), 0.01, 0.2)
    
    # Ensure validity after mutation
    if not is_valid_configuration(mutated):
        # If invalid, revert to a valid configuration - this is a simplified approach
        mutated = individual.copy() 
        
    return mutated

def evaluate_fitness_parallel(population: List[np.ndarray]) -> List[float]:
    """Evaluate fitness of entire population in parallel."""
    fitnesses = Parallel(n_jobs=-1)(
        delayed(get_fitness)(individual) for individual in population
    )
    return fitnesses

def optimize_individual(individual: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Apply local optimization to increase radii while maintaining constraints."""
    optimized = individual.copy()

    for iteration in range(max_iterations):
        improved = False
        # Try to increase each radius
        for i in range(len(optimized)):
            x, y, r = optimized[i]

            # Store original values
            original_r = r

            # Try to increase radius slightly
            max_increase = 0.01
            test_r = min(r + max_increase, 0.2)  # Cap at reasonable maximum

            # Check if we can increase the radius
            valid = True
            for j in range(len(optimized)):
                if i != j:
                    x2, y2, r2 = optimized[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < test_r + r2:
                        valid = False
                        break

            # Check boundary constraints
            if valid and (test_r <= x <= 1 - test_r) and (test_r <= y <= 1 - test_r):
                optimized[i, 2] = test_r
                improved = True

        # If no improvements were made, stop
        if not improved:
            break

    return optimized

def evolve_circles(n_circles: int = 32, pop_size: int = 50, 
                  generations: int = 100, elite_size: int = 5) -> np.ndarray:
    """Main evolutionary algorithm to pack circles optimally."""
    # Initialize population
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
            
            # Mutation with adaptive rate
            mutation_rate = max(0.05, 0.1 - (generation / generations) * 0.08)
            child1 = mutate(child1, mutation_rate=mutation_rate)
            child2 = mutate(child2, mutation_rate=mutation_rate)
            
            # Local optimization to refine solutions
            child1 = optimize_individual(child1)
            child2 = optimize_individual(child2)
            
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
        circles = evolve_circles(n_circles=32, pop_size=50, generations=100, elite_size=5)
        # Ensure the result is valid
        if not is_valid_configuration(circles):
            # Fallback to basic configuration if something went wrong
            circles = np.zeros((32, 3))
            for i in range(32):
                circles[i] = [0.1 + i * 0.03, 0.1 + (i % 4) * 0.1, 0.05]
    except Exception as e:
        # On error, fallback to basic configuration
        print(f"Error during evolution: {e}")
        circles = np.zeros((32, 3))
        for i in range(32):
            circles[i] = [0.1 + i * 0.03, 0.1 + (i % 4) * 0.1, 0.05]
    
    return circles

# EVOLVE-BLOCK-END