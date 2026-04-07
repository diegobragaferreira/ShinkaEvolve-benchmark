# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import random
import math
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def get_fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii for valid configurations."""
    if not is_valid_configuration(circles):
        return 0.0
    return np.sum(circles[:, 2])

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if circles are within bounds and non-overlapping."""
    n = len(circles)

    # Check containment constraints first (faster)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints using KDTree for efficiency
    if n < 2:
        return True
    
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

def initialize_hexagonal_grid(n_circles: int) -> np.ndarray:
    """Create initial configuration using optimized hexagonal packing."""
    # Calculate grid dimensions
    rows = int(math.sqrt(n_circles)) + 1
    cols = int(math.ceil(n_circles / rows)) + 1

    # Adjust spacing to fit within unit square with margin
    spacing_x = 0.95 / cols
    spacing_y = 0.95 / rows

    # Use hexagon radius based on grid spacing, slightly more conservative
    hex_radius = min(spacing_x, spacing_y) * 0.45

    circles = []

    # Fill grid with circles in hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n_circles:
                break

            # Offset every other row for hexagonal pattern
            x_offset = (i % 2) * (spacing_x / 2)
            x = (j * spacing_x) + x_offset + 0.025
            y = (i * spacing_y) + 0.025

            # Ensure circle stays within bounds
            if x <= 1 - hex_radius and y <= 1 - hex_radius:
                circles.append([x, y, hex_radius])

    # Fill remaining circles with small radii if needed
    while len(circles) < n_circles:
        circles.append([0.5, 0.5, 0.02])

    return np.array(circles[:n_circles])

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Generate initial population of circle configurations."""
    # Start with hexagonal grid as base
    base_population = [initialize_hexagonal_grid(n_circles) for _ in range(pop_size // 2)]
    
    # Add diversity with random initialization
    for _ in range(pop_size // 2):
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            max_attempts = 1000
            attempts = 0
            placed = False
            
            while attempts < max_attempts and not placed:
                # Generate random position and radius
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = np.random.uniform(0.01, 0.15)
                
                # Check if this would be valid with existing circles
                temp_circles = circles.copy()
                temp_circles[i] = [x, y, r]
                
                # Validate with boundary check first
                valid = True
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    valid = False
                
                if valid:
                    # Then check overlap with existing circles
                    if i > 0:
                        points = temp_circles[:i, :2]
                        radii = temp_circles[:i, 2]
                        tree = cKDTree(points)
                        nearby_indices = tree.query_ball_point([x, y], 2 * r)
                        
                        for j in nearby_indices:
                            distance = np.sqrt((x - points[j][0])**2 + (y - points[j][1])**2)
                            if distance < r + radii[j]:
                                valid = False
                                break
                
                if valid:
                    circles[i] = [x, y, r]
                    placed = True
                else:
                    attempts += 1

            # If could not place, use minimal valid configuration
            if not placed:
                circles[i] = [0.1, 0.1, 0.05]

        base_population.append(circles)

    return base_population

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
    for i in range(n):
        if np.random.random() < 0.5:
            child1[i, :2] = parent2[i, :2]
            child2[i, :2] = parent1[i, :2]

    # Keep original radii
    return child1, child2

def mutate(individual: np.ndarray, mutation_rate: float = 0.1,
          max_position_change: float = 0.03, max_radius_change: float = 0.02) -> np.ndarray:
    """Mutate an individual configuration."""
    mutated = individual.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Change position
            mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, max_position_change), 0, 1)
            mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, max_position_change), 0, 1)

            # Change radius with constraints
            mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, max_radius_change), 0.005, 0.3)

    # Ensure validity after mutation
    if not is_valid_configuration(mutated):
        # Revert to original if mutation caused invalidity
        mutated = individual.copy()

    return mutated

def evaluate_fitness_parallel(population: List[np.ndarray]) -> List[float]:
    """Evaluate fitness of entire population in parallel."""
    fitnesses = Parallel(n_jobs=-1)(
        delayed(get_fitness)(individual) for individual in population
    )
    return fitnesses

def optimize_individual(individual: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Apply local optimization to increase radii while maintaining constraints."""
    optimized = individual.copy()
    
    # Pre-compute KDTree for overlap checking
    points = optimized[:, :2]
    radii = optimized[:, 2]
    tree = cKDTree(points)
    
    for iteration in range(max_iterations):
        improved = False
        # Try to increase each radius
        for i in range(len(optimized)):
            x, y, r = optimized[i]

            # Store original values
            original_r = r

            # Try to increase radius slightly
            test_r = min(r + 0.005, 0.2)  # Cap at reasonable maximum

            # Check boundary constraints
            if not (test_r <= x <= 1 - test_r) or not (test_r <= y <= 1 - test_r):
                continue

            # Check overlap with all other circles
            valid = True
            neighbors = tree.query_ball_point([x, y], 2 * test_r)
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = optimized[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < test_r + r2:
                        valid = False
                        break

            # If valid, update radius
            if valid:
                optimized[i, 2] = test_r
                improved = True

        # If no improvements were made, stop
        if not improved:
            break

    return optimized

def evolve_circles(n_circles: int = 32, pop_size: int = 100,
                  generations: int = 150, elite_size: int = 10) -> np.ndarray:
    """Main evolutionary algorithm to pack circles optimally."""
    # Initialize population
    population = initialize_population(pop_size, n_circles)

    # Evolution loop
    best_fitness_history = []
    
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = evaluate_fitness_parallel(population)

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)
        
        # Print progress every 20 generations
        if generation % 20 == 0:
            print(f"Gen {generation}: Best fitness = {best_fitness:.6f}")

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
        circles = evolve_circles(n_circles=32, pop_size=100, generations=150, elite_size=10)
        # Ensure the result is valid
        if not is_valid_configuration(circles):
            # Fallback to basic configuration if something went wrong
            circles = initialize_hexagonal_grid(32)
    except Exception as e:
        # On error, fallback to basic configuration
        print(f"Error during evolution: {e}")
        circles = initialize_hexagonal_grid(32)

    return circles

# EVOLVE-BLOCK-END