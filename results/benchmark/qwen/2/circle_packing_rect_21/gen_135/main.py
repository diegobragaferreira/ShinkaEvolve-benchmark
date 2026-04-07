# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
from math import sqrt, pi

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses a hexagonal grid initialization with evolutionary refinement.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Constants for the problem
    n = 21
    perimeter = 4.0
    target_sum = 2.3658321334167627  # Benchmark
    
    # Try different container aspect ratios to find optimal
    best_ratio = 1.0
    best_sum = 0.0
    best_config = None
    
    # Test several aspect ratios
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for ratio in ratios:
        width = 1.0
        height = width / ratio
        
        if width + height <= perimeter / 2:  # Valid container half-perimeter constraint
            # Create hexagonal grid configuration
            config = _create_hexagonal_grid(n, width, height)
            if config is not None:
                sum_radii = np.sum(config[:, 2])
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_ratio = ratio
                    best_config = config
    
    # Use the best configuration found
    if best_config is None:
        # Fallback to simple initialization
        circles = np.zeros((n, 3))
        circles[:, 0] = np.random.uniform(0.01, 0.99, n)
        circles[:, 1] = np.random.uniform(0.01, 0.99, n)
        circles[:, 2] = np.random.uniform(0.01, 0.1, n)
        return circles
    
    # Refine using evolutionary approach on the best configuration
    refined = _evolutionary_refinement(best_config, best_ratio)
    
    return refined

def _create_hexagonal_grid(n, width, height):
    """Creates a hexagonal grid configuration for circle packing"""
    # Determine optimal hexagonal packing parameters
    # Hexagon packing density is approximately 0.9069
    
    # Calculate approximate radius needed for n circles
    # Area needed = n * π * r^2
    # Container area = width * height
    container_area = width * height
    if container_area <= 0:
        return None
    
    # Estimate reasonable average radius
    avg_area_per_circle = container_area / n
    estimated_radius = sqrt(avg_area_per_circle / pi)
    
    # For hexagonal packing, the optimal arrangement is:
    # Row spacing = √3 * r
    # Column spacing = 2 * r
    
    # Calculate grid dimensions
    r = estimated_radius
    row_spacing = r * sqrt(3)
    col_spacing = r * 2
    
    # Ensure at least 1 row and column fit
    if row_spacing > height or col_spacing > width:
        # Scale down radius to fit
        max_r = min(height / sqrt(3), width / 2)
        if max_r <= 0:
            return None
        r = max_r
    
    # Calculate number of rows and columns
    num_cols = max(1, int(width / col_spacing))
    num_rows = max(1, int(height / row_spacing))
    
    # Make sure we have enough space for n circles
    actual_n = num_cols * num_rows
    if actual_n < n:
        # Increase spacing to accommodate more circles
        scale = sqrt(n / actual_n)
        col_spacing *= scale
        row_spacing *= scale
        num_cols = max(1, int(width / col_spacing))
        num_rows = max(1, int(height / row_spacing))
        actual_n = num_cols * num_rows
    
    # Create positions in hexagonal pattern
    positions = []
    radius = r * 0.9  # Leave some margin
    
    # Generate hexagonal grid
    for i in range(num_rows):
        row_offset = i * row_spacing
        row_y = row_offset + radius
        if row_y > height - radius:
            break
            
        # Alternate offset for hexagonal arrangement
        col_offset = (i % 2) * (col_spacing / 2)
        
        for j in range(num_cols):
            col_x = j * col_spacing + col_offset + radius
            if col_x > width - radius:
                break
                
            positions.append([col_x, row_y, radius])
            
            if len(positions) >= n:
                break
        
        if len(positions) >= n:
            break
    
    # If not enough positions, fill remaining with random
    while len(positions) < n:
        x = np.random.uniform(radius, width - radius)
        y = np.random.uniform(radius, height - radius)
        positions.append([x, y, radius])
    
    # Convert to numpy array
    circles = np.array(positions[:n])
    
    # Validate constraints
    if not _validate_placement(circles, width, height):
        # If invalid, start fresh with better approach
        circles = np.zeros((n, 3))
        # Simple random initialization with appropriate radii
        for i in range(n):
            circles[i, 0] = np.random.uniform(radius, width - radius)
            circles[i, 1] = np.random.uniform(radius, height - radius)
            circles[i, 2] = radius * (0.8 + np.random.uniform(0, 0.4))
    
    return circles

def _validate_placement(circles, width, height):
    """Validate that all circles fit within container and don't overlap"""
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Check boundary constraints
    for i in range(len(circles)):
        x, y, r = positions[i, 0], positions[i, 1], radii[i]
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            return False
    
    # Check overlap constraints efficiently using KDTree
    tree = KDTree(positions)
    pairs = tree.query_pairs(2.0 * min(radii))  # Only check nearby pairs
    
    for i, j in pairs:
        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]
        distance = sqrt(dx*dx + dy*dy)
        if distance < (radii[i] + radii[j]):
            return False
    
    return True

def _evolutionary_refinement(initial_circles, ratio):
    """Refine the initial configuration using evolutionary techniques"""
    n = len(initial_circles)
    width = 1.0
    height = width / ratio
    
    # Parameters for evolution
    population_size = 30
    generations = 100
    elite_size = 5
    mutation_rate = 0.1
    
    # Create initial population
    population = []
    for i in range(population_size):
        individual = initial_circles.copy()
        
        # Add some random perturbation to make diverse individuals
        for j in range(n):
            if random.random() < 0.3:  # 30% chance to mutate each circle
                # Perturb position
                individual[j, 0] = np.clip(individual[j, 0] + np.random.normal(0, 0.05), 
                                        individual[j, 2], width - individual[j, 2])
                individual[j, 1] = np.clip(individual[j, 1] + np.random.normal(0, 0.05), 
                                        individual[j, 2], height - individual[j, 2])
                
                # Perturb radius
                individual[j, 2] = np.clip(individual[j, 2] + np.random.normal(0, 0.01), 
                                        0.001, 0.5)
        
        population.append(individual)
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            if _validate_placement(individual, width, height):
                fitness = np.sum(individual[:, 2])
            else:
                fitness = -1.0  # Invalid solution
            fitness_scores.append(fitness)
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Select elite
        elite = sorted_population[:elite_size]
        
        # Create new population
        new_population = elite.copy()
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = sorted_population[_tournament_select(sorted_fitness)]
            parent2 = sorted_population[_tournament_select(sorted_fitness)]
            
            # Crossover
            child = _crossover(parent1, parent2)
            
            # Mutation
            child = _mutate(child, mutation_rate, width, height)
            
            # Validate and add to population
            if _validate_placement(child, width, height):
                new_population.append(child)
            else:
                # If invalid, try again with a modified version
                random_child = child.copy()
                for i in range(len(random_child)):
                    if random.random() < 0.2:
                        random_child[i, 0] = np.clip(random_child[i, 0] + np.random.normal(0, 0.02),
                                                  random_child[i, 2], width - random_child[i, 2])
                        random_child[i, 1] = np.clip(random_child[i, 1] + np.random.normal(0, 0.02),
                                                  random_child[i, 2], height - random_child[i, 2])
                if _validate_placement(random_child, width, height):
                    new_population.append(random_child)
                else:
                    # Fallback to parent1 if all failed
                    new_population.append(parent1)
        
        population = new_population[:population_size]
    
    # Return best solution
    fitness_scores = []
    for individual in population:
        if _validate_placement(individual, width, height):
            fitness = np.sum(individual[:, 2])
        else:
            fitness = -1.0
        fitness_scores.append(fitness)
    
    best_index = np.argmax(fitness_scores)
    return population[best_index]

def _tournament_select(fitness_scores, tournament_size=3):
    """Select individual using tournament selection"""
    tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index

def _crossover(parent1, parent2):
    """Perform crossover between two parents"""
    child = parent1.copy()
    # Single point crossover on circle properties
    crossover_point = random.randint(1, len(parent1) - 1)
    
    for i in range(crossover_point, len(parent1)):
        child[i, :] = parent2[i, :]
    
    return child

def _mutate(individual, mutation_rate, width, height):
    """Mutate an individual"""
    mutated = individual.copy()
    
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Mutate position
            mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.03),
                                  mutated[i, 2], width - mutated[i, 2])
            mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.03),
                                  mutated[i, 2], height - mutated[i, 2])
            
            # Mutate radius
            mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.008), 
                                  0.001, 0.5)
    
    return mutated

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")