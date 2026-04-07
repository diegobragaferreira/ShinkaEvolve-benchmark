# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time

# Global constants for optimization
MAX_GENERATIONS = 1000
POPULATION_SIZE = 50
ELITE_SIZE = 5
INITIAL_MUTATION_RATE = 0.1
MIN_MUTATION_RATE = 0.01
BOUNDARY_PENALTY_WEIGHT = 1000
OVERLAP_PENALTY_WEIGHT = 10000

def generate_voronoi_positions(n: int, seed: int = 42) -> np.ndarray:
    """Generate initial candidate positions using Voronoi tessellation."""
    np.random.seed(seed)
    
    # Generate random points inside the unit square
    points = np.random.rand(n*2, 2)
    
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    # Get Voronoi vertices
    vertices = vor.vertices
    
    # Filter vertices that are inside the unit square
    valid_vertices = vertices[
        (vertices[:, 0] >= 0) & (vertices[:, 0] <= 1) &
        (vertices[:, 1] >= 0) & (vertices[:, 1] <= 1)
    ]
    
    # If we don't have enough valid vertices, sample additional points
    if len(valid_vertices) < n:
        additional_points = np.random.rand(n - len(valid_vertices), 2)
        valid_vertices = np.vstack([valid_vertices, additional_points])
    
    # Return first n vertices or points
    return valid_vertices[:n] if len(valid_vertices) >= n else valid_vertices

def initialize_circles_voronoi(n: int) -> np.ndarray:
    """Initialize circles using Voronoi-based approach."""
    circles = np.zeros((n, 3))
    
    # Generate Voronoi-based positions
    positions = generate_voronoi_positions(n)
    
    # Initialize each circle with maximum possible radius at its position
    for i in range(n):
        x, y = positions[i]
        max_radius = min(x, 1-x, y, 1-y)
        
        # Set a reasonable initial radius (small fraction of max possible)
        circles[i] = [x, y, max_radius * 0.3]
    
    return circles

def create_spatial_grid(circles: np.ndarray, grid_size: int = 10) -> dict:
    """Create a spatial grid for efficient collision detection."""
    grid = {}
    cell_size = 1.0 / grid_size
    
    for i, (x, y, r) in enumerate(circles):
        # Determine which grid cells this circle affects
        min_cell_x = max(0, int((x - r) / cell_size))
        max_cell_x = min(grid_size - 1, int((x + r) / cell_size))
        min_cell_y = max(0, int((y - r) / cell_size))
        max_cell_y = min(grid_size - 1, int((y + r) / cell_size))
        
        for gx in range(min_cell_x, max_cell_x + 1):
            for gy in range(min_cell_y, max_cell_y + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)
    
    return grid

def update_spatial_grid(circles: np.ndarray, old_grid: dict, 
                       updated_indices: list, grid_size: int = 10) -> dict:
    """Efficiently update spatial grid when circles change."""
    grid = {}
    cell_size = 1.0 / grid_size
    
    # Rebuild grid for all circles
    for i, (x, y, r) in enumerate(circles):
        min_cell_x = max(0, int((x - r) / cell_size))
        max_cell_x = min(grid_size - 1, int((x + r) / cell_size))
        min_cell_y = max(0, int((y - r) / cell_size))
        max_cell_y = min(grid_size - 1, int((y + r) / cell_size))
        
        for gx in range(min_cell_x, max_cell_x + 1):
            for gy in range(min_cell_y, max_cell_y + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)
    
    return grid

def check_collision_with_grid(circles: np.ndarray, i: int, grid: dict, 
                            grid_size: int = 10) -> bool:
    """Fast collision check using spatial grid."""
    x, y, r = circles[i]
    cell_size = 1.0 / grid_size
    
    # Check nearby cells for potential collisions
    min_cell_x = max(0, int((x - r) / cell_size))
    max_cell_x = min(grid_size - 1, int((x + r) / cell_size))
    min_cell_y = max(0, int((y - r) / cell_size))
    max_cell_y = min(grid_size - 1, int((y + r) / cell_size))
    
    for gx in range(min_cell_x, max_cell_x + 1):
        for gy in range(min_cell_y, max_cell_y + 1):
            if (gx, gy) in grid:
                for j in grid[(gx, gy)]:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance_sq = (x - x2)**2 + (y - y2)**2
                        if distance_sq < (r + r2)**2:
                            return True
    return False

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if the configuration satisfies all constraints efficiently."""
    n = len(circles)
    
    # Early exit if any circle violates containment
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > 1-x or r > y or r > 1-y:
            return False
    
    # Use spatial grid for efficient overlap checking
    grid = create_spatial_grid(circles)
    
    # Check for overlaps using spatial grid
    for i in range(n):
        if check_collision_with_grid(circles, i, grid):
            return False
    
    return True

def calculate_fitness(circles: np.ndarray) -> float:
    """Calculate fitness with hierarchical penalty system."""
    total_radius = np.sum(circles[:, 2])
    
    # Penalty for constraint violations
    penalty = 0
    
    # Containment penalty (quadratic for severe violations)
    for i in range(len(circles)):
        x, y, r = circles[i]
        if r > x:
            penalty += BOUNDARY_PENALTY_WEIGHT * (r - x)**2
        if r > 1-x:
            penalty += BOUNDARY_PENALTY_WEIGHT * (r - (1-x))**2
        if r > y:
            penalty += BOUNDARY_PENALTY_WEIGHT * (r - y)**2
        if r > 1-y:
            penalty += BOUNDARY_PENALTY_WEIGHT * (r - (1-y))**2
    
    # Overlap penalty (linear for minor overlaps, quadratic for major ones)
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            if distance_sq < (r1 + r2)**2:
                overlap = (r1 + r2)**2 - distance_sq
                penalty += OVERLAP_PENALTY_WEIGHT * overlap
    
    # Large penalty for completely invalid configurations
    if penalty > 0:
        penalty += 1e6
    
    return total_radius - penalty

def adaptive_mutation_rate(generation: int, max_generations: int) -> float:
    """Calculate adaptive mutation rate that decreases over time."""
    progress = generation / max_generations
    return MIN_MUTATION_RATE + (INITIAL_MUTATION_RATE - MIN_MUTATION_RATE) * (1 - progress)

def mutate_circles(circles: np.ndarray, generation: int, max_generations: int = MAX_GENERATIONS) -> np.ndarray:
    """Apply mutation to circle configuration with adaptive rates."""
    new_circles = circles.copy()
    mutation_rate = adaptive_mutation_rate(generation, max_generations)
    
    for i in range(len(circles)):
        if np.random.random() < mutation_rate:
            # Mutate position
            new_circles[i, 0] += np.random.normal(0, 0.01)
            new_circles[i, 1] += np.random.normal(0, 0.01)
            
            # Ensure validity of new position
            new_circles[i, 0] = np.clip(new_circles[i, 0], 0.01, 0.99)
            new_circles[i, 1] = np.clip(new_circles[i, 1], 0.01, 0.99)
            
            # Mutate radius
            new_circles[i, 2] += np.random.normal(0, 0.005)
            new_circles[i, 2] = max(0.001, new_circles[i, 2])  # Ensure positive radius
    
    return new_circles

def crossover_circles(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two circle configurations."""
    child = parent1.copy()
    n = len(parent1)
    
    # Single point crossover for positions and radii
    crossover_point = np.random.randint(1, n)
    
    # Crossover for positions and radii
    for i in range(crossover_point, n):
        child[i] = parent2[i]
    
    return child

def tournament_selection(population: list, fitness_scores: list, tournament_size: int = 3) -> int:
    """Select an individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    
    # Initialize population with Voronoi-based starting points
    population = []
    for _ in range(POPULATION_SIZE):
        individual = initialize_circles_voronoi(n)
        population.append(individual)
    
    best_fitness = -np.inf
    best_individual = None
    
    # Evolutionary algorithm
    for generation in range(MAX_GENERATIONS):
        # Calculate fitness for all individuals
        fitness_scores = []
        valid_count = 0
        
        for individual in population:
            fitness = calculate_fitness(individual)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                
            if fitness > 0:  # Only count valid solutions
                valid_count += 1
        
        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}, Valid = {valid_count}/{POPULATION_SIZE}")
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        
        # Keep elite individuals
        new_population = population[:ELITE_SIZE]
        
        # Generate offspring through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            parent1_idx = tournament_selection(population, fitness_scores)
            parent2_idx = tournament_selection(population, fitness_scores)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = crossover_circles(parent1, parent2)
            
            # Mutation
            child = mutate_circles(child, generation)
            
            # Ensure validity with constraint checking
            if not is_valid_configuration(child):
                # If invalid, try to repair or regenerate
                child = initialize_circles_voronoi(n)
            
            new_population.append(child)
        
        population = new_population
    
    # Final validation and return
    if best_individual is not None and is_valid_configuration(best_individual):
        return best_individual
    else:
        # Return the best valid configuration found during evolution
        for individual in population:
            if is_valid_configuration(individual):
                return individual
        # If no valid configuration was found, return a default Voronoi configuration
        return initialize_circles_voronoi(n)

# EVOLVE-BLOCK-END