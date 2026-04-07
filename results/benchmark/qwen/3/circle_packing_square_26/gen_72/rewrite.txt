# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import cKDTree
import math
from typing import Tuple, List, Dict, Any

# Global constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
TOURNAMENT_SIZE = 5
BOUNDARY_PENALTY_WEIGHT = 10000.0
OVERLAP_PENALTY_WEIGHT = 10000.0

class SpatialGrid:
    """Efficient spatial indexing for collision detection."""
    
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.grid = {}
        
    def clear(self):
        self.grid = {}
        
    def add_circle(self, idx: int, x: float, y: float, r: float):
        """Add circle to spatial grid."""
        self.clear()
        grid_x = int(x * self.grid_size)
        grid_y = int(y * self.grid_size)
        
        if (grid_x, grid_y) not in self.grid:
            self.grid[(grid_x, grid_y)] = []
        self.grid[(grid_x, grid_y)].append(idx)
        
    def get_neighbors(self, x: float, y: float, r: float) -> List[int]:
        """Get potentially overlapping circles."""
        neighbors = []
        grid_x = int(x * self.grid_size)
        grid_y = int(y * self.grid_size)
        
        # Check neighboring cells including diagonals
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if (nx, ny) in self.grid:
                    neighbors.extend(self.grid[(nx, ny)])
        
        return neighbors

def is_valid(circles: np.ndarray, grid: SpatialGrid = None) -> bool:
    """Check if all circles are within bounds and non-overlapping using spatial indexing."""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Initialize spatial grid if needed
    if grid is None:
        grid = SpatialGrid()
        for i in range(n):
            x, y, r = circles[i]
            grid.add_circle(i, x, y, r)
    
    # Check overlap constraints efficiently
    for i in range(n):
        x1, y1, r1 = circles[i]
        
        # Get neighbors in spatial grid
        neighbors = grid.get_neighbors(x1, y1, r1)
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
    
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def calculate_penalty(circles: np.ndarray) -> float:
    """Calculate penalty for constraint violations."""
    penalty = 0
    
    # Boundary penalty calculation
    boundary_violations = 0
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0:
            boundary_violations += (r - x)**2
        if x + r > 1:
            boundary_violations += (x + r - 1)**2
        if y - r < 0:
            boundary_violations += (r - y)**2
        if y + r > 1:
            boundary_violations += (y + r - 1)**2
    
    penalty += BOUNDARY_PENALTY_WEIGHT * boundary_violations

    # Overlap penalty calculation
    overlap_penalty = 0
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                overlap_penalty += (r1 + r2 - distance)**2
    
    penalty += OVERLAP_PENALTY_WEIGHT * overlap_penalty
    
    return penalty

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a solution, higher is better."""
    if not is_valid(circles):
        return -calculate_penalty(circles)
    
    return calculate_sum_radii(circles)

def initialize_population(size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with better starting configurations."""
    population = []
    
    # Multi-resolution initialization approach
    # Start with coarse grid, then refine
    for _ in range(size):
        circles = np.zeros((n_circles, 3))
        
        # Coarse grid initialization
        rows = cols = int(math.ceil(math.sqrt(n_circles)))
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n_circles:
                    break
                x_base = (col + 1) * spacing_x
                y_base = (row + 1) * spacing_y
                
                # Add noise to positions
                x = max(0.01, min(0.99, x_base + random.uniform(-0.02, 0.02)))
                y = max(0.01, min(0.99, y_base + random.uniform(-0.02, 0.02)))
                
                # Set initial radius based on available space
                max_radius = min(x, 1-x, y, 1-y)
                r = max(0.01, min(0.2, max_radius * random.uniform(0.5, 0.8)))
                
                circles[count] = [x, y, r]
                count += 1
                
        # Refinement phase - local optimization
        for _ in range(3):
            optimize_local(circles)
            
        population.append(circles)
    
    return population

def optimize_local(circles: np.ndarray, iterations: int = 10):
    """Local optimization to improve circle placement."""
    n = len(circles)
    
    # Simple gradient-like approach for local refinement
    for _ in range(iterations):
        # For each circle, try to adjust position to reduce overlap
        for i in range(n):
            x, y, r = circles[i]
            best_x, best_y, best_r = x, y, r
            best_score = evaluate_fitness(circles)
            
            # Try small adjustments
            for dx in [-0.005, 0, 0.005]:
                for dy in [-0.005, 0, 0.005]:
                    new_x = max(r, min(1-r, x + dx))
                    new_y = max(r, min(1-r, y + dy))
                    
                    # Test adjustment
                    test_circles = circles.copy()
                    test_circles[i, 0] = new_x
                    test_circles[i, 1] = new_y
                    
                    if is_valid(test_circles):
                        score = evaluate_fitness(test_circles)
                        if score > best_score:
                            best_score = score
                            best_x, best_y = new_x, new_y
            
            circles[i, 0] = best_x
            circles[i, 1] = best_y

def tournament_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray, grid: SpatialGrid = None) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover with constraint awareness."""
    if random.random() > 0.8:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Uniform crossover with constraint awareness
    crossover_mask = np.random.random(n) > 0.5
    
    for i in range(n):
        if crossover_mask[i]:
            child1[i] = parent2[i].copy()
        else:
            child2[i] = parent1[i].copy()
    
    # Post-crossover validation and refinement
    if not is_valid(child1, grid):
        refine_circles(child1, grid)
    if not is_valid(child2, grid):
        refine_circles(child2, grid)
    
    return child1, child2

def refine_circles(circles: np.ndarray, grid: SpatialGrid):
    """Refine circles to satisfy constraints."""
    # Simple refinement: adjust circles to resolve conflicts
    n = len(circles)
    for iter_count in range(20):  # Limited iterations to prevent infinite loops
        grid.clear()
        valid = True
        for i in range(n):
            grid.add_circle(i, circles[i][0], circles[i][1], circles[i][2])
        
        for i in range(n):
            x, y, r = circles[i]
            # Fix boundary violations
            if x - r < 0:
                x = r
            elif x + r > 1:
                x = 1 - r
            if y - r < 0:
                y = r
            elif y + r > 1:
                y = 1 - r
                
            # Adjust radius if necessary
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            
            circles[i] = [x, y, r]

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive strategy."""
    mutated = circles.copy()
    n = len(mutated)
    
    # Adapt mutation strength based on generation
    base_mutation_rate = 0.2
    mutation_strength = base_mutation_rate * (1 - generation / total_generations)
    
    # Two-phase mutation approach
    for i in range(n):
        if random.random() < mutation_strength:
            # Choose mutation type based on generation (different strategies)
            if generation < total_generations * 0.3:  # Early generation: aggressive exploration
                # Large-scale mutations
                mutation_type = random.choice(['position', 'radius'])
                if mutation_type == 'position':
                    mutate_position_large(mutated, i)
                else:
                    mutate_radius_large(mutated, i)
            else:  # Later generation: fine-tuning
                # Small-scale mutations
                mutation_type = random.choice(['position', 'radius', 'position'])
                if mutation_type == 'position':
                    mutate_position_small(mutated, i)
                else:
                    mutate_radius_small(mutated, i)
    
    # Ensure validity after mutation
    for i in range(n):
        x, y, r = mutated[i]
        # Clamp to valid ranges
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        r = max(0.001, min(0.49, r))
        mutated[i] = [x, y, r]
        
    return mutated

def mutate_position_large(circles: np.ndarray, i: int):
    """Large position mutation for early exploration."""
    circles[i, 0] = max(0.01, min(0.99, circles[i, 0] + random.uniform(-0.1, 0.1)))
    circles[i, 1] = max(0.01, min(0.99, circles[i, 1] + random.uniform(-0.1, 0.1)))

def mutate_position_small(circles: np.ndarray, i: int):
    """Small position mutation for fine-tuning."""
    circles[i, 0] = max(0.01, min(0.99, circles[i, 0] + random.gauss(0, 0.01)))
    circles[i, 1] = max(0.01, min(0.99, circles[i, 1] + random.gauss(0, 0.01)))

def mutate_radius_large(circles: np.ndarray, i: int):
    """Large radius mutation for exploratory changes."""
    circles[i, 2] = max(0.001, min(0.49, circles[i, 2] + random.uniform(-0.1, 0.1)))

def mutate_radius_small(circles: np.ndarray, i: int):
    """Small radius mutation for fine adjustment."""
    circles[i, 2] = max(0.001, min(0.49, circles[i, 2] + random.gauss(0, 0.005)))

def get_best_individual(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Get the individual with highest fitness."""
    best_idx = fitnesses.index(max(fitnesses))
    return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 26
    population = initialize_population(POPULATION_SIZE, n)

    best_fitness_history = []

    # Pre-compute spatial grid for performance
    grid = SpatialGrid()
    
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")

        # Create new population
        new_population = []

        # Elitism: keep best individual
        best_individual = get_best_individual(population, fitnesses)
        new_population.append(best_individual)

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2, grid)

            child1 = mutate(child1, len(new_population), POPULATION_SIZE)
            child2 = mutate(child2, len(new_population) + 1, POPULATION_SIZE)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]

    # Get final best solution
    final_fitnesses = [evaluate_fitness(individual) for individual in population]
    best_solution = get_best_individual(population, final_fitnesses)

    return best_solution

# EVOLVE-BLOCK-END