# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
from collections import defaultdict

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained within the unit square."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap_grid(circles: np.ndarray, grid_size: int = 20) -> bool:
    """Check if any circles overlap using spatial grid for efficiency."""
    n = len(circles)
    if n <= 1:
        return True
        
    # Create spatial grid
    grid = defaultdict(list)
    
    # Assign circles to grid cells
    cell_size = 1.0 / grid_size
    for i in range(n):
        x, y, r = circles[i]
        # Determine which grid cells this circle touches
        min_col = max(0, int((x - r) / cell_size))
        max_col = min(grid_size - 1, int((x + r) / cell_size))
        min_row = max(0, int((y - r) / cell_size))
        max_row = min(grid_size - 1, int((y + r) / cell_size))
        
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                grid[(row, col)].append(i)
    
    # Check for overlaps within each grid cell and neighboring cells
    for (row, col), indices in grid.items():
        # Check pairs within same cell
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                x1, y1, r1 = circles[idx1]
                x2, y2, r2 = circles[idx2]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < r1 + r2:
                    return False
        
        # Check pairs with neighboring cells (8-connected)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor_cell = (row + dr, col + dc)
                if neighbor_cell in grid:
                    for idx1 in indices:
                        for idx2 in grid[neighbor_cell]:
                            x1, y1, r1 = circles[idx1]
                            x2, y2, r2 = circles[idx2]
                            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if dist < r1 + r2:
                                return False
    
    return True

def check_overlap(circles: np.ndarray) -> bool:
    """Check if any circles overlap."""
    return check_overlap_grid(circles)

def fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with hybrid grid-based approach."""
    population = []
    
    # Create a structured grid-based initialization
    def create_structured_initialization(n: int) -> np.ndarray:
        circles = np.zeros((n, 3))
        
        # Arrange in a grid pattern with some randomness
        rows = int(np.ceil(np.sqrt(n)))
        cols = rows
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Position with some jitter
                x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                
                # Initial radius calculation based on proximity to other points
                # For now use fixed radius and adjust later to avoid overlap
                radius = min(spacing_x, spacing_y) * 0.3
                
                circles[idx] = [x, y, radius]
                idx += 1
        
        # Refine to avoid overlaps using iterative improvement
        # Use simple greedy approach to adjust radii
        for i in range(n):
            # Find minimum distance to other circles
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    x1, y1, _ = circles[i]
                    x2, y2, _ = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_dist = min(min_dist, dist)
            
            # Set radius to be half the minimum distance to other circles, with bounds
            if min_dist < float('inf'):
                max_radius = min(circles[i][0], circles[i][1], 
                               1 - circles[i][0], 1 - circles[i][1])
                new_radius = min(min_dist / 2.0, max_radius * 0.8)
                circles[i][2] = max(0.001, min(new_radius, 0.4))
        
        return circles
    
    # Create multiple initializations
    for _ in range(pop_size):
        # Try structured initialization first
        circles = create_structured_initialization(n_circles)
        
        # If valid, add to population
        if check_containment(circles) and check_overlap(circles):
            population.append(circles)
        else:
            # Try to fix it by adjusting positions and radii
            circles = circles.copy()
            max_iterations = 100
            for _ in range(max_iterations):
                # Try to make it valid by reducing radii and adjusting positions
                valid = True
                for i in range(n_circles):
                    # Ensure containment
                    x, y, r = circles[i]
                    if x - r < 0:
                        x = r + 0.001
                    if x + r > 1:
                        x = 1 - r - 0.001
                    if y - r < 0:
                        y = r + 0.001
                    if y + r > 1:
                        y = 1 - r - 0.001
                    circles[i] = [x, y, r]
                
                # Reduce radii for overlap correction
                for i in range(n_circles):
                    for j in range(i+1, n_circles):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < r1 + r2:
                            # Adjust both radii to prevent overlap
                            total_radius = r1 + r2
                            new_total = dist * 0.99  # 1% margin
                            ratio = new_total / total_radius
                            
                            # Scale down radii
                            circles[i][2] *= ratio * 0.95
                            circles[j][2] *= ratio * 0.95
                
                # Ensure valid ranges
                for i in range(n_circles):
                    circles[i][2] = max(0.001, min(circles[i][2], 0.5))
                
                if check_containment(circles) and check_overlap(circles):
                    valid = True
                    break
            
            # Final fallback if still not valid
            if not (check_containment(circles) and check_overlap(circles)):
                circles = np.zeros((n_circles, 3))
                # Create uniform small circles
                for i in range(n_circles):
                    circles[i] = [0.5, 0.5, 0.01]
            
            population.append(circles)
    
    return population

def mutate(circles: np.ndarray, generation: int = 0, max_generations: int = 1000) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate."""
    # Adaptive mutation rate that decreases over generations
    mutation_rate_start = 0.15
    mutation_rate_end = 0.01
    
    # Sigmoidal decay function
    mutation_rate = mutation_rate_end + (mutation_rate_start - mutation_rate_end) * (
        1 / (1 + np.exp(10 * (generation / max_generations - 0.5)))
    )
    
    mutated = circles.copy()
    n = len(mutated)
    
    # Apply mutation to each circle
    for i in range(n):
        if random.random() < mutation_rate:
            # Choose type of mutation
            mutation_type = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
            
            if mutation_type == 0:  # Mutate x position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02),
                                      mutated[i, 2], 1 - mutated[i, 2])
            elif mutation_type == 1:  # Mutate y position  
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02),
                                      mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] * np.random.normal(1, 0.1), 0.001, 0.5)
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parent solutions."""
    # Uniform crossover with special handling for radius
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Crossover points for each circle
    for i in range(len(parent1)):
        if random.random() < 0.5:
            # Swap position and radius between parents
            child1[i, 0], child2[i, 0] = child2[i, 0], child1[i, 0]
            child1[i, 1], child2[i, 1] = child2[i, 1], child1[i, 1]
            child1[i, 2], child2[i, 2] = child2[i, 2], child1[i, 2]
    
    return child1, child2

def select_parents(population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Select two parents using tournament selection."""
    tournament_size = 3
    # Select first parent
    idx1 = random.randint(0, len(population)-1)
    best_idx = idx1
    best_fit = fitnesses[idx1]
    for _ in range(tournament_size - 1):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    parent1 = population[best_idx]
    
    # Select second parent
    idx2 = random.randint(0, len(population)-1)
    best_idx = idx2
    best_fit = fitnesses[idx2]
    for _ in range(tournament_size - 1):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    parent2 = population[best_idx]
    
    return parent1, parent2

def local_optimization(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply local optimization to improve circle packing."""
    optimized = circles.copy()
    
    # Simple local optimization using gradient descent-like approach
    for _ in range(max_iter):
        improved = False
        
        # Try to increase radii while maintaining constraints
        for i in range(len(optimized)):
            x, y, r = optimized[i]
            
            # Calculate maximum possible radius without violating boundaries
            max_radius = min(x, y, 1-x, 1-y)
            
            # Calculate minimum distance to other circles
            min_dist = float('inf')
            for j in range(len(optimized)):
                if i != j:
                    x2, y2, _ = optimized[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    min_dist = min(min_dist, dist)
            
            # If there's room to expand radius
            if min_dist > 0:
                new_radius = min((min_dist / 2.0), max_radius * 0.9)
                if new_radius > r:
                    optimized[i, 2] = new_radius
                    improved = True
        
        # If no improvement after a full iteration, stop
        if not improved:
            break
    
    return optimized

def optimize_circles_evolutionary(max_generations: int = 1000, pop_size: int = 50) -> np.ndarray:
    """Evolutionary optimization for circle packing."""
    n = 26
    
    # Initialize population
    population = initialize_population(pop_size, n)
    best_solution = None
    best_fitness = -float('inf')
    
    # Track best fitness history for early stopping
    fitness_history = []
    
    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitnesses = []
        for circles in population:
            if check_containment(circles) and check_overlap(circles):
                fit = fitness(circles)
                fitnesses.append(fit)
            else:
                fitnesses.append(-1000)  # Penalize invalid solutions
        
        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()
        
        # Store fitness history for early stopping
        fitness_history.append(best_fitness)
        if len(fitness_history) > 10:
            fitness_history.pop(0)
        
        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        # Keep best individuals (elitism) - 1/3 instead of 1/4
        sorted_indices = np.argsort(fitnesses)[::-1][:pop_size//3]
        for idx in sorted_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1, parent2 = select_parents(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation with generation info
            child1 = mutate(child1, generation, max_generations)
            child2 = mutate(child2, generation, max_generations)
            
            # Local optimization
            child1 = local_optimization(child1)
            child2 = local_optimization(child2)
            
            # Ensure validity
            if check_containment(child1) and check_overlap(child1):
                new_population.append(child1)
            else:
                # Try to fix if invalid - copy parent and slightly adjust
                fixed_child = parent1.copy()
                # Make small adjustments to the parent to make it valid
                for i in range(len(fixed_child)):
                    x, y, r = fixed_child[i]
                    # Adjust position to keep within bounds
                    fixed_child[i, 0] = np.clip(x + np.random.normal(0, 0.005), r, 1-r)
                    fixed_child[i, 1] = np.clip(y + np.random.normal(0, 0.005), r, 1-r)
                new_population.append(fixed_child)
            
            if len(new_population) < pop_size and check_containment(child2) and check_overlap(child2):
                new_population.append(child2)
            elif len(new_population) < pop_size:
                # Try to fix second child
                fixed_child = parent2.copy()
                # Make small adjustments to the parent to make it valid
                for i in range(len(fixed_child)):
                    x, y, r = fixed_child[i]
                    fixed_child[i, 0] = np.clip(x + np.random.normal(0, 0.005), r, 1-r)
                    fixed_child[i, 1] = np.clip(y + np.random.normal(0, 0.005), r, 1-r)
                new_population.append(fixed_child)
        
        population = new_population[:pop_size]
        
        # Early stopping if fitness hasn't improved in the last 200 generations
        if generation > 200 and len(fitness_history) >= 10:
            recent_improvement = max(fitness_history) - min(fitness_history)
            if recent_improvement < 0.0001:
                print(f"Early stopping at generation {generation}")
                break
    
    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run evolutionary optimization
    circles = optimize_circles_evolutionary(max_generations=500, pop_size=30)
    
    # Final validation
    if circles is None or not check_containment(circles) or not check_overlap(circles):
        # Fallback to a simple arrangement if optimization failed
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.3
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1
        
        # Adjust last few circles to fit
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.01]
    
    return circles

# EVOLVE-BLOCK-END