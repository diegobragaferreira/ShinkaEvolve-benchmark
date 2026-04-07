# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import random
import math
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def is_valid_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Check if the solution satisfies all constraints"""
    if len(circles) == 0:
        return False
    
    # Check if all circles are within boundaries
    for x, y, r in circles:
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    # Check for overlaps using efficient spatial indexing
    tree = cKDTree(circles[:, :2])
    pairs = tree.query_pairs(r=0.0001, output_type='ndarray')
    
    # Check actual overlaps
    for i, j in pairs:
        if i >= j:
            continue
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance < r1 + r2:
            return False
    
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def initialize_population(n_circles: int, n_pop: int, rect_width: float = 1.0, rect_height: float = 1.0) -> List[np.ndarray]:
    """Initialize multiple starting populations with different patterns"""
    populations = []
    
    # Pattern 1: Hexagonal packing
    def hexagonal_pattern():
        circles = []
        # Parameters for hexagonal grid
        rows = int(math.sqrt(n_circles))
        cols = (n_circles + rows - 1) // rows
        
        # Adjust spacing based on available area
        spacing_x = rect_width / (cols + 1)
        spacing_y = rect_height / (rows + 1)
        
        # Calculate max possible radius based on spacing
        max_radius = min(spacing_x, spacing_y) / 2.0
        
        count = 0
        for i in range(rows):
            for j in range(cols): 
                if count >= n_circles:
                    break
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2.0
                circles.append([x, y, max_radius * 0.8])
                count += 1
            if count >= n_circles:
                break
        
        # Ensure we have exactly n_circles
        while len(circles) < n_circles:
            circles.append([rect_width/2, rect_height/2, max_radius * 0.5])
            
        return np.array(circles[:n_circles])
    
    # Pattern 2: Triangular packing
    def triangular_pattern():
        circles = []
        # Simple triangular arrangement
        center_x, center_y = rect_width/2, rect_height/2
        base_radius = min(rect_width, rect_height) * 0.1
        angle_step = 2 * math.pi / n_circles
        
        for i in range(n_circles):
            angle = i * angle_step
            radius = base_radius * (0.7 + 0.3 * (i % 3))  # Varying radii
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            circles.append([x, y, radius])
        
        return np.array(circles)
    
    # Pattern 3: Grid pattern
    def grid_pattern():
        circles = []
        rows = int(math.sqrt(n_circles))
        cols = (n_circles + rows - 1) // rows
        
        spacing_x = rect_width / (cols + 1)
        spacing_y = rect_height / (rows + 1)
        max_radius = min(spacing_x, spacing_y) / 2.0
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n_circles:
                    break
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                circles.append([x, y, max_radius * 0.7])
                count += 1
            if count >= n_circles:
                break
        
        return np.array(circles[:n_circles])
    
    # Pattern 4: Random with constraints
    def random_pattern():
        circles = []
        for _ in range(n_circles):
            # Generate random position with some margin
            x = random.uniform(0.05, rect_width - 0.05)
            y = random.uniform(0.05, rect_height - 0.05)
            max_radius = min(x, rect_width - x, y, rect_height - y)
            radius = random.uniform(0.01, max_radius * 0.5)
            circles.append([x, y, radius])
        return np.array(circles)
    
    # Generate multiple diverse initial populations
    patterns = [hexagonal_pattern, triangular_pattern, grid_pattern, random_pattern]
    for i in range(n_pop):
        pattern_func = random.choice(patterns)
        try:
            pop = pattern_func()
            # Ensure all circles fit within boundaries
            for idx in range(len(pop)):
                if pop[idx][0] - pop[idx][2] < 0 or pop[idx][0] + pop[idx][2] > rect_width:
                    pop[idx][0] = max(pop[idx][2], min(rect_width - pop[idx][2], pop[idx][0]))
                if pop[idx][1] - pop[idx][2] < 0 or pop[idx][1] + pop[idx][2] > rect_height:
                    pop[idx][1] = max(pop[idx][2], min(rect_height - pop[idx][2], pop[idx][1]))
            populations.append(pop)
        except Exception:
            # Fallback to simple random initialization
            fallback = np.random.rand(n_circles, 3)
            fallback[:, 0] *= rect_width
            fallback[:, 1] *= rect_height
            fallback[:, 2] *= min(rect_width, rect_height) * 0.1
            populations.append(fallback)
    
    return populations

def mutate_individual(individual: np.ndarray, generation: int, max_gen: int, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Mutate an individual with adaptive mutation rate"""
    mutated = individual.copy()
    
    # Dynamic mutation rate decreasing over generations
    mutation_rate = max(0.1, 0.5 * (1 - generation / max_gen))
    
    n_circles = len(mutated)
    
    # Mutate positions
    for i in range(n_circles):
        if random.random() < mutation_rate:
            # Randomly decide what to mutate (position or radius)
            if random.random() < 0.7:  # Mutate position
                dx = random.uniform(-0.1, 0.1)
                dy = random.uniform(-0.1, 0.1)
                
                new_x = max(0.01, min(rect_width - 0.01, mutated[i][0] + dx))
                new_y = max(0.01, min(rect_height - 0.01, mutated[i][1] + dy))
                
                mutated[i][0] = new_x
                mutated[i][1] = new_y
            else:  # Mutate radius
                dr = random.uniform(-0.05, 0.05)
                new_r = max(0.01, mutated[i][2] + dr)
                mutated[i][2] = new_r
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Crossover two parents to create offspring"""
    child = parent1.copy()
    n_circles = len(parent1)
    
    # Single point crossover
    crossover_point = random.randint(1, n_circles - 1)
    
    for i in range(crossover_point, n_circles):
        child[i] = parent2[i].copy()
    
    return child

def evaluate_fitness(individual: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> float:
    """Evaluate fitness of an individual"""
    if not is_valid_solution(individual, rect_width, rect_height):
        return -float('inf')  # Penalize invalid solutions heavily
    
    return calculate_sum_radii(individual)

def local_optimization(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0, max_iter: int = 100) -> np.ndarray:
    """Perform local optimization on a solution using scipy minimize"""
    def objective(params):
        # Reshape parameters back to circles array
        new_circles = circles.copy()
        for i in range(len(new_circles)):
            new_circles[i][0] = params[3*i]
            new_circles[i][1] = params[3*i+1]
            new_circles[i][2] = params[3*i+2]
        
        # Penalize if invalid
        if not is_valid_solution(new_circles, rect_width, rect_height):
            return -1e10
            
        return -calculate_sum_radii(new_circles)  # Negative because we minimize
    
    def constraint_func(params):
        # Ensure all circles are within boundaries
        new_circles = circles.copy()
        for i in range(len(new_circles)):
            new_circles[i][0] = params[3*i]
            new_circles[i][1] = params[3*i+1]
            new_circles[i][2] = params[3*i+2]
        
        return calculate_sum_radii(new_circles)
    
    # Flatten current solution
    initial_params = []
    for circle in circles:
        initial_params.extend(circle)
    
    # Define bounds for each parameter
    bounds = []
    for i in range(len(circles)):
        bounds.append((0.01, rect_width - 0.01))      # x coordinate
        bounds.append((0.01, rect_height - 0.01))    # y coordinate
        bounds.append((0.01, min(rect_width, rect_height)/2))  # radius
    
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
        if result.success:
            # Extract optimized solution
            optimized_circles = circles.copy()
            for i in range(len(optimized_circles)):
                optimized_circles[i][0] = result.x[3*i]
                optimized_circles[i][1] = result.x[3*i+1]
                optimized_circles[i][2] = result.x[3*i+2]
            return optimized_circles
    except:
        pass
    
    return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 21
    n_pop = 20  # Population size
    n_gen = 50  # Generations
    elite_size = 4  # Number of elites to keep
    rect_width = 1.0
    rect_height = 1.0
    
    # Initialize population
    populations = initialize_population(n_circles, n_pop, rect_width, rect_height)
    
    # Evaluate initial population
    fitness_scores = []
    for pop in populations:
        fitness = evaluate_fitness(pop, rect_width, rect_height)
        fitness_scores.append(fitness)
    
    # Sort by fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]
    current_population = [populations[i] for i in sorted_indices]
    current_fitness = [fitness_scores[i] for i in sorted_indices]
    
    best_solution = current_population[0].copy()
    best_fitness = current_fitness[0]
    
    # Evolutionary loop
    for gen in range(n_gen):
        # Apply local optimization to top individuals
        for i in range(min(elite_size, len(current_population))):
            current_population[i] = local_optimization(current_population[i], rect_width, rect_height)
            new_fitness = evaluate_fitness(current_population[i], rect_width, rect_height)
            if new_fitness > current_fitness[i]:
                current_fitness[i] = new_fitness
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_solution = current_population[i].copy()
        
        # Create new population through selection, crossover and mutation
        new_population = current_population[:elite_size]  # Keep elites
        
        # Tournament selection
        def tournament_selection(tournament_size=3):
            tournament_indices = random.sample(range(len(current_population)), tournament_size)
            tournament_fitness = [current_fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            return current_population[winner_idx].copy()
        
        # Generate offspring
        while len(new_population) < n_pop:
            parent1 = tournament_selection()
            parent2 = tournament_selection()
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, gen, n_gen, rect_width, rect_height)
            
            # Local optimization
            child = local_optimization(child, rect_width, rect_height)
            
            new_population.append(child)
        
        # Update population
        current_population = new_population[:n_pop]
        current_fitness = [evaluate_fitness(pop, rect_width, rect_height) for pop in current_population]
        
        # Update best solution
        max_fitness_idx = np.argmax(current_fitness)
        if current_fitness[max_fitness_idx] > best_fitness:
            best_fitness = current_fitness[max_fitness_idx]
            best_solution = current_population[max_fitness_idx].copy()
    
    # Final local optimization
    final_solution = local_optimization(best_solution, rect_width, rect_height, max_iter=200)
    final_fitness = evaluate_fitness(final_solution, rect_width, rect_height)
    
    if final_fitness > best_fitness:
        return final_solution
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
