# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import random
from copy import deepcopy
from typing import Tuple, List
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - perimeter = 4, so width + height = 2
    # Using 2:1 ratio for better packing efficiency (width:height = 1.33:0.67)
    rect_width = 1.3333333333333333
    rect_height = 0.6666666666666667
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    n_circles = 21
    max_iterations = 1000
    population_size = 100
    elite_size = 10
    initial_mutation_rate = 0.15
    final_mutation_rate = 0.05
    
    # Enhanced grid-based initialization with multiple grid configurations
    def create_initial_population(size: int) -> List[np.ndarray]:
        population = []
        
        # Try different grid arrangements to find good starting points
        grid_configs = [
            (4, 6),   # 4 rows, 6 columns
            (3, 7),   # 3 rows, 7 columns  
            (5, 5),   # 5 rows, 5 columns
            (6, 4),   # 6 rows, 4 columns
        ]
        
        # Add some random variations to ensure diversity
        for _ in range(size):
            if len(population) >= size:
                break
                
            # Try different grid configurations
            for rows, cols in grid_configs:
                if rows * cols >= n_circles:
                    # Adjust grid size to fit within rectangle
                    cell_width = rect_width / cols
                    cell_height = rect_height / rows
                    
                    # Use smaller margin for better packing
                    radius = min(cell_width, cell_height) * 0.45
                    
                    circles = []
                    idx = 0
                    
                    for i in range(rows):
                        for j in range(cols):
                            if idx >= n_circles:
                                break
                            x = (j + 0.5) * cell_width
                            y = (i + 0.5) * cell_height
                            
                            # Make sure we're within bounds
                            if x > rect_width or y > rect_height:
                                continue
                                
                            circles.append([x, y, radius])
                            idx += 1
                            
                    if len(circles) == n_circles:
                        # Add slight random perturbations to avoid perfect grids
                        circles = np.array(circles)
                        for i in range(n_circles):
                            circles[i][0] += np.random.uniform(-0.02, 0.02)
                            circles[i][1] += np.random.uniform(-0.02, 0.02)
                            circles[i][2] *= np.random.uniform(0.9, 1.1)
                        population.append(circles)
                        break
        
        # Fill remaining population with random valid configurations
        while len(population) < size:
            circles = np.zeros((n_circles, 3))
            # Try generating random valid circles
            attempts = 0
            while attempts < 1000:
                # Random positions and radii
                for i in range(n_circles):
                    x = np.random.uniform(0.05, rect_width - 0.05)
                    y = np.random.uniform(0.05, rect_height - 0.05)
                    r = np.random.uniform(0.01, 0.2)
                    circles[i] = [x, y, r]
                
                if is_valid_layout(circles):
                    population.append(circles.copy())
                    break
                attempts += 1
        
        return population
    
    # Efficient overlap detection using KDTree
    def is_valid_layout(circles: np.ndarray) -> bool:
        # Check boundary constraints
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        
        # Check pairwise collisions efficiently using KDTree
        try:
            tree = KDTree(circles[:, :2])
            pairs = tree.query_pairs(r=0.0001, predicate=lambda x, y: x + y > 0)
            
            for i, j in pairs:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Circles should not overlap
                if dist < r1 + r2:
                    return False
        except:
            # Fallback to brute-force if KDTree fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    
                    # Distance between centers
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Circles should not overlap
                    if dist < r1 + r2:
                        return False
        
        return True
    
    # Improved fitness evaluation with penalties for constraint violations
    def evaluate_fitness(circles: np.ndarray) -> float:
        # Check validity first
        if not is_valid_layout(circles):
            # Apply strong penalty for invalid layouts
            return -1000.0
        
        # Return sum of radii for valid layouts
        return np.sum(circles[:, 2])
    
    # Local refinement function to improve promising solutions
    def local_refinement(circles: np.ndarray, iterations: int = 50) -> np.ndarray:
        """Perform local search to improve a solution"""
        current = circles.copy()
        best_fitness = evaluate_fitness(current)
        
        for _ in range(iterations):
            # Create candidate by slightly modifying one circle at a time
            candidate = current.copy()
            idx = np.random.randint(0, len(candidate))
            
            # Slightly modify position and radius
            candidate[idx][0] = np.clip(
                candidate[idx][0] + np.random.normal(0, 0.03), 
                0.05, rect_width - 0.05)
            candidate[idx][1] = np.clip(
                candidate[idx][1] + np.random.normal(0, 0.03), 
                0.05, rect_height - 0.05)
            candidate[idx][2] = np.clip(
                candidate[idx][2] + np.random.normal(0, 0.015),
                0.01, 0.3)
            
            # If valid and better, accept
            new_fitness = evaluate_fitness(candidate)
            if new_fitness > best_fitness:
                current = candidate
                best_fitness = new_fitness
        
        return current
    
    # Create initial population
    population = create_initial_population(population_size)
    
    # Genetic algorithm loop
    best_fitness = -float('inf')
    best_individual = None
    
    start_time = time.time()
    
    for generation in range(max_iterations):
        # Adaptive mutation rate
        mutation_rate = initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * (generation / max_iterations)
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            score = evaluate_fitness(individual)
            fitness_scores.append(score)
            
            if score > best_fitness:
                best_fitness = score
                best_individual = individual.copy()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        new_population = population[:elite_size]
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = tournament_selection(population, fitness_scores, 3)
            parent2_idx = tournament_selection(population, fitness_scores, 3)
            
            # Crossover
            child = crossover(population[parent1_idx], population[parent2_idx])
            
            # Local refinement on the child before mutation
            child = local_refinement(child)
            
            # Mutation with adaptive rate
            mutate(child, mutation_rate, rect_width, rect_height)
            
            # Local refinement after mutation
            child = local_refinement(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Early stopping if no improvement for many generations
        if generation > 20 and abs(best_fitness - evaluate_fitness(population[0])) < 1e-6:
            break
            
        # Timeout check
        if time.time() - start_time > 55:  # Leave 5 seconds for cleanup
            break
    
    # Final local refinement on best solution
    if best_individual is not None:
        best_individual = local_refinement(best_individual, 100)
        return np.array(best_individual)
    else:
        # Fallback to last generation if nothing was found
        return population[0]

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], k: int) -> int:
    """Select individual via tournament selection"""
    tournament_indices = np.random.choice(len(population), k)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Single point crossover on circle positions and radii"""
    child = deepcopy(parent1)
    
    # Select crossover point
    crossover_point = np.random.randint(1, len(parent1))
    
    # Cross over positions and radii
    child[crossover_point:, :2] = parent2[crossover_point:, :2]  # Positions
    child[crossover_point:, 2] = parent2[crossover_point:, 2]   # Radii
    
    return child

def mutate(individual: np.ndarray, mutation_rate: float, rect_width: float, rect_height: float) -> None:
    """Mutate circle positions and radii"""
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # Mutate position
            individual[i, 0] = np.clip(
                individual[i, 0] + np.random.normal(0, 0.05), 
                0.05, rect_width - 0.05)
            individual[i, 1] = np.clip(
                individual[i, 1] + np.random.normal(0, 0.05), 
                0.05, rect_height - 0.05)
            
            # Mutate radius
            individual[i, 2] = np.clip(
                individual[i, 2] + np.random.normal(0, 0.02),
                0.01, 0.3)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
