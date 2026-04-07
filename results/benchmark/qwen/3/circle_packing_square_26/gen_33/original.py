# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from copy import deepcopy

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 26
    max_generations = 1000
    population_size = 50
    
    def initialize_population(pop_size, n_circles):
        """Initialize population with diverse configurations"""
        population = []
        for _ in range(pop_size):
            # Hybrid initialization: grid + perturbation + voronoi-like
            circles = np.zeros((n_circles, 3))
            
            # Grid-based initial placement
            sqrt_n = int(np.ceil(np.sqrt(n_circles)))
            grid_positions = []
            for i in range(sqrt_n):
                for j in range(sqrt_n):
                    if len(grid_positions) < n_circles:
                        x = (i + 0.5) / sqrt_n
                        y = (j + 0.5) / sqrt_n
                        grid_positions.append([x, y])
            
            # Add some randomness and Voronoi-like distribution
            for i in range(n_circles):
                x, y = grid_positions[i]
                # Add small random perturbation
                x += np.random.normal(0, 0.03)
                y += np.random.normal(0, 0.03)
                
                # Clamp to unit square
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                # Initial radius based on distance to neighbors
                min_dist = float('inf')
                for other_x, other_y in grid_positions[:i]:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)
                
                # Set radius to be safe from boundaries and neighbors
                radius = min(0.05, x, 1-x, y, 1-y, min_dist/2)
                if radius <= 0:
                    radius = 0.01
                    
                circles[i] = [x, y, radius]
            
            population.append(circles)
        return population
    
    def is_valid(circles):
        """Check if all circles are within bounds and non-overlapping"""
        n = len(circles)
        
        # Check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints using KDTree for efficiency
        coords = circles[:, :2]
        try:
            tree = cKDTree(coords)
            pairs = tree.query_pairs(r=0.0001)  # This will find all points at distance 0, so we'll use a small epsilon
            
            # Check overlaps with a proper distance threshold
            distances = cdist(coords, coords)
            np.fill_diagonal(distances, float('inf'))  # Ignore self-distances
            
            # Find any overlapping circles
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i][j]
                    if dist < circles[i][2] + circles[j][2]:  # Overlapping
                        return False
                        
        except Exception:
            # Fallback to brute force if KDTree fails
            for i in range(n):
                x1, y1, r1 = circles[i]
                for j in range(i+1, n):
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        return False
        
        return True
    
    def evaluate_fitness(circles):
        """Evaluate fitness of a solution"""
        if not is_valid(circles):
            # Dynamic penalty based on constraint violations
            total_penalty = 0
            
            # Boundary violations
            for i in range(len(circles)):
                x, y, r = circles[i]
                boundary_violation = 0
                if x - r < 0:
                    boundary_violation += abs(x - r)
                if x + r > 1:
                    boundary_violation += abs(x + r - 1)
                if y - r < 0:
                    boundary_violation += abs(y - r)
                if y + r > 1:
                    boundary_violation += abs(y + r - 1)
                total_penalty += boundary_violation * 1000
            
            # Overlap violations
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    overlap = max(0, r1 + r2 - distance)
                    total_penalty += overlap * 10000
            
            return -total_penalty
        
        # Valid configuration: maximize sum of radii
        return np.sum(circles[:, 2])
    
    def mutate(circles, generation, max_generations):
        """Apply mutation to circles"""
        mutated = deepcopy(circles)
        mutation_rate = 0.15 - (0.13 * generation / max_generations)  # Adaptive mutation rate
        
        # Mutate each circle with some probability
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate position
                mutated[i, 0] += np.random.normal(0, 0.02)  # x coordinate
                mutated[i, 1] += np.random.normal(0, 0.02)  # y coordinate
                
                # Clamp to unit square
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
                
                # Mutate radius (with care to keep it positive)
                mutated[i, 2] += np.random.normal(0, 0.01)
                mutated[i, 2] = max(0.001, mutated[i, 2])
                
        return mutated
    
    def crossover(parent1, parent2):
        """Create offspring via crossover of two parents"""
        child = deepcopy(parent1)
        n = len(parent1)
        
        # Single-point crossover
        crossover_point = random.randint(1, n-1)
        
        for i in range(crossover_point, n):
            child[i] = parent2[i].copy()
            
        return child
    
    def tournament_selection(population, k=3):
        """Select individual using tournament selection"""
        selected = random.sample(population, k)
        return max(selected, key=evaluate_fitness)
    
    def refine_solution(circles):
        """Apply local refinement to fix minor constraint violations"""
        refined = deepcopy(circles)
        # Simple refinement: make sure all constraints are satisfied
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Ensure containment
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]
        return refined
    
    # Initialize population
    population = initialize_population(population_size, n_circles)
    
    # Evolve
    best_fitness = float('-inf')
    best_solution = None
    
    for generation in range(max_generations):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])
        
        # Elitism: keep top 10%
        elite_count = max(1, population_size // 10)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite = [population[i] for i in sorted_indices]
        
        # Create new population
        new_population = deepcopy(elite)
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            mutated_child = mutate(child, generation, max_generations)
            
            # Local refinement
            refined_child = refine_solution(mutated_child)
            
            new_population.append(refined_child)
        
        population = new_population[:population_size]
    
    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to first individual if nothing was found
        return population[0]

# EVOLVE-BLOCK-END
