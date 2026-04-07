# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from copy import deepcopy
from sklearn.cluster import KMeans
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Optimize rectangle dimensions using golden ratio for better packing
    phi = (1 + math.sqrt(5)) / 2
    rect_width = 2 / (1 + phi)  # Width around 0.764
    rect_height = 2 / (1 + 1/phi)  # Height around 1.236
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    n_circles = 21
    max_iterations = 500
    population_size = 100
    elite_size = 10
    initial_mutation_rate = 0.2
    min_mutation_rate = 0.05
    
    # Initialize population with multiple strategies
    def create_initial_population(size):
        population = []
        
        # Strategy 1: Hexagonal packing (good initial layout)
        while len(population) < size // 3:
            circles = []
            try:
                # Create hexagonal packing pattern
                rows = int(np.ceil(np.sqrt(n_circles)))
                cols = int(np.ceil(n_circles / rows))
                
                # Hexagonal spacing
                cell_width = rect_width / cols
                cell_height = rect_height / rows
                
                # Adjust for hexagonal packing
                hex_radius = min(cell_width, cell_height) * 0.4
                hex_spacing_x = hex_radius * 2
                hex_spacing_y = hex_radius * np.sqrt(3)
                
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if idx >= n_circles:
                            break
                        # Offset every other row
                        x_offset = (i % 2) * (hex_spacing_x / 2)
                        x = x_offset + j * hex_spacing_x + hex_radius
                        y = i * hex_spacing_y + hex_radius
                        
                        # Check bounds
                        if x > rect_width - hex_radius or y > rect_height - hex_radius:
                            continue
                            
                        circles.append([x, y, hex_radius])
                        idx += 1
                        
                if len(circles) == n_circles:
                    population.append(np.array(circles))
            except:
                pass
        
        # Strategy 2: Grid-based with slight randomization
        while len(population) < 2 * size // 3:
            circles = []
            try:
                rows = int(np.ceil(np.sqrt(n_circles)))
                cols = int(np.ceil(n_circles / rows))
                
                cell_width = rect_width / cols
                cell_height = rect_height / rows
                
                base_radius = min(cell_width, cell_height) * 0.4
                
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if idx >= n_circles:
                            break
                        x = (j + 0.5) * cell_width + np.random.normal(0, cell_width * 0.1)
                        y = (i + 0.5) * cell_height + np.random.normal(0, cell_height * 0.1)
                        
                        # Ensure within bounds
                        x = np.clip(x, base_radius, rect_width - base_radius)
                        y = np.clip(y, base_radius, rect_height - base_radius)
                        
                        r = base_radius + np.random.normal(0, base_radius * 0.1)
                        r = np.clip(r, 0.01, base_radius * 1.5)
                        
                        circles.append([x, y, r])
                        idx += 1
                        
                if len(circles) == n_circles:
                    population.append(np.array(circles))
            except:
                pass
        
        # Strategy 3: Random placement with overlap checking
        while len(population) < size:
            circles = []
            attempt = 0
            max_attempts = 1000
            
            while len(circles) < n_circles and attempt < max_attempts:
                x = np.random.uniform(0.05, rect_width - 0.05)
                y = np.random.uniform(0.05, rect_height - 0.05)
                r = np.random.uniform(0.01, 0.2)
                
                # Check if it collides with existing circles
                valid = True
                if len(circles) > 0:
                    positions = np.array([[c[0], c[1]] for c in circles])
                    distances = cdist([[x, y]], positions)[0]
                    for d, c in zip(distances, circles):
                        if d < (r + c[2]):
                            valid = False
                            break
                
                if valid:
                    # Check boundary
                    if (x - r >= 0 and x + r <= rect_width and 
                        y - r >= 0 and y + r <= rect_height):
                        circles.append([x, y, r])
                
                attempt += 1
            
            if len(circles) == n_circles:
                population.append(np.array(circles))
        
        return population
    
    # Efficient collision detection using KDTree
    def is_valid_layout_fast(circles):
        # Check boundary constraints
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        
        # Use KDTree for efficient neighbor search
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Query for neighbors within sum of radii distance
        distances = tree.query_pairs(2 * max(circles[:, 2]), p=np.inf)
        
        # Verify actual overlap
        for i, j in distances:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + dy*dy
            radii_sum = r1 + r2
            
            if dist_sq < radii_sum * radii_sum:
                return False
                
        return True
    
    # Fast fitness evaluation with early termination
    def evaluate_fitness_fast(circles):
        if not is_valid_layout_fast(circles):
            return 0.0  # Invalid layouts get zero fitness
            
        return np.sum(circles[:, 2])  # Sum of radii
    
    # Create initial population
    population = create_initial_population(population_size)
    
    # If we don't have enough initial individuals, fill with random valid ones
    while len(population) < population_size:
        circles = np.zeros((n_circles, 3))
        attempts = 0
        while attempts < 1000:
            # Random positions and radii
            for i in range(n_circles):
                x = np.random.uniform(0.05, rect_width - 0.05)
                y = np.random.uniform(0.05, rect_height - 0.05)
                r = np.random.uniform(0.01, 0.2)
                circles[i] = [x, y, r]
            
            if is_valid_layout_fast(circles):
                population.append(circles.copy())
                break
            attempts += 1
    
    # Track best solution
    best_fitness = 0.0
    best_individual = None
    
    # Early stopping condition
    patience = 20
    patience_counter = 0
    prev_best = 0.0
    
    for generation in range(max_iterations):
        # Adaptive mutation rate
        current_mutation_rate = max(min_mutation_rate, 
                                   initial_mutation_rate * (1 - generation / max_iterations))
        
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            score = evaluate_fitness_fast(individual)
            fitness_scores.append(score)
            
            if score > best_fitness:
                best_fitness = score
                best_individual = individual.copy()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Check for early stopping
        if abs(prev_best - best_fitness) < 1e-6:
            patience_counter += 1
        else:
            patience_counter = 0
            prev_best = best_fitness
            
        if patience_counter >= patience:
            break
        
        # Keep elite
        new_population = population[:elite_size]
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection with bias towards feasible solutions
            parent1_idx = tournament_selection(population, fitness_scores, 3)
            parent2_idx = tournament_selection(population, fitness_scores, 3)
            
            # Crossover
            child = crossover(population[parent1_idx], population[parent2_idx], rect_width, rect_height)
            
            # Mutation
            mutate(child, current_mutation_rate, rect_width, rect_height)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best solution found
    if best_individual is not None:
        return np.array(best_individual)
    else:
        # Fallback to last generation if nothing was found
        return population[0]

def tournament_selection(population, fitness_scores, k):
    """Select individual via tournament selection with feasibility bias"""
    # Sample k individuals
    tournament_indices = np.random.choice(len(population), k)
    
    # Filter out invalid individuals (low fitness) from consideration
    valid_indices = []
    valid_fitness = []
    
    for i in tournament_indices:
        # Only consider individuals with reasonable fitness (not zero)
        if fitness_scores[i] > 0.01:
            valid_indices.append(i)
            valid_fitness.append(fitness_scores[i])
    
    # If we have valid individuals, select from them
    if len(valid_indices) > 0:
        winner_idx = valid_indices[np.argmax(valid_fitness)]
    else:
        # Otherwise fall back to normal tournament
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        winner_idx = winner_index
    
    return winner_idx

def crossover(parent1, parent2, rect_width, rect_height):
    """Advanced crossover with better geometric properties"""
    child = deepcopy(parent1)
    
    # Uniform crossover with special care for boundaries
    crossover_mask = np.random.rand(len(parent1)) > 0.5
    
    # For positions and radii
    child[crossover_mask, :2] = parent2[crossover_mask, :2]  # Positions
    child[crossover_mask, 2] = parent2[crossover_mask, 2]   # Radii
    
    # Fix potential boundary violations
    for i in range(len(child)):
        x, y, r = child[i]
        # Ensure boundaries
        x = np.clip(x, r, rect_width - r)
        y = np.clip(y, r, rect_height - r)
        child[i] = [x, y, r]
    
    return child

def mutate(individual, mutation_rate, rect_width, rect_height):
    """Improved mutation operator with proper boundary handling"""
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # Mutate position with bounded Gaussian
            delta_x = np.random.normal(0, 0.05)
            delta_y = np.random.normal(0, 0.05)
            new_x = individual[i, 0] + delta_x
            new_y = individual[i, 1] + delta_y
            
            # Clip to boundaries with safety margin
            new_x = np.clip(new_x, 0.05, rect_width - 0.05)
            new_y = np.clip(new_y, 0.05, rect_height - 0.05)
            
            # Mutate radius with bounded Gaussian
            delta_r = np.random.normal(0, 0.02)
            new_r = individual[i, 2] + delta_r
            new_r = np.clip(new_r, 0.01, 0.3)
            
            individual[i] = [new_x, new_y, new_r]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
