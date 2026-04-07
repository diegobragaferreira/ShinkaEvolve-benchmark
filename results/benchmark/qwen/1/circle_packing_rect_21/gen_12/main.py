# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_constraints_fast(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Efficiently check constraints using spatial indexing."""
    n = len(circles)
    
    # Check boundary constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    # Use KDTree for efficient neighbor search
    if n > 1:
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Build KDTree for fast nearest neighbor queries
        tree = cKDTree(positions)
        
        # Query for neighbors within sum of radii distance
        max_radius_sum = np.max(radii) + np.max(radii)
        pairs = tree.query_pairs(max_radius_sum * 2.0, output_type='ndarray')
        
        # Check actual overlaps
        for i, j in pairs:
            if i < j:  # Avoid duplicate checks
                distance = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                if distance < (radii[i] + radii[j]):
                    return False
    
    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii with constraint validation."""
    if not check_constraints_fast(circles):
        return -np.inf
    
    return np.sum(circles[:, 2])

def create_hexagonal_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Create high-quality initial solution using hexagonal lattice pattern."""
    circles = np.zeros((21, 3))
    
    # Hexagonal packing parameters
    # Determine grid spacing based on rectangle dimensions
    rows = int(np.sqrt(21))
    cols = int(np.ceil(21 / rows))
    
    # Adjust for rectangular container
    if rect_width > rect_height:
        # Wider rectangle - more columns
        cols = min(cols, int(rect_width / (rect_height * 0.8)))
        rows = int(np.ceil(21 / cols))
    else:
        # Taller rectangle - more rows
        rows = min(rows, int(rect_height / (rect_width * 0.8)))
        cols = int(np.ceil(21 / rows))
    
    # Ensure we don't exceed 21 circles
    total_cells = rows * cols
    actual_count = min(21, total_cells)
    
    # Calculate cell size based on rectangle dimensions
    cell_width = rect_width / cols
    cell_height = rect_height / rows
    
    # Circle radius based on cell size
    max_radius = min(cell_width, cell_height) * 0.4
    
    # Place circles in hexagonal pattern
    placed = 0
    for i in range(rows):
        for j in range(cols):
            if placed >= 21:
                break
                
            # Hexagonal offset pattern
            x_offset = j * cell_width + cell_width / 2
            y_offset = i * cell_height + cell_height / 2
            
            # Offset every other row
            if i % 2 == 1:
                x_offset += cell_width / 2
            
            # Ensure within bounds
            x = max(max_radius, min(rect_width - max_radius, x_offset))
            y = max(max_radius, min(rect_height - max_radius, y_offset))
            
            # Adjust radius based on proximity to edges
            r = max_radius * min(
                x / max_radius,
                (rect_width - x) / max_radius,
                y / max_radius,
                (rect_height - y) / max_radius
            )
            
            # Add some small randomness for diversity
            r *= np.random.uniform(0.8, 1.0)
            
            circles[placed] = [x, y, r]
            placed += 1
            
        if placed >= 21:
            break
    
    # Fill remaining positions with zeros if needed
    for i in range(placed, 21):
        circles[i] = [0, 0, 0]
    
    return circles

def mutate(circles: np.ndarray, mutation_rate: float = 0.3) -> np.ndarray:
    """Improved mutation operator with constraint preservation."""
    mutated = circles.copy()
    
    for i in range(21):
        if np.random.random() < mutation_rate:
            # Choose mutation type
            mutation_type = np.random.choice(['position', 'radius'])
            
            if mutation_type == 'position':
                # Mutate position with bounded Gaussian
                step_size = 0.05 + np.random.random() * 0.05
                mutated[i, 0] += np.random.normal(0, step_size)
                mutated[i, 1] += np.random.normal(0, step_size)
                
                # Ensure bounds are respected
                r = mutated[i, 2]
                mutated[i, 0] = np.clip(mutated[i, 0], r, 1.5 - r)
                mutated[i, 1] = np.clip(mutated[i, 1], r, 0.5 - r)
            else:
                # Mutate radius with log-normal distribution to avoid negative values
                scale_factor = np.exp(np.random.normal(0, 0.2))
                mutated[i, 2] *= scale_factor
                mutated[i, 2] = max(0.001, mutated[i, 2])
                mutated[i, 2] = min(0.4, mutated[i, 2])  # Cap max radius
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced crossover operator using blend crossover."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Blend crossover with better mixing
    for i in range(21):
        # For positions, use blend crossover with parameter alpha
        alpha = np.random.random()
        child1[i, :2] = alpha * parent1[i, :2] + (1 - alpha) * parent2[i, :2]
        child2[i, :2] = (1 - alpha) * parent1[i, :2] + alpha * parent2[i, :2]
        
        # For radii, use weighted average with some uniform noise
        beta = np.random.random()
        child1[i, 2] = beta * parent1[i, 2] + (1 - beta) * parent2[i, 2]
        child2[i, 2] = (1 - beta) * parent1[i, 2] + beta * parent2[i, 2]
        
        # Add small random variation
        if np.random.random() < 0.3:
            child1[i, 2] *= np.random.normal(1, 0.1)
        if np.random.random() < 0.3:
            child2[i, 2] *= np.random.normal(1, 0.1)
        
        # Ensure positive radii
        child1[i, 2] = max(0.001, child1[i, 2])
        child2[i, 2] = max(0.001, child2[i, 2])
    
    return child1, child2

def repair_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Enhanced repair mechanism for fixing constraint violations."""
    repaired = circles.copy()
    
    # Ensure positive radii and bounds
    repaired[:, 2] = np.maximum(repaired[:, 2], 0.001)
    
    # Enforce bounds
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        x = np.clip(x, r, rect_width - r)
        y = np.clip(y, r, rect_height - r)
        repaired[i] = [x, y, r]
    
    # Resolve overlaps using iterative improvement
    for iteration in range(50):
        # Check for conflicts using spatial index
        positions = repaired[:, :2]
        radii = repaired[:, 2]
        
        # Use KDTree for efficient neighbor query
        try:
            tree = cKDTree(positions)
            # Find pairs that might be overlapping
            max_radius_sum = np.max(radii) + np.max(radii)
            pairs = tree.query_pairs(max_radius_sum * 2.0, output_type='ndarray')
            conflicts = []
            
            for i, j in pairs:
                if i < j:  # Avoid duplicate checks
                    distance = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                    if distance < (radii[i] + radii[j]):
                        conflicts.append((i, j))
        except:
            # Fallback to brute force if KDTree fails
            conflicts = []
            for i in range(len(repaired)):
                for j in range(i+1, len(repaired)):
                    distance = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
                    if distance < (radii[i] + radii[j]):
                        conflicts.append((i, j))
        
        if not conflicts:
            break
            
        # Handle conflicts by moving circles apart
        moved = False
        for i, j in conflicts:
            x1, y1, r1 = repaired[i]
            x2, y2, r2 = repaired[j]
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Move circles away from each other
                move_distance = (r1 + r2 - distance) / 2
                dx_norm = dx / distance
                dy_norm = dy / distance
                
                # Apply movement with bounded adjustment
                move1 = move_distance * r2 / (r1 + r2 + 1e-8)
                move2 = move_distance * r1 / (r1 + r2 + 1e-8)
                
                repaired[i, 0] -= dx_norm * move1 * 0.5
                repaired[i, 1] -= dy_norm * move1 * 0.5
                repaired[j, 0] += dx_norm * move2 * 0.5
                repaired[j, 1] += dy_norm * move2 * 0.5
                
                # Keep within bounds
                repaired[i, 0] = np.clip(repaired[i, 0], r1, rect_width - r1)
                repaired[i, 1] = np.clip(repaired[i, 1], r1, rect_height - r1)
                repaired[j, 0] = np.clip(repaired[j, 0], r2, rect_width - r2)
                repaired[j, 1] = np.clip(repaired[j, 1], r2, rect_height - r2)
                moved = True
        
        # Stop early if no movement occurred
        if not moved:
            break
    
    return repaired

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimized rectangle dimensions for better packing
    rect_width = 1.5
    rect_height = 0.5

    # Parameters for evolutionary algorithm
    population_size = 60
    generations = 150
    elite_size = 10
    tournament_size = 10

    # Initialize population with better quality solutions
    population = []
    for _ in range(population_size):
        solution = create_hexagonal_initial_solution(rect_width, rect_height)
        population.append(solution)

    # Track best fitness for convergence detection
    previous_best = -np.inf
    stagnation_count = 0
    
    # Evolutionary algorithm
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = population[:elite_size]
        
        # Generate new population
        new_population = elite[:]
        
        # Create offspring using tournament selection and crossover
        while len(new_population) < population_size:
            # Tournament selection - select two parents
            parent1_idx = sorted_indices[np.random.choice(min(tournament_size, len(sorted_indices)))]
            parent2_idx = sorted_indices[np.random.choice(min(tournament_size, len(sorted_indices)))]
            
            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            # Repair
            child1 = repair_solution(child1, rect_width, rect_height)
            child2 = repair_solution(child2, rect_width, rect_height)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
        
        # Convergence detection
        current_best = max(fitness_scores)
        if abs(current_best - previous_best) < 1e-5:
            stagnation_count += 1
        else:
            stagnation_count = 0
        previous_best = current_best
        
        # Early stopping if stagnated too long
        if stagnation_count > 20:
            print(f"Early stopping at generation {generation} due to convergence")
            break
            
        # Print progress
        if generation % 30 == 0:
            print(f"Generation {generation}: Best fitness = {current_best:.6f}")
    
    # Return the best solution
    fitness_scores = [evaluate_fitness(individual) for individual in population]
    best_idx = np.argmax(fitness_scores)
    best_solution = population[best_idx]
    
    # Final validation
    final_fitness = evaluate_fitness(best_solution)
    if final_fitness == -np.inf:
        print("Warning: Final solution violated constraints. Returning fallback.")
        # Fallback to best valid solution found during evolution
        for i in range(len(population)):
            if evaluate_fitness(population[i]) > -np.inf:
                return population[i]
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")