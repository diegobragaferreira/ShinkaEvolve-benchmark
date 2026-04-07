# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random
from typing import Tuple, List, Optional
import math

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles: np.ndarray, idx: int, tree: Optional[KDTree] = None) -> bool:
    """Check if circle at index idx is valid (within bounds and not overlapping)."""
    x, y, r = circles[idx]

    # Check containment constraints
    if x < r or x > 1 - r or y < r or y > 1 - r:
        return False

    # Use spatial indexing for faster overlap checking if available
    if tree is not None:
        # Query nearby circles within a reasonable distance
        nearby_indices = tree.query_ball_point([x, y], r + 1e-6)
        for i in nearby_indices:
            if i == idx:
                continue
            x_i, y_i, r_i = circles[i]
            distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
            if distance < r + r_i:
                return False
    else:
        # Fallback to brute force checking
        for i in range(len(circles)):
            if i == idx:
                continue
            x_i, y_i, r_i = circles[i]
            distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
            if distance < r + r_i:
                return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def compute_distance_matrix(circles: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between all circles."""
    return cdist(circles[:, :2], circles[:, :2])

def build_spatial_index(circles: np.ndarray) -> KDTree:
    """Build a spatial index for fast neighbor queries."""
    return KDTree(circles[:, :2])

def generate_hexagonal_grid(n_circles: int, spacing_factor: float = 1.0) -> np.ndarray:
    """Generate initial circle positions using hexagonal packing pattern."""
    circles = np.zeros((n_circles, 3))
    
    # Calculate hexagonal grid parameters
    rows = int(math.ceil(math.sqrt(n_circles)))
    cols = int(math.ceil(n_circles / rows))
    
    # Effective spacing for hexagonal arrangement
    hex_spacing = 1.0 / max(rows, cols)
    
    # Radius based on spacing
    radius = hex_spacing * 0.4 * spacing_factor
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n_circles:
                break
                
            # Hexagonal offset pattern
            offset = 0 if i % 2 == 0 else 0.5
            x = (j + offset) * hex_spacing
            y = i * hex_spacing * math.sqrt(3)/2
            
            # Ensure within bounds
            x = max(radius, min(1-radius, x))
            y = max(radius, min(1-radius, y))
            
            circles[count] = [x, y, radius]
            count += 1
            
        if count >= n_circles:
            break
            
    return circles[:count]

def initialize_population(pop_size: int, n_circles: int) -> list:
    """Create initial population with hybrid approach."""
    population = []
    
    # Phase 1: Generate diverse initial solutions using hexagonal patterns
    for i in range(pop_size // 2):
        # Start with hexagonal grid
        base_circles = generate_hexagonal_grid(n_circles, 0.8 + 0.4 * random.random())
        
        # Add some randomness
        for j in range(n_circles):
            base_circles[j, 0] += random.uniform(-0.05, 0.05)
            base_circles[j, 1] += random.uniform(-0.05, 0.05)
            base_circles[j, 2] *= (0.9 + 0.2 * random.random())  # Vary radii slightly
        
        # Clip to valid bounds
        for j in range(n_circles):
            base_circles[j, 0] = max(base_circles[j, 2], min(1 - base_circles[j, 2], base_circles[j, 0]))
            base_circles[j, 1] = max(base_circles[j, 2], min(1 - base_circles[j, 2], base_circles[j, 1]))
            
        population.append(base_circles.copy())
    
    # Phase 2: Fill remaining with random valid configurations
    for i in range(pop_size // 2, pop_size):
        circles = np.zeros((n_circles, 3))
        for j in range(n_circles):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            r = np.random.uniform(0.005, 0.05)
            circles[j] = [x, y, r]
            
        # Validate and fix invalid circles
        for j in range(n_circles):
            if not is_valid_placement(circles, j):
                # Try to find valid position
                attempts = 0
                while not is_valid_placement(circles, j) and attempts < 100:
                    circles[j, 0] = np.random.uniform(0.01, 0.99)
                    circles[j, 1] = np.random.uniform(0.01, 0.99)
                    circles[j, 2] = np.random.uniform(0.005, 0.05)
                    attempts += 1
                    
        population.append(circles.copy())
    
    return population

def smart_mutate(circles: np.ndarray, mutation_rate: float = 0.1, tree: Optional[KDTree] = None) -> np.ndarray:
    """Apply smart mutation that considers validity constraints."""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Choose type of mutation based on current state
            if mutated[i, 2] > 0.02:  # Large circle - focus on position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.015), 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.015), 0.01, 0.99)
            else:  # Small circle - focus on radius
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.005), 0.001, 0.15)
                
            # If the mutation resulted in an invalid configuration, revert to previous and try again
            temp_circle = mutated[i].copy()
            if not is_valid_placement(mutated, i, tree):
                # Try to fix by moving closer to center with smaller radius
                mutated[i, 0] = 0.5
                mutated[i, 1] = 0.5
                mutated[i, 2] = temp_circle[2] * 0.95  # Slightly reduce radius
                # If still invalid, regenerate the circle
                if not is_valid_placement(mutated, i, tree):
                    mutated[i, 0] = np.random.uniform(0.01, 0.99)
                    mutated[i, 1] = np.random.uniform(0.01, 0.99)
                    mutated[i, 2] = np.random.uniform(0.005, 0.05)

    return mutated

def adaptive_crossover(parent1: np.ndarray, parent2: np.ndarray, fitness1: float, fitness2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover with adaptive mixing based on fitness."""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Weighted crossover - better parents contribute more
    fitness_ratio = max(0.1, min(10, fitness1 / (fitness2 + 1e-8)))
    weight = 0.5 + 0.3 * (fitness_ratio - 1)/(max(fitness_ratio, 1/fitness_ratio) - 1)
    
    for i in range(len(child1)):
        if np.random.random() < 0.5:
            # Swap genes with probability based on fitness
            if np.random.random() < weight:
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()
            else:
                pass  # Keep original pairing

    return child1, child2

def tournament_selection_with_diversity(population: list, fitnesses: list, diversity_threshold: float = 0.1) -> list:
    """Select parents using tournament selection with diversity awareness."""
    selected = []
    population_size = len(population)
    
    for _ in range(population_size):
        # Adjust tournament size based on diversity
        if len(set([tuple(p.tolist()) for p in population])) > population_size * (1 - diversity_threshold):
            tournament_size = 3  # More intense selection
        else:
            tournament_size = 5  # More diverse selection
            
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx])

    return selected

def local_improve(circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
    """Apply aggressive local improvement to maximize sum of radii."""
    optimized = circles.copy()
    
    # Build spatial index for fast validation
    tree = build_spatial_index(optimized)
    
    # Iterative improvement: try to increase radii while maintaining validity
    for iteration in range(max_iterations):
        improved = False
        
        # For each circle, try to increase its radius safely
        for i in range(len(optimized)):
            original_r = optimized[i, 2]
            
            # Calculate maximum possible radius
            max_radius = 1.0  # Will be constrained by boundaries and neighbors
            
            # Boundary constraints
            max_radius = min(max_radius, optimized[i, 0] - 1e-6)  # Left boundary
            max_radius = min(max_radius, 1 - optimized[i, 0] - 1e-6)  # Right boundary
            max_radius = min(max_radius, optimized[i, 1] - 1e-6)  # Bottom boundary
            max_radius = min(max_radius, 1 - optimized[i, 1] - 1e-6)  # Top boundary
            
            # Neighbor constraints
            neighbors = tree.query_ball_point([optimized[i, 0], optimized[i, 1]], 1.0)
            for j in neighbors:
                if i != j:
                    dist = np.sqrt((optimized[i, 0] - optimized[j, 0])**2 + (optimized[i, 1] - optimized[j, 1])**2)
                    max_radius = min(max_radius, dist - optimized[j, 2] - 1e-6)
                    
            # Try to increase radius
            if max_radius > original_r + 1e-6:
                new_r = min(original_r * 1.03, max_radius)
                if new_r > original_r:
                    # Test if the change is valid
                    old_pos = optimized[i, :2].copy()
                    old_r = optimized[i, 2]
                    
                    optimized[i, 2] = new_r
                    
                    # Validate the change
                    if not is_valid_placement(optimized, i, tree):
                        # Revert if invalid
                        optimized[i, 2] = old_r
                        optimized[i, :2] = old_pos
                    else:
                        improved = True
                        
        if not improved:
            break
            
    return optimized

def optimize_circles() -> np.ndarray:
    """Main optimization function using hybrid evolutionary approach."""
    n_circles = 26
    pop_size = 60  # Increased for better exploration
    generations = 120  # Increased for more thorough search
    
    # Initialize population
    population = initialize_population(pop_size, n_circles)
    
    best_fitness = 0
    best_individual = None
    
    # Track performance over generations for adaptive behavior
    recent_fitnesses = []
    stagnation_count = 0
    max_stagnation = 20
    
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]
        
        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            stagnation_count = 0  # Reset stagnation counter
        else:
            stagnation_count += 1
            
        # Adaptive population management
        if stagnation_count > max_stagnation:
            # Increase diversity by adding random individuals
            extra_pop = []
            for _ in range(pop_size // 10):
                circles = np.zeros((n_circles, 3))
                for i in range(n_circles):
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    r = np.random.uniform(0.005, 0.05)
                    circles[i] = [x, y, r]
                extra_pop.append(circles)
                
            population.extend(extra_pop[:min(len(extra_pop), pop_size//10)])
            population = population[-pop_size:]  # Keep only population size
            stagnation_count = 0  # Reset
            
        # Select parents with diversity awareness
        parents = tournament_selection_with_diversity(population, fitnesses)
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Build spatial index once for efficiency
        current_tree = build_spatial_index(best_individual) if best_individual is not None else None
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]
            
            # Crossover with adaptive weights
            fitness1 = evaluate_fitness(parent1)
            fitness2 = evaluate_fitness(parent2)
            child1, child2 = adaptive_crossover(parent1, parent2, fitness1, fitness2)
            
            # Smart mutation
            mutation_rate = 0.15 * np.exp(-generation / 80.0) + 0.02  # Adaptive mutation rate
            child1 = smart_mutate(child1, mutation_rate, current_tree)
            child2 = smart_mutate(child2, mutation_rate, current_tree)
            
            # Apply local improvement to offspring
            child1 = local_improve(child1)
            child2 = local_improve(child2)
            
            # Validation check - add only valid individuals
            if is_valid_placement(child1, len(child1)-1, current_tree):
                new_population.append(child1)
            if len(new_population) < pop_size and is_valid_placement(child2, len(child2)-1, current_tree):
                new_population.append(child2)
        
        # Trim to population size
        population = new_population[:pop_size]
        
        # Store recent fitnesses for diversity tracking
        recent_fitnesses.append(best_fitness)
        if len(recent_fitnesses) > 10:
            recent_fitnesses.pop(0)
            
    # Final local optimization on the best solution
    if best_individual is not None:
        best_individual = local_improve(best_individual, max_iterations=50)
        
    return best_individual if best_individual is not None else population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = optimize_circles()
        return circles
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to improved heuristic
        circles = np.zeros((26, 3))
        
        # Create a more sophisticated but still valid fallback
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x / 3.0

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                # Add more variation to avoid perfectly regular patterns
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break

        # Final cleanup to ensure constraints are satisfied
        for i in range(count):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])

        return circles


# EVOLVE-BLOCK-END