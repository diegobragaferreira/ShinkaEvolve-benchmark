# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Global constants
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE_START = 0.1
MUTATION_RATE_END = 0.01
TOURNAMENT_SIZE = 3
MAX_EVALUATIONS = 5000
BENCHMARK = 2.6358627564136983

def initialize_voronoi_points(n: int, seed: int = 42) -> np.ndarray:
    """Initialize points using a Voronoi-like distribution."""
    np.random.seed(seed)
    # Generate points that are somewhat uniformly distributed
    points = []
    # Create a grid with some randomness
    grid_size = int(np.ceil(np.sqrt(n)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n:
                break
            x = (i + np.random.random() * 0.8 + 0.1) / grid_size
            y = (j + np.random.random() * 0.8 + 0.1) / grid_size
            if x <= 1 and y <= 1:
                points.append([x, y])
    
    # If we don't have enough points, add more randomly
    while len(points) < n:
        x = np.random.random()
        y = np.random.random()
        points.append([x, y])
        
    return np.array(points[:n])

def generate_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Generate initial population with good starting configurations."""
    population = []
    for _ in range(pop_size):
        # Use Voronoi-based initialization
        centers = initialize_voronoi_points(n_circles)
        # Assign initial radii (uniformly small)
        radii = np.full(n_circles, 0.02)
        circles = np.column_stack([centers, radii])
        population.append(circles)
    return population

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained within the unit square."""
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]
    return np.all((radii <= x_coords) & 
                  (x_coords <= 1 - radii) & 
                  (radii <= y_coords) & 
                  (y_coords <= 1 - radii))

def get_distance_matrix(circles: np.ndarray) -> np.ndarray:
    """Get distance matrix efficiently using KDTree."""
    positions = circles[:, :2]
    tree = cKDTree(positions)
    distances = tree.sparse_distance_matrix(tree, max_distance=2.0, p=2)
    return distances

def calculate_overlap_penalty(circles: np.ndarray) -> float:
    """Calculate penalty based on overlap amounts."""
    if len(circles) < 2:
        return 0.0
    
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Use spatial indexing for efficient overlap detection
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=2.0, output_type='ndarray')
    
    total_penalty = 0.0
    for i, j in pairs:
        if i < j:  # Only check each pair once
            dist = np.linalg.norm(positions[i] - positions[j])
            r_i, r_j = radii[i], radii[j]
            overlap = max(0, r_i + r_j - dist)
            if overlap > 0:
                total_penalty += overlap ** 2  # Squared penalty to emphasize large overlaps
                
    return total_penalty

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a solution with penalties."""
    if not check_containment(circles):
        return -1000.0
    
    # Calculate sum of radii
    total_radius = np.sum(circles[:, 2])
    
    # Add penalty for overlaps
    overlap_penalty = calculate_overlap_penalty(circles)
    
    # Return fitness (negative because we're minimizing in optimization context)
    return total_radius - overlap_penalty * 1000.0

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Apply mutation to a circle configuration."""
    mutated = circles.copy()
    # Adaptive mutation rate
    mutation_rate = MUTATION_RATE_START - (generation / total_generations) * (MUTATION_RATE_START - MUTATION_RATE_END)
    
    n_circles = len(mutated)
    for i in range(n_circles):
        if random.random() < mutation_rate:
            # Mutate position
            if random.random() < 0.5:
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + np.random.normal(0, 0.02)))
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + np.random.normal(0, 0.02)))
            else:
                # Mutate radius
                delta = np.random.normal(0, 0.01)
                mutated[i, 2] = max(0.001, min(0.5, mutated[i, 2] + delta))
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Create offspring via crossover."""
    n_circles = len(parent1)
    child = parent1.copy()
    
    # Single point crossover
    crossover_point = random.randint(1, n_circles - 1)
    child[crossover_point:, :] = parent2[crossover_point:, :]
    
    # Apply local refinement to handle potential overlaps
    refined_child = local_refinement(child)
    return refined_child

def local_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to resolve constraint violations."""
    refined = circles.copy()
    
    # Simple iterative refinement to fix containment and overlap issues
    for _ in range(10):  # Limit iterations
        # Fix containment first
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Ensure containment
            x = max(r, min(1 - r, x))
            y = max(r, min(1 - r, y))
            refined[i] = [x, y, r]
            
        # Slightly adjust overlapping circles
        positions = refined[:, :2]
        radii = refined[:, 2]
        
        # Detect and resolve overlaps
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=0.001, output_type='ndarray')  # Very small threshold
        
        # Simple repulsion for very close circles
        for i, j in pairs:
            if i < j:
                pos_i = positions[i]
                pos_j = positions[j] 
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < radii[i] + radii[j]:  # Overlapping
                    # Move them apart
                    direction = pos_i - pos_j
                    if np.linalg.norm(direction) > 0:
                        direction /= np.linalg.norm(direction)
                        move_dist = (radii[i] + radii[j] - dist) / 2.0
                        refined[i, :2] += direction * move_dist * 0.5
                        refined[j, :2] -= direction * move_dist * 0.5
                        
    return refined

def tournament_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Perform tournament selection."""
    selected_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    selected_fitnesses = [fitnesses[i] for i in selected_indices]
    winner_idx = selected_indices[np.argmax(selected_fitnesses)]
    return population[winner_idx]

def optimize_circles_evolutionary(n_circles: int = 26, 
                                pop_size: int = POPULATION_SIZE,
                                generations: int = GENERATIONS) -> np.ndarray:
    """Main evolutionary optimization loop."""
    # Initialize population
    population = generate_initial_population(pop_size, n_circles)
    
    best_solution = None
    best_fitness = float('-inf')
    
    for gen in range(generations):
        # Evaluate fitness for entire population
        fitnesses = [evaluate_fitness(individual) for individual in population]
        
        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(best_solution.copy())
        
        # Generate rest of population through selection, crossover, and mutation
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            child = crossover(parent1, parent2)
            child = mutate(child, gen, generations)
            
            # Ensure child is valid
            child = local_refinement(child)
            
            new_population.append(child)
            
        population = new_population
    
    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    try:
        # Run optimization
        result = optimize_circles_evolutionary(
            n_circles=26,
            pop_size=50,
            generations=100
        )
        
        # Final validation
        if result is not None:
            # Check final fitness
            final_fitness = evaluate_fitness(result)
            elapsed_time = time.time() - start_time
            
            # Print info for debugging
            print(f"Final fitness: {final_fitness}")
            print(f"Sum of radii: {np.sum(result[:, 2])}")
            print(f"Execution time: {elapsed_time:.2f}s")
            print(f"Benchmark ratio: {np.sum(result[:, 2]) / BENCHMARK:.4f}")
            
            # Final refinement
            final_result = local_refinement(result)
            
            return final_result
            
        else:
            # Fallback to simple initialization if optimization fails
            print("Fallback to simple initialization...")
            circles = np.zeros((26, 3))
            # Simple grid initialization
            grid_size = 5
            count = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if count >= 26:
                        break
                    x = (i + 0.5) / grid_size
                    y = (j + 0.5) / grid_size
                    r = 0.05
                    circles[count] = [x, y, r]
                    count += 1
            return circles
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to simple initialization
        circles = np.zeros((26, 3))
        grid_size = 5
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                r = 0.05
                circles[count] = [x, y, r]
                count += 1
        return circles

# EVOLVE-BLOCK-END
