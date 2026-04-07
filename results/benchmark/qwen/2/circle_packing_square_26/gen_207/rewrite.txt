# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import math

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles: np.ndarray, idx: int) -> bool:
    """Check if circle at index idx is valid (within bounds and not overlapping)."""
    x, y, r = circles[idx]

    # Check containment constraints
    if x < r or x > 1 - r or y < r or y > 1 - r:
        return False

    # Check overlap constraints with existing circles
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

def local_optimize(circles: np.ndarray, max_iterations: int = 20) -> np.ndarray:
    """Apply local optimization to improve a solution by making small adjustments."""
    optimized = circles.copy()

    # Try to increase radii where possible
    for _ in range(max_iterations):
        improved = False

        # For each circle, try to increase its radius
        for i in range(len(optimized)):
            current_r = optimized[i, 2]

            # Try increasing radius slightly
            new_r = min(current_r * 1.05, 0.5)  # Cap at reasonable size

            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(len(optimized)):
                if i != j:
                    dist = np.sqrt((optimized[i, 0] - optimized[j, 0])**2 +
                                 (optimized[i, 1] - optimized[j, 1])**2)
                    if dist < new_r + optimized[j, 2]:
                        valid = False
                        break

            # Update if improvement is possible and within bounds
            if valid and new_r <= 1 - max(optimized[i, 0], 1 - optimized[i, 0]) and \
               new_r <= 1 - max(optimized[i, 1], 1 - optimized[i, 1]):
                optimized[i, 2] = new_r
                improved = True

        # If no improvement made, stop early
        if not improved:
            break

    return optimized

def smart_local_refinement(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """
    Enhanced local optimization that combines several refinement techniques.
    """
    refined = circles.copy()
    
    # Phase 1: Basic radius expansion with overlap checking
    for _ in range(10):
        improved = False
        for i in range(len(refined)):
            current_r = refined[i, 2]
            max_possible_r = min(
                refined[i, 0], 
                refined[i, 1], 
                1 - refined[i, 0], 
                1 - refined[i, 1]
            )
            
            # Try to expand radius
            new_r = min(current_r * 1.1, max_possible_r)
            
            # Check constraints
            valid = True
            for j in range(len(refined)):
                if i != j:
                    dist = np.sqrt((refined[i, 0] - refined[j, 0])**2 + 
                                 (refined[i, 1] - refined[j, 1])**2)
                    if dist < new_r + refined[j, 2]:
                        valid = False
                        break
            
            if valid and new_r > current_r:
                refined[i, 2] = new_r
                improved = True
        
        if not improved:
            break
    
    # Phase 2: Physics-based repulsion for minor adjustments
    for iteration in range(30):
        moved = False
        for i in range(len(refined)):
            total_force_x, total_force_y = 0.0, 0.0
            
            # Calculate repulsive forces from overlapping circles
            for j in range(len(refined)):
                if i != j:
                    x1, y1, r1 = refined[i]
                    x2, y2, r2 = refined[j]
                    
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if dist < r1 + r2:
                        # Apply repulsive force
                        if dist > 0.001:
                            force_magnitude = (r1 + r2 - dist) * 0.1
                            dx = (x1 - x2) / dist
                            dy = (y1 - y2) / dist
                            
                            total_force_x += dx * force_magnitude
                            total_force_y += dy * force_magnitude
            
            # Apply the forces if there are any
            if abs(total_force_x) > 0.0001 or abs(total_force_y) > 0.0001:
                # Move circle in direction of net force
                refined[i, 0] += total_force_x * 0.1
                refined[i, 1] += total_force_y * 0.1
                
                # Keep within bounds
                refined[i, 0] = np.clip(refined[i, 0], refined[i, 2], 1 - refined[i, 2])
                refined[i, 1] = np.clip(refined[i, 1], refined[i, 2], 1 - refined[i, 2])
                
                moved = True
        
        if not moved:
            break
    
    return refined

def create_hybrid_initialization(pop_size: int, n_circles: int) -> list:
    """
    Create initial population using hybrid approach:
    1. Spatial partitioning for initial distribution
    2. Physics-based repulsion to resolve overlaps
    3. Radius optimization based on neighborhood constraints
    """
    population = []
    
    for _ in range(pop_size):
        # Step 1: Create rough grid layout
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Initialize circles
        circles = np.zeros((n_circles, 3))
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_circles:
                    break
                x = (i + 1) * spacing_x + np.random.uniform(-spacing_x/8, spacing_x/8)
                y = (j + 1) * spacing_y + np.random.uniform(-spacing_y/8, spacing_y/8)
                radius = min(spacing_x, spacing_y) * 0.3
                circles[count] = [x, y, radius]
                count += 1
            if count >= n_circles:
                break
        
        # Step 2: Apply physics-based repulsion to reduce overlaps
        for _ in range(50):
            improved = False
            for i in range(n_circles):
                x1, y1, r1 = circles[i]
                for j in range(n_circles):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        min_distance = r1 + r2
                        
                        if distance < min_distance:
                            if distance > 0.001:
                                # Apply repulsive force
                                dx = (x1 - x2) / distance
                                dy = (y1 - y2) / distance
                                overlap = min_distance - distance
                                move_amount = overlap * 0.2
                                
                                circles[i, 0] += dx * move_amount
                                circles[i, 1] += dy * move_amount
                                
                                # Keep within bounds
                                circles[i, 0] = np.clip(circles[i, 0], r1, 1 - r1)
                                circles[i, 1] = np.clip(circles[i, 1], r1, 1 - r1)
                                improved = True
            
            if not improved:
                break
        
        # Step 3: Optimize radii based on neighbors and boundaries
        for i in range(n_circles):
            # Calculate maximum possible radius considering neighbors and bounds
            max_radius = min(
                circles[i, 0],
                circles[i, 1],
                1 - circles[i, 0],
                1 - circles[i, 1]
            )
            
            # Check neighbor constraints
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    max_radius = min(max_radius, dist - circles[j, 2] - 0.001)
            
            # Set a reasonable radius
            if max_radius > 0.001:
                circles[i, 2] = max_radius * 0.95
        
        # Step 4: Final validation and cleanup
        for i in range(n_circles):
            # Ensure within bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        # Step 5: Apply smart local refinement
        circles = smart_local_refinement(circles)
        
        population.append(circles.copy())
    
    return population

def adaptive_mutation(circles: np.ndarray, generation: int, population_std: float) -> np.ndarray:
    """Apply adaptive mutation with dynamic rate based on generation and population diversity."""
    
    # Dynamic mutation rate based on generation and population diversity
    base_mutation_rate = 0.15
    diversity_factor = max(0.1, 1.0 - population_std / 0.1) if population_std > 0 else 1.0
    generation_factor = max(0.1, 1.0 - generation / 150.0)
    mutation_rate = base_mutation_rate * diversity_factor * generation_factor
    
    mutated = circles.copy()
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate either position or radius
            if np.random.random() < 0.5:
                # Mutate position with adaptive step size
                step_size = 0.02 * (1.0 - generation / 150.0)
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, step_size), 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, step_size), 0.01, 0.99)
            else:
                # Mutate radius with adaptive step size
                step_size = 0.01 * (1.0 - generation / 150.0)
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, step_size), 0.001, 0.3)
    
    return mutated

def constraint_aware_crossover(parent1: np.ndarray, parent2: np.ndarray, fitness1: float, fitness2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Crossover that favors better-performing parents and reduces overlap risk."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Weighted crossover based on fitness
    fitness_weight1 = fitness1 / (fitness1 + fitness2 + 1e-8)
    fitness_weight2 = fitness2 / (fitness1 + fitness2 + 1e-8)
    
    for i in range(len(child1)):
        # Probability of inheriting from parent1 is proportional to fitness
        crossover_prob = fitness_weight1
        
        if np.random.random() < crossover_prob:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()
    
    return child1, child2

def calculate_population_diversity(population: list) -> float:
    """Calculate the standard deviation of radii across the entire population."""
    if len(population) == 0:
        return 0.0
    
    all_radii = []
    for individual in population:
        all_radii.extend(individual[:, 2])
    
    if len(all_radii) < 2:
        return 0.0
    
    return np.std(all_radii)

def optimize_circles() -> np.ndarray:
    """Main optimization function using advanced evolutionary algorithm."""
    n_circles = 26
    pop_size = 60
    generations = 120

    # Create initial population
    population = create_hybrid_initialization(pop_size, n_circles)

    best_fitness = 0
    best_individual = None

    for generation in range(generations):
        # Calculate population diversity
        pop_diversity = calculate_population_diversity(population)
        
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Create new population
        new_population = []

        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual.copy())

        # Generate offspring through tournament selection and reproduction
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 5
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            parent1_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            parent2_idx = tournament_indices[np.argmax(tournament_fitnesses)]

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Constraint-aware crossover
            child1, child2 = constraint_aware_crossover(parent1, parent2, fitnesses[parent1_idx], fitnesses[parent2_idx])

            # Adaptive mutation
            child1 = adaptive_mutation(child1, generation, pop_diversity)
            child2 = adaptive_mutation(child2, generation, pop_diversity)

            # Apply enhanced local refinement
            child1 = smart_local_refinement(child1)
            child2 = smart_local_refinement(child2)

            # Ensure children meet constraints
            valid_child1 = True
            valid_child2 = True
            
            # Validate children
            for i in range(n_circles):
                if not is_valid_placement(child1, i):
                    valid_child1 = False
                    break
                if not is_valid_placement(child2, i):
                    valid_child2 = False
                    break
            
            if valid_child1:
                new_population.append(child1)
            if len(new_population) < pop_size and valid_child2:
                new_population.append(child2)

        # Trim population to exact size
        population = new_population[:pop_size]

    # Final refinement of best solution
    if best_individual is not None:
        best_individual = smart_local_refinement(best_individual, max_iterations=100)

    return best_individual if best_individual is not None else population[0] if population else np.zeros((26, 3))

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
        # Fallback to improved heuristic with better spatial distribution
        circles = np.zeros((26, 3))

        # Use a more sophisticated pattern with staggered grid + boundary padding
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = min(spacing_x, spacing_y) * 0.35

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                # Create staggered pattern
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                
                # Apply varying perturbations based on grid position
                if (i + j) % 2 == 0:
                    x += np.random.uniform(-spacing_x/6, spacing_x/6)
                    y += np.random.uniform(-spacing_y/6, spacing_y/6)
                else:
                    x += np.random.uniform(-spacing_x/8, spacing_x/8)
                    y += np.random.uniform(-spacing_y/8, spacing_y/8)
                
                # Ensure it's within bounds
                x = np.clip(x, radius, 1 - radius)
                y = np.clip(y, radius, 1 - radius)
                
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break

        # Apply final refinement
        for _ in range(20):
            improved = False
            for i in range(26):
                x1, y1, r1 = circles[i]
                for j in range(26):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        min_distance = r1 + r2

                        if distance < min_distance:
                            if distance > 0.001:
                                dx = (x1 - x2) / distance
                                dy = (y1 - y2) / distance
                                overlap = min_distance - distance
                                move_amount = overlap * 0.3

                                circles[i, 0] += dx * move_amount
                                circles[i, 1] += dy * move_amount

                                # Keep within bounds
                                circles[i, 0] = np.clip(circles[i, 0], r1, 1 - r1)
                                circles[i, 1] = np.clip(circles[i, 1], r1, 1 - r1)
                                improved = True

            if not improved:
                break

        # Final cleanup to ensure all constraints are met
        for i in range(26):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])

        return circles

# EVOLVE-BLOCK-END