# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from scipy.spatial import KDTree
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def check_containment(circles):
    """Check if all circles are fully contained within the unit square"""
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap_efficient(circles):
    """Check if any circles overlap using KDTree for efficiency"""
    n = len(circles)
    if n <= 1:
        return True
    
    # Extract positions and radii
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Build KDTree for efficient neighbor search
    tree = KDTree(positions)
    
    # For each circle, check if it overlaps with any other circle
    for i in range(n):
        x, y, r = circles[i]
        # Find neighbors within distance 2*r
        indices = tree.query_ball_point([x, y], 2 * r)
        
        # Check overlaps with neighbors
        for j in indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False
    return True

def check_overlap(circles):
    """Check if any circles overlap (legacy version for compatibility)"""
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
    return True

def evaluate_fitness(circles):
    """Evaluate fitness as sum of radii, with penalty for constraint violations"""
    if not check_containment(circles) or not check_overlap_efficient(circles):
        # Large negative penalty for constraint violations
        return -1000.0

    total_radius = np.sum(circles[:, 2])
    return total_radius

def constraint_aware_crossover(parent1, parent2):
    """Crossover that prioritizes valid configurations"""
    n = len(parent1)
    child = np.zeros_like(parent1)

    # For each circle, randomly choose from parent1 or parent2
    for i in range(n):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    # If child violates constraints, try to fix it by taking the better parent
    if not check_containment(child) or not check_overlap_efficient(child):
        # Score both parents
        parent1_score = evaluate_fitness(parent1)
        parent2_score = evaluate_fitness(parent2)
        
        # Choose the better parent to use as base
        if parent1_score >= parent2_score:
            return parent1.copy()
        else:
            return parent2.copy()
    
    return child

def mutate(circles, mutation_rate=0.1, max_mutation=0.05):
    """Apply mutation to circles with better constraint handling"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly mutate position and/or radius
            if random.random() < 0.5:
                # Mutate position
                mutated[i, 0] += np.random.normal(0, max_mutation)
                mutated[i, 1] += np.random.normal(0, max_mutation)
                # Ensure it stays within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], 0, 1)
                mutated[i, 1] = np.clip(mutated[i, 1], 0, 1)
            else:
                # Mutate radius
                mutated[i, 2] += np.random.normal(0, max_mutation/2)
                # Ensure radius remains positive
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def refine_solution(circles):
    """Apply simple local optimization to improve a solution"""
    refined = circles.copy()
    
    # Simple greedy improvement: try to slightly increase radii
    # while maintaining constraints
    for i in range(len(refined)):
        original_radius = refined[i, 2]
        best_radius = original_radius
        
        # Try to increase radius slightly
        for _ in range(10):
            new_radius = original_radius + np.random.uniform(0, 0.01)
            if new_radius > best_radius:
                # Temporarily update
                refined[i, 2] = new_radius
                
                # Check constraints
                if check_containment(refined) and check_overlap_efficient(refined):
                    best_radius = new_radius
                else:
                    # Revert if constraint violated
                    refined[i, 2] = original_radius
        
        # Set the best radius found
        refined[i, 2] = best_radius
    
    return refined

def initialize_population(pop_size, n_circles):
    """Initialize population with multi-scale grid-based approach"""
    population = []

    for _ in range(pop_size):
        # Start with grid-based initialization
        circles = np.zeros((n_circles, 3))
        
        # Create a grid layout first
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        circle_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if circle_idx >= n_circles:
                    break
                    
                base_x = (i + 1) * spacing_x
                base_y = (j + 1) * spacing_y
                
                # Add some randomness to avoid perfect grid
                x = np.clip(base_x + np.random.normal(0, spacing_x * 0.2), 0, 1)
                y = np.clip(base_y + np.random.normal(0, spacing_y * 0.2), 0, 1)
                
                # Initial radius estimate based on spacing
                max_radius = min(x, 1-x, y, 1-y)
                r = np.clip(max_radius * np.random.uniform(0.3, 0.7), 0.001, 0.1)
                
                circles[circle_idx] = [x, y, r]
                circle_idx += 1
            
            if circle_idx >= n_circles:
                break
        
        # Refine the initial configuration
        circles = refine_solution(circles)
        
        # Apply local optimization
        for _ in range(5):
            circles = refine_solution(circles)
            
        population.append(circles)

    return population

def tournament_selection(population, fitnesses, tournament_size=5):
    """Select an individual using tournament selection with larger tournaments"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n_circles = 26
    pop_size = 100  # Increased population size
    generations = 200
    elite_size = 10  # Increased elite size

    # Initialize population
    population = initialize_population(pop_size, n_circles)

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Create new population
        new_population = []

        # Elitism: keep the best individuals
        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Calculate adaptive mutation rate (exponential decay)
        mutation_rate = 0.15 * (0.1 ** (gen / 75.0))  # Decay from 0.15 to ~0.015 over 75 gens
        max_mutation = 0.05 * (0.8 ** (gen / 50.0))   # Gradually reduce mutation magnitude

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            child = constraint_aware_crossover(parent1, parent2)

            # Mutation
            child = mutate(child, mutation_rate=mutation_rate, max_mutation=max_mutation)

            # Local refinement
            child = refine_solution(child)

            new_population.append(child)

        population = new_population[:pop_size]  # Ensure exact population size

        # Print progress
        if gen % 20 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness:.4f}")

    # Return the best solution found
    final_fitnesses = [evaluate_fitness(individual) for individual in population]
    best_idx = np.argmax(final_fitnesses)
    best_solution = population[best_idx]

    # Final validation
    if not check_containment(best_solution) or not check_overlap_efficient(best_solution):
        print("Warning: Best solution violates constraints")

    print(f"Final fitness: {evaluate_fitness(best_solution):.6f}")

    return best_solution


# EVOLVE-BLOCK-END