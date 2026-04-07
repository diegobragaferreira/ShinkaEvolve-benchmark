# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import math
from typing import Tuple, List
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container dimensions - rectangle with perimeter 4, so width + height = 2
    # Using 1.3:0.7 ratio for better packing efficiency
    container_width = 1.3
    container_height = 0.7
    
    # Parameters
    n_circles = 21
    max_generations = 100
    population_size = 40
    elite_count = 5
    mutation_rate = 0.2
    crossover_rate = 0.7
    
    # Initialize population with high-quality starting configurations
    population = []
    for _ in range(population_size):
        circles = generate_hexagonal_initial_solution(container_width, container_height, n_circles)
        population.append(circles)
    
    best_solution = None
    best_sum = 0
    convergence_threshold = 1e-6
    stagnation_counter = 0
    max_stagnation = 20
    
    # Evolutionary loop with physics-based refinement
    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness_with_physics(individual, container_width, container_height)
            fitness_scores.append(fitness)
        
        # Update best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_sum:
            best_sum = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()
            stagnation_counter = 0  # Reset stagnation counter
        else:
            stagnation_counter += 1
        
        # Check for early termination
        if stagnation_counter >= max_stagnation:
            break
            
        # Selection using tournament selection
        selected = tournament_selection(population, fitness_scores, population_size)
        
        # Crossover and mutation to create new population
        new_population = []
        
        # Elitism - keep best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1_idx = random.randint(0, len(selected)-1)
            parent2_idx = random.randint(0, len(selected)-1)
            parent1 = selected[parent1_idx]
            parent2 = selected[parent2_idx]
            
            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            mutate(child1, container_width, container_height, mutation_rate)
            mutate(child2, container_width, container_height, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
    
    # Final refinement of best solution
    if best_solution is not None:
        refined_solution = refine_solution_physics(best_solution, container_width, container_height, 100)
        return refined_solution
    else:
        # Fallback to initial solution
        return generate_hexagonal_initial_solution(container_width, container_height, n_circles)

def generate_hexagonal_initial_solution(width: float, height: float, n_circles: int) -> np.ndarray:
    """Generate initial solution using hexagonal packing pattern."""
    circles = np.zeros((n_circles, 3))
    
    # Create hexagonal grid pattern
    # Base spacing for hexagonal lattice
    side_length = min(width, height) * 0.15  # Initial spacing
    
    # Determine grid dimensions for hexagonal packing
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    # Adjust for hexagonal packing efficiency
    hex_radius = side_length * 0.8  # Effective radius for hexagonal packing
    
    # Generate hexagonal grid points
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
                
            # Hexagonal offset pattern
            x_offset = j * side_length * 1.5
            y_offset = i * side_length * np.sqrt(3) / 2
            
            # Alternate rows for hexagonal pattern
            if i % 2 == 1:
                x_offset += side_length * 0.75
            
            # Add slight random perturbation to avoid perfect regularity
            x = x_offset + random.uniform(-side_length*0.3, side_length*0.3)
            y = y_offset + random.uniform(-side_length*0.3, side_length*0.3)
            
            # Ensure valid bounds
            x = max(hex_radius, min(width - hex_radius, x))
            y = max(hex_radius, min(height - hex_radius, y))
            
            # Initial radius - adjusted for proximity to edges and optimal packing
            max_radius = min(x, width - x, y, height - y) * 0.3
            radius = min(max_radius, hex_radius * random.uniform(0.7, 1.2))
            
            circles[idx] = [x, y, radius]
            idx += 1
    
    return circles

def evaluate_fitness_with_physics(circles: np.ndarray, width: float, height: float) -> float:
    """Evaluate fitness using physics-based refinement."""
    # First pass: basic physics simulation
    refined_circles = refine_solution_physics(circles.copy(), width, height, 50)
    
    # Evaluate sum of radii
    sum_radii = np.sum(refined_circles[:, 2])
    
    # Add penalty for constraint violations
    penalty = 0
    violations = 0
    
    # Boundary violations
    for i in range(len(refined_circles)):
        x, y, r = refined_circles[i]
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            penalty += 1000
            violations += 1
    
    # Overlap violations
    for i in range(len(refined_circles)):
        for j in range(i+1, len(refined_circles)):
            x1, y1, r1 = refined_circles[i]
            x2, y2, r2 = refined_circles[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < (r1 + r2):
                penalty += 1000
                violations += 1
    
    return sum_radii - penalty

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], k: int) -> List[np.ndarray]:
    """Perform tournament selection."""
    selected = []
    for _ in range(k):
        tournament_indices = random.sample(range(len(population)), 3)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index])
    return selected

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform uniform crossover between two parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # For each circle, randomly choose from either parent
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()
    
    return child1, child2

def mutate(individual: np.ndarray, width: float, height: float, rate: float) -> None:
    """Mutate an individual with adaptive mutation."""
    for i in range(len(individual)):
        if random.random() < rate:
            # Determine mutation type and magnitude
            mutation_type = random.choice(['position', 'radius'])
            
            if mutation_type == 'position':
                # Mutate position with adaptive magnitude based on circle size
                max_delta = min(0.05, individual[i][2] * 0.5)
                individual[i][0] += random.uniform(-max_delta, max_delta)
                individual[i][1] += random.uniform(-max_delta, max_delta)
                
                # Keep within bounds
                individual[i][0] = max(0.01, min(width - 0.01, individual[i][0]))
                individual[i][1] = max(0.01, min(height - 0.01, individual[i][1]))
            else:
                # Mutate radius
                max_delta = individual[i][2] * 0.3
                individual[i][2] *= random.uniform(0.7, 1.3)
                
                # Ensure positive radius
                individual[i][2] = max(0.001, min(0.3, individual[i][2]))

def refine_solution_physics(circles: np.ndarray, width: float, height: float, iterations: int) -> np.ndarray:
    """Refine solution using physics simulation with forces."""
    # Copy circles to avoid modifying originals
    refined_circles = circles.copy()
    
    # Physics parameters
    repulsion_strength = 1.0
    boundary_strength = 2.0
    attraction_strength = 0.1
    damping_factor = 0.8
    
    for iteration in range(iterations):
        # Calculate forces for each circle
        forces = np.zeros_like(refined_circles)
        
        # Compute repulsive forces (between circles)
        for i in range(len(refined_circles)):
            x1, y1, r1 = refined_circles[i]
            
            # Check nearby circles using spatial tree for efficiency
            points = refined_circles[:, :2]
            tree = cKDTree(points)
            neighbors = tree.query_ball_point([x1, y1], 2 * max(r1, 0.01), p=2)
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = refined_circles[j]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0:
                        # Repulsion force inversely proportional to distance squared
                        overlap_distance = (r1 + r2) - distance
                        if overlap_distance > 0:
                            # Stronger repulsion when overlapping
                            force_magnitude = repulsion_strength * overlap_distance / (distance + 0.001) ** 2
                            
                            if force_magnitude > 0:
                                forces[i, 0] -= force_magnitude * dx / distance
                                forces[i, 1] -= force_magnitude * dy / distance
        
        # Compute boundary forces
        for i in range(len(refined_circles)):
            x, y, r = refined_circles[i]
            
            # Boundary forces - push away from walls
            boundary_forces = boundary_strength
            
            # Left boundary
            if x - r < 0:
                forces[i, 0] += boundary_forces * (r - x)
            # Right boundary
            if x + r > width:
                forces[i, 0] -= boundary_forces * (x + r - width)
            # Bottom boundary
            if y - r < 0:
                forces[i, 1] += boundary_forces * (r - y)
            # Top boundary
            if y + r > height:
                forces[i, 1] -= boundary_forces * (y + r - height)
        
        # Apply forces and update positions
        step_size = 0.01
        for i in range(len(refined_circles)):
            # Apply forces with damping
            refined_circles[i, 0] += forces[i, 0] * step_size
            refined_circles[i, 1] += forces[i, 1] * step_size
            
            # Keep radii positive
            refined_circles[i, 2] = max(0.001, min(0.3, refined_circles[i, 2]))
            
            # Enforce valid positions
            x, y, r = refined_circles[i]
            refined_circles[i, 0] = max(r, min(width - r, x))
            refined_circles[i, 1] = max(r, min(height - r, y))
        
        # Early exit if converged (minimal force change)
        if iteration > 10 and np.all(np.abs(forces) < 1e-6):
            break
    
    # Final optimization of individual radii
    final_circles = optimize_individual_radii(refined_circles, width, height)
    
    return final_circles

def optimize_individual_radii(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize individual circle radii to maximally increase total sum."""
    optimized_circles = circles.copy()
    
    # Iteratively improve radii by trying to increase each one while respecting constraints
    improved = True
    iterations = 0
    max_iterations = 100
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to improve each circle's radius
        for i in range(len(optimized_circles)):
            x, y, r = optimized_circles[i]
            
            # Maximum possible radius at this position
            max_radius = min(x, width - x, y, height - y)
            
            # Check overlap with all others
            valid_radius = max_radius
            for j in range(len(optimized_circles)):
                if i != j:
                    x2, y2, r2 = optimized_circles[j]
                    dx = x - x2
                    dy = y - y2
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Cannot get closer than sum of radii
                    if distance < (r + r2):
                        # Reduce the radius to maintain separation
                        max_possible = distance - r2 - 0.001  # Small buffer
                        if max_possible > 0:
                            valid_radius = min(valid_radius, max_possible)
                        else:
                            valid_radius = 0.001  # Minimum radius
                    else:
                        # If far enough away, we can potentially increase radius
                        pass
            
            # Try to increase radius if beneficial
            if valid_radius > r and valid_radius > 0.001:
                # Try increasing radius up to valid limit
                new_r = min(valid_radius, r * 1.1)  # Small increase
                if new_r > r + 0.001:
                    optimized_circles[i, 2] = new_r
                    improved = True
    
    return optimized_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")