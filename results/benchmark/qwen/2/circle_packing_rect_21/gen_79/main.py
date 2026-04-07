# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container dimensions - rectangle with perimeter 4, so width + height = 2
    # Using 1:1 ratio for simplicity (square)
    container_width = 1.0
    container_height = 1.0

    # Parameters
    n_circles = 21
    max_iterations = 2000
    population_size = 80
    elite_size = 8
    mutation_rate = 0.15
    crossover_rate = 0.8

    # Initialize population
    population = []
    for _ in range(population_size):
        circles = generate_initial_solution(container_width, container_height, n_circles)
        population.append(circles)

    best_solution = None
    best_sum = 0

    # Evolutionary loop with alternating phases
    for generation in range(max_iterations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual, container_width, container_height)
            fitness_scores.append(fitness)

        # Update best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_sum:
            best_sum = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Apply hybrid optimization - local refinement before selection
        refined_population = []
        for individual in population:
            # Apply local optimization to improve individual quality
            refined = local_refinement(individual.copy(), container_width, container_height)
            refined_population.append(refined)

        # Update population with refined individuals
        population = refined_population

        # Selection (tournament)
        selected = tournament_selection(population, fitness_scores, population_size)

        # Crossover and mutation
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % population_size]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            mutate(child1, container_width, container_height, mutation_rate)
            mutate(child2, container_width, container_height, mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Final local optimization on best solution
    if best_solution is not None:
        final_solution = local_refinement(best_solution.copy(), container_width, container_height)
        return final_solution
    else:
        # Fallback to initial solution with final refinement
        initial = generate_initial_solution(container_width, container_height, n_circles)
        return local_refinement(initial, container_width, container_height)

def generate_initial_solution(width: float, height: float, n_circles: int) -> np.ndarray:
    """Generate initial solution using hexagonal packing with better spacing."""
    circles = np.zeros((n_circles, 3))

    # More structured grid arrangement
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))

    # Adjust for rectangle dimensions with better spacing
    cell_width = width / (cols + 1)
    cell_height = height / (rows + 1)

    # Place circles in a grid pattern with strategic randomness
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break

            # Add more systematic positioning with less randomness for better initial packing
            x = (j + 1) * cell_width + (random.random() - 0.5) * cell_width * 0.2
            y = (i + 1) * cell_height + (random.random() - 0.5) * cell_height * 0.2

            # Ensure valid bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))

            # Initial radius based on available space, but with better scaling
            min_radius = min(x, width - x, y, height - y) * 0.4

            # Use more informed radius selection
            radius = min_radius * random.uniform(0.3, 0.7)

            circles[idx] = [x, y, radius]
            idx += 1

    # Apply initial refinement to handle overlaps
    circles = refine_positions(circles, width, height)

    return circles

def evaluate_fitness(circles: np.ndarray, width: float, height: float) -> float:
    """Evaluate fitness based on sum of radii, with improved penalty system."""
    sum_radii = np.sum(circles[:, 2])

    # Better constraint checking with prioritized penalties
    penalty = 0

    # Boundary violations - higher penalty for severe violations
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Calculate how much we're violating boundaries
        left_violation = max(0, r - x)
        right_violation = max(0, x + r - width)
        bottom_violation = max(0, r - y)
        top_violation = max(0, y + r - height)

        # Higher penalty for severe violations
        boundary_penalty = (left_violation + right_violation + bottom_violation + top_violation) * 500
        penalty += boundary_penalty

    # Overlap violations - check with early termination
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]

            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < (r1 + r2):
                # Penalty based on how much they overlap
                overlap = (r1 + r2) - distance
                penalty += overlap * 1000

    # Return fitness with stronger penalty for constraint violations
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

    # Apply crossover with more structured approach
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def mutate(individual: np.ndarray, width: float, height: float, rate: float) -> None:
    """Mutate an individual with more intelligent mutation strategy."""
    for i in range(len(individual)):
        if random.random() < rate:
            # Mutate position or radius with different strategies
            if random.random() < 0.6:  # 60% chance to mutate position
                # Mutate position with adaptive step size based on container size
                step_x = 0.05 * width
                step_y = 0.05 * height
                individual[i][0] += (random.random() - 0.5) * step_x
                individual[i][1] += (random.random() - 0.5) * step_y

                # Keep within bounds
                individual[i][0] = max(0.01, min(width - 0.01, individual[i][0]))
                individual[i][1] = max(0.01, min(height - 0.01, individual[i][1]))
            else:  # 40% chance to mutate radius
                # Mutate radius with more controlled change
                scale_factor = random.uniform(0.7, 1.3)
                individual[i][2] *= scale_factor

                # Ensure positive radius
                individual[i][2] = max(0.001, individual[i][2])

def refine_positions(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Refine positions to ensure no overlaps and respect boundaries."""
    # Use a more sophisticated iterative improvement approach
    for _ in range(50):  # Reduced iterations for performance
        updated = False
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Calculate maximum possible radius at this location
            max_r = min(x, width - x, y, height - y)

            # Check for overlap with other circles
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x2 - x)**2 + (y2 - y)**2)

                    # Can't get closer than sum of radii
                    if distance < (r + r2):
                        max_r = min(max_r, distance - r2)

            # Increase radius if beneficial and safe
            if max_r > r and max_r > 0.001:
                circles[i][2] = min(max_r, max_r * 1.05)  # Slightly favor increase
                updated = True

        if not updated:
            break

    return circles

def local_refinement(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Apply advanced local refinement to improve solution quality."""
    # Phase 1: Physics-based relaxation to resolve overlaps
    circles = physics_based_relaxation(circles, width, height)

    # Phase 2: Iterative radius optimization
    circles = optimize_radii_iteratively(circles, width, height)

    # Phase 3: Final boundary correction
    circles = enforce_boundaries(circles, width, height)

    return circles

def physics_based_relaxation(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Apply physics-inspired relaxation to resolve overlap constraints."""
    # Simple force-based approach to move overlapping circles apart
    max_iterations = 100
    step_size = 0.01

    for iteration in range(max_iterations):
        forces = np.zeros_like(circles)

        # Calculate forces between overlapping circles
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]

            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = np.sqrt(dx*dx + dy*dy)

                    # Only apply force if circles are overlapping
                    if distance > 0 and distance < (r1 + r2):
                        force_magnitude = (r1 + r2 - distance) / (distance + 0.001)
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance

        # Apply forces and correct boundary violations
        for i in range(len(circles)):
            # Apply forces
            circles[i, 0] += forces[i, 0] * step_size
            circles[i, 1] += forces[i, 1] * step_size

            # Enforce boundary constraints
            x, y, r = circles[i]
            circles[i, 0] = max(r, min(width - r, x))
            circles[i, 1] = max(r, min(height - r, y))

            # Adjust radius to fit in available space
            max_radius = min(x, width - x, y, height - y)
            circles[i, 2] = min(circles[i, 2], max_radius * 0.95)

    return circles

def optimize_radii_iteratively(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Iteratively optimize radii to maximize sum while respecting constraints."""
    # Try to increase radii one by one
    improved = True
    attempts = 0
    max_attempts = 30

    while improved and attempts < max_attempts:
        improved = False
        attempts += 1

        # Try to increase each radius individually
        for i in range(len(circles)):
            old_radius = circles[i, 2]
            x, y, r = circles[i]

            # Calculate max possible radius
            max_radius = min(x, width - x, y, height - y)

            # Check overlap with all others
            valid_max_radius = max_radius
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x2 - x)**2 + (y2 - y)**2)
                    # Maximum radius without overlap
                    overlap_radius = distance - r2
                    if overlap_radius > 0:
                        valid_max_radius = min(valid_max_radius, overlap_radius)

            # Try to increase radius if beneficial
            if valid_max_radius > r and valid_max_radius > 0.001:
                # Try increasing to a fraction of the maximum possible
                new_radius = min(valid_max_radius, r * 1.1)
                if new_radius > r + 0.001:
                    circles[i, 2] = new_radius
                    improved = True

    return circles

def enforce_boundaries(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Ensure all circles respect boundary constraints."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Constrain to valid region
        x = max(r, min(width - r, x))
        y = max(r, min(height - r, y))
        # Adjust radius if necessary
        max_radius = min(x, width - x, y, height - y)
        r = min(r, max_radius * 0.99)
        circles[i] = [x, y, r]
    return circles

def generate_initial_solution(width: float, height: float, n_circles: int) -> np.ndarray:
    """Generate initial solution using hexagonal packing."""
    circles = np.zeros((n_circles, 3))

    # Try different grid arrangements to find a good initial configuration
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))

    # Adjust for rectangle dimensions
    cell_width = width / cols
    cell_height = height / rows

    # Place circles in a grid pattern with some randomness
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break

            x = (j + 0.5) * cell_width + (random.random() - 0.5) * cell_width * 0.3
            y = (i + 0.5) * cell_height + (random.random() - 0.5) * cell_height * 0.3

            # Ensure valid bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))

            # Initial radius based on available space
            min_radius = min(x, width - x, y, height - y) * 0.3

            # Random small radius to encourage evolution
            radius = min_radius * random.uniform(0.2, 0.8)

            circles[idx] = [x, y, radius]
            idx += 1

    # Refine using local optimization to ensure no overlaps
    circles = refine_positions(circles, width, height)

    return circles

def evaluate_fitness(circles: np.ndarray, width: float, height: float) -> float:
    """Evaluate fitness based on sum of radii, penalizing constraint violations."""
    sum_radii = np.sum(circles[:, 2])

    # Check constraints and apply penalties
    penalty = 0

    # Boundary violations
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            penalty += 1000

    # Overlap violations
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]

            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < (r1 + r2):
                penalty += 1000

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
    """Mutate an individual."""
    for i in range(len(individual)):
        if random.random() < rate:
            # Mutate position or radius
            if random.random() < 0.5:
                # Mutate position
                individual[i][0] += (random.random() - 0.5) * 0.1 * width
                individual[i][1] += (random.random() - 0.5) * 0.1 * height

                # Keep within bounds
                individual[i][0] = max(0.01, min(width - 0.01, individual[i][0]))
                individual[i][1] = max(0.01, min(height - 0.01, individual[i][1]))
            else:
                # Mutate radius
                individual[i][2] *= random.uniform(0.8, 1.2)

                # Ensure positive radius
                individual[i][2] = max(0.001, individual[i][2])

def refine_positions(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Refine positions to ensure no overlaps and respect boundaries."""
    # Use a simple iterative improvement approach
    for _ in range(100):  # Limited iterations for performance
        # Try to increase radii while maintaining no overlaps
        updated = False
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Calculate maximum possible radius at this location
            max_r = min(x, width - x, y, height - y)

            # Check for overlap with other circles
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x2 - x)**2 + (y2 - y)**2)

                    # Can't get closer than sum of radii
                    if distance < (r + r2):
                        max_r = min(max_r, distance - r2)

            # Increase radius if beneficial and safe
            if max_r > r and max_r > 0.001:
                circles[i][2] = min(max_r, max_r * 1.05)  # Slightly favor increase
                updated = True

        if not updated:
            break

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")