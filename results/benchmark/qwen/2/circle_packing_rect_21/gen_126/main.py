# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

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
    max_iterations = 5000
    population_size = 100
    elite_size = 10
    mutation_rate = 0.1
    crossover_rate = 0.8

    # Adaptive parameters based on optimization progress
    initial_mutation_rate = 0.1
    final_mutation_rate = 0.01
    adaptive_mutation = True

    # Initialize population
    population = []
    for _ in range(population_size):
        circles = generate_initial_solution(container_width, container_height, n_circles)
        population.append(circles)

    best_solution = None
    best_sum = 0

    # Evolutionary loop
    for generation in range(max_iterations):
        # Adaptive mutation rate
        if adaptive_mutation:
            current_mutation_rate = initial_mutation_rate + (final_mutation_rate - initial_mutation_rate) * (generation / max_iterations)
        else:
            current_mutation_rate = mutation_rate

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
            mutate(child1, container_width, container_height, current_mutation_rate)
            mutate(child2, container_width, container_height, current_mutation_rate)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Return best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to initial solution
        return generate_initial_solution(container_width, container_height, n_circles)

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

    # Boundary violations - more severe penalty for large violations
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0:
            penalty += (r - x) ** 2 * 1000
        if x + r > width:
            penalty += (x + r - width) ** 2 * 1000
        if y - r < 0:
            penalty += (r - y) ** 2 * 1000
        if y + r > height:
            penalty += (y + r - height) ** 2 * 1000

    # Overlap violations - use squared penalty to emphasize importance
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]

            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < (r1 + r2):
                overlap = (r1 + r2) - distance
                penalty += overlap ** 2 * 1000

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
                # Mutate position with adaptive magnitude
                mutation_magnitude = 0.05 * width if random.random() < 0.5 else 0.05 * height
                individual[i][0] += (random.random() - 0.5) * mutation_magnitude
                individual[i][1] += (random.random() - 0.5) * mutation_magnitude

                # Keep within bounds
                individual[i][0] = max(0.01, min(width - 0.01, individual[i][0]))
                individual[i][1] = max(0.01, min(height - 0.01, individual[i][1]))
            else:
                # Mutate radius with larger range for better exploration
                individual[i][2] *= random.uniform(0.5, 2.0)

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