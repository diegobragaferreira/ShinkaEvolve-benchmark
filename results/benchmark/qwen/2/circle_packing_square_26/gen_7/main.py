# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def calculate_fitness(circles: np.ndarray) -> Tuple[float, bool]:
    """
    Calculate fitness for a set of circles.

    Returns:
        fitness: Sum of radii (positive value to maximize)
        valid: Whether the configuration is valid (no overlaps, fully contained)
    """
    n = len(circles)

    # Check containment constraints
    valid = True
    total_radius = 0.0

    for i in range(n):
        x, y, r = circles[i]
        if r <= x <= 1-r and r <= y <= 1-r:
            total_radius += r
        else:
            valid = False
            break

    if not valid:
        return -1.0, False

    # Check overlap constraints
    positions = circles[:, :2]
    radii = circles[:, 2]

    distances = cdist(positions, positions)

    for i in range(n):
        for j in range(i+1, n):
            distance = distances[i, j]
            min_distance = radii[i] + radii[j]
            if distance < min_distance:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return -1.0, False

    return total_radius, True

def create_initial_population(pop_size: int, n_circles: int) -> np.ndarray:
    """Create initial population of random circle arrangements."""
    population = []

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Generate random placements and radii
        for i in range(n_circles):
            # Random radius between 0 and 0.1 (arbitrary initial upper bound)
            r = np.random.uniform(0.001, 0.1)

            # Random position ensuring full containment
            x = np.random.uniform(r, 1-r)
            y = np.random.uniform(r, 1-r)

            circles[i] = [x, y, r]

        population.append(circles)

    return np.array(population)

def mutate_circle(circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to a circle configuration."""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate either position or radius
            if np.random.random() < 0.5:
                # Mutate position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02),
                                      mutated[i, 2], 1-mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02),
                                      mutated[i, 2], 1-mutated[i, 2])
            else:
                # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.01),
                                      0.001, 0.5)

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Single-point crossover between two parent solutions."""
    n = len(parent1)
    crossover_point = np.random.randint(1, n)

    child = np.zeros_like(parent1)

    # Take first part from parent1, second part from parent2
    child[:crossover_point] = parent1[:crossover_point]
    child[crossover_point:] = parent2[crossover_point:]

    return child

def local_optimization(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Perform local optimization to improve circle packing."""
    current = circles.copy()
    best_fitness, _ = calculate_fitness(current)

    for _ in range(max_iter):
        # Try small adjustments to each circle
        candidate = current.copy()

        for i in range(len(candidate)):
            # Small random perturbation
            dx = np.random.normal(0, 0.005)
            dy = np.random.normal(0, 0.005)
            dr = np.random.normal(0, 0.001)

            candidate[i, 0] = np.clip(candidate[i, 0] + dx,
                                    candidate[i, 2], 1-candidate[i, 2])
            candidate[i, 1] = np.clip(candidate[i, 1] + dy,
                                    candidate[i, 2], 1-candidate[i, 2])
            candidate[i, 2] = np.clip(candidate[i, 2] + dr,
                                    0.001, 0.5)

        fitness, valid = calculate_fitness(candidate)
        if valid and fitness > best_fitness:
            current = candidate
            best_fitness = fitness

    return current

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    pop_size = 50
    generations = 100
    elite_size = 10

    # Create initial population
    population = create_initial_population(pop_size, n)

    best_solution = None
    best_fitness = -float('inf')

    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        valid_individuals = []

        for individual in population:
            fitness, valid = calculate_fitness(individual)
            fitness_scores.append(fitness)

            if valid:
                valid_individuals.append((individual, fitness))

        # Sort by fitness
        valid_individuals.sort(key=lambda x: x[1], reverse=True)

        # Keep track of best solution
        if valid_individuals and valid_individuals[0][1] > best_fitness:
            best_fitness = valid_individuals[0][1]
            best_solution = valid_individuals[0][0].copy()

        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}, Best fitness: {best_fitness}")

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        for i in range(elite_size):
            if i < len(valid_individuals):
                new_population.append(valid_individuals[i][0])

        # Fill rest of population through selection, crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parent1_idx = np.random.randint(0, len(valid_individuals))
            parent2_idx = np.random.randint(0, len(valid_individuals))

            parent1 = valid_individuals[parent1_idx][0]
            parent2 = valid_individuals[parent2_idx][0]

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate_circle(child)

            # Local optimization
            child = local_optimization(child)

            new_population.append(child)

        population = np.array(new_population)

    # Final local optimization on best solution
    if best_solution is not None:
        final_solution = local_optimization(best_solution)
        fitness, _ = calculate_fitness(final_solution)
        print(f"Final fitness: {fitness}")
        return final_solution
    else:
        # Fallback to a reasonable arrangement if no valid solution was found
        circles = np.zeros((n, 3))
        # Place circles in a grid-like pattern
        rows = cols = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        radius = spacing_x / 2

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

        return circles


# EVOLVE-BLOCK-END