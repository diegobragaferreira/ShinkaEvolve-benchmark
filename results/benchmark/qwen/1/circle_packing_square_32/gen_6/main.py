# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# Global constants for the optimization
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5

def initialize_population(n: int, population_size: int) -> np.ndarray:
    """Initialize a population of circle configurations"""
    population = []

    # Create a few good starting solutions using a greedy approach
    for _ in range(population_size // 4):
        circles = np.zeros((n, 3))
        # Place circles in a grid-like pattern first, then optimize
        placed = 0
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)

        for i in range(grid_size):
            for j in range(grid_size):
                if placed >= n:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                # Set initial radius to allow some overlap for now
                r = min(spacing / 2, 0.5)
                circles[placed] = [x, y, r]
                placed += 1
            if placed >= n:
                break

        # Randomize positions slightly to get diversity
        for i in range(placed):
            circles[i][0] += random.uniform(-0.05, 0.05)
            circles[i][1] += random.uniform(-0.05, 0.05)
            circles[i][0] = max(0.01, min(0.99, circles[i][0]))
            circles[i][1] = max(0.01, min(0.99, circles[i][1]))

        population.append(circles)

    # Fill remaining population with random configurations
    for _ in range(population_size - len(population)):
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i][0] = random.random()
            circles[i][1] = random.random()
            circles[i][2] = random.uniform(0.01, 0.1)
        population.append(circles)

    return population

def is_valid_placement(circles: np.ndarray, index: int) -> bool:
    """Check if a circle at given index is properly contained and doesn't overlap"""
    x, y, r = circles[index]

    # Check containment
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False

    # Check overlap with existing circles
    for i in range(len(circles)):
        if i == index:
            continue
        x1, y1, r1 = circles[i]
        dist = np.sqrt((x - x1)**2 + (y - y1)**2)
        if dist < r + r1:
            return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate the fitness of a circle configuration"""
    total_radius = np.sum(circles[:, 2])

    # Penalty for invalid placements
    penalty = 0
    for i in range(len(circles)):
        if not is_valid_placement(circles, i):
            penalty += 1000  # Large penalty for invalid configurations

    return total_radius - penalty

def mutate(circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to a circle configuration"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate either position or radius
            if random.random() < 0.5:
                # Mutate position
                mutated[i][0] += random.gauss(0, 0.02)
                mutated[i][1] += random.gauss(0, 0.02)
                # Keep within bounds
                mutated[i][0] = max(0.01, min(0.99, mutated[i][0]))
                mutated[i][1] = max(0.01, min(0.99, mutated[i][1]))
            else:
                # Mutate radius
                mutated[i][2] += random.gauss(0, 0.01)
                mutated[i][2] = max(0.001, mutated[i][2])

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents"""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    # Single-point crossover on the whole array
    point = random.randint(1, len(parent1) - 1)

    child1 = np.vstack([parent1[:point], parent2[point:]])
    child2 = np.vstack([parent2[:point], parent1[point:]])

    return child1, child2

def tournament_selection(population: list, fitnesses: list, tournament_size: int = 5) -> np.ndarray:
    """Select an individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates
        of the i-th circle of radius r.
    """
    n = 32
    population = initialize_population(n, POPULATION_SIZE)

    best_solution = None
    best_fitness = float('-inf')

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness for entire population
        fitnesses = [evaluate_fitness(circles) for circles in population]

        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Create new population
        new_population = []

        # Elitism: keep best solution
        new_population.append(best_solution)

        # Generate rest of population through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)

            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)

            new_population.extend([child1, child2])

        # Trim to exact population size if needed
        population = new_population[:POPULATION_SIZE]

    # Final refinement of best solution
    refined_best = best_solution.copy()
    # Ensure final validity
    for i in range(len(refined_best)):
        if not is_valid_placement(refined_best, i):
            # Try to fix by reducing radius and adjusting position
            refined_best[i][2] *= 0.9
            # Move to avoid conflicts if possible
            refined_best[i][0] = max(0.01, min(0.99, refined_best[i][0]))
            refined_best[i][1] = max(0.01, min(0.99, refined_best[i][1]))

    return refined_best


# EVOLVE-BLOCK-END