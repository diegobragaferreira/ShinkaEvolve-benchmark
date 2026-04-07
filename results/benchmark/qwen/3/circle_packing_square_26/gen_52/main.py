# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple
from collections import defaultdict

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 26

    # Parameters for evolutionary algorithm
    population_size = 100
    generations = 500
    mutation_rate = 0.1
    elite_size = 10

    # Initialize population
    population = []
    for _ in range(population_size):
        individual = initialize_individual(n_circles)
        population.append(individual)

    # Evolutionary loop
    best_fitness_history = []
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitness_scores.append(fitness)

        # Track best fitness
        best_fitness = max(fitness_scores)
        best_fitness_history.append(best_fitness)

        # Select top individuals (elitism)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite = [population[i] for i in sorted_indices[:elite_size]]

        # Generate new population
        new_population = elite.copy()

        # Fill rest of population through crossover and mutation
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            if random.random() < 0.8:  # Crossover probability
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutation
            if random.random() < mutation_rate:
                mutate_individual(child1, mutation_rate)
            if random.random() < mutation_rate:
                mutate_individual(child2, mutation_rate)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:population_size]

        # Print progress
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

    # Return best solution
    final_fitness_scores = [evaluate_fitness(ind) for ind in population]
    best_index = np.argmax(final_fitness_scores)
    return population[best_index]


def initialize_individual(n_circles: int) -> np.ndarray:
    """Initialize a random individual (set of circles)"""
    individual = np.zeros((n_circles, 3))

    # Initialize positions and radii randomly
    for i in range(n_circles):
        # Random position within unit square with safety margin
        individual[i, 0] = np.random.uniform(0.01, 0.99)  # x coordinate
        individual[i, 1] = np.random.uniform(0.01, 0.99)  # y coordinate

        # Random radius, small enough to fit in square
        max_radius = min(0.5 - individual[i, 0], 0.5 - individual[i, 1],
                         individual[i, 0], individual[i, 1])
        individual[i, 2] = np.random.uniform(0.001, max_radius * 0.8)

    # Ensure no overlaps by applying simple adjustment
    adjust_for_overlaps(individual)

    return individual


def evaluate_fitness(individual: np.ndarray) -> float:
    """Evaluate fitness of an individual (sum of radii) with penalties for violations"""
    total_radius = np.sum(individual[:, 2])

    # Check constraints and apply penalties
    penalty = 0

    # Check containment constraints
    for i in range(len(individual)):
        x, y, r = individual[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 1000  # Heavy penalty for containment violation

    # Check overlap constraints using spatial grid for efficiency
    penalty += check_overlaps_spatial_grid(individual)

    return total_radius - penalty


def check_overlaps_spatial_grid(individual: np.ndarray) -> float:
    """Check overlaps using spatial grid indexing to improve efficiency"""
    penalty = 0
    n_circles = len(individual)

    # Define grid size (adjust based on typical circle sizes)
    grid_size = 0.1

    # Create grid mapping
    grid = defaultdict(list)

    # Place circles in grid cells
    for i in range(n_circles):
        x, y, r = individual[i]

        # Find grid cell boundaries
        min_cell_x = int(max(0, (x - r) / grid_size))
        max_cell_x = int(min(10, (x + r) / grid_size))  # 10x10 grid for unit square
        min_cell_y = int(max(0, (y - r) / grid_size))
        max_cell_y = int(min(10, (y + r) / grid_size))

        # Add circle to all relevant grid cells
        for gx in range(min_cell_x, max_cell_x + 1):
            for gy in range(min_cell_y, max_cell_y + 1):
                grid[(gx, gy)].append(i)

    # Check overlaps within and between adjacent cells
    for cell_key, circle_indices in grid.items():
        # Check all pairs in this cell
        for i in range(len(circle_indices)):
            idx1 = circle_indices[i]
            x1, y1, r1 = individual[idx1]

            for j in range(i + 1, len(circle_indices)):
                idx2 = circle_indices[j]
                x2, y2, r2 = individual[idx2]

                # Calculate distance between centers
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)

                # Check if overlapping
                if distance < r1 + r2:
                    overlap = (r1 + r2) - distance
                    penalty += overlap * 100

    return penalty


def tournament_selection(population: list, fitness_scores: list, tournament_size: int = 3) -> np.ndarray:
    """Select an individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index]


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents"""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Single point crossover on positions and radii
    crossover_point = random.randint(1, len(parent1) - 1)

    # Swap positions and radii for half the circles
    child1[crossover_point:, :2] = parent2[crossover_point:, :2]
    child1[crossover_point:, 2] = parent2[crossover_point:, 2]

    child2[crossover_point:, :2] = parent1[crossover_point:, :2]
    child2[crossover_point:, 2] = parent1[crossover_point:, 2]

    return child1, child2


def mutate_individual(individual: np.ndarray, mutation_rate: float):
    """Mutate an individual"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Mutate position slightly
            individual[i, 0] += np.random.normal(0, 0.01)
            individual[i, 1] += np.random.normal(0, 0.01)

            # Keep within bounds
            individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
            individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)

            # Mutate radius
            individual[i, 2] += np.random.normal(0, 0.005)
            individual[i, 2] = max(0.001, individual[i, 2])


def adjust_for_overlaps(individual: np.ndarray):
    """Adjust positions to prevent overlaps"""
    # Simple overlap resolution: move overlapping circles apart
    for i in range(len(individual)):
        for j in range(i+1, len(individual)):
            x1, y1, r1 = individual[i]
            x2, y2, r2 = individual[j]

            # Calculate distance between centers
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx*dx + dy*dy)

            # If circles overlap
            if distance < r1 + r2:
                # Move them apart
                overlap = (r1 + r2) - distance
                if distance > 0:
                    # Push them apart along the line connecting centers
                    push_x = dx / distance * overlap * 0.5
                    push_y = dy / distance * overlap * 0.5

                    individual[i, 0] -= push_x
                    individual[i, 1] -= push_y
                    individual[j, 0] += push_x
                    individual[j, 1] += push_y
                else:
                    # If they're at the same position, push them apart randomly
                    angle = np.random.uniform(0, 2*np.pi)
                    push_dist = overlap * 0.5
                    individual[i, 0] -= push_dist * np.cos(angle)
                    individual[i, 1] -= push_dist * np.sin(angle)
                    individual[j, 0] += push_dist * np.cos(angle)
                    individual[j, 1] += push_dist * np.sin(angle)

                # Keep within bounds
                individual[i, 0] = np.clip(individual[i, 0], r1, 1-r1)
                individual[i, 1] = np.clip(individual[i, 1], r1, 1-r1)
                individual[j, 0] = np.clip(individual[j, 0], r2, 1-r2)
                individual[j, 1] = np.clip(individual[j, 1], r2, 1-r2)


# EVOLVE-BLOCK-END