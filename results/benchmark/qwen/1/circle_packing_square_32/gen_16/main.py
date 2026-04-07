# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def evaluate_fitness(circles):
    """Evaluate fitness as sum of radii with penalty for constraint violations."""
    # Calculate sum of radii
    total_radius = np.sum(circles[:, 2])

    # Penalty for overlapping circles
    penalty = 0
    n = len(circles)

    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]

            # Distance between centers
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

            # Overlap penalty
            if dist < (r1 + r2):
                overlap = (r1 + r2) - dist
                penalty += overlap**2

    # Penalty for circles outside bounds
    for i in range(n):
        x, y, r = circles[i]
        if x-r < 0 or x+r > 1 or y-r < 0 or y+r > 1:
            penalty += 1000

    return total_radius - penalty * 0.01

def generate_initial_solution():
    """Generate initial solution using greedy placement in corners and center."""
    circles = []
    n = 32

    # Corner placements
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i, (x, y) in enumerate(corners):
        if i < n:
            circles.append([x, y, min(x, y, 1-x, 1-y)])

    # Edge placements
    edges = [
        (0.5, 0.1), (0.5, 0.9),  # top and bottom midpoints
        (0.1, 0.5), (0.9, 0.5)   # left and right midpoints
    ]
    for i, (x, y) in enumerate(edges):
        if len(circles) < n:
            circles.append([x, y, min(x, y, 1-x, 1-y)])

    # Center placement
    if len(circles) < n:
        circles.append([0.5, 0.5, 0.2])

    # Fill remaining spots with greedy placement
    while len(circles) < n:
        max_radius = 0
        best_x, best_y = 0.5, 0.5

        # Try several random positions to find good candidate
        for _ in range(500):
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)

            # Find minimum distance to existing circles
            min_dist = float('inf')
            for cx, cy, r in circles:
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                min_dist = min(min_dist, dist)

            # Maximum possible radius at this point
            max_r = min(x, 1-x, y, 1-y)

            # If we can place a circle here
            if min_dist >= max_r and max_r > 0:
                # Prefer positions that don't overlap with existing circles
                if min_dist > max_radius:
                    max_radius = min_dist
                    best_x, best_y = x, y

        # Add circle with reasonable radius
        if max_radius > 0:
            r = min(max_radius, 0.15)
            circles.append([best_x, best_y, r])

    return np.array(circles)

def mutate_solution(circles, mutation_rate=0.1):
    """Apply mutation to solution."""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position
            mutated[i, 0] += random.uniform(-0.02, 0.02)
            mutated[i, 1] += random.uniform(-0.02, 0.02)
            # Keep within bounds
            mutated[i, 0] = np.clip(mutated[i, 0], 0.01, 0.99)
            mutated[i, 1] = np.clip(mutated[i, 1], 0.01, 0.99)

            # Mutate radius (with bounds)
            mutated[i, 2] += random.uniform(-0.01, 0.01)
            mutated[i, 2] = max(0.01, min(0.4, mutated[i, 2]))

    return mutated

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    child = parent1.copy()

    # Single point crossover
    crossover_point = random.randint(1, len(parent1)-1)

    for i in range(crossover_point, len(parent1)):
        child[i] = parent2[i].copy()

    return child

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    population_size = 50
    generations = 100

    # Initialize population
    population = []
    for _ in range(population_size):
        sol = generate_initial_solution()
        population.append(sol)

    # Evolutionary optimization
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = []
        for sol in population:
            fitness = evaluate_fitness(sol)
            fitness_scores.append(fitness)

        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Keep top 50% and create offspring
        elite_count = population_size // 2
        new_population = population[:elite_count]

        # Create offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = random.randint(0, elite_count-1)
            parent2_idx = random.randint(0, elite_count-1)

            child = crossover(population[parent1_idx], population[parent2_idx])
            child = mutate_solution(child)
            new_population.append(child)

        population = new_population

    # Return best solution
    best_solution = population[0]
    return best_solution


# EVOLVE-BLOCK-END