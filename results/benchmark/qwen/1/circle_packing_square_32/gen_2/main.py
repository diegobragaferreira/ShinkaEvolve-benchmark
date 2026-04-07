# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import Tuple

# Global constants for the optimization
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_valid_circle(circle: Tuple[float, float, float]) -> bool:
    """Check if a circle is within the unit square bounds."""
    x, y, r = circle
    return (r <= x <= 1 - r) and (r <= y <= 1 - r)

def check_overlap(circle1: Tuple[float, float, float], circle2: Tuple[float, float, float]) -> bool:
    """Check if two circles overlap."""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    return distance((x1, y1), (x2, y2)) < (r1 + r2)

def calculate_fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii with penalty for overlaps."""
    # Sum of all radii
    total_radius = np.sum(circles[:, 2])

    # Penalty for overlaps
    penalty = 0
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap((circles[i][0], circles[i][1], circles[i][2]),
                           (circles[j][0], circles[j][1], circles[j][2])):
                # Penalty proportional to the amount of overlap
                overlap = (circles[i][2] + circles[j][2]) - distance((circles[i][0], circles[i][1]),
                                                                    (circles[j][0], circles[j][1]))
                penalty += max(0, overlap) ** 2

    # Return fitness (higher is better)
    return total_radius - penalty * 1000

def create_random_individual() -> np.ndarray:
    """Create a random valid individual (32 circles)."""
    circles = np.zeros((32, 3))
    attempts = 0
    max_attempts = 1000

    # Try to place circles without overlapping
    for i in range(32):
        placed = False
        attempt_count = 0
        while not placed and attempt_count < max_attempts:
            # Random position and radius
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            r = random.uniform(0.01, 0.2)

            # Check if this circle is valid
            if not is_valid_circle((x, y, r)):
                attempt_count += 1
                continue

            # Check if it overlaps with any existing circles
            overlaps = False
            for j in range(i):
                if check_overlap((x, y, r), (circles[j][0], circles[j][1], circles[j][2])):
                    overlaps = True
                    break

            if not overlaps:
                circles[i] = [x, y, r]
                placed = True
            else:
                attempt_count += 1

    # If we couldn't place all circles, try to improve what we have
    return circles

def mutate(individual: np.ndarray) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = individual.copy()
    if random.random() < MUTATION_RATE:
        # Select a random circle to mutate
        idx = random.randint(0, 31)
        # Slightly perturb position and radius
        mutated[idx][0] = max(0.01, min(0.99, mutated[idx][0] + random.gauss(0, 0.02)))
        mutated[idx][1] = max(0.01, min(0.99, mutated[idx][1] + random.gauss(0, 0.02)))
        mutated[idx][2] = max(0.01, min(0.4, mutated[idx][2] + random.gauss(0, 0.01)))

        # Ensure validity of mutated circle
        if not is_valid_circle((mutated[idx][0], mutated[idx][1], mutated[idx][2])):
            # Reset to valid state if necessary
            mutated[idx][0] = max(0.01, min(0.99, mutated[idx][0]))
            mutated[idx][1] = max(0.01, min(0.99, mutated[idx][1]))
            mutated[idx][2] = max(0.01, min(0.4, mutated[idx][2]))

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    # Single point crossover on circle positions
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover point (split at some circle index)
    crossover_point = random.randint(0, 31)

    # Swap circles from crossover point onwards
    child1[crossover_point:] = parent2[crossover_point:]
    child2[crossover_point:] = parent1[crossover_point:]

    return child1, child2

def tournament_selection(population: list, fitnesses: list) -> np.ndarray:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx]

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize population
    population = []
    for _ in range(POPULATION_SIZE):
        individual = create_random_individual()
        population.append(individual)

    # Evolution loop
    best_fitness_history = []
    for generation in range(GENERATIONS):
        # Calculate fitness for all individuals
        fitnesses = [calculate_fitness(individual) for individual in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")

        # Create new population
        new_population = []

        # Elitism: keep the best individual
        best_idx = np.argmax(fitnesses)
        new_population.append(population[best_idx].copy())

        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.extend([child1, child2])

        # Trim population to exact size
        population = new_population[:POPULATION_SIZE]

    # Return the best solution found
    final_fitnesses = [calculate_fitness(individual) for individual in population]
    best_idx = np.argmax(final_fitnesses)
    best_solution = population[best_idx]

    return best_solution


# EVOLVE-BLOCK-END