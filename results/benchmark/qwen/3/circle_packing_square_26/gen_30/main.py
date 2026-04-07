# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained within the unit square."""
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap(circles: np.ndarray) -> bool:
    """Check if any circles overlap."""
    # Calculate pairwise distances
    positions = circles[:, :2]
    radii = circles[:, 2]

    # Create distance matrix
    distances = cdist(positions, positions)

    # Check for overlaps (distance between centers < sum of radii)
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] < radii[i] + radii[j]:
                return False
    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a circle configuration."""
    if not check_containment(circles) or not check_overlap(circles):
        # Penalize constraint violations heavily
        return -1000.0

    # Return the sum of radii as fitness
    return np.sum(circles[:, 2])

def initialize_population(pop_size: int, n_circles: int) -> list:
    """Initialize a population with valid random configurations."""
    population = []

    for _ in range(pop_size):
        # Try to create a valid configuration
        max_attempts = 1000
        attempts = 0
        valid = False

        while not valid and attempts < max_attempts:
            # Generate random circles with small radii
            circles = np.zeros((n_circles, 3))

            # Random positions and radii
            for i in range(n_circles):
                # Random radius between 0.01 and 0.15
                r = random.uniform(0.01, 0.15)
                # Random position ensuring containment
                x = random.uniform(r, 1-r)
                y = random.uniform(r, 1-r)
                circles[i] = [x, y, r]

            # If valid, accept this configuration
            if check_containment(circles) and check_overlap(circles):
                valid = True
                population.append(circles)
            else:
                attempts += 1

        # If we couldn't generate a valid configuration, use a grid approach
        if not valid:
            circles = np.zeros((n_circles, 3))
            # Place circles in a grid pattern
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = int(np.ceil(n_circles / rows))
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)

            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x = spacing_x * (j + 1)
                    y = spacing_y * (i + 1)
                    r = min(spacing_x, spacing_y) * 0.4  # Radius smaller than spacing
                    circles[idx] = [x, y, r]
                    idx += 1

            # Adjust radii to prevent overlaps
            for i in range(n_circles):
                circles[i, 2] = min(
                    circles[i, 0],
                    1 - circles[i, 0],
                    circles[i, 1],
                    1 - circles[i, 1]
                ) * 0.8  # Slightly smaller to ensure containment

            population.append(circles)

    return population

def tournament_selection(population: list, fitnesses: list, tournament_size: int = 3) -> np.ndarray:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two parents."""
    n = len(parent1)
    child = np.copy(parent1)

    # Single point crossover for circle positions and radii
    crossover_point = random.randint(1, n - 1)

    # Swap positions and radii for circles after crossover point
    child[crossover_point:, :2] = parent2[crossover_point:, :2]
    child[crossover_point:, 2] = parent2[crossover_point:, 2]

    # Ensure child is valid (containment and overlap constraints)
    # Apply a small correction to make sure circles stay within bounds
    for i in range(n):
        x, y, r = child[i]
        # Clip to ensure containment
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        child[i] = [x, y, r]

    # If there are overlaps, try to resolve them
    if not check_overlap(child):
        # Simple approach - adjust positions to avoid overlaps
        # This is a basic fix, could be improved with better optimization
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap(child):
                    # Move circles apart if they overlap
                    x1, y1, r1 = child[i]
                    x2, y2, r2 = child[j]
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if distance < r1 + r2:
                        # Move them apart along the line connecting their centers
                        dx = x2 - x1
                        dy = y2 - y1
                        if distance > 0:
                            scale = (r1 + r2 - distance) / distance * 0.5
                            child[i, 0] -= dx * scale
                            child[i, 1] -= dy * scale
                            child[j, 0] += dx * scale
                            child[j, 1] += dy * scale

    return child

def mutate(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = np.copy(individual)
    n = len(mutated)

    for i in range(n):
        if random.random() < mutation_rate:
            # Mutate either position or radius
            if random.random() < 0.5:
                # Mutate position
                mutated[i, 0] += random.gauss(0, 0.02)
                mutated[i, 1] += random.gauss(0, 0.02)
                # Ensure containment
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1-mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1-mutated[i, 2])
            else:
                # Mutate radius
                mutated[i, 2] *= np.exp(random.gauss(0, 0.2))  # Log-normal mutation
                # Ensure valid range
                mutated[i, 2] = np.clip(mutated[i, 2], 0.01, 0.3)

    return mutated

def optimize_circles_evolutionary(n_circles: int = 26, pop_size: int = 50,
                                generations: int = 100,
                                mutation_rate: float = 0.1) -> np.ndarray:
    """Main evolutionary optimization function."""
    # Initialize population
    population = initialize_population(pop_size, n_circles)

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(ind) for ind in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Print progress every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness}")

        # Create new population
        new_population = []

        # Keep the best individual (elitism)
        best_idx = np.argmax(fitnesses)
        new_population.append(population[best_idx])

        # Generate rest of population through selection, crossover, and mutation
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population

        # Adaptive mutation rate - decrease over time
        if gen > 0 and gen % 20 == 0:
            mutation_rate *= 0.95

    # Return the best individual
    final_fitnesses = [evaluate_fitness(ind) for ind in population]
    best_idx = np.argmax(final_fitnesses)
    return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run evolutionary optimization
    circles = optimize_circles_evolutionary(n_circles=26, pop_size=50, generations=100)

    return circles


# EVOLVE-BLOCK-END