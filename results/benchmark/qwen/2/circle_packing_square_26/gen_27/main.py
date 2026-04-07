# EVOLVE-BLOCK-START
import numpy as np
from typing import Tuple, List
import random
import copy
from scipy.spatial.distance import cdist

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 5
ELITISM_COUNT = 10
MAX_REPAIR_ATTEMPTS = 100

def initialize_population(n_circles: int, pop_size: int) -> np.ndarray:
    """Initialize population with random valid circle configurations"""
    population = []

    for _ in range(pop_size):
        # Generate random circles with valid positions and radii
        circles = np.zeros((n_circles, 3))

        # Start with coarse grid initialization
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y

                # Add some randomness to avoid perfect grid
                x += random.uniform(-spacing_x/4, spacing_x/4)
                y += random.uniform(-spacing_y/4, spacing_y/4)

                # Ensure within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))

                # Set initial radius
                radius = min(x, 1-x, y, 1-y) * 0.4
                radius = max(0.01, min(0.2, radius))  # Clip to reasonable range

                circles[idx] = [x, y, radius]
                idx += 1

            if idx >= n_circles:
                break

        # Add some random circles if needed
        for i in range(idx, n_circles):
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            radius = min(x, 1-x, y, 1-y) * 0.4
            radius = max(0.01, min(0.2, radius))
            circles[i] = [x, y, radius]

        population.append(circles)

    return np.array(population)

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration satisfies all constraints"""
    n_circles = len(circles)

    # Check containment constraints
    for i in range(n_circles):
        x, y, r = circles[i]
        if r > x or r > 1-x or r > y or r > 1-y:
            return False

    # Check overlap constraints
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2:
                return False

    return True

def calculate_fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii"""
    return np.sum(circles[:, 2])

def repair_invalid_configuration(circles: np.ndarray) -> np.ndarray:
    """Attempt to repair invalid configuration"""
    repaired = copy.deepcopy(circles)

    for attempt in range(MAX_REPAIR_ATTEMPTS):
        # Try to resolve overlaps by adjusting radii and positions
        valid = True

        # Check for overlaps and fix them
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]

            # Ensure containment
            new_r = min(r1, x1, 1-x1, y1, 1-y1)
            if new_r < r1:
                repaired[i, 2] = new_r
                valid = False

            # Check overlaps with other circles
            for j in range(len(repaired)):
                if i != j:
                    x2, y2, r2 = repaired[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    min_dist = r1 + r2

                    if dist < min_dist:
                        # Reduce radius or move circle
                        # Try reducing both radii proportionally
                        reduction = (min_dist - dist) / 2.0

                        # Don't let radius go below minimum
                        if r1 - reduction > 0.001:
                            repaired[i, 2] -= reduction
                            repaired[j, 2] -= reduction
                            valid = False
                        else:
                            # If we can't reduce, try moving circles
                            # Move them apart along the line connecting centers
                            dx = x2 - x1
                            dy = y2 - y1
                            dist = max(0.001, np.sqrt(dx*dx + dy*dy))
                            dx /= dist
                            dy /= dist

                            move_amount = (min_dist - dist) / 2.0
                            if r1 > 0.001:
                                repaired[i, 0] -= dx * move_amount
                                repaired[i, 1] -= dy * move_amount
                                repaired[j, 0] += dx * move_amount
                                repaired[j, 1] += dy * move_amount
                                valid = False

        # Check containment again after adjustments
        for k in range(len(repaired)):
            x, y, r = repaired[k]
            new_r = min(r, x, 1-x, y, 1-y)
            if new_r < r:
                repaired[k, 2] = new_r
                valid = False

        if valid:
            break

    return repaired

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Create offspring via uniform crossover"""
    n_circles = len(parent1)
    child = np.zeros_like(parent1)

    # Uniform crossover for each circle
    for i in range(n_circles):
        # Randomly choose which parent to take each component from
        if random.random() < 0.5:
            child[i, 0] = parent1[i, 0]  # x
            child[i, 1] = parent1[i, 1]  # y
            child[i, 2] = parent1[i, 2]  # r
        else:
            child[i, 0] = parent2[i, 0]
            child[i, 1] = parent2[i, 1]
            child[i, 2] = parent2[i, 2]

    # Add some random variation to encourage exploration
    for i in range(n_circles):
        if random.random() < 0.1:  # 10% chance per circle
            # Slight perturbation
            child[i, 0] += random.uniform(-0.02, 0.02)
            child[i, 1] += random.uniform(-0.02, 0.02)
            child[i, 2] += random.uniform(-0.01, 0.01)

            # Ensure bounds
            child[i, 0] = max(0.01, min(0.99, child[i, 0]))
            child[i, 1] = max(0.01, min(0.99, child[i, 1]))
            child[i, 2] = max(0.001, min(0.3, child[i, 2]))

    return child

def mutate(circles: np.ndarray, mutation_rate: float = MUTATION_RATE) -> np.ndarray:
    """Apply mutations to circles"""
    mutated = copy.deepcopy(circles)

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly modify x, y, or r
            choice = random.randint(0, 2)

            if choice == 0:  # Modify x
                mutated[i, 0] += random.uniform(-0.05, 0.05)
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
            elif choice == 1:  # Modify y
                mutated[i, 1] += random.uniform(-0.05, 0.05)
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
            else:  # Modify r
                mutated[i, 2] += random.uniform(-0.02, 0.02)
                mutated[i, 2] = max(0.001, min(0.3, mutated[i, 2]))

    return mutated

def select_tournament(population: np.ndarray, fitnesses: np.ndarray, tournament_size: int = TOURNAMENT_SIZE) -> int:
    """Select parent using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return winner_index

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses evolutionary algorithm approach for optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    # Initialize population
    population = initialize_population(26, POPULATION_SIZE)

    best_fitness = 0
    best_individual = None

    # Main evolutionary loop
    for generation in range(GENERATIONS):
        # Evaluate fitness for all individuals
        fitnesses = []
        for individual in population:
            if not is_valid_configuration(individual):
                individual = repair_invalid_configuration(individual)
            fitness = calculate_fitness(individual)
            fitnesses.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = copy.deepcopy(individual)

        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = population[sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]

        # Print progress every 20 generations
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        for i in range(ELITISM_COUNT):
            new_population.append(copy.deepcopy(population[i]))

        # Generate offspring through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            parent1_idx = select_tournament(population, fitnesses)
            parent2_idx = select_tournament(population, fitnesses)

            # Crossover
            child = crossover(population[parent1_idx], population[parent2_idx])

            # Mutation
            child = mutate(child, MUTATION_RATE * (1 - generation/GENERATIONS))

            # Repair if necessary
            if not is_valid_configuration(child):
                child = repair_invalid_configuration(child)

            new_population.append(child)

        # Ensure we have correct population size
        population = new_population[:POPULATION_SIZE]

    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback: return the best from final population
        return population[0]


# EVOLVE-BLOCK-END