# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def check_containment(circles):
    """Check if all circles are fully contained within the unit square"""
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap(circles):
    """Check if any circles overlap"""
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
    return True

def evaluate_fitness(circles):
    """Evaluate fitness as sum of radii, with penalty for constraint violations"""
    if not check_containment(circles) or not check_overlap(circles):
        # Large negative penalty for constraint violations
        return -1000.0

    total_radius = np.sum(circles[:, 2])
    return total_radius

def crossover(parent1, parent2):
    """Uniform crossover for circle packing"""
    n = len(parent1)
    child = np.zeros_like(parent1)

    # For each circle, randomly choose from parent1 or parent2
    for i in range(n):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    return child

def mutate(circles, mutation_rate=0.1, max_mutation=0.05):
    """Apply mutation to circles"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly mutate position and/or radius
            if random.random() < 0.5:
                # Mutate position
                mutated[i, 0] += np.random.normal(0, max_mutation)
                mutated[i, 1] += np.random.normal(0, max_mutation)
                # Ensure it stays within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], 0, 1)
                mutated[i, 1] = np.clip(mutated[i, 1], 0, 1)
            else:
                # Mutate radius
                mutated[i, 2] += np.random.normal(0, max_mutation/2)
                # Ensure radius remains positive
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def initialize_population(pop_size, n_circles):
    """Initialize population with random valid configurations"""
    population = []

    for _ in range(pop_size):
        # Start with random positions and small radii
        circles = np.zeros((n_circles, 3))

        # Initialize positions and radii
        for i in range(n_circles):
            # Try to place circle in valid location
            attempts = 0
            valid_placement = False

            while not valid_placement and attempts < 100:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = np.random.uniform(0.01, 0.1)

                # Check if this circle would fit with existing ones
                temp_circles = circles.copy()
                temp_circles[i] = [x, y, r]

                # Simple check: ensure it fits in the square and doesn't overlap with others
                if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                    valid_placement = True

                    # Check overlap with existing circles
                    for j in range(i):
                        existing_x, existing_y, existing_r = temp_circles[j]
                        distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if distance < r + existing_r:
                            valid_placement = False
                            break

                attempts += 1

            if valid_placement:
                circles[i] = [x, y, r]
            else:
                # If we couldn't place it properly, just use random values
                circles[i] = [np.random.uniform(0.05, 0.95),
                             np.random.uniform(0.05, 0.95),
                             np.random.uniform(0.01, 0.1)]

        population.append(circles)

    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n_circles = 26
    pop_size = 50
    generations = 200
    elite_size = 5

    # Initialize population
    population = initialize_population(pop_size, n_circles)

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Create new population
        new_population = []

        # Elitism: keep the best individuals
        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child, mutation_rate=0.1, max_mutation=0.05)

            new_population.append(child)

        population = new_population[:pop_size]  # Ensure exact population size

        # Print progress
        if gen % 20 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness:.4f}")

    # Return the best solution found
    final_fitnesses = [evaluate_fitness(individual) for individual in population]
    best_idx = np.argmax(final_fitnesses)
    best_solution = population[best_idx]

    # Final validation
    if not check_containment(best_solution) or not check_overlap(best_solution):
        print("Warning: Best solution violates constraints")

    print(f"Final fitness: {evaluate_fitness(best_solution):.6f}")

    return best_solution


# EVOLVE-BLOCK-END