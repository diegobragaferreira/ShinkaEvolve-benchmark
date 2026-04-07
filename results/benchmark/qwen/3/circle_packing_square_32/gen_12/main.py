# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def evaluate_fitness(circles):
    """Evaluate fitness of circle configuration - sum of radii with penalties"""
    # Extract radii
    radii = circles[:, 2]
    sum_radii = np.sum(radii)

    # Penalty for overlapping circles
    penalty = 0

    # Calculate pairwise distances
    positions = circles[:, :2]
    distances = cdist(positions, positions)

    # Apply penalty for overlaps
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            dist = distances[i, j]
            r_i = radii[i]
            r_j = radii[j]
            # If circles overlap, apply penalty
            if dist < (r_i + r_j):
                overlap = (r_i + r_j) - dist
                penalty += overlap * 1000  # Large penalty for overlaps

    # Penalty for circles going outside bounds
    for i, (x, y, r) in enumerate(circles):
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 10000  # Large penalty for boundary violations

    return sum_radii - penalty

def mutate_individual(circles, mutation_rate=0.1, max_mutation=0.05):
    """Mutate circle positions and radii"""
    mutated = circles.copy()

    for i in range(len(circles)):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate
            choice = random.randint(0, 2)

            if choice == 0:  # Mutate x position
                mutated[i, 0] = max(0, min(1, mutated[i, 0] + random.uniform(-max_mutation, max_mutation)))
            elif choice == 1:  # Mutate y position
                mutated[i, 1] = max(0, min(1, mutated[i, 1] + random.uniform(-max_mutation, max_mutation)))
            else:  # Mutate radius
                mutated[i, 2] = max(0.001, min(0.5, mutated[i, 2] + random.uniform(-max_mutation, max_mutation)))

    return mutated

def crossover(parent1, parent2):
    """Create offspring from two parents using uniform crossover"""
    offspring = parent1.copy()

    for i in range(len(parent1)):
        if random.random() > 0.5:
            offspring[i] = parent2[i]

    return offspring

def create_initial_population(pop_size, n_circles):
    """Create initial random population of circle configurations"""
    population = []

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Generate random valid circles
        for i in range(n_circles):
            # Ensure circle fits in unit square
            r = random.uniform(0.01, 0.2)  # Reasonable starting radius
            x = random.uniform(r, 1-r)
            y = random.uniform(r, 1-r)
            circles[i] = [x, y, r]

        population.append(circles)

    return population

def check_validity(circles):
    """Check if circle configuration is valid"""
    # Check bounds
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlaps
    positions = circles[:, :2]
    radii = circles[:, 2]
    distances = cdist(positions, positions)

    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            dist = distances[i, j]
            r_i = radii[i]
            r_j = radii[j]
            if dist < (r_i + r_j):
                return False

    return True

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Genetic Algorithm parameters
    pop_size = 50
    generations = 200
    elite_size = 5
    mutation_rate = 0.1
    max_mutation = 0.05

    # Create initial population
    population = create_initial_population(pop_size, 32)

    best_fitness = float('-inf')
    best_solution = None

    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for ind in population:
            fitness = evaluate_fitness(ind)
            fitness_scores.append(fitness)

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores.sort(reverse=True)

        # Track best solution
        if fitness_scores[0] > best_fitness:
            best_fitness = fitness_scores[0]
            best_solution = population[0].copy()

        # Print progress every 20 generations
        if gen % 20 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness:.4f}")

        # Create new population with elitism
        new_population = population[:elite_size]

        # Generate offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitness_scores, 3)
            parent2 = tournament_selection(population, fitness_scores, 3)

            # Crossover
            offspring = crossover(parent1, parent2)

            # Mutation
            offspring = mutate_individual(offspring, mutation_rate, max_mutation)

            new_population.append(offspring)

        population = new_population

    print(f"Final best fitness: {best_fitness:.6f}")

    # Final validation
    if check_validity(best_solution):
        print("Solution is valid")
    else:
        print("Warning: Solution may be invalid")

    return best_solution

def tournament_selection(population, fitness_scores, tournament_size):
    """Select individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = [fitness_scores[i] for i in selected_indices]
    winner_index = selected_indices[np.argmax(selected_fitness)]
    return population[winner_index]


# EVOLVE-BLOCK-END