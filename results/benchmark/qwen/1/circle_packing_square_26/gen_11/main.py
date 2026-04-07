# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def check_collision(circle1, circle2):
    """Check if two circles collide"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance < (r1 + r2)

def is_valid_placement(circle, circles):
    """Check if a circle placement is valid (within bounds and no collisions)"""
    x, y, r = circle

    # Check containment
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False

    # Check collisions with existing circles
    for c in circles:
        if check_collision(circle, c):
            return False

    return True

def calculate_sum_radii(circles):
    """Calculate the sum of all radii"""
    return sum(circle[2] for circle in circles)

def mutate_circle(circle, mutation_strength=0.05):
    """Mutate a single circle (randomly change position and radius)"""
    x, y, r = circle.copy()

    # Mutate position
    x += np.random.normal(0, mutation_strength)
    y += np.random.normal(0, mutation_strength)

    # Mutate radius (ensure positive)
    r *= np.random.uniform(0.8, 1.2)
    r = max(0.001, r)

    return np.array([x, y, r])

def crossover_circles(circle1, circle2):
    """Perform crossover between two circles"""
    # Simple average crossover
    x = (circle1[0] + circle2[0]) / 2
    y = (circle1[1] + circle2[1]) / 2
    r = (circle1[2] + circle2[2]) / 2
    return np.array([x, y, r])

def create_initial_population(n_circles, population_size=100):
    """Create initial population of circle arrangements"""
    population = []
    for _ in range(population_size):
        circles = []
        while len(circles) < n_circles:
            # Random placement
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            r = np.random.uniform(0.01, 0.1)

            candidate = np.array([x, y, r])
            if is_valid_placement(candidate, circles):
                circles.append(candidate)

        # If we couldn't place enough circles, try again with smaller radii
        if len(circles) < n_circles:
            circles = []
            for i in range(n_circles):
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                r = np.random.uniform(0.01, 0.05)
                circles.append(np.array([x, y, r]))

        population.append(np.array(circles))
    return population

def evaluate_fitness(circles):
    """Evaluate fitness as sum of radii for valid solutions"""
    if len(circles) == 0:
        return 0

    # Calculate sum of radii
    total_radius = calculate_sum_radii(circles)

    # Penalize invalid placements
    penalty = 0
    for i, circle in enumerate(circles):
        x, y, r = circle
        # Check containment
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty -= 1000

        # Check collisions with others
        for j in range(i):
            other = circles[j]
            if check_collision(circle, other):
                penalty -= 1000

    return total_radius + penalty

def select_parents(population, fitnesses, tournament_size=3):
    """Tournament selection for parents"""
    selected = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_index].copy())
    return selected

def evolve_circles(n_generations=500, population_size=100, elite_size=10):
    """Main evolutionary algorithm for circle packing"""
    n_circles = 26
    best_solution = None
    best_fitness = -np.inf

    # Create initial population
    population = create_initial_population(n_circles, population_size)

    for generation in range(n_generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Print progress
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_indices]
        fitnesses = [fitnesses[i] for i in sorted_indices]

        # Create new population
        new_population = []

        # Elite preservation
        for i in range(elite_size):
            new_population.append(population[i].copy())

        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Select parents
            parents = select_parents(population, fitnesses)
            parent1 = parents[0]
            parent2 = parents[1]

            # Crossover
            child = crossover_circles(parent1, parent2)

            # Mutation
            if np.random.rand() < 0.8:  # 80% chance of mutation
                child = mutate_circle(child)

            # Ensure validity
            if is_valid_placement(child, new_population):
                new_population.append(child)
            else:
                # Try to make a valid version
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                r = np.random.uniform(0.001, 0.05)
                new_population.append(np.array([x, y, r]))

        population = new_population[:population_size]

    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Run evolutionary algorithm
    circles = evolve_circles(n_generations=300, population_size=100, elite_size=10)

    # If we got a valid solution, return it; otherwise return zeros
    if circles is not None and len(circles) == 26:
        return circles
    else:
        # Fallback to simple initialization
        circles = np.zeros((26, 3))
        # Place circles in a grid-like pattern
        n_per_row = int(np.ceil(np.sqrt(26)))
        spacing = 1.0 / (n_per_row + 1)
        radius = spacing * 0.4

        idx = 0
        for i in range(n_per_row):
            for j in range(n_per_row):
                if idx >= 26:
                    break
                x = (j + 1) * spacing
                y = (i + 1) * spacing
                circles[idx] = [x, y, radius]
                idx += 1

        return circles


# EVOLVE-BLOCK-END