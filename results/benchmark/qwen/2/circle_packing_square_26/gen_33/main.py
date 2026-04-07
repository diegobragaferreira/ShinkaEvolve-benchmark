# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import warnings

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles: np.ndarray, idx: int) -> bool:
    """Check if circle at index idx is valid (within bounds and not overlapping)."""
    x, y, r = circles[idx]

    # Check containment constraints
    if x < r or x > 1 - r or y < r or y > 1 - r:
        return False

    # Check overlap constraints with existing circles
    for i in range(len(circles)):
        if i == idx:
            continue
        x_i, y_i, r_i = circles[i]
        distance = np.sqrt((x - x_i)**2 + (y - y_i)**2)
        if distance < r + r_i:
            return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def create_initial_grid_population(pop_size: int, n_circles: int) -> list:
    """Create initial population using grid-based initialization."""
    population = []

    # Use a more systematic grid initialization
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)

    # Start with a good baseline grid
    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))
        count = 0

        # Create initial grid pattern
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y

                # Apply small random perturbations to avoid perfect grid issues
                perturbation = 0.1 * spacing_x
                x += np.random.uniform(-perturbation, perturbation)
                y += np.random.uniform(-perturbation, perturbation)

                # Ensure it's within bounds
                x = np.clip(x, spacing_x/2, 1 - spacing_x/2)
                y = np.clip(y, spacing_y/2, 1 - spacing_y/2)

                # Initial radius - start with a reasonable size
                r = min(spacing_x, spacing_y) / 3.0

                circles[count] = [x, y, r]
                count += 1
            if count >= n_circles:
                break

        # Optimize each circle's radius to maximize total sum
        # This is a simple greedy approach to improve initial population
        for i in range(n_circles):
            current_radius = circles[i, 2]
            max_radius = min(circles[i, 0], 1 - circles[i, 0], circles[i, 1], 1 - circles[i, 1])

            # Try to increase radius while maintaining validity
            if max_radius > current_radius:
                # Binary search for maximum valid radius
                low, high = current_radius, max_radius
                best_radius = current_radius

                # Simple binary search for better radius
                for _ in range(10):
                    mid = (low + high) / 2
                    circles[i, 2] = mid

                    # Check if this works with others
                    valid = True
                    for j in range(n_circles):
                        if i != j:
                            dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 +
                                         (circles[i, 1] - circles[j, 1])**2)
                            if dist < mid + circles[j, 2]:
                                valid = False
                                break

                    if valid:
                        best_radius = mid
                        low = mid
                    else:
                        high = mid

                circles[i, 2] = best_radius

            # Ensure it's still valid
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])

        population.append(circles.copy())

    return population

def mutate_individual(circles: np.ndarray, mutation_rate: float = 0.1, generation: int = 0, max_generations: int = 100) -> np.ndarray:
    """Apply mutation to an individual with adaptive rates."""
    mutated = circles.copy()

    # Adaptive mutation rate - decreases over generations
    adaptive_rate = mutation_rate * (1.0 - generation / max_generations)

    for i in range(len(mutated)):
        if np.random.random() < adaptive_rate:
            # Mutate either position or radius with preference
            if np.random.random() < 0.7:  # 70% chance to mutate position
                # Mutate position with adaptive step sizes
                step_size = 0.02 * (1.0 - generation / max_generations)  # Smaller steps later
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, step_size), 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, step_size), 0.01, 0.99)
            else:  # 30% chance to mutate radius
                # Mutate radius with adaptive step size
                step_size = 0.01 * (1.0 - generation / max_generations)
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, step_size), 0.001, 0.2)

    # Local optimization: try to increase radii where possible
    for i in range(len(mutated)):
        if np.random.random() < 0.1:  # Small chance to optimize radius
            original_radius = mutated[i, 2]
            current_pos = mutated[i, :2]

            # Try to increase radius without violating constraints
            max_possible_radius = min(
                mutated[i, 0],
                1 - mutated[i, 0],
                mutated[i, 1],
                1 - mutated[i, 1]
            )

            # Binary search for best radius
            low, high = original_radius, max_possible_radius
            found_valid = False

            for _ in range(15):
                mid = (low + high) / 2
                mutated[i, 2] = mid

                # Check overlap with all others
                valid = True
                for j in range(len(mutated)):
                    if i != j:
                        dist = np.sqrt((current_pos[0] - mutated[j, 0])**2 +
                                     (current_pos[1] - mutated[j, 1])**2)
                        if dist < mid + mutated[j, 2]:
                            valid = False
                            break

                if valid:
                    found_valid = True
                    low = mid
                else:
                    high = mid

            if found_valid:
                mutated[i, 2] = low

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents with better constraint handling."""
    # Use a more sophisticated crossover that prioritizes better parents
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Use a weighted crossover strategy based on fitness
    # Parents with higher fitness contribute more to the child
    fitness1 = evaluate_fitness(parent1)
    fitness2 = evaluate_fitness(parent2)

    # Prefer better parent for crossover
    if fitness1 > fitness2:
        better_parent = parent1
        worse_parent = parent2
    else:
        better_parent = parent2
        worse_parent = parent1

    # Crossover strategy: take 70% from better parent, 30% from worse
    for i in range(len(child1)):
        if np.random.random() < 0.7:  # 70% from better parent
            child1[i] = better_parent[i].copy()
            child2[i] = better_parent[i].copy()
        else:  # 30% from worse parent
            child1[i] = worse_parent[i].copy()
            child2[i] = worse_parent[i].copy()

    return child1, child2

def select_parents(population: list, fitnesses: list, tournament_size: int = 3) -> list:
    """Select parents using tournament selection with better diversity."""
    selected = []

    for _ in range(len(population)):
        # Tournament selection with better diversity consideration
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        # Select based on fitness but with some randomness
        probabilities = np.array(tournament_fitnesses) / np.sum(tournament_fitnesses)
        probabilities = probabilities / np.sum(probabilities)  # Normalize

        winner_idx = np.random.choice(tournament_indices, p=probabilities)
        selected.append(population[winner_idx])

    return selected

def optimize_circles() -> np.ndarray:
    """Main optimization function using enhanced evolutionary algorithm."""
    n_circles = 26
    pop_size = 50
    generations = 100
    initial_mutation_rate = 0.1

    # Create better initial population
    population = create_initial_grid_population(pop_size, n_circles)

    best_fitness = 0
    best_individual = None

    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Select parents
        parents = select_parents(population, fitnesses)

        # Create new population through crossover and mutation
        new_population = []

        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)

        # Generate offspring
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation with adaptive rate
            child1 = mutate_individual(child1, initial_mutation_rate, generation, generations)
            child2 = mutate_individual(child2, initial_mutation_rate, generation, generations)

            # Ensure children meet constraints
            if is_valid_placement(child1, len(child1)-1):  # Check last circle
                new_population.append(child1)
            if len(new_population) < pop_size and is_valid_placement(child2, len(child2)-1):
                new_population.append(child2)

        # Trim to population size
        population = new_population[:pop_size]

    return best_individual if best_individual is not None else population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = optimize_circles()
        return circles
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to improved heuristic
        circles = np.zeros((26, 3))

        # Try to create a more organized pattern
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x / 2.0

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                # Slightly randomize to avoid perfect grid issues
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break

        # Ensure constraints are satisfied and try to optimize radii
        for i in range(count):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])

        return circles


# EVOLVE-BLOCK-END