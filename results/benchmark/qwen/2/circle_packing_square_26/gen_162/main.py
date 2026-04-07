# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple

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

def local_optimize(circles: np.ndarray, max_iterations: int = 20) -> np.ndarray:
    """Apply local optimization to improve a solution by making small adjustments."""
    optimized = circles.copy()

    # Try to increase radii where possible
    for _ in range(max_iterations):
        improved = False

        # For each circle, try to increase its radius
        for i in range(len(optimized)):
            current_r = optimized[i, 2]

            # Try increasing radius slightly
            new_r = min(current_r * 1.05, 0.5)  # Cap at reasonable size

            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(len(optimized)):
                if i != j:
                    dist = np.sqrt((optimized[i, 0] - optimized[j, 0])**2 +
                                 (optimized[i, 1] - optimized[j, 1])**2)
                    if dist < new_r + optimized[j, 2]:
                        valid = False
                        break

            # Update if improvement is possible and within bounds
            if valid and new_r <= 1 - max(optimized[i, 0], 1 - optimized[i, 0]) and \
               new_r <= 1 - max(optimized[i, 1], 1 - optimized[i, 1]):
                optimized[i, 2] = new_r
                improved = True

        # If no improvement made, stop early
        if not improved:
            break

    return optimized

def create_initial_population(pop_size: int, n_circles: int) -> list:
    """Create initial population of valid circle arrangements."""
    population = []

    # Use Voronoi-based initialization for better initial spread
    for _ in range(pop_size):
        # Generate initial points using a more sophisticated Voronoi approach
        points = generate_voronoi_points(n_circles)

        # Create circles from the Voronoi points
        circles = np.zeros((n_circles, 3))
        for i, (x, y) in enumerate(points):
            # Assign radius based on Voronoi cell concept but simplified for practicality
            # Larger circles in less dense regions
            radius = min(0.1, max(0.01, 0.05 * (1.0 + np.random.normal(0, 0.1))))

            # Ensure it stays within bounds
            x = np.clip(x, radius, 1 - radius)
            y = np.clip(y, radius, 1 - radius)

            circles[i] = [x, y, radius]

        # Apply local optimization to improve the initial solution
        circles = local_optimize(circles)

        population.append(circles.copy())

    return population

def generate_voronoi_points(n_points: int) -> np.ndarray:
    """Generate points that are well spread using a Voronoi-inspired approach."""
    # Use a combination of grid and perturbation to create well-distributed points
    points = []

    # Start with a grid pattern
    grid_size = int(np.ceil(np.sqrt(n_points)))
    spacing = 1.0 / (grid_size + 1)

    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n_points:
                break

            x = (i + 1) * spacing
            y = (j + 1) * spacing

            # Add strategic perturbations for better spread
            if i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1:
                # Edge points get larger perturbations
                x += np.random.uniform(-spacing/8, spacing/8)
                y += np.random.uniform(-spacing/8, spacing/8)
            elif i == grid_size // 2 and j == grid_size // 2:
                # Center gets moderate perturbation
                x += np.random.uniform(-spacing/16, spacing/16)
                y += np.random.uniform(-spacing/16, spacing/16)
            else:
                # Other points get small perturbations
                x += np.random.uniform(-spacing/24, spacing/24)
                y += np.random.uniform(-spacing/24, spacing/24)

            points.append([x, y])

    if len(points) < n_points:
        # Fill remaining points randomly but within bounds
        for _ in range(n_points - len(points)):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])

    return np.array(points[:n_points])

def mutate_individual(circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to an individual."""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate either position or radius
            if np.random.random() < 0.5:
                # Mutate position with better bounds
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02), 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02), 0.01, 0.99)
            else:
                # Mutate radius with better bounds
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.01), 0.001, 0.2)

    # Fix any invalid placements
    for i in range(len(mutated)):
        if not is_valid_placement(mutated, i):
            # Try to fix by adjusting position and radius
            attempts = 0
            while not is_valid_placement(mutated, i) and attempts < 100:
                mutated[i, 0] = np.random.uniform(0.01, 0.99)
                mutated[i, 1] = np.random.uniform(0.01, 0.99)
                mutated[i, 2] = np.random.uniform(0.001, 0.1)
                attempts += 1

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    # Uniform crossover with better mixing strategy
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover based on fitness difference to prioritize better parents
    for i in range(len(child1)):
        if np.random.random() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def select_parents(population: list, fitnesses: list, tournament_size: int = 3) -> list:
    """Select parents using tournament selection."""
    selected = []

    for _ in range(len(population)):
        # Tournament selection with better probability calculation
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx])

    return selected

def optimize_circles() -> np.ndarray:
    """Main optimization function using evolutionary algorithm."""
    n_circles = 26
    pop_size = 50
    generations = 100

    # Create initial population with better starting points
    population = create_initial_population(pop_size, n_circles)

    best_fitness = 0
    best_individual = None

    for generation in range(generations):
        # Adaptive mutation rate with exponential decay
        mutation_rate = 0.1 * np.exp(-generation / 75.0) + 0.01

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

            # Mutation
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)

            # Apply local optimization to refined solutions
            child1 = local_optimize(child1)
            child2 = local_optimize(child2)

            # Ensure children meet constraints
            if is_valid_placement(child1, len(child1)-1):  # Check last circle
                new_population.append(child1)
            if len(new_population) < pop_size and is_valid_placement(child2, len(child2)-1):
                new_population.append(child2)

        # Trim to population size
        population = new_population[:pop_size]

    # Final local optimization on the best solution
    if best_individual is not None:
        best_individual = local_optimize(best_individual, max_iterations=50)

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
        radius = spacing_x / 3.0

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                # Slightly randomize to avoid perfect grid issues
                x += np.random.uniform(-spacing_x/10, spacing_x/10)
                y += np.random.uniform(-spacing_y/10, spacing_y/10)
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break

        # Ensure constraints are satisfied
        for i in range(count):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])

        return circles


# EVOLVE-BLOCK-END