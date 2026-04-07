# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from copy import deepcopy

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - perimeter = 4, so width + height = 2
    # Using optimized rectangle dimensions from successful implementations
    rect_width = 1.3
    rect_height = 0.7

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Parameters
    n_circles = 21
    max_iterations = 500
    population_size = 50
    elite_size = 5
    mutation_rate = 0.1

    # Initialize population with adaptive grid layouts
    def create_adaptive_grid(n, width, height):
        """
        Create an adaptive grid layout based on circle count and rectangle dimensions
        """
        # Calculate optimal grid dimensions based on circle count and rectangle aspect ratio
        # Use the area-based approach from successful implementations
        spacing = np.sqrt((width * height) / n) * 0.8  # Adjust spacing factor

        # Compute grid dimensions
        cols = max(1, int(np.ceil(width / spacing)))
        rows = max(1, int(np.ceil(height / spacing)))

        # Adjust grid to fit exactly n circles
        if cols * rows < n:
            cols = max(1, int(np.ceil(np.sqrt(n * width / height))))
            rows = max(1, int(np.ceil(n / cols)))

        # Ensure we don't go over the rectangle bounds
        cell_width = width / cols
        cell_height = height / rows

        # Use minimum of cell dimensions for radius with safety margin
        max_radius = min(cell_width, cell_height) * 0.45

        # Generate grid layout with slight randomization to avoid symmetry issues
        circles = []
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Add slight random offset to break symmetry and improve packing
                x = (j + 0.5 + np.random.uniform(-0.15, 0.15)) * cell_width
                y = (i + 0.5 + np.random.uniform(-0.15, 0.15)) * cell_height
                # Ensure positions stay within bounds
                x = np.clip(x, max_radius, width - max_radius)
                y = np.clip(y, max_radius, height - max_radius)
                circles.append([x, y, max_radius])
                idx += 1

        return np.array(circles)

    # Initialize population with adaptive grid layouts
    def create_initial_population(size):
        population = []
        # Try different grid arrangements to find good starting points
        for _ in range(size):
            # Use adaptive grid initialization
            circles = create_adaptive_grid(n_circles, rect_width, rect_height)

            # Add slight randomization to avoid local optima
            for i in range(len(circles)):
                circles[i, 0] += np.random.uniform(-0.02, 0.02)
                circles[i, 1] += np.random.uniform(-0.02, 0.02)
                circles[i, 2] *= np.random.uniform(0.95, 1.05)

            population.append(circles)

        return population

    # Check if circles collide with each other or boundaries
    def is_valid_layout(circles):
        # Check boundary constraints
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False

        # Check pairwise collisions
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]

                # Distance between centers
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)

                # Circles should not overlap
                if dist < r1 + r2:
                    return False

        return True

    # Calculate fitness (sum of radii) for valid layouts only
    def evaluate_fitness(circles):
        if not is_valid_layout(circles):
            return 0.0  # Invalid layouts get zero fitness

        return np.sum(circles[:, 2])  # Sum of radii

    # Create initial population
    population = create_initial_population(population_size)

    # If we don't have enough initial individuals, fill with random valid ones
    while len(population) < population_size:
        circles = np.zeros((n_circles, 3))
        # Try generating random valid circles
        attempts = 0
        while attempts < 1000:
            # Random positions and radii
            for i in range(n_circles):
                x = np.random.uniform(0.05, rect_width - 0.05)
                y = np.random.uniform(0.05, rect_height - 0.05)
                r = np.random.uniform(0.01, 0.2)
                circles[i] = [x, y, r]

            if is_valid_layout(circles):
                population.append(circles.copy())
                break
            attempts += 1

    # Genetic algorithm loop
    best_fitness = 0.0
    best_individual = None

    for generation in range(max_iterations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            score = evaluate_fitness(individual)
            fitness_scores.append(score)

            if score > best_fitness:
                best_fitness = score
                best_individual = individual.copy()

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Keep elite
        new_population = population[:elite_size]

        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = tournament_selection(population, fitness_scores, 3)
            parent2_idx = tournament_selection(population, fitness_scores, 3)

            # Crossover
            child = crossover(population[parent1_idx], population[parent2_idx])

            # Mutation
            mutate(child, mutation_rate, rect_width, rect_height)

            new_population.append(child)

        population = new_population

    # Return best solution found
    if best_individual is not None:
        return np.array(best_individual)
    else:
        # Fallback to last generation if nothing was found
        return population[0]

def tournament_selection(population, fitness_scores, k):
    """Select individual via tournament selection"""
    tournament_indices = np.random.choice(len(population), k)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return winner_index

def crossover(parent1, parent2):
    """Single point crossover on circle positions and radii"""
    child = deepcopy(parent1)

    # Select crossover point
    crossover_point = np.random.randint(1, len(parent1))

    # Cross over positions and radii
    child[crossover_point:, :2] = parent2[crossover_point:, :2]  # Positions
    child[crossover_point:, 2] = parent2[crossover_point:, 2]   # Radii

    return child

def mutate(individual, mutation_rate, rect_width, rect_height):
    """Mutate circle positions and radii"""
    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # Mutate position
            individual[i, 0] = np.clip(
                individual[i, 0] + np.random.normal(0, 0.05),
                0.05, rect_width - 0.05)
            individual[i, 1] = np.clip(
                individual[i, 1] + np.random.normal(0, 0.05),
                0.05, rect_height - 0.05)

            # Mutate radius
            individual[i, 2] = np.clip(
                individual[i, 2] + np.random.normal(0, 0.02),
                0.01, 0.3)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")