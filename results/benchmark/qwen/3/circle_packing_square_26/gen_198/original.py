# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.8
BOUNDARY_PENALTY_WEIGHT = 10000.0
OVERLAP_PENALTY_WEIGHT = 10000.0
GRID_SIZE = 25  # Grid cells per dimension for spatial indexing

def create_grid(circles: np.ndarray) -> dict:
    """Create a spatial grid for efficient overlap checking."""
    grid = {}

    # Precompute grid boundaries for all circles to avoid repeated calculations
    for i, (x, y, r) in enumerate(circles):
        # Determine which grid cells this circle touches
        min_x_cell = max(0, int((x - r) * GRID_SIZE))
        max_x_cell = min(GRID_SIZE - 1, int((x + r) * GRID_SIZE))
        min_y_cell = max(0, int((y - r) * GRID_SIZE))
        max_y_cell = min(GRID_SIZE - 1, int((y + r) * GRID_SIZE))

        # Add circle to all relevant grid cells
        for gx in range(min_x_cell, max_x_cell + 1):
            for gy in range(min_y_cell, max_y_cell + 1):
                grid.setdefault((gx, gy), []).append(i)

    return grid

def check_overlap_with_grid(circles: np.ndarray, grid: dict) -> bool:
    """Check overlaps using spatial grid for improved efficiency."""
    # For each cell in the grid, check if any pairs of circles overlap
    for cell, circle_indices in grid.items():
        # Only check pairs within the same grid cell
        for i in range(len(circle_indices)):
            idx1 = circle_indices[i]
            x1, y1, r1 = circles[idx1]

            for j in range(i + 1, len(circle_indices)):
                idx2 = circle_indices[j]
                x2, y2, r2 = circles[idx2]

                # Calculate distance between circle centers
                dx = x1 - x2
                dy = y1 - y2
                distance_squared = dx*dx + dy*dy

                # Check if circles overlap
                if distance_squared < (r1 + r2)**2:
                    return False

    return True

def is_valid(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping."""
    n = len(circles)

    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints using spatial grid
    grid = create_grid(circles)
    return check_overlap_with_grid(circles, grid)

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a solution, higher is better."""
    if not is_valid(circles):
        # Apply penalty for constraint violations
        penalty = 0

        # Boundary penalty
        boundary_violations = 0
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0:
                boundary_violations += (r - x)**2
            if x + r > 1:
                boundary_violations += (x + r - 1)**2
            if y - r < 0:
                boundary_violations += (r - y)**2
            if y + r > 1:
                boundary_violations += (y + r - 1)**2

        penalty += BOUNDARY_PENALTY_WEIGHT * boundary_violations

        # Overlap penalty
        overlap_penalty = 0
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    overlap_penalty += (r1 + r2 - distance)**2

        penalty += OVERLAP_PENALTY_WEIGHT * overlap_penalty

        return -penalty

    return calculate_sum_radii(circles)

def initialize_population(size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with valid configurations."""
    population = []

    # Generate better initial distribution using a more structured approach
    def generate_initial_distribution(n_points: int) -> np.ndarray:
        # Use a grid-based approach combined with some randomness
        points = []
        grid_size = max(3, int(np.ceil(np.sqrt(n_points))))

        # Create grid points with some jitter
        spacing = 1.0 / (grid_size + 1)
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n_points:
                    break
                x = (j + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                y = (i + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                # Ensure points stay within bounds
                x = np.clip(x, spacing, 1 - spacing)
                y = np.clip(y, spacing, 1 - spacing)
                points.append([x, y])

        # Add extra points randomly to fill gaps
        while len(points) < n_points:
            x = np.random.random()
            y = np.random.random()
            points.append([x, y])

        return np.array(points[:n_points])

    for _ in range(size):
        # Generate initial points
        points = generate_initial_distribution(n_circles)

        # Create circles with smarter radius assignment
        circles = np.zeros((n_circles, 3))

        for i in range(n_circles):
            # Calculate minimum distance to all other points
            distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
            distances[i] = np.inf  # Exclude self-distance
            min_distance = np.min(distances)

            # Calculate maximum allowable radius based on containment
            max_allowable_radius = min(points[i][0], points[i][1],
                                     1 - points[i][0], 1 - points[i][1])

            # Use a more conservative approach for better initial configurations
            if min_distance > 0:
                # Radius should be about 1/4 of minimum neighbor distance, bounded by containment
                proposed_radius = min(min_distance / 4.0, max_allowable_radius * 0.6)
            else:
                proposed_radius = max_allowable_radius * 0.4

            # Clamp radius to reasonable bounds
            radius = max(0.001, min(proposed_radius, 0.4))

            circles[i] = [points[i][0], points[i][1], radius]

        # Validate and refine
        if not is_valid(circles):
            # Try to make it valid with minimal changes
            for i in range(n_circles):
                x, y, r = circles[i]
                # Constrain to valid region
                x = np.clip(x, r + 0.001, 1 - r - 0.001)
                y = np.clip(y, r + 0.001, 1 - r - 0.001)
                circles[i] = [x, y, r]

        population.append(circles)

    return population

def optimize_initial_config(circles: np.ndarray):
    """Simple local optimization to improve initial configuration."""
    # This is a placeholder for more sophisticated optimization
    # In a real implementation, this would involve some local search
    pass

def tournament_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Single point crossover on genes (x, y, r) for each circle
    crossover_point = random.randint(1, n - 1)

    # Swap genes after the crossover point
    for i in range(crossover_point, n):
        child1[i] = parent2[i].copy()
        child2[i] = parent1[i].copy()

    return child1, child2

def mutate(circles: np.ndarray, generation: int = 0, max_generations: int = 1000,
           diversity_factor: float = 1.0) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate and diversity-aware strategy."""
    # Adaptive mutation rate that decreases over generations
    mutation_rate_start = 0.25
    mutation_rate_end = 0.02
    generation_progress = generation / max_generations

    # Sigmoidal decay with custom parameters for better control
    mutation_rate = mutation_rate_end + (mutation_rate_start - mutation_rate_end) * (
        1 / (1 + np.exp(12 * (generation_progress - 0.5)))
    )

    # Adjust based on population diversity
    mutation_rate *= diversity_factor

    mutated = circles.copy()
    n = len(mutated)

    for i in range(n):
        if random.random() < mutation_rate:
            # Determine what kind of mutation to apply based on progress
            if generation < max_generations * 0.3:
                # Early generations: explore with larger mutations
                strength = 0.06
                # Prioritize position changes
                choice = random.choices([0, 1, 2], weights=[0.6, 0.6, 0.1])[0]
            elif generation < max_generations * 0.7:
                # Middle generations: balanced mutations
                strength = 0.03
                choice = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.4])[0]
            else:
                # Late generations: exploit with small, precise mutations
                strength = 0.015
                # Prioritize radius changes for fine-tuning
                choice = random.choices([0, 1, 2], weights=[0.2, 0.2, 0.8])[0]

            if choice == 0:  # Mutate x coordinate
                mutated[i, 0] = max(0.001, min(0.999, mutated[i, 0] + random.gauss(0, strength)))
            elif choice == 1:  # Mutate y coordinate
                mutated[i, 1] = max(0.001, min(0.999, mutated[i, 1] + random.gauss(0, strength)))
            else:  # Mutate radius with log-normal distribution to avoid extreme values
                log_factor = random.gauss(0, 0.15)
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] * np.exp(log_factor)))

    return mutated

def get_best_individual(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Get the individual with highest fitness."""
    best_idx = fitnesses.index(max(fitnesses))
    return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 26
    population = initialize_population(POPULATION_SIZE, n)

    best_fitness_history = []

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")

        # Create new population
        new_population = []

        # Elitism: keep best individual
        best_individual = get_best_individual(population, fitnesses)
        new_population.append(best_individual)

        # Calculate diversity for adaptive mutation rate
        if len(population) > 1:
            # Simple diversity calculation: average distance between individuals
            total_dist = 0
            count = 0
            for i in range(len(population)):
                for j in range(i+1, len(population)):
                    dist = np.mean(np.abs(population[i] - population[j]))
                    total_dist += dist
                    count += 1
            diversity = total_dist / count if count > 0 else 1.0
            diversity_factor = max(0.5, 1.0 - diversity * 5.0)  # Lower diversity = higher mutation
        else:
            diversity_factor = 1.0

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, generation, NUM_GENERATIONS, diversity_factor)
            child2 = mutate(child2, generation, NUM_GENERATIONS, diversity_factor)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]

    # Get final best solution
    final_fitnesses = [evaluate_fitness(individual) for individual in population]
    best_solution = get_best_individual(population, final_fitnesses)

    # Ensure the final solution is valid, if not try to fix it
    if not is_valid(best_solution):
        # Apply a more sophisticated repair approach
        for i in range(len(best_solution)):
            x, y, r = best_solution[i]
            # Ensure it stays within bounds
            if x - r < 0:
                r = x
            if x + r > 1:
                r = 1 - x
            if y - r < 0:
                r = y
            if y + r > 1:
                r = 1 - y
            # Apply new radius
            best_solution[i, 2] = max(0.001, r - 0.005)

    # Return the best solution found
    return best_solution


# EVOLVE-BLOCK-END