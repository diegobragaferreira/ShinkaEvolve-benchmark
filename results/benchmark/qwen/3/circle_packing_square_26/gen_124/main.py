# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.15
MUTATION_RATE_END = 0.01
CROSSOVER_RATE = 0.8
BOUNDARY_PENALTY_WEIGHT = 10000.0
OVERLAP_PENALTY_WEIGHT = 100000.0
# Progressive constraint relaxation parameters
RELAXATION_START_GEN = 50
RELAXATION_END_GEN = 200
INITIAL_PENALTY_WEIGHTS = [1.0, 1.0]  # [boundary, overlap]
FINAL_PENALTY_WEIGHTS = [6.0, 6.0]

def is_valid(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping using spatial indexing."""
    n = len(circles)

    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Use spatial indexing for efficient overlap checking
    if n > 1:
        try:
            positions = [(x, y) for x, y, r in circles]
            tree = cKDTree(positions)

            # Query pairs within sum of radii distance
            pairs = tree.query_pairs(r=2.0, output_type='ndarray')

            # Check each potential overlapping pair
            for i, j in pairs:
                if i < j:  # Avoid duplicate checking
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        return False
        except Exception:
            # Fallback to brute force if tree fails
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def evaluate_fitness(circles: np.ndarray, generation: int = 0) -> float:
    """Evaluate fitness of a solution, higher is better."""
    if not is_valid(circles):
        # Apply penalty for constraint violations
        penalty = 0

        # Dynamic penalty weighting based on generation
        if generation < RELAXATION_START_GEN:
            # Early generations: lighter penalties for exploration
            boundary_weight = INITIAL_PENALTY_WEIGHTS[0]
            overlap_weight = INITIAL_PENALTY_WEIGHTS[1]
        elif generation < RELAXATION_END_GEN:
            # Middle generations: gradual transition
            progress = (generation - RELAXATION_START_GEN) / (RELAXATION_END_GEN - RELAXATION_START_GEN)
            boundary_weight = INITIAL_PENALTY_WEIGHTS[0] + progress * (FINAL_PENALTY_WEIGHTS[0] - INITIAL_PENALTY_WEIGHTS[0])
            overlap_weight = INITIAL_PENALTY_WEIGHTS[1] + progress * (FINAL_PENALTY_WEIGHTS[1] - INITIAL_PENALTY_WEIGHTS[1])
        else:
            # Later generations: strict penalties for convergence
            boundary_weight = FINAL_PENALTY_WEIGHTS[0]
            overlap_weight = FINAL_PENALTY_WEIGHTS[1]

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

        penalty += boundary_weight * boundary_violations

        # Overlap penalty - compute based on actual overlap amounts
        overlap_penalty = 0
        n = len(circles)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    overlap = (r1 + r2 - distance)
                    overlap_penalty += overlap**2

        penalty += overlap_weight * overlap_penalty

        return -penalty

    return calculate_sum_radii(circles)

def initialize_population(size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with improved Voronoi-inspired grid placement."""
    population = []

    # Use hexagonal grid pattern for better spatial distribution
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))

    # Grid spacing calculation
    spacing_x = 0.9 / cols  # Leave margin for boundary
    spacing_y = 0.9 / rows

    for _ in range(size):
        circles = np.zeros((n_circles, 3))

        # Generate grid points with hexagonal offset
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break

                # Hexagonal offset for even rows
                x_offset = (j + 0.5 * (i % 2)) * spacing_x
                y_offset = i * spacing_y * 0.866  # sqrt(3)/2 for hexagon height

                # Add some randomness to avoid perfect grid
                x = 0.05 + min(0.95, x_offset + random.uniform(-spacing_x*0.1, spacing_x*0.1))
                y = 0.05 + min(0.95, y_offset + random.uniform(-spacing_y*0.1, spacing_y*0.1))

                # Set initial radius with variation
                r = 0.01 + random.uniform(0.01, 0.04)  # Variable radius

                circles[idx] = [x, y, r]
                idx += 1

        # Ensure all circles fit within bounds
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Adjust radius to fit within bounds
            max_radius = min(x, 1-x, y, 1-y)
            circles[i, 2] = min(r, max_radius)

        # Apply small local optimization to improve initial placement
        optimize_initial_config(circles)
        population.append(circles)

    return population

def optimize_initial_config(circles: np.ndarray):
    """Simple local optimization to improve initial configuration."""
    # This could be expanded with more sophisticated methods
    # For now, it just enforces bounds and resolves trivial overlaps
    n = len(circles)

    # Enforce boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        # Adjust for boundaries
        if x - r < 0:
            x = r
        if x + r > 1:
            x = 1 - r
        if y - r < 0:
            y = r
        if y + r > 1:
            y = 1 - r

        circles[i] = [x, y, r]

    # Basic overlap resolution
    if n > 1:
        try:
            positions = [(x, y) for x, y, r in circles]
            tree = cKDTree(positions)
            pairs = tree.query_pairs(r=0.01, output_type='ndarray')  # Small threshold

            for i, j in pairs:
                if i < j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    if distance < r1 + r2:
                        # Push circles apart
                        if distance > 0.001:
                            dx = (x2 - x1) / distance
                            dy = (y2 - y1) / distance
                            move_dist = (r1 + r2 - distance) * 0.5
                            circles[i, 0] -= dx * move_dist * 0.3
                            circles[i, 1] -= dy * move_dist * 0.3
                            circles[j, 0] += dx * move_dist * 0.3
                            circles[j, 1] += dy * move_dist * 0.3
        except Exception:
            pass

def tournament_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Select an individual using tournament selection."""
    # Adapt tournament size based on population diversity
    tournament_size = max(3, TOURNAMENT_SIZE - int(len(population) * 0.01))
    tournament_indices = random.sample(range(len(population)), tournament_size)
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

    # Multi-point crossover for better exploration
    crossover_points = sorted(random.sample(range(1, n), min(5, n//3)))
    crossover_points = [0] + crossover_points + [n]

    # Alternate segments between parents
    for i in range(len(crossover_points) - 1):
        start = crossover_points[i]
        end = crossover_points[i + 1]
        if i % 2 == 0:
            child1[start:end] = parent2[start:end].copy()
            child2[start:end] = parent1[start:end].copy()
        else:
            child1[start:end] = parent1[start:end].copy()
            child2[start:end] = parent2[start:end].copy()

    return child1, child2

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive rate."""
    mutated = circles.copy()
    n = len(mutated)

    # Adaptive mutation rate that decreases over generations
    mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * (
        1 / (1 + np.exp(10 * (generation / total_generations - 0.5)))
    )

    # Mutate each circle with adaptive probability
    for i in range(n):
        if random.random() < mutation_rate:
            # Choose which component to mutate
            component = random.randint(0, 2)

            if component == 0:  # x coordinate
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, 0.02)))
            elif component == 1:  # y coordinate
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, 0.02)))
            else:  # radius
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.01)))

    # Apply post-mutation refinement to fix constraint violations
    mutated = refine_circles(mutated)
    return mutated

def refine_circles(circles: np.ndarray) -> np.ndarray:
    """Refine circles to ensure valid constraints."""
    refined = circles.copy()

    # First enforce boundary constraints
    for i in range(len(refined)):
        x, y, r = refined[i]
        # Ensure circle fits in unit square
        max_radius = min(x, 1-x, y, 1-y)
        refined[i, 2] = min(r, max_radius)
        # Clamp coordinates
        refined[i, 0] = max(refined[i, 2], min(1-refined[i, 2], x))
        refined[i, 1] = max(refined[i, 2], min(1-refined[i, 2], y))

    # Resolve overlaps through iterative adjustment
    n = len(refined)
    max_iterations = 10
    for _ in range(max_iterations):
        any_changed = False
        positions = [(x, y) for x, y, r in refined]
        if len(positions) > 1:
            try:
                tree = cKDTree(positions)
                pairs = tree.query_pairs(r=0.01, output_type='ndarray')

                for i, j in pairs:
                    if i < j:
                        x1, y1, r1 = refined[i]
                        x2, y2, r2 = refined[j]
                        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                        if distance < r1 + r2:
                            # Adjust positions to separate circles
                            if distance > 0.001:
                                dx = (x2 - x1) / distance
                                dy = (y2 - y1) / distance
                                move_dist = (r1 + r2 - distance) * 0.5

                                # Apply adjustment with small damping
                                refined[i, 0] -= dx * move_dist * 0.2
                                refined[i, 1] -= dy * move_dist * 0.2
                                refined[j, 0] += dx * move_dist * 0.2
                                refined[j, 1] += dy * move_dist * 0.2
                                any_changed = True
            except Exception:
                pass

        if not any_changed:
            break

    return refined

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
        fitnesses = [evaluate_fitness(individual, generation) for individual in population]

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

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, generation, NUM_GENERATIONS)
            child2 = mutate(child2, generation, NUM_GENERATIONS)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]

    # Get final best solution
    final_fitnesses = [evaluate_fitness(individual, NUM_GENERATIONS) for individual in population]
    best_solution = get_best_individual(population, final_fitnesses)

    # Final validation and repair if needed
    if not is_valid(best_solution):
        # Apply final refinement
        best_solution = refine_circles(best_solution)

    # Ensure everything is within bounds
    for i in range(len(best_solution)):
        x, y, r = best_solution[i]
        # Ensure it stays within bounds
        best_solution[i, 0] = max(r, min(1-r, x))
        best_solution[i, 1] = max(r, min(1-r, y))
        best_solution[i, 2] = max(0.001, min(0.49, r))

    # Return the best solution found
    return best_solution


# EVOLVE-BLOCK-END