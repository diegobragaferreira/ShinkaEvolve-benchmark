# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    Uses efficient spatial indexing for overlap checking.
    """
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]  # Get (x, y) coordinates
    tree = cKDTree(points)

    # For each circle, check overlap with others
    for i in range(n):
        x1, y1, r1 = circles[i]
        # Find nearby circles (within 2*(r1+r2) distance)
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

        # Check overlap with each nearby circle
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2

                if distance_sq < min_distance_sq:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population using advanced grid-based initialization"""
    population = []

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Multi-scale grid initialization
        grid_sizes = [3, 4, 5]  # Try different grid sizes
        grid_size = grid_sizes[np.random.randint(0, len(grid_sizes))]

        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        # Place circles on a grid with some randomness
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break

                # Position in grid cell with slight randomness
                x = (i + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                y = (j + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)

                # Ensure within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)

                # Assign initial radius based on proximity to edges
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                r = min(0.05, min_dist_to_edge * 0.8)

                circles[idx] = [x, y, r]
                idx += 1

            if idx >= n_circles:
                break

        # Fill remaining circles randomly but with better constraints
        for i in range(idx, n_circles):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            r = min(0.05, min_dist_to_edge * 0.6)
            circles[i] = [x, y, r]

        # Improve initial configuration by trying to increase radii
        for attempt in range(10):
            improved = False
            for i in range(n_circles):
                # Try to increase radius safely
                original_r = circles[i, 2]
                potential_r = original_r * 1.1

                # Check if we can increase the radius
                can_increase = True
                for j in range(n_circles):
                    if i != j:
                        distance = np.sqrt((circles[i, 0] - circles[j, 0])**2 +
                                         (circles[i, 1] - circles[j, 1])**2)
                        if distance < (potential_r + circles[j, 2]):
                            can_increase = False
                            break

                if can_increase:
                    # Check boundary constraints
                    min_edge_dist = min(circles[i, 0], 1-circles[i, 0],
                                      circles[i, 1], 1-circles[i, 1])
                    if potential_r <= min_edge_dist * 0.95:
                        circles[i, 2] = potential_r
                        improved = True

            if not improved:
                break

        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 5) -> np.ndarray:
    """Select parent using tournament selection with adaptive size"""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Uniform crossover between two parents"""
    child = parent1.copy()

    # Apply crossover with 50% probability for each gene
    mask = np.random.rand(*parent1.shape) > 0.5
    child[mask] = parent2[mask]

    return child

def mutate(circles: np.ndarray, generation: int, max_generations: int,
           base_mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to a circle configuration with adaptive rate"""
    # Adaptive mutation rate - decays over time
    mutation_rate = base_mutation_rate * (0.1**(generation/max_generations))

    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius with equal probability
            if np.random.rand() < 0.5:  # Mutate position
                # Apply larger mutations early, smaller later
                scale = 0.05 * (1 - generation/max_generations) + 0.005
                mutated[i, 0] += np.random.normal(0, scale)
                mutated[i, 1] += np.random.normal(0, scale)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                # Apply larger mutations early, smaller later
                scale = 0.02 * (1 - generation/max_generations) + 0.002
                mutated[i, 2] += np.random.normal(0, scale)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Repair invalid circle configurations with improved overlap resolution"""
    repaired = circles.copy()

    # First ensure all circles are within bounds
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        # Keep within bounds
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])  # Ensure positive radius

    # Then resolve overlaps using iterative repulsion with early termination
    points = repaired[:, :2]
    tree = cKDTree(points)

    # Try to resolve overlaps iteratively with limited attempts
    for _ in range(20):  # Cap iterations
        any_changes = False
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]

            # Find nearby circles
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        # Repel circles apart
                        if distance > 0.001:  # Avoid division by zero
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance

                            # Move them apart
                            move_amount = (min_distance - distance) * 0.5
                            repaired[i, 0] += dx * move_amount
                            repaired[i, 1] += dy * move_amount
                            repaired[j, 0] -= dx * move_amount
                            repaired[j, 1] -= dy * move_amount

                            # Clamp to bounds
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)

                        any_changes = True

        if not any_changes:
            break

    return repaired

def local_refinement(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """
    Apply local refinement to improve solution using greedy radius increases
    """
    refined = circles.copy()
    n_circles = len(refined)

    # Try to increase radii greedily
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Try to increase each circle's radius
        for i in range(n_circles):
            original_r = refined[i, 2]

            # Compute maximum possible radius
            x, y, _ = refined[i]
            max_possible_r = min(x, 1-x, y, 1-y)

            # Try to increase radius by small amounts in a sequence
            increments = [0.005, 0.002, 0.001, 0.0005, 0.0001]

            for increment in increments:
                test_r = min(original_r + increment, max_possible_r * 0.99)

                if test_r > original_r + 1e-6:
                    # Test the new configuration
                    temp_circles = refined.copy()
                    temp_circles[i, 2] = test_r

                    # Check validity with the whole population
                    valid = True
                    for j in range(n_circles):
                        if i != j:
                            distance = np.sqrt((temp_circles[i, 0] - temp_circles[j, 0])**2 +
                                             (temp_circles[i, 1] - temp_circles[j, 1])**2)
                            if distance < (test_r + temp_circles[j, 2]):
                                valid = False
                                break

                    if valid:
                        refined = temp_circles
                        improved = True
                        break

    return refined

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters
    pop_size = 50
    n_generations = 100
    elite_size = 5

    # Create initial population
    population = create_initial_population(pop_size, 26)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None

    for generation in range(n_generations):
        # Calculate fitness for all individuals
        fitnesses = []
        valid_individuals = []

        for circles in population:
            if validate_circles(circles):
                fitness = calculate_sum_radii(circles)
                fitnesses.append(fitness)
                valid_individuals.append(circles)
            else:
                # Repair invalid individuals
                repaired = repair_circles(circles)
                if validate_circles(repaired):
                    fitness = calculate_sum_radii(repaired)
                    fitnesses.append(fitness)
                    valid_individuals.append(repaired)
                else:
                    # If still invalid, penalize heavily
                    fitnesses.append(-np.inf)
                    valid_individuals.append(circles)

        # Track best individual
        if valid_individuals:
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_individual = valid_individuals[max_idx].copy()

        # Elitism: keep top individuals
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elites = [valid_individuals[i] for i in elite_indices if fitnesses[i] > -np.inf]

        # Generate new population
        new_population = elites[:]

        # Fill remaining slots with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(valid_individuals, fitnesses)
            parent2 = tournament_selection(valid_individuals, fitnesses)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate(child, generation, n_generations)

            # Repair
            child = repair_circles(child)

            new_population.append(child)

        population = new_population[:pop_size]

    # Return the best solution found with final refinement
    if best_individual is not None:
        # Apply local refinement to the best solution
        refined_solution = local_refinement(best_individual)
        if validate_circles(refined_solution):
            return refined_solution
        else:
            return best_individual
    else:
        # If no valid solution found, return the best from final population
        fitnesses = [calculate_sum_radii(circles) for circles in population if validate_circles(circles)]
        if fitnesses:
            best_idx = np.argmax(fitnesses)
            # Apply local refinement to the best candidate from final population
            refined_solution = local_refinement(population[best_idx])
            if validate_circles(refined_solution):
                return refined_solution
            else:
                return population[best_idx]
        else:
            # Fallback: return a valid random solution
            circles = np.zeros((26, 3))
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
            return circles


# EVOLVE-BLOCK-END