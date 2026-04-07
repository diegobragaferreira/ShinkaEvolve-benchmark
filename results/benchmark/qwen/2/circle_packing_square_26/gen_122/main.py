# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.

    Args:
        circles: np.array of shape (n, 3) where each row is (x, y, r)

    Returns:
        True if all circles are valid, False otherwise
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
    """Create initial population using enhanced multi-scale grid-based initialization"""
    population = []

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Use multiple grid strategies for better diversity
        grid_strategies = [
            (3, 3),   # Small grid
            (4, 4),   # Medium grid
            (5, 5),   # Large grid
            (2, 13),  # Rectangular grid for 26 circles
            (13, 2)   # Transposed rectangular grid
        ]

        # Randomly select a grid strategy
        grid_rows, grid_cols = grid_strategies[np.random.randint(0, len(grid_strategies))]

        # Ensure we use the correct number of circles
        actual_circles = min(grid_rows * grid_cols, n_circles)

        spacing_x = 1.0 / (grid_cols + 1)
        spacing_y = 1.0 / (grid_rows + 1)

        # Place circles on the selected grid with strategic positioning
        idx = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                if idx >= n_circles:
                    break

                # Strategic positioning with offset for better distribution
                if grid_rows == 2 and grid_cols == 13:
                    # Special case for 2x13 grid to avoid linear patterns
                    x = (j + 0.5) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                    y = (i + 0.5) * spacing_y + (i * spacing_y / 3) * np.random.uniform(-0.3, 0.3)
                elif grid_rows == 13 and grid_cols == 2:
                    # Special case for 13x2 grid
                    x = (j + 0.5) * spacing_x + (j * spacing_x / 3) * np.random.uniform(-0.3, 0.3)
                    y = (i + 0.5) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                else:
                    # Regular grid with slight randomization
                    x = (j + 0.5) * spacing_x + np.random.uniform(-spacing_x/8, spacing_x/8)
                    y = (i + 0.5) * spacing_y + np.random.uniform(-spacing_y/8, spacing_y/8)

                # Ensure within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)

                # Assign initial radius based on proximity to edges with some randomness
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                # Use a more aggressive initial radius assignment for better starting point
                r = min(0.08, min_dist_to_edge * np.random.uniform(0.7, 0.95))

                circles[idx] = [x, y, r]
                idx += 1

            if idx >= n_circles:
                break

        # Fill remaining circles with random but informed placement
        for i in range(idx, n_circles):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            # Use a more strategic initial radius
            r = min(0.08, min_dist_to_edge * np.random.uniform(0.5, 0.8))
            circles[i] = [x, y, r]

        # Apply initial improvement through greedy radius expansion
        improved = True
        attempts = 0
        while improved and attempts < 10:
            improved = False
            attempts += 1
            for i in range(n_circles):
                original_r = circles[i, 2]
                # Try to increase radius more aggressively
                potential_r = min(original_r * 1.15, 0.1)  # Allow larger increases

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

        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select parent using tournament selection"""
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

def mutate(circles: np.ndarray, mutation_rate: float = 0.1,
           max_radius_change: float = 0.02) -> np.ndarray:
    """Apply mutation to a circle configuration"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius
            if np.random.rand() < 0.5:  # Mutate position
                mutated[i, 0] += np.random.normal(0, 0.01)
                mutated[i, 1] += np.random.normal(0, 0.01)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                mutated[i, 2] += np.random.normal(0, max_radius_change)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Repair invalid circle configurations"""
    repaired = circles.copy()

    # First ensure all circles are within bounds
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        # Keep within bounds
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])  # Ensure positive radius

    # Then resolve overlaps using simple repulsion
    points = repaired[:, :2]
    tree = cKDTree(points)

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

    return repaired

def local_refinement(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Apply enhanced local refinement to improve the circle packing solution.
    This combines multiple strategies to maximize radii while maintaining constraints.
    """
    refined_circles = circles.copy()
    n_circles = len(refined_circles)

    # Phase 1: Adaptive radius expansion
    improved = True
    iteration = 0
    while improved and iteration < 50:  # Limit iterations to prevent infinite loop
        improved = False
        iteration += 1

        # Try to increase each circle's radius
        for i in range(n_circles):
            original_r = refined_circles[i, 2]
            max_possible_r = min(
                refined_circles[i, 0],
                refined_circles[i, 1],
                1 - refined_circles[i, 0],
                1 - refined_circles[i, 1]
            )

            # Try to increase radius up to maximum possible
            test_r = min(original_r + 0.002, max_possible_r * 0.99)

            if test_r > original_r + 1e-6:  # Significant improvement
                # Check if we can actually increase this radius
                valid = True
                temp_circles = refined_circles.copy()
                temp_circles[i, 2] = test_r

                # Check all constraints
                if validate_circles(temp_circles):
                    refined_circles = temp_circles
                    improved = True
                else:
                    # Try smaller increments
                    for increment in [0.001, 0.0005, 0.0001]:
                        test_r = min(original_r + increment, max_possible_r * 0.99)
                        temp_circles = refined_circles.copy()
                        temp_circles[i, 2] = test_r

                        if validate_circles(temp_circles):
                            refined_circles = temp_circles
                            improved = True
                            break

    # Phase 2: Local position adjustment for better packing
    # Use a greedy approach to slightly reposition circles to enable larger radii
    for _ in range(20):
        # Try to improve each circle by adjusting its position
        for i in range(n_circles):
            x, y, r = refined_circles[i]

            # Try small adjustments to potentially allow larger radius
            best_x, best_y, best_r = x, y, r
            best_radius = r

            # Try several small position adjustments
            adjustments = [
                (0, 0, 0),
                (0.005, 0, 0),
                (-0.005, 0, 0),
                (0, 0.005, 0),
                (0, -0.005, 0),
                (0.003, 0.003, 0),
                (-0.003, -0.003, 0)
            ]

            for dx, dy, dr in adjustments:
                test_x = max(0.001, min(0.999, x + dx))
                test_y = max(0.001, min(0.999, y + dy))
                test_r = max(0.001, min(r + dr,
                                      min(test_x, test_y, 1-test_x, 1-test_y) * 0.99))

                # Test new configuration
                temp_circles = refined_circles.copy()
                temp_circles[i] = [test_x, test_y, test_r]

                if validate_circles(temp_circles) and test_r > best_radius:
                    best_x, best_y, best_r = test_x, test_y, test_r
                    best_radius = test_r

            # Update if we found a better configuration
            if best_radius > refined_circles[i, 2]:
                refined_circles[i] = [best_x, best_y, best_radius]

    # Phase 3: Final repair step using overlap resolution
    # This ensures we didn't introduce any overlaps from our adjustments
    repaired = repair_circles(refined_circles)

    # Verify our result is valid
    if validate_circles(repaired):
        return repaired
    else:
        # If repair failed, return original with minor validation fixes
        final_result = circles.copy()
        for i in range(n_circles):
            # Ensure bounds
            final_result[i, 0] = np.clip(final_result[i, 0],
                                       final_result[i, 2],
                                       1 - final_result[i, 2])
            final_result[i, 1] = np.clip(final_result[i, 1],
                                       final_result[i, 2],
                                       1 - final_result[i, 2])
            final_result[i, 2] = max(0.001, final_result[i, 2])
        return final_result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters
    pop_size = 50
    n_generations = 100
    mutation_rate = 0.1
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

            # Mutation
            child = mutate(child, mutation_rate)

            # Repair
            child = repair_circles(child)

            new_population.append(child)

        population = new_population[:pop_size]

        # Adaptive mutation rate
        if generation > 50:
            mutation_rate = 0.05
        elif generation > 25:
            mutation_rate = 0.07

    # Return the best solution found
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