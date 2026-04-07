# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles: np.ndarray, idx: int) -> bool:
    """Check if circle at index idx is valid (within bounds and not overlapping)."""
    x, y, r = circles[idx]

    # Check containment constraints
    if x < r or x > 1 - r or y < r or y > 1 - r:
        return False

    # Check overlap constraints with existing circles using KDTree for efficiency
    if len(circles) > 1:
        points = circles[:, :2]  # Get (x, y) coordinates
        tree = cKDTree(points)
        # Find nearby circles (within 2*(r_max) distance)
        nearby_indices = tree.query_ball_point([x, y], 2 * (r + 0.001))

        for j in nearby_indices:
            if j == idx:
                continue
            x_j, y_j, r_j = circles[j]
            distance = np.sqrt((x - x_j)**2 + (y - y_j)**2)
            if distance < r + r_j:
                return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def create_initial_population(pop_size: int, n_circles: int) -> list:
    """Create initial population of valid circle arrangements with better initialization."""
    population = []

    for _ in range(pop_size):
        # Create random circles with smart initialization
        circles = np.zeros((n_circles, 3))

        # More intelligent placement strategy
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        # Place circles in grid pattern with jitter
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_circles:
                    break
                # Base grid position
                x_base = (i + 1) * spacing_x
                y_base = (j + 1) * spacing_y

                # Add jitter and ensure valid bounds
                x = np.clip(x_base + np.random.normal(0, spacing_x * 0.2), 0.01, 0.99)
                y = np.clip(y_base + np.random.normal(0, spacing_y * 0.2), 0.01, 0.99)

                # Initial radius based on available space
                max_r = min(x, 1-x, y, 1-y) * 0.8
                r = np.clip(np.random.uniform(0.01, max_r * 0.5), 0.001, 0.2)

                circles[count] = [x, y, r]
                count += 1
            if count >= n_circles:
                break

        # Validate the initial configuration
        valid = True
        for i in range(n_circles):
            if not is_valid_placement(circles, i):
                valid = False
                break

        if valid:
            population.append(circles.copy())
        else:
            # Fallback to simpler initialization
            circles = np.zeros((n_circles, 3))
            for i in range(n_circles):
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                r = np.random.uniform(0.001, 0.1)
                circles[i] = [x, y, r]
            population.append(circles)

    # Ensure we have enough valid individuals
    while len(population) < pop_size:
        # Add random valid individuals
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            r = np.random.uniform(0.001, 0.1)
            circles[i] = [x, y, r]
            # Ensure validity by fixing if needed
            if not is_valid_placement(circles, i):
                # Simple fix by adjusting position
                max_r = min(x, 1-x, y, 1-y)
                if max_r > 0.001:
                    circles[i, 2] = min(r, max_r * 0.9)
                circles[i, 0] = np.clip(x, circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(y, circles[i, 2], 1 - circles[i, 2])
        population.append(circles)

    return population

def mutate_individual(circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to an individual with improved strategies."""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate either position or radius with weighted probability
            mutation_type = np.random.random()

            if mutation_type < 0.6:  # Position mutation (more likely)
                # Mutate position with better bounds
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.03), 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.03), 0.01, 0.99)
            elif mutation_type < 0.9:  # Radius mutation (moderate)
                # Mutate radius with better bounds
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.015), 0.001, 0.2)
            else:  # Combined mutation (small chance)
                # Mutate both position and radius
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02), 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02), 0.01, 0.99)
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.01), 0.001, 0.2)

    # Fix any invalid placements using a more robust approach
    for i in range(len(mutated)):
        if not is_valid_placement(mutated, i):
            # Try to fix by adjusting to nearest valid position
            x, y, r = mutated[i]

            # Try to find a better radius that fits
            max_r = min(x, 1-x, y, 1-y) * 0.95
            if max_r > 0.001:
                mutated[i, 2] = np.clip(r, 0.001, max_r)

            # Adjust position to ensure bounds
            mutated[i, 0] = np.clip(x, mutated[i, 2], 1 - mutated[i, 2])
            mutated[i, 1] = np.clip(y, mutated[i, 2], 1 - mutated[i, 2])

            # If still invalid, use backup method
            if not is_valid_placement(mutated, i):
                mutated[i, 0] = np.random.uniform(mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.random.uniform(mutated[i, 2], 1 - mutated[i, 2])

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents with improved mixing."""
    # Uniform crossover with smarter adaptation
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover with bias towards better-performing parents based on fitness
    # We'll consider a simple version here that still maintains good variety
    for i in range(len(child1)):
        if np.random.random() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def select_parents(population: list, fitnesses: list, tournament_size: int = 3) -> list:
    """Select parents using tournament selection with adaptive size."""
    selected = []

    # Adaptive tournament size based on population diversity
    if len(fitnesses) > 1:
        fitness_std = np.std(fitnesses)
        # Larger tournament size when diversity is low (more selection pressure)
        # Smaller tournament size when diversity is high (more exploration)
        tournament_size = max(2, min(len(population), int(3 + fitness_std * 5)))

    for _ in range(len(population)):
        # Tournament selection with adaptive size
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx])

    return selected

def local_refinement(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Enhanced local refinement to maximize radii while maintaining constraints."""
    refined = circles.copy()

    # Phase 1: Increase radii as much as possible while maintaining constraints
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1

        # Try to maximize each circle's radius
        for i in range(len(refined)):
            x, y, r = refined[i]

            # Compute maximum possible radius at current position
            max_radius = min(x, 1-x, y, 1-y)

            # Try to increase radius up to feasible amount
            new_r = min(r + 0.005, max_radius * 0.99)  # Leave some margin

            # Check if new radius is valid
            valid = True
            for j in range(len(refined)):
                if i != j:
                    x_j, y_j, r_j = refined[j]
                    distance = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                    if distance < new_r + r_j:
                        valid = False
                        break

            if valid and new_r > r + 1e-6:
                refined[i, 2] = new_r
                improved = True

    # Phase 2: Fine-tune positions to potentially enable even larger radii
    # This is more complex but can help improve results further
    for _ in range(10):
        improved = False

        # For each circle, see if moving it slightly helps increase radii
        for i in range(len(refined)):
            x, y, r = refined[i]

            # Try small adjustments to potentially free up space for larger radius
            best_x, best_y, best_r = x, y, r

            # Try several small position adjustments
            adjustments = [
                (0.001, 0, 0),
                (-0.001, 0, 0),
                (0, 0.001, 0),
                (0, -0.001, 0),
                (0.0005, 0.0005, 0),
                (-0.0005, -0.0005, 0),
                (0.0005, -0.0005, 0),
                (-0.0005, 0.0005, 0)
            ]

            for dx, dy, _ in adjustments:
                test_x = np.clip(x + dx, r, 1 - r)
                test_y = np.clip(y + dy, r, 1 - r)

                # Check if this new position would allow larger radius
                max_new_r = min(test_x, 1-test_x, test_y, 1-test_y)

                # Try to increase radius at new position
                test_r = min(max_new_r * 0.99, r + 0.002)

                # Check constraints
                valid = True
                for j in range(len(refined)):
                    if i != j:
                        x_j, y_j, r_j = refined[j]
                        distance = np.sqrt((test_x - x_j)**2 + (test_y - y_j)**2)
                        if distance < test_r + r_j:
                            valid = False
                            break

                if valid and test_r > best_r:
                    best_x, best_y, best_r = test_x, test_y, test_r

            if best_r > refined[i, 2]:
                refined[i, 0] = best_x
                refined[i, 1] = best_y
                refined[i, 2] = best_r
                improved = True

    return refined

def optimize_circles() -> np.ndarray:
    """Main optimization function using evolutionary algorithm with enhancements."""
    n_circles = 26
    pop_size = 60  # Increased population size for better exploration
    generations = 120  # More generations for better convergence
    mutation_rate = 0.15  # Higher mutation rate for more exploration initially

    # Create initial population with better starting points
    population = create_initial_population(pop_size, n_circles)

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

        # Apply local refinement to best individual before selection
        if best_individual is not None:
            refined_best = local_refinement(best_individual)
            if evaluate_fitness(refined_best) > best_fitness:
                best_fitness = evaluate_fitness(refined_best)
                best_individual = refined_best.copy()

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
            # Decrease mutation rate over time to improve exploitation
            adaptive_mutation = max(0.05, mutation_rate * (1 - generation / generations))
            child1 = mutate_individual(child1, adaptive_mutation)
            child2 = mutate_individual(child2, adaptive_mutation)

            # Ensure children meet constraints and apply local refinement
            if is_valid_placement(child1, len(child1)-1):  # Check last circle
                # Apply local refinement to improve quality
                refined_child1 = local_refinement(child1)
                if evaluate_fitness(refined_child1) > evaluate_fitness(child1):
                    new_population.append(refined_child1)
                else:
                    new_population.append(child1)
            if len(new_population) < pop_size and is_valid_placement(child2, len(child2)-1):
                # Apply local refinement to improve quality
                refined_child2 = local_refinement(child2)
                if evaluate_fitness(refined_child2) > evaluate_fitness(child2):
                    new_population.append(refined_child2)
                else:
                    new_population.append(child2)

        # Trim to population size
        population = new_population[:pop_size]

    # Final refinement of best solution
    if best_individual is not None:
        final_refined = local_refinement(best_individual)
        return final_refined
    else:
        # Return first valid individual from final population
        for individual in population:
            if evaluate_fitness(individual) > 0:
                return local_refinement(individual)
        return population[0] if population else np.zeros((26, 3))

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