# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 200
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.1
ELITISM_COUNT = 5
MAX_ATTEMPTS = 1000

def validate_circle_placement(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and don't overlap"""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    for i in range(n):
        x, y, r = circles[i]
        # Find nearby circles (within 2*r distance)
        indices = tree.query_ball_point([x, y], 2*r)
        for j in indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate sum of all radii"""
    return np.sum(circles[:, 2])

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population with improved initialization"""
    population = []

    for _ in range(pop_size):
        # Start with structured grid initialization
        circles = np.zeros((n_circles, 3))

        # Try to place circles in a grid-like pattern with random perturbations
        attempts = 0
        success = False

        while not success and attempts < MAX_ATTEMPTS:
            # Initialize with smaller radii in grid-like formation
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = int(np.ceil(n_circles / rows))

            # Create a grid of positions
            grid_positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(grid_positions) >= n_circles:
                        break
                    x = (j + 0.5) / cols
                    y = (i + 0.5) / rows
                    grid_positions.append((x, y))

            # Fill circles with grid positions
            for i in range(n_circles):
                if i < len(grid_positions):
                    x, y = grid_positions[i]
                    # Small random offset
                    x += (random.random() - 0.5) * 0.05
                    y += (random.random() - 0.5) * 0.05
                    # Small random radius
                    r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                    circles[i] = [x, y, r]
                else:
                    # Random placement for extra circles
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                    r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                    circles[i] = [x, y, r]

            # Try to improve the configuration with local optimization
            improved = local_improvement(circles)

            if validate_circle_placement(improved):
                circles = improved
                success = True
            else:
                # Try again with different initialization
                attempts += 1

        if not success:
            # Fallback to random initialization if everything fails
            circles = np.zeros((n_circles, 3))
            for i in range(n_circles):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles[i] = [x, y, r]

        population.append(circles)

    return population

def local_improvement(circles: np.ndarray) -> np.ndarray:
    """Apply enhanced local improvement using simulated annealing approach"""
    n = len(circles)
    circles_copy = circles.copy()

    # Simulated Annealing parameters
    initial_temp = 0.1
    cooling_rate = 0.95
    min_temp = 1e-6
    max_iterations = 300

    best_solution = circles_copy.copy()
    best_score = calculate_sum_radii(best_solution)

    current_solution = circles_copy.copy()
    current_score = best_score
    temperature = initial_temp

    iteration = 0
    while temperature > min_temp and iteration < max_iterations:
        # Create candidate solution by making small random changes
        candidate = current_solution.copy()

        # Choose a random circle to modify
        idx = random.randint(0, n-1)
        x, y, r = candidate[idx]

        # Try different types of modifications
        if random.random() < 0.5:  # Modify radius
            # Try to increase radius significantly
            max_r = min(x, 1-x, y, 1-y)
            step_size = random.uniform(0.001, 0.01)
            new_r = min(r + step_size, max_r)

            # Check if this radius change is valid
            valid = True
            for j in range(n):
                if i != j:
                    x2, y2, r2 = candidate[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < new_r + r2:
                        valid = False
                        break

            if valid:
                candidate[idx, 2] = new_r

        else:  # Modify position
            # Slightly perturb position
            dx = (random.random() - 0.5) * 0.02
            dy = (random.random() - 0.5) * 0.02
            new_x = x + dx
            new_y = y + dy

            # Keep within bounds
            new_x = np.clip(new_x, r, 1-r)
            new_y = np.clip(new_y, r, 1-r)

            # Check if new position is valid
            valid = True
            for j in range(n):
                if i != j:
                    x2, y2, r2 = candidate[j]
                    dist = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                    if dist < r + r2:
                        valid = False
                        break

            if valid:
                candidate[idx, 0] = new_x
                candidate[idx, 1] = new_y

        # Calculate new score
        new_score = calculate_sum_radii(candidate)

        # Accept or reject the candidate
        if new_score > current_score:
            current_solution = candidate
            current_score = new_score
            if new_score > best_score:
                best_solution = candidate
                best_score = new_score
        else:
            # Accept worse solution with probability based on temperature
            delta = new_score - current_score
            if random.random() < np.exp(delta / temperature):
                current_solution = candidate
                current_score = new_score

        temperature *= cooling_rate
        iteration += 1

    # Final validation and repair
    if validate_circle_placement(best_solution):
        return best_solution

    # If not valid, apply constraint repair
    return repair_constraints(best_solution)

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int) -> np.ndarray:
    """Select individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform uniform crossover between two parents"""
    n = len(parent1)
    child = np.zeros_like(parent1)

    for i in range(n):
        # Each element has 50% chance of coming from parent1 or parent2
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    return child

def mutate(individual: np.ndarray) -> np.ndarray:
    """Apply mutation to an individual"""
    mutated = individual.copy()
    n = len(mutated)

    for i in range(n):
        if random.random() < MUTATION_RATE:
            # Mutate either position or radius
            if random.random() < 0.5:
                # Mutate position
                mutated[i, 0] += (random.random() - 0.5) * 0.1
                mutated[i, 1] += (random.random() - 0.5) * 0.1

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1], 0.01, 0.99)
            else:
                # Mutate radius
                mutated[i, 2] += (random.random() - 0.5) * 0.05

                # Ensure positive radius
                mutated[i, 2] = max(0.001, mutated[i, 2])

    # Repair any constraint violations
    repaired = repair_constraints(mutated)
    return repaired

def repair_constraints(circles: np.ndarray) -> np.ndarray:
    """Repair any constraint violations"""
    repaired = circles.copy()
    n = len(repaired)

    # Ensure all circles are within bounds
    for i in range(n):
        x, y, r = repaired[i]
        r = max(0.001, r)
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        repaired[i] = [x, y, r]

    # Apply constraint repair iteration
    for _ in range(10):
        any_changes = False
        for i in range(n):
            x, y, r = repaired[i]
            # Check overlaps and adjust if needed
            for j in range(n):
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    min_distance = r + r2
                    if distance < min_distance:
                        # Move circle away from overlapping one
                        dx = x2 - x
                        dy = y2 - y
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 0:
                            factor = (min_distance - distance) / dist * 0.1
                            x += dx * factor
                            y += dy * factor
                            any_changes = True

            # Keep within bounds
            r = max(0.001, r)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]

        if not any_changes:
            break

    return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    n_circles = 26

    # Create initial population
    population = create_initial_population(POPULATION_SIZE, n_circles)

    best_solution = None
    best_fitness = -np.inf

    for generation in range(GENERATIONS):
        # Evaluate fitness of each individual
        fitnesses = []
        for individual in population:
            if validate_circle_placement(individual):
                fitness = calculate_sum_radii(individual)
                fitnesses.append(fitness)
            else:
                # Invalid solutions get very low fitness
                fitnesses.append(-1000000)

        # Track best solution so far
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        elites = [population[i].copy() for i in elite_indices]

        # Create new population
        new_population = elites.copy()

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            # Crossover
            child = uniform_crossover(parent1, parent2)

            # Mutation
            child = mutate(child)

            # Add to new population
            new_population.append(child)

        population = new_population[:POPULATION_SIZE]

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to final population if no valid solution was found
        return population[0]


# EVOLVE-BLOCK-END