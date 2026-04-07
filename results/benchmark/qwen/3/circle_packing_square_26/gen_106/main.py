# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Global constants for optimization
POP_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
BOUNDARY_PENALTY = 1000.0
OVERLAP_PENALTY = 10000.0

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with Voronoi-based distribution and Poisson disk refinement"""
    population = []

    # Use a larger number of points for better Voronoi coverage
    n_points = max(n_circles * 2, 50)

    # Generate initial points using a combination of structured grid and random sampling
    # This creates a good distribution that will form meaningful Voronoi cells
    points = []

    # Create a structured grid with some randomness
    grid_size = int(np.ceil(np.sqrt(n_points * 2)))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)

    # Create grid points
    for x in x_coords:
        for y in y_coords:
            if len(points) < n_points:
                points.append([x, y])

    # Add extra random points to improve distribution
    while len(points) < n_points:
        points.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])

    # Generate initial population with better distributed points
    for _ in range(pop_size):
        # Select n_circles points from our generated points
        selected_points = random.sample(points, n_circles)

        individual = np.zeros((n_circles, 3))

        # Assign positions with slight perturbation and appropriate radii
        for i in range(n_circles):
            x, y = selected_points[i]

            # Perturb slightly with more controlled variation
            perturbation_x = random.uniform(-0.015, 0.015)
            perturbation_y = random.uniform(-0.015, 0.015)
            individual[i, 0] = max(0.05, min(0.95, x + perturbation_x))
            individual[i, 1] = max(0.05, min(0.95, y + perturbation_y))

            # Assign radius based on position and proximity to boundaries
            margin = min(individual[i, 0], individual[i, 1], 1 - individual[i, 0], 1 - individual[i, 1])
            # Base radius inversely proportional to distance to edge but with minimum
            base_radius = min(0.15, margin / 2.0)
            # Add some randomness to radius (but keep it within reasonable bounds)
            individual[i, 2] = max(0.01, min(0.2, base_radius * random.uniform(0.8, 1.2)))

        # Apply refinement to make sure initial configuration is valid
        individual = refine_solution(individual)
        population.append(individual)

    return population

def is_valid_circle(circle: np.ndarray, other_circles: np.ndarray) -> bool:
    """Check if a circle is valid (contained and not overlapping)"""
    x, y, r = circle

    # Check containment
    if r > x or r > y or r > (1 - x) or r > (1 - y):
        return False

    # Check overlap with existing circles
    for other in other_circles:
        if other[2] > 0:  # Only check non-zero radius circles
            ox, oy, oradius = other
            distance = np.sqrt((x - ox)**2 + (y - oy)**2)
            if distance < (r + oradius):
                return False

    return True

def calculate_penalty(circles: np.ndarray) -> float:
    """Calculate penalty based on constraint violations"""
    penalty = 0.0
    n = len(circles)

    # Check containment penalties
    for circle in circles:
        x, y, r = circle
        # Calculate how much it violates boundaries
        left_violation = max(0, r - x)
        right_violation = max(0, r - (1 - x))
        bottom_violation = max(0, r - y)
        top_violation = max(0, r - (1 - y))

        # Apply penalty proportional to violation
        penalty += BOUNDARY_PENALTY * (left_violation + right_violation +
                                      bottom_violation + top_violation)

    # Check overlap penalties using spatial indexing for efficiency
    valid_circles = [c for c in circles if c[2] > 0]
    if len(valid_circles) > 1:
        tree = cKDTree([c[:2] for c in valid_circles])
        pairs = tree.query_pairs(0.001)  # Find nearby points

        for i, j in pairs:
            ci = valid_circles[i]
            cj = valid_circles[j]
            dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
            min_dist = ci[2] + cj[2]

            if dist < min_dist:
                # Apply penalty proportional to overlap
                overlap = min_dist - dist
                penalty += OVERLAP_PENALTY * overlap

    return penalty

def evaluate_fitness(circles: np.ndarray) -> Tuple[float, float]:
    """Evaluate the fitness of a solution"""
    # Sum of radii (primary objective)
    total_radius = np.sum(circles[:, 2])

    # Penalty for constraint violations
    penalty = calculate_penalty(circles)

    # Fitness is total radius minus penalty
    fitness = total_radius - penalty

    return fitness, total_radius

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float],
                        tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Select an individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = [fitness_scores[i] for i in selected_indices]

    winner_idx = selected_indices[np.argmax(selected_fitness)]
    return population[winner_idx].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray,
             crossover_rate: float = CROSSOVER_RATE) -> np.ndarray:
    """Perform crossover between two parents"""
    if random.random() > crossover_rate:
        return parent1.copy()

    n = len(parent1)
    child = np.zeros_like(parent1)

    # Single-point crossover on radii
    crossover_point = random.randint(1, n - 1)

    # Copy positions from parent1
    child[:crossover_point, :2] = parent1[:crossover_point, :2]
    child[crossover_point:, :2] = parent2[crossover_point:, :2]

    # Copy radii from parent1
    child[:crossover_point, 2] = parent1[:crossover_point, 2]
    child[crossover_point:, 2] = parent2[crossover_point:, 2]

    # Local refinement to fix any constraint violations
    child = refine_solution(child)

    return child

def refine_solution(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to fix constraint violations"""
    refined = circles.copy()

    # Ensure containment and fix overlaps iteratively
    for i in range(len(refined)):
        x, y, r = refined[i]

        # Fix containment
        if r > x:
            r = min(r, x)
        if r > y:
            r = min(r, y)
        if r > (1 - x):
            r = min(r, 1 - x)
        if r > (1 - y):
            r = min(r, 1 - y)

        # Ensure positive radius
        r = max(0.001, r)

        refined[i, 2] = r

    # Perform iterative overlap removal
    MAX_ITER = 100
    for _ in range(MAX_ITER):
        changed = False
        for i in range(len(refined)):
            x, y, r = refined[i]

            # Check overlap with all others
            for j in range(len(refined)):
                if i != j:
                    ox, oy, oradius = refined[j]
                    distance = np.sqrt((x - ox)**2 + (y - oy)**2)

                    if distance < (r + oradius):
                        # Move circle away from overlapping one
                        if distance > 0.001:
                            dx = (x - ox) / distance
                            dy = (y - oy) / distance

                            # Reduce radius to prevent further overlap
                            new_r = min(r, oradius)
                            if new_r > 0.001:
                                refined[i, 2] = new_r * 0.99

                                # Adjust position slightly to separate
                                refined[i, 0] = x + dx * 0.001
                                refined[i, 1] = y + dy * 0.001

                                # Ensure containment after adjustment
                                refined[i, 0] = np.clip(refined[i, 0], refined[i, 2], 1 - refined[i, 2])
                                refined[i, 1] = np.clip(refined[i, 1], refined[i, 2], 1 - refined[i, 2])

                        changed = True

        if not changed:
            break

    return refined

def mutate(individual: np.ndarray, mutation_rate: float = MUTATION_RATE) -> np.ndarray:
    """Mutate an individual"""
    mutated = individual.copy()
    n = len(mutated)

    for i in range(n):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate
            choice = random.randint(0, 2)

            if choice == 0:  # Mutate x position
                mutated[i, 0] = np.clip(mutated[i, 0] + random.gauss(0, 0.02), 0.05, 0.95)
            elif choice == 1:  # Mutate y position
                mutated[i, 1] = np.clip(mutated[i, 1] + random.gauss(0, 0.02), 0.05, 0.95)
            else:  # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, 0.01), 0.001, 0.2)

    # Local refinement after mutation
    mutated = refine_solution(mutated)
    return mutated

def evolve_population(population: List[np.ndarray]) -> Tuple[List[np.ndarray], float, float]:
    """Evolve the population for one generation"""
    # Evaluate fitness
    fitness_scores = []
    total_radii = []

    for individual in population:
        fitness, total_radius = evaluate_fitness(individual)
        fitness_scores.append(fitness)
        total_radii.append(total_radius)

    # Track best individual
    best_idx = np.argmax(fitness_scores)
    best_fitness = fitness_scores[best_idx]
    best_total_radius = total_radii[best_idx]

    # Create new population
    new_population = []

    # Elitism: keep the best individual
    new_population.append(population[best_idx].copy())

    # Generate rest of population
    while len(new_population) < len(population):
        # Selection
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)

        # Crossover
        child = crossover(parent1, parent2)

        # Mutation
        child = mutate(child)

        new_population.append(child)

    return new_population, best_fitness, best_total_radius

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 26
    population = initialize_population(POP_SIZE, n)

    best_total_radius = 0.0
    best_individual = None

    # Evolution loop
    for generation in range(GENERATIONS):
        population, gen_fitness, gen_radius = evolve_population(population)

        if gen_radius > best_total_radius:
            best_total_radius = gen_radius
            best_individual = population[0]  # Keep track of best individual

        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best radius sum = {gen_radius:.6f}")

    print(f"Final result: Best radius sum = {best_total_radius:.6f}")

    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to returning first individual if something went wrong
        return population[0]

# EVOLVE-BLOCK-END