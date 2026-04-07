# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple
import math

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 250
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.2
MUTATION_RATE_END = 0.005
ELITISM_COUNT = 10
BOUNDARY_MARGIN = 0.01
MAX_EVAL_TIME = 60.0
POISSON_DISK_MIN_DISTANCE = 0.1
LOCAL_REFINEMENT_FREQUENCY = 20  # How often to apply local refinement
LOCAL_REFINEMENT_ITERATIONS = 50  # Number of local optimization iterations

def poisson_disk_sampling(width: float, height: float, min_distance: float, max_attempts: int = 30) -> np.ndarray:
    """Generate points using Poisson disk sampling for uniform distribution"""
    # Grid to track occupied cells
    cell_size = min_distance / math.sqrt(2)
    grid_width = int(math.ceil(width / cell_size))
    grid_height = int(math.ceil(height / cell_size))
    grid = np.full((grid_height, grid_width), -1, dtype=int)

    # List of points and active list
    points = []
    active_list = []

    # Add first point randomly
    first_point = np.random.rand(2) * [width, height]
    first_point[0] = np.clip(first_point[0], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
    first_point[1] = np.clip(first_point[1], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)

    points.append(first_point)
    active_list.append(0)

    # Index for grid
    def get_grid_index(point):
        x, y = point
        grid_x = int(x / cell_size)
        grid_y = int(y / cell_size)
        return grid_y, grid_x

    grid[get_grid_index(first_point)] = 0

    attempts = 0
    while active_list and attempts < max_attempts:
        # Pick random point from active list
        idx = np.random.randint(len(active_list))
        point_idx = active_list[idx]
        point = points[point_idx]

        # Try to generate new point
        found = False
        for _ in range(max_attempts):
            angle = np.random.rand() * 2 * np.pi
            radius = np.random.uniform(min_distance, 2 * min_distance)

            new_point = point + np.array([radius * np.cos(angle), radius * np.sin(angle)])

            # Check boundaries
            if (new_point[0] < BOUNDARY_MARGIN or new_point[0] > 1 - BOUNDARY_MARGIN or
                new_point[1] < BOUNDARY_MARGIN or new_point[1] > 1 - BOUNDARY_MARGIN):
                continue

            # Check grid for nearby points
            grid_y, grid_x = get_grid_index(new_point)
            valid = True

            # Check nearby cells
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = grid_y + dy, grid_x + dx
                    if 0 <= ny < grid_height and 0 <= nx < grid_width:
                        neighbor_idx = grid[ny, nx]
                        if neighbor_idx != -1:
                            neighbor = points[neighbor_idx]
                            dist = np.linalg.norm(new_point - neighbor)
                            if dist < min_distance:
                                valid = False
                                break
                if not valid:
                    break

            if valid:
                points.append(new_point)
                active_list.append(len(points) - 1)
                grid[grid_y, grid_x] = len(points) - 1
                found = True
                break

        if not found:
            active_list.pop(idx)

        attempts += 1

    return np.array(points)

def initialize_population(n_circles: int, pop_size: int) -> np.ndarray:
    """Initialize population with well-distributed circle configurations"""
    population = []

    # Generate initial points using Poisson disk sampling
    initial_points = poisson_disk_sampling(1.0, 1.0, POISSON_DISK_MIN_DISTANCE)

    # If we don't have enough points, fill with grid-based pattern
    if len(initial_points) < n_circles:
        # Complete with grid pattern
        remaining = n_circles - len(initial_points)
        grid_size = int(np.ceil(np.sqrt(remaining)))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= remaining:
                    break
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                x = np.clip(x, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                y = np.clip(y, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                initial_points = np.vstack([initial_points, [[x, y]]])
                count += 1

    # Take only the required number of points
    initial_points = initial_points[:n_circles]

    for _ in range(pop_size):
        # Initialize circles with selected points
        circles = np.zeros((n_circles, 3))

        for i in range(n_circles):
            x, y = initial_points[i]
            # Small initial radius
            r = np.random.uniform(0.005, 0.03)
            circles[i] = [x, y, r]

        # Adjust radii to be feasible and distribute more evenly
        for i in range(n_circles):
            x, y, r = circles[i]
            # Constrain radius by proximity to boundaries
            max_r = min(x, 1-x, y, 1-y) * 0.8
            circles[i, 2] = min(r, max_r)

        population.append(circles)

    return np.array(population)

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration is valid (no overlaps, fully contained)"""
    n_circles = len(circles)

    # Check boundary containment
    for i in range(n_circles):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlaps using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    # Find all pairs within distance of (r_i + r_j)
    pairs = tree.query_pairs(2 * np.max(circles[:, 2]), p=np.inf)

    for i, j in pairs:
        if i != j:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False

    return True

def calculate_fitness(circles: np.ndarray, generation: int = 0, total_generations: int = 1) -> Tuple[float, float]:
    """
    Calculate fitness with penalty for constraint violations
    Returns (total_radius, penalty_score)
    """
    total_radius = np.sum(circles[:, 2])

    if not is_valid_configuration(circles):
        # More aggressive penalty for constraint violations
        penalty = 100000
        return total_radius - penalty, penalty

    return total_radius, 0.0

def tournament_selection(population: np.ndarray, fitness_scores: np.ndarray,
                        tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Select individual using tournament selection"""
    selected_idx = np.random.randint(0, len(population), tournament_size)
    best_idx = selected_idx[np.argmax(fitness_scores[selected_idx])]
    return population[best_idx]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Uniform crossover between two parents"""
    n_circles = len(parent1)
    child = np.zeros_like(parent1)

    # Randomly choose which parent contributes each circle
    mask = np.random.rand(n_circles) > 0.5

    child[mask] = parent1[mask]
    child[~mask] = parent2[~mask]

    # Apply slight adjustments to fix any constraint violations
    for i in range(n_circles):
        x, y, r = child[i]
        # Ensure circle stays within bounds
        max_r = min(x, 1-x, y, 1-y)
        if r > max_r:
            child[i, 2] = max_r * 0.9  # Scale down radius slightly

    return child

def mutate(individual: np.ndarray, generation: int, total_generations: int, elite: bool = False) -> np.ndarray:
    """Apply mutation to an individual with dual strategy"""
    n_circles = len(individual)
    mutated = individual.copy()

    # Adaptive mutation rate
    mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * (generation / total_generations)

    # For elite individuals, use smaller mutations
    pos_mutation_scale = 0.015 if elite else 0.03
    rad_mutation_scale = 0.01 if elite else 0.02

    # Mutate each circle
    for i in range(n_circles):
        if np.random.rand() < mutation_rate:
            # Mutate position with larger scale
            mutated[i, 0] += np.random.normal(0, pos_mutation_scale)  # X position
            mutated[i, 1] += np.random.normal(0, pos_mutation_scale)  # Y position
            # Mutate radius with smaller scale
            mutated[i, 2] += np.random.normal(0, rad_mutation_scale)  # Radius

            # Clamp values to valid ranges
            mutated[i, 0] = np.clip(mutated[i, 0], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            mutated[i, 1] = np.clip(mutated[i, 1], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
            mutated[i, 2] = max(mutated[i, 2], 0.001)

            # Fix potential boundary constraint violations
            x, y, r = mutated[i]
            max_r = min(x, 1-x, y, 1-y)
            mutated[i, 2] = min(r, max_r * 0.95)

    return mutated

def local_refinement(circles: np.ndarray, max_iterations: int = LOCAL_REFINEMENT_ITERATIONS) -> np.ndarray:
    """
    Apply local optimization to improve a given solution
    Uses scipy optimize to maximize sum of radii with constraints
    """
    def objective(params):
        # Extract parameters (x, y, r for each circle)
        circles_copy = circles.copy()
        idx = 0
        for i in range(len(circles)):
            circles_copy[i, 0] = params[idx]  # x
            circles_copy[i, 1] = params[idx + 1]  # y
            circles_copy[i, 2] = params[idx + 2]  # r
            idx += 3

        # Objective is negative sum of radii (we minimize negative to maximize positive)
        return -np.sum(circles_copy[:, 2])

    def constraint_func(params):
        # Check if all constraints are satisfied
        circles_copy = circles.copy()
        idx = 0
        for i in range(len(circles)):
            circles_copy[i, 0] = params[idx]  # x
            circles_copy[i, 1] = params[idx + 1]  # y
            circles_copy[i, 2] = params[idx + 2]  # r
            idx += 3

        # Check boundary constraints
        for i in range(len(circles_copy)):
            x, y, r = circles_copy[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return -1  # Invalid

        # Check overlap constraints
        points = circles_copy[:, :2]
        tree = cKDTree(points)
        pairs = tree.query_pairs(2 * np.max(circles_copy[:, 2]), p=np.inf)

        for i, j in pairs:
            if i != j:
                x1, y1, r1 = circles_copy[i]
                x2, y2, r2 = circles_copy[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return -1  # Overlap

        return 1  # Valid

    # Flatten initial solution
    initial_params = circles.flatten()

    # Define bounds for each parameter (x, y, r for each circle)
    bounds = []
    for i in range(len(circles)):
        bounds.extend([(BOUNDARY_MARGIN, 1-BOUNDARY_MARGIN),  # x bounds
                       (BOUNDARY_MARGIN, 1-BOUNDARY_MARGIN),  # y bounds
                       (0.001, 0.5)])  # r bounds (reasonable max radius)

    # Create constraints
    constraints = {'type': 'ineq', 'fun': constraint_func}

    try:
        # Apply local optimization
        result = minimize(objective, initial_params, method='L-BFGS-B',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': max_iterations})

        if result.success:
            # Convert back to circles format
            refined_circles = circles.copy()
            idx = 0
            for i in range(len(circles)):
                refined_circles[i, 0] = result.x[idx]
                refined_circles[i, 1] = result.x[idx + 1]
                refined_circles[i, 2] = result.x[idx + 2]
                idx += 3
            return refined_circles
    except:
        pass  # If optimization fails, return original

    return circles

def evolve_circles(n_circles: int = 26, generations: int = GENERATIONS) -> np.ndarray:
    """Main evolutionary algorithm"""
    # Initialize population
    population = initialize_population(n_circles, POPULATION_SIZE)

    best_fitness_history = []

    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = np.array([calculate_fitness(ind, gen, generations)[0] for ind in population])

        # Track best fitness
        best_fitness = np.max(fitness_scores)
        best_fitness_history.append(best_fitness)

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
        elites = population[elite_indices].copy()

        # Create new population
        new_population = []

        # Add elites first
        new_population.extend(elites)

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation with different strategies for elites vs others
            elite_flag = len(new_population) < ELITISM_COUNT
            child = mutate(child, gen, generations, elite_flag)

            new_population.append(child)

        # Trim to exact population size
        population = np.array(new_population[:POPULATION_SIZE])

        # Apply local refinement periodically
        if gen % LOCAL_REFINEMENT_FREQUENCY == 0 and gen > 0:
            # Refine the best individuals in the population
            refined_population = []
            for ind in population:
                refined = local_refinement(ind)
                refined_population.append(refined)
            population = np.array(refined_population)

        # Early stopping when improvement plateaus
        if len(best_fitness_history) > 15:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-15]
            if recent_improvement < 0.001:
                break

    # Return best individual
    final_fitness_scores = np.array([calculate_fitness(ind, generations, generations)[0] for ind in population])
    best_idx = np.argmax(final_fitness_scores)
    return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    try:
        circles = evolve_circles(n_circles=26, generations=GENERATIONS)
        return circles
    except Exception as e:
        # Fallback to simple initialization if evolution fails
        print(f"Evolution failed: {e}")
        circles = np.zeros((26, 3))
        # Simple grid initialization
        grid_size = int(np.ceil(np.sqrt(26)))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                r = 0.02
                circles[count] = [x, y, r]
                count += 1
        return circles

# EVOLVE-BLOCK-END