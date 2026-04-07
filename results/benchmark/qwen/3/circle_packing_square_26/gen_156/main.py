# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def _initialize_voronoi_seeds(n_points, boundary_margin=0.1):
    """Generate well-distributed seed points using Voronoi diagram."""
    points = np.random.rand(n_points, 2) * (1 - 2*boundary_margin) + boundary_margin
    return points

def _compute_voronoi_radii(points, boundary_margin=0.1):
    """Compute radii based on Voronoi cell areas."""
    vor = Voronoi(points)
    radii = []
    for i, point in enumerate(points):
        distances = cdist([point], np.delete(points, i, axis=0))[0]
        min_distance = np.min(distances)
        max_radius = min(point[0], 1-point[0], point[1], 1-point[1])
        estimated_radius = min(min_distance/2.0, max_radius)
        radii.append(max(estimated_radius, 0.001))
    return np.array(radii)

def _initialize_structured_placement(n_circles: int) -> np.ndarray:
    """Initialize circles with structured placement for better distribution."""
    individual = np.zeros((n_circles, 3))

    # Use structured grid placement
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))

    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)

    for i in range(n_circles):
        row = i // cols
        col = i % cols

        base_x = (col + 1) * spacing_x
        base_y = (row + 1) * spacing_y

        # Add small random perturbation
        individual[i, 0] = np.clip(base_x + np.random.uniform(-spacing_x/4, spacing_x/4), 0.01, 0.99)
        individual[i, 1] = np.clip(base_y + np.random.uniform(-spacing_y/4, spacing_y/4), 0.01, 0.99)

        # Set radius with proper bounds
        max_radius = min(0.5 - individual[i, 0], 0.5 - individual[i, 1],
                       individual[i, 0], individual[i, 1])
        individual[i, 2] = np.random.uniform(0.001, max_radius * 0.8)

    return individual

def _initialize_population(n_circles: int, population_size: int) -> List[np.ndarray]:
    """Create initial population with mixed strategies."""
    population = []
    for _ in range(population_size):
        # Alternate between Voronoi-based and structured initialization
        if len(population) % 2 == 0:
            # Voronoi-based initialization
            seed_points = _initialize_voronoi_seeds(n_circles)
            radii = _compute_voronoi_radii(seed_points)
            individual = np.zeros((n_circles, 3))
            for i in range(n_circles):
                individual[i] = [seed_points[i][0], seed_points[i][1], radii[i]]
        else:
            # Structured initialization
            individual = _initialize_structured_placement(n_circles)

        # Apply overlap adjustment
        _adjust_for_overlaps(individual)
        population.append(individual)

    return population

def _check_containment_constraints(individual: np.ndarray) -> float:
    """Check containment constraints and return penalty."""
    penalty = 0.0
    for i in range(len(individual)):
        x, y, r = individual[i]
        # Calculate boundary violations
        left_violation = max(0, r - x)
        right_violation = max(0, x + r - 1)
        bottom_violation = max(0, r - y)
        top_violation = max(0, y + r - 1)

        if left_violation > 0 or right_violation > 0 or bottom_violation > 0 or top_violation > 0:
            penalty += 1000 * (left_violation + right_violation + bottom_violation + top_violation)

    return penalty

def _check_overlap_constraints(individual: np.ndarray) -> float:
    """Check overlap constraints efficiently using cKDTree."""
    penalty = 0.0
    n = len(individual)

    if n <= 1:
        return penalty

    # Use cKDTree for efficient neighbor queries
    tree = cKDTree(individual[:, :2])

    # Query neighbors within a reasonable distance
    for i in range(n):
        x1, y1, r1 = individual[i]
        # Find all neighbors within double the combined radius
        max_radius = np.max(individual[:, 2])
        neighbors = tree.query_ball_point([x1, y1], 2 * (r1 + max_radius), p=2)

        for j in neighbors:
            if i >= j:  # Skip self-comparison and duplicate pairs
                continue

            x2, y2, r2 = individual[j]

            # Fast distance check using squared distance
            dx = x2 - x1
            dy = y2 - y1
            distance_squared = dx*dx + dy*dy

            # Check if circles are close enough to potentially overlap
            combined_radius = r1 + r2
            if distance_squared < combined_radius * combined_radius:
                actual_distance = math.sqrt(distance_squared)
                if actual_distance < combined_radius:
                    overlap = combined_radius - actual_distance
                    penalty += overlap * 1000 * (1.0 + overlap * 0.01)

    return penalty

def evaluate_fitness(individual: np.ndarray, population: List[np.ndarray] = None,
                    generation: int = 0, max_generations: int = 500) -> float:
    """Evaluate fitness of an individual (sum of radii) with penalties for violations."""
    total_radius = np.sum(individual[:, 2])

    # Calculate penalties with progressive constraint relaxation
    containment_penalty = _check_containment_constraints(individual)
    overlap_penalty = _check_overlap_constraints(individual)

    # Progressive constraint relaxation - start with loose constraints in early generations
    relaxation_factor = 1.0
    if generation < 50:
        relaxation_factor = 0.2
    elif generation < 200:
        relaxation_factor = 0.2 + (1.0 - 0.2) * (generation - 50) / (200 - 50)

    penalty = (containment_penalty + overlap_penalty) * relaxation_factor

    # Add fitness sharing penalty if population is provided
    diversity_penalty = 0.0
    if population is not None and len(population) > 1:
        # Calculate similarity to other individuals using normalized distance
        individual_norm = individual.flatten() / np.linalg.norm(individual.flatten())
        for other in population:
            other_norm = other.flatten() / np.linalg.norm(other.flatten())
            similarity = np.dot(individual_norm, other_norm)
            # Penalize high similarity (low distance)
            if similarity > 0.95:  # Very similar individuals
                diversity_penalty += 1000 * (1 - similarity)

    return total_radius - penalty - diversity_penalty

def _adjust_for_overlaps(individual: np.ndarray):
    """Improve overlap resolution with better geometric handling."""
    max_iterations = 100
    for iteration in range(max_iterations):
        any_changes = False
        # Check each pair of circles
        for i in range(len(individual)):
            for j in range(i+1, len(individual)):
                x1, y1, r1 = individual[i]
                x2, y2, r2 = individual[j]

                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)

                # If circles overlap
                if distance < r1 + r2:
                    overlap = (r1 + r2) - distance
                    if distance > 0:
                        # Push them apart along the line connecting centers
                        push_x = dx / distance * overlap * 0.5
                        push_y = dy / distance * overlap * 0.5

                        individual[i, 0] -= push_x
                        individual[i, 1] -= push_y
                        individual[j, 0] += push_x
                        individual[j, 1] += push_y
                    else:
                        # If they're at the same position, push them apart randomly
                        angle = np.random.uniform(0, 2*np.pi)
                        push_dist = overlap * 0.5
                        individual[i, 0] -= push_dist * np.cos(angle)
                        individual[i, 1] -= push_dist * np.sin(angle)
                        individual[j, 0] += push_dist * np.cos(angle)
                        individual[j, 1] += push_dist * np.sin(angle)

                    # Keep within bounds
                    individual[i, 0] = np.clip(individual[i, 0], r1, 1-r1)
                    individual[i, 1] = np.clip(individual[i, 1], r1, 1-r1)
                    individual[j, 0] = np.clip(individual[j, 0], r2, 1-r2)
                    individual[j, 1] = np.clip(individual[j, 1], r2, 1-r2)
                    any_changes = True

        if not any_changes:
            break

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], tournament_size: int = 3) -> np.ndarray:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Single point crossover on positions and radii
    crossover_point = random.randint(1, len(parent1) - 1)

    # Swap positions and radii for half the circles
    child1[crossover_point:, :2] = parent2[crossover_point:, :2]
    child1[crossover_point:, 2] = parent2[crossover_point:, 2]

    child2[crossover_point:, :2] = parent1[crossover_point:, :2]
    child2[crossover_point:, 2] = parent1[crossover_point:, 2]

    return child1, child2

def mutate_individual(individual: np.ndarray, mutation_rate: float):
    """Mutate an individual."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Mutate position slightly
            individual[i, 0] += np.random.normal(0, 0.01)
            individual[i, 1] += np.random.normal(0, 0.01)

            # Keep within bounds
            individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
            individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)

            # Mutate radius
            individual[i, 2] += np.random.normal(0, 0.005)
            individual[i, 2] = max(0.001, individual[i, 2])

def run_evolution(n_circles: int = 26, population_size: int = 200, generations: int = 500) -> np.ndarray:
    """Run the evolutionary optimization process."""
    # Initialize population
    population = _initialize_population(n_circles, population_size)

    # Evolutionary loop
    best_fitness_history = []
    for generation in range(generations):
        # Evaluate fitness with population context for diversity preservation
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual, population, generation, generations)
            fitness_scores.append(fitness)

        # Track best fitness
        best_fitness = max(fitness_scores)
        best_fitness_history.append(best_fitness)

        # Select top individuals (elitism)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite = [population[i] for i in sorted_indices[:20]]  # Fixed elite size

        # Generate new population
        new_population = elite.copy()

        # Adaptive mutation rate: decrease over time with smoother decay
        adaptive_mutation_rate = 0.1 * (1 - generation / generations) ** 2
        if adaptive_mutation_rate < 0.01:
            adaptive_mutation_rate = 0.01

        # Fill rest of population through crossover and mutation
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            if random.random() < 0.8:  # Crossover probability
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutation with adaptive rate
            if random.random() < adaptive_mutation_rate:
                mutate_individual(child1, adaptive_mutation_rate)
            if random.random() < adaptive_mutation_rate:
                mutate_individual(child2, adaptive_mutation_rate)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:population_size]

        # Print progress
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

    # Return best solution
    final_fitness_scores = [evaluate_fitness(ind, population, generations, generations) for ind in population]
    best_index = np.argmax(final_fitness_scores)
    return population[best_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    return run_evolution(n_circles=26, population_size=200, generations=500)

# EVOLVE-BLOCK-END