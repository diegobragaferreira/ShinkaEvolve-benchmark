# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import time

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def create_voronoi_initialization(n: int) -> np.ndarray:
    """Create initial circle positions using Voronoi diagram"""
    # Generate random points inside unit square
    points = np.random.rand(n, 2)

    # Create Voronoi diagram
    vor = Voronoi(points)

    # Get Voronoi vertices as candidate circle centers
    vertices = vor.vertices

    # Filter vertices that are inside the unit square
    valid_vertices = []
    for vertex in vertices:
        if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
            valid_vertices.append(vertex)

    # If we don't have enough vertices, use original points
    if len(valid_vertices) < n:
        # Use all original points and fill with random points if needed
        candidates = points.copy()
        if len(valid_vertices) > 0:
            candidates = np.vstack([np.array(valid_vertices), points])
        while len(candidates) < n:
            candidates = np.vstack([candidates, np.random.rand(1, 2)])
        centers = candidates[:n]
    else:
        # Use Voronoi vertices
        centers = np.array(valid_vertices[:n])

    # Create initial circles with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = centers[i][0]  # x coordinate
        circles[i, 1] = centers[i][1]  # y coordinate
        circles[i, 2] = 0.01  # initial small radius

    return circles

def check_constraints(circles: np.ndarray) -> bool:
    """Check if all circles satisfy containment and non-overlap constraints"""
    n = circles.shape[0]

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > y or r > (1 - x) or r > (1 - y):
            return False

    # Check non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            if dist_sq < (r1 + r2)**2:
                return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all circle radii"""
    return np.sum(circles[:, 2])

def mutate_circle(circles: np.ndarray, idx: int, max_radius_change: float = 0.01) -> np.ndarray:
    """Mutate a single circle by changing its position and radius"""
    mutated = circles.copy()

    # Mutate position slightly
    mutated[idx, 0] += np.random.normal(0, 0.001)
    mutated[idx, 1] += np.random.normal(0, 0.001)

    # Mutate radius
    mutated[idx, 2] += np.random.normal(0, max_radius_change)

    # Ensure bounds
    mutated[idx, 0] = np.clip(mutated[idx, 0], 0, 1)
    mutated[idx, 1] = np.clip(mutated[idx, 1], 0, 1)
    mutated[idx, 2] = np.maximum(mutated[idx, 2], 0.001)

    return mutated

def crossover_circles(parent1: np.ndarray, parent2: np.ndarray,
                     crossover_rate: float = 0.5) -> np.ndarray:
    """Perform crossover between two circle configurations"""
    child = parent1.copy()

    for i in range(len(parent1)):
        if np.random.random() < crossover_rate:
            child[i] = parent2[i].copy()

    return child

def local_refinement(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Apply local refinement to improve packing"""
    refined = circles.copy()

    for _ in range(max_iterations):
        improved = False

        # Try to increase radius of each circle without violating constraints
        for i in range(len(refined)):
            # Save current state
            original = refined[i].copy()

            # Try to increase radius
            max_radius = min(
                refined[i, 0],  # Distance to left edge
                refined[i, 1],  # Distance to bottom edge
                1 - refined[i, 0],  # Distance to right edge
                1 - refined[i, 1]   # Distance to top edge
            )

            # Find minimum distance to other circles
            min_distance = float('inf')
            for j in range(len(refined)):
                if i != j:
                    dist = np.sqrt((refined[i, 0] - refined[j, 0])**2 +
                                 (refined[i, 1] - refined[j, 1])**2)
                    min_distance = min(min_distance, dist)

            if min_distance < max_radius:
                max_radius = min(max_radius, min_distance - 0.001)

            # Try to increase radius up to the limit
            old_radius = refined[i, 2]
            refined[i, 2] = min(old_radius + 0.005, max_radius)

            # Check if this improves the configuration
            if check_constraints(refined):
                improved = True
            else:
                # Revert if constraint violated
                refined[i] = original

        if not improved:
            break

    return refined

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
        of the i-th circle of radius r.
    """
    start_time = time.time()
    n = 26

    # Parameters for evolutionary algorithm
    population_size = 50
    generations = 50
    elite_size = 5
    mutation_rate = 0.3
    crossover_rate = 0.7

    # Initialize population
    population = []
    for _ in range(population_size):
        circles = create_voronoi_initialization(n)
        # Add some randomization to initial positions
        for i in range(n):
            circles[i, 0] += np.random.normal(0, 0.01)
            circles[i, 1] += np.random.normal(0, 0.01)
            circles[i, 0] = np.clip(circles[i, 0], 0, 1)
            circles[i, 1] = np.clip(circles[i, 1], 0, 1)
        population.append(circles)

    best_solution = None
    best_fitness = 0

    # Evolutionary loop
    for generation in range(generations):
        # Evaluate fitness of each individual
        fitness_scores = []
        for circles in population:
            if check_constraints(circles):
                fitness = calculate_sum_radii(circles)
                fitness_scores.append(fitness)
            else:
                # Penalize infeasible solutions
                fitness_scores.append(0)

        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Update best solution
        if fitness_scores[0] > best_fitness:
            best_fitness = fitness_scores[0]
            best_solution = population[0].copy()

        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        for i in range(elite_size):
            new_population.append(population[i].copy())

        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.randint(0, population_size // 2)
            parent2_idx = np.random.randint(0, population_size // 2)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Crossover
            if np.random.random() < crossover_rate:
                child = crossover_circles(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            if np.random.random() < mutation_rate:
                # Mutate random circle
                circle_idx = np.random.randint(n)
                child = mutate_circle(child, circle_idx)

            # Local refinement
            child = local_refinement(child)

            # Add to new population
            new_population.append(child)

        population = new_population[:population_size]

        # Early stopping condition
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break

    # Final local refinement on best solution
    if best_solution is not None:
        best_solution = local_refinement(best_solution)
        print(f"Final solution with sum of radii: {calculate_sum_radii(best_solution):.6f}")

    if best_solution is None:
        # Return default initialization if no good solution found
        best_solution = create_voronoi_initialization(n)
        print("Using default initialization")

    return best_solution


# EVOLVE-BLOCK-END