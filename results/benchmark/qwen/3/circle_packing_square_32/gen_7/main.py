# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def get_fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii for valid configurations."""
    if not is_valid_configuration(circles):
        return 0.0

    return np.sum(circles[:, 2])

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if circles are within bounds and non-overlapping."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    # For each circle, check if it overlaps with others
    for i in range(n):
        x, y, r = circles[i]
        # Find nearby points (within 2*r distance)
        nearby_indices = tree.query_ball_point([x, y], 2 * r)

        # Check each nearby circle for overlap
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False

    return True

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Generate initial population of circle configurations with better initialization."""
    population = []

    # Use a more structured approach for initialization
    # Start with a grid-based initialization
    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Create a grid pattern for initial placement
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))

        # Calculate spacing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        margin = 0.1  # Leave some margin for radii

        for i in range(n_circles):
            row = i // cols
            col = i % cols

            # Position with slight randomness to avoid perfect grids
            x = (col + 0.5) * spacing_x + np.random.normal(0, 0.01)
            y = (row + 0.5) * spacing_y + np.random.normal(0, 0.01)

            # Clip to ensure within bounds
            x = np.clip(x, margin, 1 - margin)
            y = np.clip(y, margin, 1 - margin)

            # Set initial radius based on proximity to other circles
            # Start with smaller radius and adjust
            r = min(0.05, 0.5 * min(spacing_x, spacing_y)) + np.random.uniform(-0.01, 0.01)
            r = max(0.005, min(r, 0.15))

            circles[i] = [x, y, r]

        # Now improve the configuration through local optimization
        circles = optimize_local(circles)

        population.append(circles)

    return population

def optimize_local(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply local optimization to improve the configuration."""
    circles = circles.copy()

    # Simple local optimization: try to increase radii where possible
    for iteration in range(max_iter):
        improved = False
        for i in range(len(circles)):
            x, y, r = circles[i]

            # Calculate maximum possible radius
            max_radius = min(x, 1 - x, y, 1 - y)

            # Check neighbors to see if we can increase radius
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_radius_for_this = distance - r2
                    if max_radius_for_this > 0:
                        max_radius = min(max_radius, max_radius_for_this)

            # Try to increase radius but keep within bounds
            if max_radius > r and max_radius > 0.001:
                new_r = min(max_radius, r + 0.01)
                if new_r > r:
                    circles[i, 2] = new_r
                    improved = True

        if not improved:
            break

    # Ensure validity after optimization
    if not is_valid_configuration(circles):
        # If invalid due to optimization, revert to original valid state
        pass  # We'll handle this at a higher level

    return circles

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray,
             crossover_rate: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Perform specialized crossover for circle packing."""
    if np.random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Use uniform crossover for positions - mix positions equally from both parents
    mask = np.random.random(n) < 0.5
    child1[mask, :2] = parent2[mask, :2]
    child2[mask, :2] = parent1[mask, :2]

    # For radii, take average with some variation
    child1[:, 2] = (parent1[:, 2] + parent2[:, 2]) / 2 + np.random.normal(0, 0.005, n)
    child1[:, 2] = np.clip(child1[:, 2], 0.005, 0.2)

    child2[:, 2] = (parent1[:, 2] + parent2[:, 2]) / 2 + np.random.normal(0, 0.005, n)
    child2[:, 2] = np.clip(child2[:, 2], 0.005, 0.2)

    # Apply local optimization to clean up after crossover
    child1 = optimize_local(child1)
    child2 = optimize_local(child2)

    return child1, child2

def mutate(individual: np.ndarray, mutation_rate: float = 0.1,
          max_radius_change: float = 0.02) -> np.ndarray:
    """Apply specialized mutation for circle packing."""
    mutated = individual.copy()

    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Mutate position
            mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02), 0, 1)
            mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02), 0, 1)

            # Mutate radius with better distribution
            mutation_amount = np.random.normal(0, max_radius_change)
            mutated[i, 2] = np.clip(mutated[i, 2] + mutation_amount, 0.005, 0.2)

    # After mutation, try to improve the configuration
    mutated = optimize_local(mutated)

    return mutated

def evaluate_fitness_parallel(population: List[np.ndarray]) -> List[float]:
    """Evaluate fitness of entire population in parallel."""
    fitnesses = Parallel(n_jobs=-1)(
        delayed(get_fitness)(individual) for individual in population
    )
    return fitnesses

def evolve_circles(n_circles: int = 32, pop_size: int = 100,
                  generations: int = 200, elite_size: int = 10) -> np.ndarray:
    """Main evolutionary algorithm to pack circles optimally with improvements."""
    # Initialize population
    population = initialize_population(pop_size, n_circles)

    best_fitness_history = []

    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = evaluate_fitness_parallel(population)

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Print progress every 20 generations
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

        # Get best individuals
        sorted_indices = np.argsort(fitnesses)[::-1]
        best_individuals = [population[i] for i in sorted_indices[:elite_size]]

        # Create new population with elitism
        new_population = best_individuals.copy()

        # Fill rest of population with offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)

            # Add to new population if valid
            if is_valid_configuration(child1):
                new_population.append(child1)
            if len(new_population) < pop_size and is_valid_configuration(child2):
                new_population.append(child2)

        population = new_population[:pop_size]

    # Return best solution
    final_fitnesses = evaluate_fitness_parallel(population)
    best_idx = np.argmax(final_fitnesses)

    return population[best_idx]

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = evolve_circles(n_circles=32, pop_size=100, generations=200, elite_size=10)
        # Ensure the result is valid
        if not is_valid_configuration(circles):
            # Fallback to basic configuration if something went wrong
            circles = np.zeros((32, 3))
            for i in range(32):
                circles[i] = [0.1 + i * 0.03, 0.1 + (i % 4) * 0.1, 0.05]
    except Exception as e:
        # On error, fallback to basic configuration
        print(f"Error during evolution: {e}")
        circles = np.zeros((32, 3))
        for i in range(32):
            circles[i] = [0.1 + i * 0.03, 0.1 + (i % 4) * 0.1, 0.05]

    return circles

# EVOLVE-BLOCK-END