# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Uses a Voronoi-guided evolutionary algorithm approach:
    1. Generate initial candidate positions via Voronoi tessellation
    2. Apply evolutionary algorithm with constraint-aware fitness
    3. Use spatial indexing for efficient collision checking

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    def check_collision(circle1, circle2):
        """Check if two circles collide with early termination"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2

        # Early termination: Check if circles are too far apart
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # If maximum possible distance is less than sum of radii, they can't collide
        if dx > (r1 + r2) or dy > (r1 + r2):
            return False

        # Quick check using squared distances to avoid sqrt computation
        dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return dist_sq < (r1 + r2) ** 2

    def check_containment(circle):
        """Check if circle is fully contained in unit square"""
        x, y, r = circle
        return r <= x <= 1-r and r <= y <= 1-r

    def get_total_radius(circles_array):
        """Calculate sum of all radii"""
        return np.sum(circles_array[:, 2])

    def build_grid(circles_array, grid_size=15):
        """Build a spatial hash grid for efficient collision detection"""
        grid = {}
        cell_size = 1.0 / grid_size

        for i, circle in enumerate(circles_array):
            x, y, r = circle
            # Determine which cells this circle spans
            min_x_cell = int(max(0, (x - r) / cell_size))
            max_x_cell = int(min(grid_size - 1, (x + r) / cell_size))
            min_y_cell = int(max(0, (y - r) / cell_size))
            max_y_cell = int(min(grid_size - 1, (y + r) / cell_size))

            for gx in range(min_x_cell, max_x_cell + 1):
                for gy in range(min_y_cell, max_y_cell + 1):
                    if (gx, gy) not in grid:
                        grid[(gx, gy)] = []
                    grid[(gx, gy)].append(i)

        return grid, cell_size

    def get_candidates_from_grid(circle_idx, grid, cell_size, circles_array, grid_size=15):
        """Get candidate circles from neighboring grid cells"""
        x, y, r = circles_array[circle_idx]
        cell_size = 1.0 / grid_size
        min_x_cell = int(max(0, (x - r) / cell_size))
        max_x_cell = int(min(grid_size - 1, (x + r) / cell_size))
        min_y_cell = int(max(0, (y - r) / cell_size))
        max_y_cell = int(min(grid_size - 1, (y + r) / cell_size))

        candidates = []
        for gx in range(min_x_cell, max_x_cell + 1):
            for gy in range(min_y_cell, max_y_cell + 1):
                if (gx, gy) in grid:
                    candidates.extend(grid[(gx, gy)])

        # Remove self
        candidates = [idx for idx in candidates if idx != circle_idx]
        return candidates

    def is_valid_configuration_fast(circles_array):
        """Fast validation using spatial hashing"""
        n = len(circles_array)

        # Check containment first
        for circle in circles_array:
            if not check_containment(circle):
                return False

        # Build spatial grid
        grid_size = 15
        grid, cell_size = build_grid(circles_array, grid_size)

        # Check collisions using spatial hashing
        for i in range(n):
            circle_i = circles_array[i]
            candidates = get_candidates_from_grid(i, grid, cell_size, circles_array, grid_size)

            for j in candidates:
                if check_collision(circle_i, circles_array[j]):
                    return False

        return True

    def evaluate_fitness_fast(circles_array):
        """Fast fitness evaluation with early termination"""
        if not is_valid_configuration_fast(circles_array):
            return -np.inf

        return get_total_radius(circles_array)

    def is_valid_configuration(circles_array):
        """Check if configuration is valid (no overlaps, fully contained)"""
        n = len(circles_array)

        # Check containment
        for circle in circles_array:
            if not check_containment(circle):
                return False

        # Check collisions
        for i in range(n):
            for j in range(i+1, n):
                if check_collision(circles_array[i], circles_array[j]):
                    return False

        return True

    def evaluate_fitness(circles_array):
        """Fitness function that heavily penalizes invalid configurations"""
        if not is_valid_configuration(circles_array):
            return -np.inf  # Penalize invalid configurations heavily

        return get_total_radius(circles_array)

    def initialize_with_voronoi():
        """Initialize circle positions using Voronoi-based approach"""
        # Generate initial points using a grid-like pattern with some randomness
        n_points = 64  # More points for better Voronoi coverage
        points = []

        # Create a systematic point distribution
        for i in range(8):
            for j in range(8):
                x = 0.1 + i * 0.125 + np.random.uniform(-0.02, 0.02)
                y = 0.1 + j * 0.125 + np.random.uniform(-0.02, 0.02)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append([x, y])

        points = np.array(points)

        # Create Voronoi diagram
        try:
            vor = Voronoi(points)

            # Get Voronoi vertices as candidate centers
            candidates = []
            for vertex in vor.vertices:
                if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                    candidates.append(vertex)

            # If we have enough candidates, sample them; otherwise use original points
            if len(candidates) >= 32:
                selected_indices = np.random.choice(len(candidates), 32, replace=False)
                centers = np.array([candidates[i] for i in selected_indices])
            else:
                # Fall back to sampling original points
                selected_indices = np.random.choice(len(points), 32, replace=False)
                centers = points[selected_indices]

        except:
            # Fallback to simple random initialization
            centers = np.random.rand(32, 2)

        # Initialize with small radii
        circles = np.zeros((32, 3))
        circles[:, 0] = centers[:, 0]
        circles[:, 1] = centers[:, 1]
        circles[:, 2] = 0.02  # Small initial radius

        return circles

    def mutate(circles_array, mutation_rate=0.1, max_radius_change=0.02):
        """Mutate the circles array"""
        mutated = circles_array.copy()

        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Mutate center position
                mutated[i, 0] += np.random.normal(0, 0.01)
                mutated[i, 1] += np.random.normal(0, 0.01)

                # Clamp positions to valid range
                mutated[i, 0] = np.clip(mutated[i, 0], 0, 1)
                mutated[i, 1] = np.clip(mutated[i, 1], 0, 1)

                # Mutate radius
                mutated[i, 2] += np.random.normal(0, max_radius_change/4)
                mutated[i, 2] = max(0.001, mutated[i, 2])  # Ensure positive radius

        return mutated

    def crossover(parent1, parent2, crossover_rate=0.3):
        """Crossover two parent solutions"""
        if np.random.random() > crossover_rate:
            return parent1.copy()

        child = parent1.copy()
        for i in range(len(child)):
            if np.random.random() < 0.5:
                child[i] = parent2[i].copy()
        return child

    def optimize_single_circle(circles_array, idx, max_iter=50):
        """Optimize a single circle's position and radius"""
        original = circles_array[idx].copy()
        best = circles_array.copy()
        best_fitness = evaluate_fitness(best)

        for _ in range(max_iter):
            # Try small perturbations
            test = circles_array.copy()
            test[idx, 0] += np.random.normal(0, 0.005)
            test[idx, 1] += np.random.normal(0, 0.005)
            test[idx, 2] += np.random.normal(0, 0.002)

            # Clamp to bounds
            test[idx, 0] = np.clip(test[idx, 0], 0, 1)
            test[idx, 1] = np.clip(test[idx, 1], 0, 1)
            test[idx, 2] = max(0.001, test[idx, 2])

            # Check if it's valid
            if is_valid_configuration(test):
                fitness = evaluate_fitness(test)
                if fitness > best_fitness:
                    best = test.copy()
                    best_fitness = fitness

        return best

    # Main evolutionary algorithm
    population_size = 20
    generations = 300
    elite_size = 4

    # Initialize population
    population = []
    for _ in range(population_size):
        circles = initialize_with_voronoi()
        population.append(circles)

    # Evolution loop
    best_fitness = float('-inf')
    best_solution = None

    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for i, circles in enumerate(population):
            fitness = evaluate_fitness(circles)
            fitness_scores.append((fitness, i))

        # Sort by fitness
        fitness_scores.sort(reverse=True)

        # Track best solution
        current_best_fitness = fitness_scores[0][0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[fitness_scores[0][1]].copy()

        # Print progress
        if gen % 50 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness:.6f}")

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        for i in range(elite_size):
            new_population.append(population[fitness_scores[i][1]])

        # Generate offspring through selection and crossover
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = [(fitness_scores[i][0], i) for i in tournament_indices]
            tournament_fitness.sort(reverse=True)

            parent1_idx = tournament_fitness[0][1]
            parent2_idx = tournament_fitness[1][1]

            # Crossover
            child = crossover(population[parent1_idx], population[parent2_idx])

            # Mutation
            child = mutate(child)

            # Local optimization for some children
            if np.random.random() < 0.3:
                # Optimize each circle individually
                for i in range(len(child)):
                    child = optimize_single_circle(child, i)

            new_population.append(child)

        population = new_population

    # Final refinement
    if best_solution is not None:
        # Apply final local optimization to best solution
        refined = best_solution.copy()
        for i in range(len(refined)):
            refined = optimize_single_circle(refined, i, max_iter=100)

        # Ensure final validity
        if is_valid_configuration(refined):
            return refined
        else:
            # Return the last valid solution found
            return best_solution
    else:
        # Return the best initialization
        return initialize_with_voronoi()

# EVOLVE-BLOCK-END