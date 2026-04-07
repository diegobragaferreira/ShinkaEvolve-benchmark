# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time

def calculate_max_radius_at_position(circles, index, x, y, width, height):
    """Fast calculation of maximum radius for circle at given position without overlapping others."""
    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)

    # Vectorized overlap checking for efficiency
    if len(circles) > 1:
        # Get other circles' positions and radii
        other_positions = circles[[i for i in range(len(circles)) if i != index], :2]
        other_radii = circles[[i for i in range(len(circles)) if i != index], 2]

        # Calculate distances to all other circles
        distances = np.sqrt(np.sum((other_positions - [x, y])**2, axis=1))

        # Maximum radius that avoids overlap with all other circles
        max_radius_overlap = np.min(distances - other_radii)

        max_radius = min(max_radius_bound, max_radius_overlap)
    else:
        max_radius = max_radius_bound

    return max(max_radius, 0.001)

def initialize_with_voronoi_based_placement(n_circles, width, height):
    """Initialize circles using Voronoi-based strategic placement"""
    np.random.seed(42)

    # Start with corner points for boundary coverage
    corner_points = [
        [0.1, 0.1],           # Bottom-left
        [width-0.1, 0.1],     # Bottom-right
        [0.1, height-0.1],    # Top-left
        [width-0.1, height-0.1], # Top-right
    ]

    # Add edge midpoints
    edge_points = [
        [width/2, 0.1],       # Bottom-middle
        [width/2, height-0.1], # Top-middle
        [0.1, height/2],      # Left-middle
        [width-0.1, height/2], # Right-middle
    ]

    # Combine corner and edge points
    init_points = corner_points + edge_points

    # Fill remaining slots with hexagonal grid pattern
    rows = 5
    cols = 5
    estimated_radius = 0.08

    hex_points = []
    for i in range(rows):
        for j in range(cols):
            if len(hex_points) >= (n_circles - len(init_points)):
                break
            x = j * 2 * estimated_radius + (i % 2) * estimated_radius
            y = i * np.sqrt(3) * estimated_radius

            # Add point if it's within bounds
            if 0.01 <= x <= width - 0.01 and 0.01 <= y <= height - 0.01:
                hex_points.append([x, y])
        if len(hex_points) >= (n_circles - len(init_points)):
            break

    # Combine all initial points
    all_init_points = init_points + hex_points[:n_circles-len(init_points)]

    # Add any remaining points randomly if needed
    while len(all_init_points) < n_circles:
        x = np.random.uniform(0.05, width - 0.05)
        y = np.random.uniform(0.05, height - 0.05)
        all_init_points.append([x, y])

    # Initialize with small radii
    circles = np.zeros((n_circles, 3))
    for i in range(n_circles):
        circles[i] = [all_init_points[i][0], all_init_points[i][1], 0.01]

    return circles

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.2, 0.8  # Optimized aspect ratio

    # Set seed for reproducibility
    np.random.seed(42)

    # Phase 1: Voronoi-based initialization for better spatial distribution
    n_circles = 21
    circles = initialize_with_voronoi_based_placement(n_circles, width, height)

    # Phase 2: Enhanced evolutionary optimization with adaptive parameters
    best_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Evolutionary parameters
    population_size = 25
    generations = 150
    mutation_rate = 0.15
    elite_size = 3

    # Adaptive parameters
    max_iterations = 1500
    convergence_threshold = 1e-6
    patience = 80

    start_time = time.time()

    # Generate initial population
    population = [circles.copy()]
    for _ in range(population_size - 1):
        # Create mutated version of the best solution
        mutant = circles.copy()
        # Mutate a few circles
        mutate_indices = np.random.choice(n_circles, size=max(1, int(mutation_rate * n_circles)), replace=False)
        for idx in mutate_indices:
            # Add small random perturbation
            dx = np.random.uniform(-0.03, 0.03)
            dy = np.random.uniform(-0.03, 0.03)

            new_x = max(0.05, min(width - 0.05, mutant[idx, 0] + dx))
            new_y = max(0.05, min(height - 0.05, mutant[idx, 1] + dy))

            # Recalculate radius at new position
            new_r = calculate_max_radius_at_position(mutant, idx, new_x, new_y, width, height)

            mutant[idx] = [new_x, new_y, new_r]
        population.append(mutant)

    # Evolutionary loop
    last_improvement = 0

    for gen in range(generations):
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break

        # Evaluate fitness of each individual in population
        fitness_scores = []
        for individual in population:
            fitness = np.sum(individual[:, 2])
            fitness_scores.append(fitness)

        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Update best solution
        current_best_fitness = sorted_fitness[0]
        if current_best_fitness > best_sum:
            best_sum = current_best_fitness
            best_circles = sorted_population[0].copy()
            last_improvement = gen

        # Check for convergence
        if gen - last_improvement > patience:
            break

        # Elitism: keep best individuals
        new_population = sorted_population[:elite_size]

        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(sorted_population), size=tournament_size, replace=False)
            tournament_fitness = [sorted_fitness[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            parent = sorted_population[winner_index].copy()

            # Create offspring by mutation
            offspring = parent.copy()

            # Apply random mutations to some circles
            mutate_count = max(1, int(mutation_rate * n_circles))
            mutate_indices = np.random.choice(n_circles, size=mutate_count, replace=False)

            for idx in mutate_indices:
                # Random perturbation
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)

                new_x = max(0.05, min(width - 0.05, offspring[idx, 0] + dx))
                new_y = max(0.05, min(height - 0.05, offspring[idx, 1] + dy))

                # Recalculate radius at new position
                new_r = calculate_max_radius_at_position(offspring, idx, new_x, new_y, width, height)

                offspring[idx] = [new_x, new_y, new_r]

            new_population.append(offspring)

        population = new_population[:population_size]

    # Phase 3: Adaptive local refinement with multi-scale search
    max_local_iterations = 500
    last_improvement = 0
    tolerance = 1e-6

    for iteration in range(max_local_iterations):
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break

        current_sum = np.sum(circles[:, 2])

        # Multi-scale adaptive search with decreasing step sizes
        scales = [0.1, 0.05, 0.02, 0.01]  # Different step sizes
        scale_idx = min(iteration // 50, len(scales) - 1)  # Gradually decrease scale

        step_size = scales[scale_idx]

        improved = False

        # Try to improve each circle
        for i in range(n_circles):
            current_x, current_y, current_r = circles[i]

            # Track best improvement
            best_pos = [current_x, current_y, current_r]
            best_radius = current_r

            # Search in a grid with adaptive scale
            search_range = step_size * 2
            search_points = np.arange(-search_range, search_range + step_size/2, step_size)

            for dx in search_points:
                for dy in search_points:
                    new_x = current_x + dx
                    new_y = current_y + dy

                    # Keep within bounds
                    new_x = max(0.05, min(width - 0.05, new_x))
                    new_y = max(0.05, min(height - 0.05, new_y))

                    # Calculate max radius at new position
                    new_r = calculate_max_radius_at_position(circles, i, new_x, new_y, width, height)

                    # If this gives us a better radius
                    if new_r > best_radius:
                        best_radius = new_r
                        best_pos = [new_x, new_y, new_r]

            # Apply improvement if found
            if best_radius > circles[i][2]:
                circles[i] = best_pos
                improved = True

        # Check for convergence
        new_sum = np.sum(circles[:, 2])
        if new_sum > best_sum:
            best_sum = new_sum
            best_circles = circles.copy()
            last_improvement = iteration
        elif abs(new_sum - current_sum) < tolerance:
            break
        elif iteration - last_improvement > patience:
            break

    # Final validation
    for i in range(n_circles):
        # Ensure minimum radius
        circles[i][2] = max(circles[i][2], 0.001)

        # Ensure circles stay within bounds
        circles[i][0] = np.clip(circles[i][0], 0.001, width - 0.001)
        circles[i][1] = np.clip(circles[i][1], 0.001, height - 0.001)

    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")