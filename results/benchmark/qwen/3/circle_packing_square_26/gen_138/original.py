# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained within the unit square."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap(circles: np.ndarray) -> bool:
    """Check if any circles overlap."""
    n = len(circles)
    # Calculate pairwise distances
    positions = circles[:, :2]
    radii = circles[:, 2]

    # Create distance matrix
    distances = cdist(positions, positions)

    # Check for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            if dist < radii[i] + radii[j]:
                return False
    return True

def fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with enhanced Voronoi-based distribution for better spatial coverage."""
    population = []

    # Generate improved Voronoi-like distribution
    def generate_improved_voronoi_points(n_points: int) -> np.ndarray:
        """Generate better distributed points using honeycomb/grid approach."""
        # Use a hexagonal grid approach for better spatial distribution
        grid_size = int(np.ceil(np.sqrt(n_points)))

        # Create a more regular hexagonal-like distribution
        points = []
        hex_radius = 1.0 / (grid_size + 2)

        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n_points:
                    break
                # Offset every other row for hexagonal arrangement
                x_offset = (j + 0.5 * (i % 2)) * hex_radius * 2
                y_offset = i * hex_radius * np.sqrt(3)

                # Add some randomness to create more natural distribution
                x = x_offset + np.random.uniform(-hex_radius*0.3, hex_radius*0.3)
                y = y_offset + np.random.uniform(-hex_radius*0.3, hex_radius*0.3)

                # Ensure points stay within bounds
                x = np.clip(x, hex_radius, 1 - hex_radius)
                y = np.clip(y, hex_radius, 1 - hex_radius)

                points.append([x, y])

        # Fill remaining points randomly if needed
        while len(points) < n_points:
            points.append([np.random.random(), np.random.random()])

        return np.array(points[:n_points])

    for _ in range(pop_size):
        # Generate improved Voronoi-like points
        points = generate_improved_voronoi_points(n_circles)

        # Create initial circles with better radii assignment
        circles = np.zeros((n_circles, 3))

        # Assign radii with better consideration of spatial relationships
        for i in range(n_circles):
            # Calculate minimum distance to all other points (excluding self)
            distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
            distances[i] = np.inf  # Exclude self-distance
            min_distance = np.min(distances)

            # Calculate maximum allowable radius based on containment
            max_allowable_radius = min(points[i][0], points[i][1],
                                     1 - points[i][0], 1 - points[i][1])

            # Use a smarter approach for radius assignment:
            # - Base radius on available space to neighbors
            # - Consider both local density and containment constraints
            if min_distance > 0:
                # Radius should be about 1/3 of minimum neighbor distance, bounded by containment
                proposed_radius = min(min_distance / 3.0, max_allowable_radius * 0.7)
            else:
                # Fallback if we can't compute distance properly
                proposed_radius = max_allowable_radius * 0.5

            # Clamp radius to reasonable bounds
            radius = max(0.001, min(proposed_radius, 0.4))

            circles[i] = [points[i][0], points[i][1], radius]

        # If valid, add to population
        if check_containment(circles) and check_overlap(circles):
            population.append(circles)
        else:
            # More robust fallback approach
            circles = np.zeros((n_circles, 3))

            # Try a more systematic approach
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = rows
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)

            # Use slightly different spacing for better distribution
            radius = min(spacing_x, spacing_y) * 0.35

            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                    y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                    circles[idx] = [x, y, radius]
                    idx += 1

            # Final validation and refinement
            if check_containment(circles) and check_overlap(circles):
                population.append(circles)
            else:
                # Last resort - create a configuration that's guaranteed to be valid
                circles = np.zeros((n_circles, 3))
                # Place circles in a circular pattern to ensure good distribution
                angle_step = 2 * np.pi / n_circles
                center = 0.5
                radius_factor = 0.35  # To keep within bounds

                for i in range(n_circles):
                    angle = i * angle_step
                    x = center + radius_factor * np.cos(angle)
                    y = center + radius_factor * np.sin(angle)
                    # Make radii progressively smaller to fit more circles
                    r = 0.05 - (i * 0.001)  # Decreasing radii
                    r = max(0.01, r)  # Minimum radius
                    circles[i] = [x, y, r]

                # If still invalid, just use uniform small radii
                if not check_containment(circles) or not check_overlap(circles):
                    for i in range(n_circles):
                        circles[i] = [0.5, 0.5, 0.01]

                population.append(circles)

    return population

def mutate(circles: np.ndarray, generation: int = 0, max_generations: int = 1000) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate."""
    # Adaptive mutation rate that decreases over generations
    mutation_rate_start = 0.15
    mutation_rate_end = 0.01

    # Sigmoidal decay function
    mutation_rate = mutation_rate_end + (mutation_rate_start - mutation_rate_end) * (
        1 / (1 + np.exp(10 * (generation / max_generations - 0.5)))
    )

    mutated = circles.copy()
    n = len(mutated)

    for i in range(n):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate
            choice = random.randint(0, 2)
            if choice == 0:  # Mutate x position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02),
                                      mutated[i, 2], 1 - mutated[i, 2])
            elif choice == 1:  # Mutate y position
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02),
                                      mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.01), 0.001, 0.5)

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parent solutions."""
    # Simple uniform crossover
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover points for each circle
    for i in range(len(parent1)):
        if random.random() < 0.5:
            # Swap position and radius between parents
            child1[i, 0], child2[i, 0] = child2[i, 0], child1[i, 0]
            child1[i, 1], child2[i, 1] = child2[i, 1], child1[i, 1]
            child1[i, 2], child2[i, 2] = child2[i, 2], child1[i, 2]

    return child1, child2

def select_parents(population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Select two parents using tournament selection."""
    tournament_size = 3
    # Select first parent
    idx1 = random.randint(0, len(population)-1)
    best_idx = idx1
    best_fit = fitnesses[idx1]
    for _ in range(tournament_size - 1):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    parent1 = population[best_idx]

    # Select second parent
    idx2 = random.randint(0, len(population)-1)
    best_idx = idx2
    best_fit = fitnesses[idx2]
    for _ in range(tournament_size - 1):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    parent2 = population[best_idx]

    return parent1, parent2

def optimize_circles_evolutionary(max_generations: int = 1000, pop_size: int = 50) -> np.ndarray:
    """Evolutionary optimization for circle packing."""
    n = 26

    # Initialize population
    population = initialize_population(pop_size, n)
    best_solution = None
    best_fitness = -float('inf')

    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitnesses = []
        for circles in population:
            if check_containment(circles) and check_overlap(circles):
                fit = fitness(circles)
                fitnesses.append(fit)
            else:
                fitnesses.append(-1000)  # Penalize invalid solutions

        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

        # Create new population through selection, crossover, and mutation
        new_population = []

        # Keep best individuals (elitism) - increase from 1/4 to 1/2
        sorted_indices = np.argsort(fitnesses)[::-1][:pop_size//2]
        for idx in sorted_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1, parent2 = select_parents(population, fitnesses)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation with generation info
            child1 = mutate(child1, generation, max_generations)
            child2 = mutate(child2, generation, max_generations)

            # Ensure validity
            if check_containment(child1) and check_overlap(child1):
                new_population.append(child1)
            else:
                # Try to fix if invalid - use a more robust approach
                # Try to repair the child by copying from parent and making small adjustments
                repaired_child = parent1.copy()
                # Make small adjustments to the repaired copy
                if random.random() < 0.5:
                    # Adjust one circle to try to fix
                    idx = random.randint(0, len(repaired_child) - 1)
                    repaired_child[idx, 0] = np.clip(repaired_child[idx, 0] + np.random.normal(0, 0.01),
                                                   repaired_child[idx, 2], 1 - repaired_child[idx, 2])
                    repaired_child[idx, 1] = np.clip(repaired_child[idx, 1] + np.random.normal(0, 0.01),
                                                   repaired_child[idx, 2], 1 - repaired_child[idx, 2])
                new_population.append(repaired_child)

            if len(new_population) < pop_size and check_containment(child2) and check_overlap(child2):
                new_population.append(child2)
            elif len(new_population) < pop_size:
                # Try to fix second child
                repaired_child = parent2.copy()
                if random.random() < 0.5:
                    idx = random.randint(0, len(repaired_child) - 1)
                    repaired_child[idx, 0] = np.clip(repaired_child[idx, 0] + np.random.normal(0, 0.01),
                                                   repaired_child[idx, 2], 1 - repaired_child[idx, 2])
                    repaired_child[idx, 1] = np.clip(repaired_child[idx, 1] + np.random.normal(0, 0.01),
                                                   repaired_child[idx, 2], 1 - repaired_child[idx, 2])
                new_population.append(repaired_child)

        population = new_population[:pop_size]

    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run evolutionary optimization
    circles = optimize_circles_evolutionary(max_generations=500, pop_size=30)

    # Final validation
    if circles is None or not check_containment(circles) or not check_overlap(circles):
        # Fallback to a simple arrangement if optimization failed
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.3

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

        # Adjust last few circles to fit
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.01]

    return circles


# EVOLVE-BLOCK-END