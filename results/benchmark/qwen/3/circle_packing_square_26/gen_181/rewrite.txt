# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Spatial grid parameters for efficient overlap checking
GRID_SIZE = 20  # Grid cells per dimension

def create_spatial_grid(circles: np.ndarray) -> dict:
    """Create a spatial grid for efficient overlap checking."""
    grid = {}

    for i, (x, y, r) in enumerate(circles):
        # Determine which grid cells this circle touches
        min_x_cell = max(0, int((x - r) * GRID_SIZE))
        max_x_cell = min(GRID_SIZE - 1, int((x + r) * GRID_SIZE))
        min_y_cell = max(0, int((y - r) * GRID_SIZE))
        max_y_cell = min(GRID_SIZE - 1, int((y + r) * GRID_SIZE))

        for gx in range(min_x_cell, max_x_cell + 1):
            for gy in range(min_y_cell, max_y_cell + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)

    return grid

def check_overlap_efficient(circles: np.ndarray, grid: dict = None) -> bool:
    """Check if any circles overlap using spatial grid for improved efficiency."""
    if grid is None:
        grid = create_spatial_grid(circles)

    n = len(circles)

    # For each cell in the grid, check if any pairs of circles overlap
    for cell, circle_indices in grid.items():
        # Only check pairs within the same grid cell
        for i in range(len(circle_indices)):
            idx1 = circle_indices[i]
            x1, y1, r1 = circles[idx1]

            for j in range(i + 1, len(circle_indices)):
                idx2 = circle_indices[j]
                x2, y2, r2 = circles[idx2]

                # Calculate distance between circle centers
                dx = x1 - x2
                dy = y1 - y2
                distance_squared = dx*dx + dy*dy

                # Check if circles overlap
                if distance_squared < (r1 + r2)**2:
                    return False

    return True

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained within the unit square."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def is_valid(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping."""
    n = len(circles)

    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints using spatial grid
    grid = create_spatial_grid(circles)
    return check_overlap_efficient(circles, grid)

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a solution, higher is better."""
    if not is_valid(circles):
        # Apply penalty for constraint violations
        penalty = 0

        # Boundary penalty
        boundary_violations = 0
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0:
                boundary_violations += (r - x)**2
            if x + r > 1:
                boundary_violations += (x + r - 1)**2
            if y - r < 0:
                boundary_violations += (r - y)**2
            if y + r > 1:
                boundary_violations += (y + r - 1)**2

        penalty += 10000.0 * boundary_violations

        # Overlap penalty
        overlap_penalty = 0
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    overlap_penalty += (r1 + r2 - distance)**2

        penalty += 10000.0 * overlap_penalty

        return -penalty

    return calculate_sum_radii(circles)

def generate_better_voronoi_points(n_points: int) -> np.ndarray:
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

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with enhanced Voronoi-based distribution for better spatial coverage."""
    population = []

    for _ in range(pop_size):
        # Generate improved Voronoi-like points
        points = generate_better_voronoi_points(n_circles)

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
        if is_valid(circles):
            population.append(circles)
        else:
            # More robust fallback approach - direct grid-based initialization
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
            if is_valid(circles):
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
                if not is_valid(circles):
                    for i in range(n_circles):
                        circles[i] = [0.5, 0.5, 0.01]

                population.append(circles)

    return population

def mutate(circles: np.ndarray, generation: int = 0, max_generations: int = 1000) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate and dual mutation strategy."""
    # Adaptive mutation rate that decreases over generations
    mutation_rate_start = 0.20
    mutation_rate_end = 0.02

    # Sigmoidal decay function
    mutation_rate = mutation_rate_end + (mutation_rate_start - mutation_rate_end) * (
        1 / (1 + np.exp(10 * (generation / max_generations - 0.5)))
    )

    mutated = circles.copy()
    n = len(mutated)

    # Two-stage mutation:
    # Stage 1: Large-scale mutations for exploration (higher variance)
    # Stage 2: Fine-tuning mutations for exploitation (lower variance)

    for i in range(n):
        if random.random() < mutation_rate:
            # Determine mutation type based on generation progress
            # Early generations: favor larger changes for exploration
            # Later generations: favor smaller changes for exploitation
            if generation < max_generations * 0.3:
                # Exploration phase - larger mutations
                mutation_strength = 0.05
                mutation_type = "large"
            else:
                # Exploitation phase - smaller mutations
                mutation_strength = 0.01
                mutation_type = "small"

            # Randomly choose what to mutate with bias towards position for early stages
            if generation < max_generations * 0.5:
                # Prioritize position changes in early stages
                choice = random.choices([0, 1, 2], weights=[0.5, 0.5, 0.1])[0]
            else:
                # Equal weights in later stages
                choice = random.randint(0, 2)

            if choice == 0:  # Mutate x position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, mutation_strength),
                                      mutated[i, 2], 1 - mutated[i, 2])
            elif choice == 1:  # Mutate y position
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, mutation_strength),
                                      mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                # Apply bounded mutation to radius
                mutated[i, 2] = np.clip(mutated[i, 2] * np.random.normal(1, 0.1), 0.001, 0.4)

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

def refine_invalid_configuration(circles: np.ndarray) -> np.ndarray:
    """Apply geometric corrections to make configuration valid."""
    refined = circles.copy()

    # Apply containment correction first
    for i in range(len(refined)):
        x, y, r = refined[i]

        # Ensure containment with boundary padding
        x = np.clip(x, r + 0.001, 1 - r - 0.001)
        y = np.clip(y, r + 0.001, 1 - r - 0.001)

        refined[i] = [x, y, r]

    # Apply overlap correction by reducing radii where necessary
    # We do this iteratively to ensure we don't create new overlaps
    for _ in range(5):  # Limit iterations to prevent infinite loops
        if check_overlap_efficient(refined):
            break

        # Reduce radii of circles that cause overlaps
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Reduce radius slightly to allow for overlap resolution
            refined[i, 2] = max(0.001, r * 0.95)

    return refined

def specialize_repair(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Apply specialized repair strategies when standard methods fail."""
    # Try several repair approaches
    attempts = []

    # Approach 1: Take the better parent
    fit1 = calculate_sum_radii(parent1)
    fit2 = calculate_sum_radii(parent2)
    if fit1 >= fit2:
        attempts.append(parent1.copy())
    else:
        attempts.append(parent2.copy())

    # Approach 2: Create a hybrid that's more constrained
    hybrid = parent1.copy()
    # Mix positions more conservatively
    for i in range(len(hybrid)):
        if random.random() < 0.3:  # 30% chance to use parent2's data
            hybrid[i, 0] = parent2[i, 0]
            hybrid[i, 1] = parent2[i, 1]
            # Keep parent1's radius for stability
    attempts.append(hybrid)

    # Approach 3: Use a simple grid arrangement for guarantee
    grid_arrangement = np.zeros((len(parent1), 3))
    rows = int(np.ceil(np.sqrt(len(parent1))))
    cols = rows
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    radius = min(spacing_x, spacing_y) * 0.25
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= len(parent1):
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            grid_arrangement[idx] = [x, y, radius]
            idx += 1
    attempts.append(grid_arrangement)

    # Test all approaches and return the best valid one
    best_attempt = attempts[0]
    best_fitness = -float('inf')

    for attempt in attempts:
        if is_valid(attempt):
            fit = calculate_sum_radii(attempt)
            if fit > best_fitness:
                best_fitness = fit
                best_attempt = attempt

    return best_attempt

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
            if is_valid(circles):
                fit = calculate_sum_radii(circles)
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
        # Also keep top 20% as elite to preserve quality
        elite_count = pop_size // 5
        sorted_indices = np.argsort(fitnesses)[::-1][:elite_count]
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

            # Special refinement step for children that may have become invalid
            # Apply geometric corrections for containment and overlap
            child1 = refine_invalid_configuration(child1)
            child2 = refine_invalid_configuration(child2)

            # Ensure validity
            if is_valid(child1):
                new_population.append(child1)
            else:
                # If still invalid, apply specialized repair
                repaired_child = specialize_repair(parent1, parent2)
                new_population.append(repaired_child)

            if len(new_population) < pop_size and is_valid(child2):
                new_population.append(child2)
            elif len(new_population) < pop_size:
                # Try to fix second child
                repaired_child = specialize_repair(parent1, parent2)
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
    if circles is None or not is_valid(circles):
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