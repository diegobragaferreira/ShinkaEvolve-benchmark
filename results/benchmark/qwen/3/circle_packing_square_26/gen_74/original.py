# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
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
    """Check if any circles overlap using efficient spatial indexing."""
    n = len(circles)
    if n < 2:
        return True

    # Use cKDTree for efficient nearest neighbor search
    positions = circles[:, :2]
    radii = circles[:, 2]

    try:
        # Build spatial tree once
        tree = cKDTree(positions)

        # Query pairs within sum of radii distance
        pairs = tree.query_pairs(r=2.0, output_type='ndarray')

        # Check each potential overlapping pair
        for i, j in pairs:
            if i < j:  # Avoid duplicate checking
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < radii[i] + radii[j]:
                    return False

    except Exception:
        # Fallback to original method if tree fails
        # Calculate pairwise distances
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

    # Generate improved Voronoi-like distribution using a more sophisticated approach
    def generate_better_voronoi_points(n_points: int) -> np.ndarray:
        """Generate better distributed points using Poisson disk sampling with grid refinement."""
        # Use a combination of grid sampling and Poisson disk refinement
        points = []

        # Start with a coarse grid
        grid_size = max(3, int(np.ceil(np.sqrt(n_points))))
        spacing = 1.0 / (grid_size + 1)

        # Add points on grid with some jitter
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n_points:
                    break
                x = (j + 1) * spacing + np.random.uniform(-spacing/3, spacing/3)
                y = (i + 1) * spacing + np.random.uniform(-spacing/3, spacing/3)
                # Ensure points stay within bounds
                x = np.clip(x, spacing, 1 - spacing)
                y = np.clip(y, spacing, 1 - spacing)
                points.append([x, y])

        # Fill remaining points with more random sampling
        # but biased towards less dense areas by using rejection sampling
        while len(points) < n_points:
            x = np.random.random()
            y = np.random.random()
            points.append([x, y])

        return np.array(points[:n_points])

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
                # Use a more conservative approach for better initial configurations
                # Radius should be about 1/4 of minimum neighbor distance, bounded by containment
                proposed_radius = min(min_distance / 4.0, max_allowable_radius * 0.6)
            else:
                # Fallback if we can't compute distance properly
                proposed_radius = max_allowable_radius * 0.4

            # Clamp radius to reasonable bounds
            radius = max(0.001, min(proposed_radius, 0.4))

            circles[i] = [points[i][0], points[i][1], radius]

        # If valid, add to population
        if check_containment(circles) and check_overlap(circles):
            population.append(circles)
        else:
            # More robust fallback approach - direct grid-based initialization
            circles = np.zeros((n_circles, 3))

            # Use a more structured approach for grid
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = rows
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)

            # Use tighter spacing for better distribution
            radius = min(spacing_x, spacing_y) * 0.3

            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/8, spacing_x/8)
                    y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/8, spacing_y/8)
                    circles[idx] = [x, y, radius]
                    idx += 1

            # Final validation and refinement
            if check_containment(circles) and check_overlap(circles):
                population.append(circles)
            else:
                # Last resort - create a configuration that's guaranteed to be valid
                circles = np.zeros((n_circles, 3))
                # Place circles in a spiral pattern to get more even distribution
                angle_step = 2 * np.pi / n_circles
                center = 0.5
                radius_factor = 0.4  # To keep within bounds

                for i in range(n_circles):
                    angle = i * angle_step
                    # Spiral pattern with varying radii to spread out better
                    radial_factor = 0.8 + 0.2 * np.sin(i * 0.5)
                    x = center + radius_factor * radial_factor * np.cos(angle)
                    y = center + radius_factor * radial_factor * np.sin(angle)
                    # Make radii progressively smaller to fit more circles
                    r = 0.08 - (i * 0.002)  # Decreasing radii
                    r = max(0.01, r)  # Minimum radius
                    circles[i] = [x, y, r]

                # If still invalid, just use uniform small radii
                if not check_containment(circles) or not check_overlap(circles):
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
            if check_containment(child1) and check_overlap(child1):
                new_population.append(child1)
            else:
                # If still invalid, apply specialized repair
                repaired_child = specialize_repair(parent1, parent2)
                new_population.append(repaired_child)

            if len(new_population) < pop_size and check_containment(child2) and check_overlap(child2):
                new_population.append(child2)
            elif len(new_population) < pop_size:
                # Try to fix second child
                repaired_child = specialize_repair(parent1, parent2)
                new_population.append(repaired_child)

        population = new_population[:pop_size]

    return best_solution

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
        if check_overlap(refined):
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
    fit1 = fitness(parent1)
    fit2 = fitness(parent2)
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
        if check_containment(attempt) and check_overlap(attempt):
            fit = fitness(attempt)
            if fit > best_fitness:
                best_fitness = fit
                best_attempt = attempt

    return best_attempt

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