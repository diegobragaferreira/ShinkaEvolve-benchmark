# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Core optimization parameters
GRID_SIZE = 30  # Increased grid resolution for better spatial indexing
POP_SIZE = 60   # Increased population size
MAX_GEN = 800   # More generations for deeper exploration
ELITE_COUNT = 12  # More elite individuals preserved

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

def fitness(circles: np.ndarray, generation: int = 0, total_generations: int = 1000) -> float:
    """Calculate fitness as sum of radii with adaptive penalty scaling."""
    # Check if solution is valid
    if not check_containment(circles) or check_overlap_efficient(circles):
        # Apply adaptive penalty for constraint violations
        penalty = 0

        # Calculate severity of violations
        boundary_violations = 0
        overlap_violations = 0

        # Boundary violations
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

        # Overlap violations
        grid = create_spatial_grid(circles)
        if check_overlap_efficient(circles, grid):
            # Simple overlap check if spatial grid fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        overlap_violations += (r1 + r2 - distance)**2

        # Adaptive penalty scaling based on generation and violation severity
        gen_factor = min(1.0, generation / total_generations)

        # Early generations: lighter penalties to encourage exploration
        # Later generations: heavier penalties to enforce constraints
        penalty_weight = 100000 * (0.5 + 0.5 * gen_factor)

        # Severity-based scaling - more severe violations get heavier penalties
        total_boundary_penalty = boundary_violations * penalty_weight
        total_overlap_penalty = overlap_violations * penalty_weight * 10  # Overlaps more critical

        penalty = total_boundary_penalty + total_overlap_penalty

        return -penalty - 1000000

    return np.sum(circles[:, 2])

def generate_improved_voronoi_points(n_points: int) -> np.ndarray:
    """Generate better distributed points using a more sophisticated Voronoi approach."""
    # Create a hexagonal grid pattern for better spatial distribution
    grid_size = int(np.ceil(np.sqrt(n_points)))

    # Create a honeycomb-like structure
    points = []
    hex_radius = 0.8 / (grid_size + 2)  # Adjusted for better spacing

    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n_points:
                break
            # Offset every other row for hexagonal arrangement
            x_offset = (j + 0.5 * (i % 2)) * hex_radius * 2
            y_offset = i * hex_radius * np.sqrt(3)

            # Add some randomness to create more natural distribution
            x = 0.1 + x_offset + np.random.uniform(-hex_radius*0.2, hex_radius*0.2)
            y = 0.1 + y_offset + np.random.uniform(-hex_radius*0.2, hex_radius*0.2)

            # Ensure points stay within bounds
            x = np.clip(x, hex_radius, 1 - hex_radius)
            y = np.clip(y, hex_radius, 1 - hex_radius)

            points.append([x, y])

    # Fill remaining points randomly but with some bias toward underpopulated areas
    while len(points) < n_points:
        points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])

    return np.array(points[:n_points])

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with enhanced Voronoi-based distribution for better spatial coverage."""
    population = []

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
            # Base radius on available space to neighbors, with better constraint consideration
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
        if check_containment(circles) and check_overlap_efficient(circles):
            population.append(circles)
        else:
            # Robust fallback approach - direct grid-based initialization
            circles = np.zeros((n_circles, 3))

            # Use systematic grid distribution
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = rows
            spacing_x = 0.8 / (cols + 1)  # Leave margin for boundaries
            spacing_y = 0.8 / (rows + 1)

            # Use slightly more compact spacing for better density
            radius = min(spacing_x, spacing_y) * 0.4

            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x = 0.1 + (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                    y = 0.1 + (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                    circles[idx] = [x, y, radius]
                    idx += 1

            # Final validation and refinement
            if check_containment(circles) and check_overlap_efficient(circles):
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
                    r = 0.06 - (i * 0.001)  # Decreasing radii
                    r = max(0.01, r)  # Minimum radius
                    circles[i] = [x, y, r]

                # If still invalid, just use uniform small radii
                if not check_containment(circles) or not check_overlap_efficient(circles):
                    for i in range(n_circles):
                        circles[i] = [0.5, 0.5, 0.01]

                population.append(circles)

    return population

def mutate(circles: np.ndarray, generation: int = 0, max_generations: int = 1000,
           diversity_factor: float = 1.0) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate and smart mutation strategy."""
    # Adaptive mutation rate that decreases over generations (but with steeper curve)
    mutation_rate_start = 0.30
    mutation_rate_end = 0.01

    # Steeper sigmoidal decay for quicker convergence
    mutation_rate = mutation_rate_end + (mutation_rate_start - mutation_rate_end) * (
        1 / (1 + np.exp(20 * (generation / max_generations - 0.5)))
    )

    # Adjust based on population diversity
    mutation_rate *= diversity_factor

    mutated = circles.copy()
    n = len(mutated)

    # Dual mutation strategy with better phase control
    for i in range(n):
        if random.random() < mutation_rate:
            # Determine mutation phase based on generation
            gen_progress = generation / max_generations

            if gen_progress < 0.35:  # Early exploration phase
                # Larger mutations with more position change
                mutation_strength = 0.08
                # Prefer position changes for exploration
                choice = random.choices([0, 1, 2], weights=[0.7, 0.7, 0.2])[0]
            elif gen_progress < 0.75:  # Mid-exploitation phase
                # Moderate mutations
                mutation_strength = 0.04
                # Balanced approach
                choice = random.choices([0, 1, 2], weights=[0.5, 0.5, 0.5])[0]
            else:  # Late exploitation phase
                # Fine-tuning mutations
                mutation_strength = 0.02
                # Prefer radius changes for fine-tuning
                choice = random.choices([0, 1, 2], weights=[0.2, 0.2, 0.8])[0]

            if choice == 0:  # Mutate x position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, mutation_strength),
                                      mutated[i, 2] + 0.001, 1 - mutated[i, 2] - 0.001)
            elif choice == 1:  # Mutate y position
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, mutation_strength),
                                      mutated[i, 2] + 0.001, 1 - mutated[i, 2] - 0.001)
            else:  # Mutate radius
                # Use log-normal mutation for better radius control
                log_factor = np.random.normal(0, 0.2)
                mutated[i, 2] = np.clip(mutated[i, 2] * np.exp(log_factor), 0.001, 0.4)

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parent solutions with fitness awareness."""
    # Fitness-aware crossover with preference for better parents
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover points for each circle
    for i in range(len(parent1)):
        # Use fitness-based probability for inheritance
        fit1 = parent1[i, 2]  # Radius as proxy for fitness
        fit2 = parent2[i, 2]

        # Calculate probability of inheriting from parent1
        prob_parent1 = fit1 / (fit1 + fit2) if (fit1 + fit2) > 0 else 0.5

        if random.random() < prob_parent1:
            # Inherit from parent1
            pass
        else:
            # Inherit from parent2
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def select_parents(population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Select two parents using tournament selection with diversity consideration."""
    tournament_size = 5  # Increased tournament size for better selection pressure

    # Select first parent with better probability of selecting high fitness
    candidates1 = random.sample(range(len(population)), tournament_size)
    fitness_scores1 = [fitnesses[i] for i in candidates1]
    best_idx1 = candidates1[fitness_scores1.index(max(fitness_scores1))]
    parent1 = population[best_idx1]

    # Select second parent, ensuring it's different from first
    candidates2 = random.sample(range(len(population)), tournament_size)
    fitness_scores2 = [fitnesses[i] for i in candidates2]
    best_idx2 = candidates2[fitness_scores2.index(max(fitness_scores2))]

    # Ensure we don't pick the same parent twice
    if best_idx1 == best_idx2 and len(population) > 1:
        # If they're the same, find a different one
        for i in range(len(population)):
            if i != best_idx1:
                parent2 = population[i]
                break
        else:
            # Fallback if all are the same (should rarely happen)
            parent2 = population[random.choice([i for i in range(len(population)) if i != best_idx1])]
    else:
        parent2 = population[best_idx2]

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
    for _ in range(15):  # Increase iterations for better overlap resolution
        if check_overlap_efficient(refined):
            break

        # Reduce radii of circles that cause overlaps
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Reduce radius more aggressively to resolve overlaps quickly
            refined[i, 2] = max(0.001, r * 0.92)

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

    # Approach 2: Create a hybrid
    hybrid = parent1.copy()
    for i in range(len(hybrid)):
        if random.random() < 0.3:  # 30% chance to use parent2's data
            hybrid[i, 0] = parent2[i, 0]
            hybrid[i, 1] = parent2[i, 1]
    attempts.append(hybrid)

    # Approach 3: Grid arrangement
    grid_arrangement = np.zeros((len(parent1), 3))
    rows = int(np.ceil(np.sqrt(len(parent1))))
    cols = rows
    spacing_x = 0.8 / (cols + 1)
    spacing_y = 0.8 / (rows + 1)
    radius = min(spacing_x, spacing_y) * 0.3
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= len(parent1):
                break
            x = 0.1 + (j + 1) * spacing_x
            y = 0.1 + (i + 1) * spacing_y
            grid_arrangement[idx] = [x, y, radius]
            idx += 1
    attempts.append(grid_arrangement)

    # Test all approaches and return the best valid one
    best_attempt = attempts[0]
    best_fitness = -float('inf')

    for attempt in attempts:
        if check_containment(attempt) and check_overlap_efficient(attempt):
            fit = fitness(attempt)
            if fit > best_fitness:
                best_fitness = fit
                best_attempt = attempt

    return best_attempt

def calculate_diversity(population: List[np.ndarray]) -> float:
    """Calculate population diversity as average distance between individuals."""
    if len(population) < 2:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(len(population)):
        for j in range(i+1, len(population)):
            # Calculate average distance between circles in different individuals
            distances = 0.0
            for k in range(len(population[i])):
                dist = np.sqrt(np.sum((population[i][k] - population[j][k])**2))
                distances += dist
            total_distance += distances / len(population[i])
            count += 1

    return total_distance / count if count > 0 else 0.0

def optimize_circles_evolutionary(max_generations: int = 800, pop_size: int = 60) -> np.ndarray:
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
            if check_containment(circles) and check_overlap_efficient(circles):
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

        # Calculate diversity for adaptive mutation rate
        diversity = calculate_diversity(population)
        # Diversity is inversely related to how much variety there is
        # When diversity is low, we want to increase mutation to restore variety
        diversity_factor = max(0.7, 1.0 - diversity * 8)  # More aggressive boost at low diversity

        # Keep best individuals (elitism) - increased from 1/5 to 1/4
        sorted_indices = np.argsort(fitnesses)[::-1][:ELITE_COUNT]
        for idx in sorted_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1, parent2 = select_parents(population, fitnesses)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation with generation info and diversity factor
            child1 = mutate(child1, generation, max_generations, diversity_factor)
            child2 = mutate(child2, generation, max_generations, diversity_factor)

            # Special refinement step for children that may have become invalid
            child1 = refine_invalid_configuration(child1)
            child2 = refine_invalid_configuration(child2)

            # Ensure validity
            if check_containment(child1) and check_overlap_efficient(child1):
                new_population.append(child1)
            else:
                # If still invalid, apply specialized repair
                repaired_child = specialize_repair(parent1, parent2)
                new_population.append(repaired_child)

            if len(new_population) < pop_size and check_containment(child2) and check_overlap_efficient(child2):
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
    # Run evolutionary optimization with improved parameters
    circles = optimize_circles_evolutionary(max_generations=MAX_GEN, pop_size=POP_SIZE)

    # Final validation
    if circles is None or not check_containment(circles) or not check_overlap_efficient(circles):
        # Fallback to a simple arrangement if optimization failed
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 0.8 / (cols + 1)
        spacing_y = 0.8 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.3

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = 0.1 + (j + 1) * spacing_x
                y = 0.1 + (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

        # Adjust last few circles to fit
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.01]

    return circles

# EVOLVE-BLOCK-END