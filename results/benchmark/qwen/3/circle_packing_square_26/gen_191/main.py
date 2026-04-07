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

# Optimization parameters
GRID_SIZE = 35  # Increased grid resolution for better efficiency
POP_SIZE = 50   # Population size
MAX_GEN = 800   # Generations
ELITE_COUNT = 15  # Elite individuals preserved

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

    # Early exit if no grid cells contain multiple circles
    for cell, circle_indices in grid.items():
        if len(circle_indices) > 1:
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
    # Check boundary constraints
    for i in range(len(circles)):
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
                boundary_violations += (r - x)**2 * 100000
            if x + r > 1:
                boundary_violations += (x + r - 1)**2 * 100000
            if y - r < 0:
                boundary_violations += (r - y)**2 * 100000
            if y + r > 1:
                boundary_violations += (y + r - 1)**2 * 100000

        penalty += boundary_violations

        # Overlap penalty
        overlap_penalty = 0
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    overlap_penalty += (r1 + r2 - distance)**2 * 100000

        penalty += overlap_penalty

        return -penalty - 1000000

    return calculate_sum_radii(circles)

def generate_improved_voronoi_points(n_points: int) -> np.ndarray:
    """Generate better distributed points using Voronoi plus Poisson disk sampling."""
    from scipy.spatial import Voronoi
    import numpy as np

    # Generate initial points using a combination of structured grid and Poisson disk sampling
    points = []

    # Step 1: Create a structured grid of points
    grid_size = max(5, int(np.ceil(np.sqrt(n_points * 1.5))))
    spacing = 1.0 / (grid_size + 1)

    # Generate grid points with some jitter
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n_points * 1.2:  # Generate extra points
                break
            x = (j + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2)
            y = (i + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2)
            # Ensure points are within valid range
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            points.append([x, y])

    # Step 2: Use Voronoi diagram to get better distribution
    if len(points) >= n_points:
        points_array = np.array(points[:n_points * 2])  # Get more points for Voronoi

        try:
            # Create Voronoi diagram
            vor = Voronoi(points_array)

            # Use Voronoi vertices as potential positions
            voronoi_points = []
            for vertex in vor.vertices:
                x, y = vertex
                if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                    voronoi_points.append([x, y])

            # If we got enough Voronoi points, use them; otherwise fall back to original
            if len(voronoi_points) >= n_points:
                points = voronoi_points[:n_points]
            else:
                # Mix original grid points with Voronoi points
                mixed_points = list(points_array[:n_points])
                # Add some Voronoi points if available
                for vp in voronoi_points[:min(5, n_points - len(mixed_points))]:
                    mixed_points.append(vp)
                # Fill remaining with original points
                while len(mixed_points) < n_points:
                    mixed_points.append(points_array[len(mixed_points) % len(points_array)])
                points = mixed_points[:n_points]
        except:
            # If Voronoi creation fails, use original points
            pass

    # Step 3: Apply Poisson disk sampling to further enhance distribution
    # Only apply if we don't have enough points yet
    if len(points) < n_points:
        # Use Poisson disk sampling to distribute remaining points
        poisson_points = []
        active_list = []

        # Start with a random point
        start_x = np.random.uniform(0.05, 0.95)
        start_y = np.random.uniform(0.05, 0.95)
        poisson_points.append([start_x, start_y])
        active_list.append(0)

        # Try to generate more points using Poisson disk sampling
        max_attempts = 50
        attempts = 0
        while len(poisson_points) < n_points and attempts < max_attempts:
            if not active_list:
                break

            # Pick a random active point
            idx = np.random.choice(active_list)
            x, y = poisson_points[idx]

            # Try to generate a new point
            found = False
            for _ in range(10):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0.05, 0.2)

                new_x = x + radius * np.cos(angle)
                new_y = y + radius * np.sin(angle)

                # Check bounds
                if new_x < 0.05 or new_x > 0.95 or new_y < 0.05 or new_y > 0.95:
                    continue

                # Check distance to existing points
                too_close = False
                for px, py in poisson_points:
                    dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                    if dist < 0.1:  # Minimum distance threshold
                        too_close = True
                        break

                if not too_close:
                    poisson_points.append([new_x, new_y])
                    active_list.append(len(poisson_points) - 1)
                    found = True
                    break

            if not found:
                active_list.remove(idx)
            attempts += 1

        # Fill remaining points
        while len(poisson_points) < n_points:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            poisson_points.append([x, y])

        points = poisson_points[:n_points]

    # Ensure we have exactly n_points
    if len(points) < n_points:
        # Fill remaining with random points in valid range
        while len(points) < n_points:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])
    elif len(points) > n_points:
        points = points[:n_points]

    return np.array(points)

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with advanced Voronoi-based distribution."""
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

            # Enhanced radius assignment based on both spatial relationships and containment
            if min_distance > 0:
                # Use a more sophisticated radius assignment - consider both neighbor spacing and container bounds
                proposed_radius = min(min_distance / 3.5, max_allowable_radius * 0.7)
            else:
                proposed_radius = max_allowable_radius * 0.5

            # Clamp radius to reasonable bounds
            radius = max(0.001, min(proposed_radius, 0.4))

            circles[i] = [points[i][0], points[i][1], radius]

        # If valid, add to population
        if is_valid(circles):
            population.append(circles)
        else:
            # Robust fallback approach - direct grid-based initialization
            circles = np.zeros((n_circles, 3))

            # Use a more strategic grid arrangement
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = rows
            spacing_x = 0.9 / (cols + 1)  # Leave more margin
            spacing_y = 0.9 / (rows + 1)

            # Use tighter spacing for better density
            radius = min(spacing_x, spacing_y) * 0.35

            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x = 0.05 + (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                    y = 0.05 + (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                    circles[idx] = [x, y, radius]
                    idx += 1

            # Final validation and refinement
            if is_valid(circles):
                population.append(circles)
            else:
                # Last resort - create a configuration that's guaranteed to be valid
                circles = np.zeros((n_circles, 3))

                # Place circles in a more balanced arrangement
                angle_step = 2 * np.pi / n_circles
                center = 0.5
                radius_factor = 0.4  # To keep within bounds

                for i in range(n_circles):
                    angle = i * angle_step
                    x = center + radius_factor * np.cos(angle)
                    y = center + radius_factor * np.sin(angle)
                    # Use consistent, larger radii for better packing
                    r = 0.08 - (i * 0.0015)
                    r = max(0.015, r)  # Minimum radius
                    circles[i] = [x, y, r]

                # If still invalid, use uniform small radii
                if not is_valid(circles):
                    for i in range(n_circles):
                        circles[i] = [0.5, 0.5, 0.015]

                population.append(circles)

    return population

def mutate(circles: np.ndarray, generation: int = 0, max_generations: int = 1000,
           diversity_factor: float = 1.0) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate and dual strategy."""
    # More aggressive adaptive mutation rate
    mutation_rate_start = 0.3
    mutation_rate_end = 0.01

    # Steeper decay curve for faster convergence in later generations
    generation_progress = generation / max_generations

    # Use exponential decay for more rapid reduction
    mutation_rate = mutation_rate_end + (mutation_rate_start - mutation_rate_end) * np.exp(-10 * generation_progress)

    # Adjust based on population diversity
    mutation_rate *= diversity_factor

    mutated = circles.copy()
    n = len(mutated)

    for i in range(n):
        if random.random() < mutation_rate:
            # Phase-based mutation strategy with more distinct behavior
            if generation < max_generations * 0.2:
                # Exploration phase - larger mutations, more position changes
                mutation_strength = 0.08
                choice = random.choices([0, 1, 2], weights=[0.7, 0.7, 0.1])[0]
            elif generation < max_generations * 0.6:
                # Exploitation phase - balanced mutations
                mutation_strength = 0.04
                choice = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.4])[0]
            else:
                # Fine-tuning phase - smaller mutations, more radius changes
                mutation_strength = 0.02
                choice = random.choices([0, 1, 2], weights=[0.2, 0.2, 0.8])[0]

            if choice == 0:  # Mutate x position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, mutation_strength),
                                      mutated[i, 2] + 0.001, 1 - mutated[i, 2] - 0.001)
            elif choice == 1:  # Mutate y position
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, mutation_strength),
                                      mutated[i, 2] + 0.001, 1 - mutated[i, 2] - 0.001)
            else:  # Mutate radius with more controlled log-normal distribution
                log_factor = np.random.normal(0, 0.2)
                mutated[i, 2] = np.clip(mutated[i, 2] * np.exp(log_factor), 0.001, 0.4)

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover with fitness-aware selection and better gene mixing."""
    child1 = parent1.copy()
    child2 = parent2.copy()

    # More sophisticated crossover with fitness-weighted selection
    for i in range(len(parent1)):
        # Weighted selection based on parent fitness (radius)
        fit1 = parent1[i, 2]  # Radius as fitness proxy
        fit2 = parent2[i, 2]

        # Probability of inheriting from parent1
        prob_parent1 = fit1 / (fit1 + fit2 + 1e-8)  # Prevent division by zero

        if random.random() < prob_parent1:
            # Inherit from parent1
            pass
        else:
            # Inherit from parent2
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()

    return child1, child2

def select_parents(population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Select two parents using enhanced tournament selection."""
    tournament_size = 6  # Increase tournament size for better selection pressure

    # Select first parent - more selective
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    best_idx1 = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    parent1 = population[best_idx1]

    # Select second parent - ensure different from first
    while True:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        best_idx2 = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        if best_idx2 != best_idx1:
            break

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
    # Use more aggressive overlap resolution
    for _ in range(15):  # Increase iterations
        if check_overlap_efficient(refined):
            break

        # Reduce radii more aggressively
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Reduce radius more significantly to resolve overlaps
            refined[i, 2] = max(0.001, r * 0.90)

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

    # Approach 2: Create a hybrid with more conservative mixing
    hybrid = parent1.copy()
    for i in range(len(hybrid)):
        if random.random() < 0.25:  # 25% chance to use parent2's data
            hybrid[i, 0] = parent2[i, 0]
            hybrid[i, 1] = parent2[i, 1]
    attempts.append(hybrid)

    # Approach 3: Use a more structured grid arrangement for guarantee
    grid_arrangement = np.zeros((len(parent1), 3))
    rows = int(np.ceil(np.sqrt(len(parent1))))
    cols = rows
    spacing_x = 0.85 / (cols + 1)
    spacing_y = 0.85 / (rows + 1)
    radius = min(spacing_x, spacing_y) * 0.3
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= len(parent1):
                break
            x = 0.075 + (j + 1) * spacing_x
            y = 0.075 + (i + 1) * spacing_y
            grid_arrangement[idx] = [x, y, radius]
            idx += 1
    attempts.append(grid_arrangement)

    # Approach 4: Improved local optimization
    if len(attempts) > 0:
        best_candidate = attempts[0]
        optimized = best_candidate.copy()
        for i in range(len(optimized)):
            # Smaller, more controlled adjustments for tuning
            optimized[i, 0] = np.clip(optimized[i, 0] + np.random.normal(0, 0.003),
                                   optimized[i, 2] + 0.001, 1 - optimized[i, 2] - 0.001)
            optimized[i, 1] = np.clip(optimized[i, 1] + np.random.normal(0, 0.003),
                                   optimized[i, 2] + 0.001, 1 - optimized[i, 2] - 0.001)
        attempts.append(optimized)

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

def optimize_circles_evolutionary(max_generations: int = 800, pop_size: int = 50) -> np.ndarray:
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

        # Early stopping condition if no improvement for many generations
        if generation > 200 and generation % 50 == 0:
            # Check if we've stopped improving
            recent_improvement = False
            for i in range(1, 5):
                if generation - i * 50 >= 0:
                    prev_fitness = fitnesses[np.argmax(fitnesses)]
                    if abs(prev_fitness - best_fitness) < 0.01:
                        recent_improvement = True
                        break
            if not recent_improvement:
                print("Early stopping: No significant improvement in recent generations")
                break

        # Create new population through selection, crossover, and mutation
        new_population = []

        # Calculate diversity for adaptive mutation rate
        diversity = calculate_diversity(population)
        diversity_factor = max(0.6, 1.0 - diversity * 8)  # Adjust diversity impact

        # Keep best individuals (elitism) - increased from 1/5 to 1/3
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
    # Run evolutionary optimization with improved parameters
    circles = optimize_circles_evolutionary(max_generations=MAX_GEN, pop_size=POP_SIZE)

    # Final validation
    if circles is None or not is_valid(circles):
        # Fallback to a simple arrangement if optimization failed
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 0.9 / (cols + 1)
        spacing_y = 0.9 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.35

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = 0.05 + (j + 1) * spacing_x
                y = 0.05 + (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

        # Adjust last few circles to fit
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.015]

    return circles


# EVOLVE-BLOCK-END