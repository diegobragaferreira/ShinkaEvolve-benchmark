# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import time

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
    """Check if any circles overlap using spatial indexing for efficiency."""
    if len(circles) <= 1:
        return True

    # Use KDTree for efficient neighbor queries
    positions = circles[:, :2]
    radii = circles[:, 2]

    # Build KDTree for spatial indexing
    tree = cKDTree(positions)

    # For each circle, query nearby points within reach of potential overlap
    for i, (x, y, r) in enumerate(circles):
        # Query points within distance 2*(r_max) to avoid full O(n^2) search
        nearby_indices = tree.query_ball_point([x, y], 2 * (r + 0.01))

        # Check overlaps with nearby points
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < (r + r2 - 1e-8):
                    return False
    return True

def fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def evaluate_fitness_with_penalty(circles: np.ndarray) -> Tuple[float, float, float]:
    """Evaluate fitness with penalties for constraint violations."""
    total_radius = np.sum(circles[:, 2])
    
    # Check constraints
    valid = check_containment(circles) and check_overlap(circles)
    
    if valid:
        return total_radius, total_radius, 0.0
    
    # Calculate penalties for constraint violations
    penalty = 0.0
    
    # Boundary penalty calculation
    boundary_penalty = 0.0
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0:
            boundary_penalty += (r - x)**2
        if x + r > 1:
            boundary_penalty += (x + r - 1)**2
        if y - r < 0:
            boundary_penalty += (r - y)**2
        if y + r > 1:
            boundary_penalty += (y + r - 1)**2
    
    # Overlap penalty calculation
    overlap_penalty = 0.0
    if len(circles) > 1:
        positions = circles[:, :2]
        radii = circles[:, 2]
        tree = cKDTree(positions)
        
        for i, (x, y, r) in enumerate(circles):
            nearby_indices = tree.query_ball_point([x, y], 2 * (r + 0.01))
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < (r + r2 - 1e-8):
                        overlap_penalty += (r + r2 - distance)**2
    
    penalty = 10000.0 * boundary_penalty + 100000.0 * overlap_penalty
    return total_radius - penalty, total_radius, penalty

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with Voronoi-based distribution for better spatial coverage."""
    population = []

    # Generate Voronoi-like distribution using a grid-based approach with jitter
    def generate_voronoi_points(n_points: int) -> np.ndarray:
        """Generate approximately Voronoi-distributed points."""
        # Create a grid of points
        grid_size = int(np.ceil(np.sqrt(n_points)))
        spacing = 1.0 / (grid_size + 1)
        points = []

        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n_points:
                    break
                x = (i + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                y = (j + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                points.append([x, y])

        # Ensure we have exactly n_points
        while len(points) < n_points:
            points.append([np.random.random(), np.random.random()])

        return np.array(points[:n_points])

    for _ in range(pop_size):
        # Generate Voronoi-like points
        points = generate_voronoi_points(n_circles)

        # Create initial circles with appropriate radii based on proximity
        circles = np.zeros((n_circles, 3))

        # Assign radii using a more intelligent approach based on available space
        for i in range(n_circles):
            # Calculate minimum distance to all other points
            distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
            # Exclude self-distance
            distances[i] = np.inf
            min_distance = np.min(distances)

            # Set radius to be a fraction of the minimum distance to nearby points
            # but bounded by containment constraints
            max_allowable_radius = min(points[i][0], points[i][1],
                                     1 - points[i][0], 1 - points[i][1])

            # Use half of the minimum distance to neighbors as radius
            proposed_radius = min(min_distance / 2.0, max_allowable_radius * 0.8)
            radius = max(0.001, min(proposed_radius, 0.3))  # Clamp between reasonable values

            circles[i] = [points[i][0], points[i][1], radius]

        # If valid, add to population
        if check_containment(circles) and check_overlap(circles):
            population.append(circles)
        else:
            # Fallback to a simpler approach if initial attempt fails
            circles = np.zeros((n_circles, 3))
            rows = int(np.ceil(np.sqrt(n_circles)))
            cols = rows
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)
            radius = min(spacing_x, spacing_y) * 0.4

            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x = (j + 1) * spacing_x
                    y = (i + 1) * spacing_y
                    circles[idx] = [x, y, radius]
                    idx += 1

            # Final validation and refinement
            if check_containment(circles) and check_overlap(circles):
                population.append(circles)
            else:
                # Last resort - minimal valid configuration
                circles = np.zeros((n_circles, 3))
                for i in range(n_circles):
                    circles[i] = [0.5, 0.5, 0.01]
                population.append(circles)

    return population

def mutate(circles: np.ndarray, generation: int = 0, max_generations: int = 1000) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate."""
    # Adaptive mutation rate that decreases over generations
    mutation_rate_start = 0.25
    mutation_rate_end = 0.02
    
    # Sigmoidal decay function
    mutation_rate = mutation_rate_end + (mutation_rate_start - mutation_rate_end) * (
        1 / (1 + np.exp(10 * (generation / max_generations - 0.5)))
    )
    
    mutated = circles.copy()
    n = len(mutated)

    for i in range(n):
        if random.random() < mutation_rate:
            # Choose mutation type with preference for position changes
            mutation_type = random.choices(
                [0, 1, 2, 3],
                weights=[0.5, 0.5, 0.2, 0.3]  # Position more likely
            )[0]

            if mutation_type == 0:  # Mutate x position (larger change)
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.05), 0.05, 0.95)
            elif mutation_type == 1:  # Mutate y position (larger change)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.05), 0.05, 0.95)
            elif mutation_type == 2:  # Mutate radius (smaller change)
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.01), 0.001, 0.2)
            else:  # Mutate both position and radius
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.02), 0.05, 0.95)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.02), 0.05, 0.95)
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.005), 0.001, 0.2)

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parent solutions."""
    # Multi-point crossover that respects both positions and radii
    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Multi-point crossover
    crossover_points = sorted(random.sample(range(1, n), min(3, n-1)))

    # Alternate between parents for segments
    last_point = 0
    use_parent1 = True
    for point in crossover_points:
        if use_parent1:
            child1[last_point:point, :] = parent1[last_point:point, :]
        else:
            child2[last_point:point, :] = parent2[last_point:point, :]
        last_point = point
        use_parent1 = not use_parent1

    # Handle final segment
    if use_parent1:
        child1[last_point:, :] = parent1[last_point:, :]
    else:
        child2[last_point:, :] = parent2[last_point:, :]

    return child1, child2

def tournament_selection(population: List[np.ndarray], fitnesses: List[float], 
                        tournament_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Select two parents using tournament selection."""
    # Select first parent
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    parent1 = population[winner_idx].copy()

    # Select second parent (different from first)
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    parent2 = population[winner_idx].copy()

    return parent1, parent2

def refine_solution(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to fix constraint violations."""
    refined = circles.copy()

    # Phase 1: Fix containment violations first
    for i in range(len(refined)):
        x, y, r = refined[i]
        if r > x:
            x = r + 0.001
        if r > y:
            y = r + 0.001
        if r > (1 - x):
            x = 1 - r - 0.001
        if r > (1 - y):
            y = 1 - r - 0.001
        refined[i, 0] = x
        refined[i, 1] = y

    # Phase 2: Iterative overlap resolution
    max_iter = 100
    for iteration in range(max_iter):
        changed = False
        
        # Build list of valid circles for overlap checking
        valid_indices = [i for i in range(len(refined)) if refined[i, 2] > 0]

        for i in valid_indices:
            x, y, r = refined[i]

            # Check overlap with all others
            for j in valid_indices:
                if i != j:
                    ox, oy, oradius = refined[j]
                    distance = np.sqrt((x - ox)**2 + (y - oy)**2)

                    if distance < (r + oradius):
                        # Try to resolve overlap by adjusting current circle
                        if distance > 0.0001:
                            # Normalize direction vector
                            dx = (x - ox) / distance
                            dy = (y - oy) / distance

                            # Reduce radius to prevent further overlap
                            new_r = max(0.001, (r + oradius) * 0.99 - distance)
                            if new_r < r and new_r > 0.001:
                                refined[i, 2] = new_r
                                changed = True

                            # Adjust position to separate circles
                            separation = 0.001
                            refined[i, 0] = x + dx * separation
                            refined[i, 1] = y + dy * separation

                            # Ensure containment after adjustment
                            refined[i, 0] = np.clip(refined[i, 0], refined[i, 2], 1 - refined[i, 2])
                            refined[i, 1] = np.clip(refined[i, 1], refined[i, 2], 1 - refined[i, 2])
                        else:
                            # If circles are at same position, move one slightly
                            refined[i, 0] += random.uniform(-0.001, 0.001)
                            refined[i, 1] += random.uniform(-0.001, 0.001)
                            changed = True

        if not changed:
            break

    # Final containment check and correction
    for i in range(len(refined)):
        x, y, r = refined[i]
        r = max(0.001, r)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        refined[i, 0] = x
        refined[i, 1] = y
        refined[i, 2] = r

    return refined

def optimize_circles_evolutionary(max_generations: int = 500, pop_size: int = 120) -> np.ndarray:
    """Evolutionary optimization for circle packing."""
    n = 26

    # Initialize population
    population = initialize_population(pop_size, n)
    best_solution = None
    best_fitness = -float('inf')
    best_total_radius = 0.0

    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        total_radii = []
        penalties = []

        for circles in population:
            fitness_val, total_radius, penalty = evaluate_fitness_with_penalty(circles)
            fitness_scores.append(fitness_val)
            total_radii.append(total_radius)
            penalties.append(penalty)

        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_total_radius = total_radii[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, "
                  f"Total radius = {best_total_radius:.6f}")

        # Create new population through selection, crossover, and mutation
        new_population = []

        # Elitism: keep the best individuals
        sorted_indices = np.argsort(fitness_scores)[::-1][:pop_size//5]
        for idx in sorted_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1, parent2 = tournament_selection(population, fitness_scores)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation with generation info
            child1 = mutate(child1, generation, max_generations)
            child2 = mutate(child2, generation, max_generations)

            # Apply local refinement to ensure validity
            child1 = refine_solution(child1)
            child2 = refine_solution(child2)

            # Add children to new population
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population[:pop_size]

    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run evolutionary optimization
    circles = optimize_circles_evolutionary(max_generations=500, pop_size=120)

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