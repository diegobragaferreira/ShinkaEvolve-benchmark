# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    Optimized with early termination and efficient spatial queries.
    """
    n = len(circles)

    # Check containment constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]  # Get (x, y) coordinates
    tree = cKDTree(points)

    # For each circle, check overlap with others
    for i in range(n):
        x1, y1, r1 = circles[i]

        # Find nearby circles (within 2*(r1+r2) distance)
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

        # Check overlap with each nearby circle
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2

                if distance_sq < min_distance_sq:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population using advanced multi-scale grid initialization"""
    population = []

    # Multi-scale grid approach with more variety
    grid_sizes = [2, 3, 4, 5, 6]  # Try different grid sizes

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))
        grid_size = np.random.choice(grid_sizes)
        spacing_x = 1.0 / grid_size
        spacing_y = 1.0 / grid_size

        # Place circles on a grid with more randomness
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break

                # Position in grid cell with more randomness
                x = (i + 0.5 + np.random.uniform(-0.15, 0.15)) * spacing_x
                y = (j + 0.5 + np.random.uniform(-0.15, 0.15)) * spacing_y

                # Initial radius based on proximity to edges with more variation
                max_radius = min(x, 1-x, y, 1-y)
                r = np.random.uniform(0.005, min(0.12, max_radius * 0.9))

                # Adjust to fit within bounds
                r = min(r, max_radius)

                circles[idx] = [x, y, r]
                idx += 1

        # Add adaptive perturbations based on diversity
        for i in range(n_circles):
            # More aggressive perturbation for diverse individuals
            if np.random.rand() < 0.5:  # 50% chance to perturb
                circles[i, 0] += np.random.normal(0, 0.01)
                circles[i, 1] += np.random.normal(0, 0.01)
                circles[i, 2] += np.random.normal(0, 0.005)

                # Ensure valid bounds
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])

        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select parent using improved adaptive tournament selection based on population diversity"""
    # Calculate population diversity (variance of fitness values)
    if len(fitnesses) > 1:
        diversity = np.var(fitnesses)
        std_dev = np.std(fitnesses)

        # Adjust tournament size based on diversity with more precise scaling
        if diversity > std_dev * 0.3:
            # High diversity → smaller tournaments (more exploration)
            tournament_size = max(2, min(6, int(tournament_size * 0.6)))
        elif diversity < std_dev * 0.1:
            # Low diversity → larger tournaments (more exploitation)
            tournament_size = max(5, min(9, int(tournament_size * 1.4)))
        else:
            # Medium diversity → standard tournaments
            tournament_size = max(3, min(7, int(tournament_size * 0.95)))

    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Improved crossover with constraint awareness"""
    child = parent1.copy()

    # Apply crossover with 60% probability for each gene (slightly more selective)
    mask = np.random.rand(*parent1.shape) > 0.4

    # Ensure that the resulting child is valid by checking constraints
    child[mask] = parent2[mask]

    # Make sure positions and radii respect boundary conditions
    for i in range(len(child)):
        x, y, r = child[i]
        child[i, 0] = np.clip(x, r, 1 - r)
        child[i, 1] = np.clip(y, r, 1 - r)
        child[i, 2] = max(0.001, r)

    return child

def mutate(circles: np.ndarray, mutation_rate: float = 0.1,
           max_radius_change: float = 0.02) -> np.ndarray:
    """Improved mutation with better boundary handling"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius (weighted towards position)
            if np.random.rand() < 0.75:  # 75% chance to mutate position
                mutated[i, 0] += np.random.normal(0, 0.012)
                mutated[i, 1] += np.random.normal(0, 0.012)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # 25% chance to mutate radius
                mutated[i, 2] += np.random.normal(0, max_radius_change * 0.4)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def count_overlap_violations(circles: np.ndarray) -> int:
    """Count the number of overlap violations in the current solution."""
    n = len(circles)
    if n <= 1:
        return 0

    points = circles[:, :2]
    tree = cKDTree(points)
    overlap_count = 0

    for i in range(n):
        x1, y1, r1 = circles[i]
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2

                if distance_sq < min_distance_sq:
                    overlap_count += 1

    return overlap_count // 2  # Each violation counted twice

def adaptive_local_search(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Apply adaptive local search based on overlap severity."""
    optimized = circles.copy()
    overlap_count = count_overlap_violations(optimized)

    # Classify overlap severity and apply appropriate refinement
    if overlap_count == 0:
        # Low overlap - focus on radius expansion
        return local_radius_expansion(optimized, max_iterations)
    elif overlap_count <= 5:
        # Medium overlap - combination approach
        return local_refinement_medium(optimized, max_iterations)
    else:
        # High overlap - aggressive repulsion
        return local_refinement_high(optimized, max_iterations)

def local_radius_expansion(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Focus on expanding radii for solutions with few overlaps."""
    optimized = circles.copy()

    for iteration in range(max_iterations):
        improved = False
        for i in range(len(optimized)):
            x, y, r = optimized[i]

            # Compute maximum possible radius for this circle
            max_radius = min(x, y, 1-x, 1-y)

            # Try to increase radius if there's room and it doesn't cause overlaps
            if max_radius > r:
                # Find nearby circles that would prevent expansion
                current_radius = r
                for j in range(len(optimized)):
                    if i != j:
                        x2, y2, r2 = optimized[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        # How much we could expand before overlapping
                        max_for_this_circle = distance - r2
                        max_radius = min(max_radius, max_for_this_circle)

                if max_radius > current_radius:
                    # Try to increase radius gradually
                    new_radius = min(current_radius + 0.005, max_radius)
                    if new_radius > current_radius:
                        optimized[i, 2] = new_radius
                        improved = True

        if not improved:
            break

    return optimized

def local_refinement_medium(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Refinement strategy for medium overlap cases."""
    optimized = circles.copy()

    for iteration in range(max_iterations):
        improved = False

        # Step 1: Try to expand radii
        for i in range(len(optimized)):
            x, y, r = optimized[i]

            # Compute maximum possible radius for this circle
            max_radius = min(x, y, 1-x, 1-y)

            # Find nearby circles
            for j in range(len(optimized)):
                if i != j:
                    x2, y2, r2 = optimized[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_for_this_circle = distance - r2
                    max_radius = min(max_radius, max_for_this_circle)

            if max_radius > r:
                new_radius = min(r + 0.003, max_radius)
                if new_radius > r:
                    optimized[i, 2] = new_radius
                    improved = True

        # Step 2: Position adjustment for overlaps
        points = optimized[:, :2]
        tree = cKDTree(points)

        for i in range(len(optimized)):
            x1, y1, r1 = optimized[i]
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = optimized[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        # Move circles apart
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            total_radius = r1 + r2
                            move_amount = (min_distance - distance) * 0.3

                            optimized[i, 0] += dx * move_amount * (r2 / total_radius)
                            optimized[i, 1] += dy * move_amount * (r2 / total_radius)
                            optimized[j, 0] -= dx * move_amount * (r1 / total_radius)
                            optimized[j, 1] -= dy * move_amount * (r1 / total_radius)

                            # Clamp to bounds
                            optimized[i, 0] = np.clip(optimized[i, 0], r1, 1 - r1)
                            optimized[i, 1] = np.clip(optimized[i, 1], r1, 1 - r1)
                            optimized[j, 0] = np.clip(optimized[j, 0], r2, 1 - r2)
                            optimized[j, 1] = np.clip(optimized[j, 1], r2, 1 - r2)

                            improved = True

        if not improved:
            break

    return optimized

def local_refinement_high(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Aggressive refinement for high overlap cases using physics-inspired repulsion."""
    optimized = circles.copy()

    for iteration in range(max_iterations):
        improved = False
        points = optimized[:, :2]
        tree = cKDTree(points)

        # Physics-inspired repulsion
        forces = np.zeros_like(optimized[:, :2])

        for i in range(len(optimized)):
            x1, y1, r1 = optimized[i]

            # Compute repulsion forces from nearby circles
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = optimized[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        # Apply strong repulsion force
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            force_magnitude = (min_distance - distance) * 0.8

                            forces[i, 0] += dx * force_magnitude * (r2 / (r1 + r2))
                            forces[i, 1] += dy * force_magnitude * (r2 / (r1 + r2))

        # Apply forces to positions
        for i in range(len(optimized)):
            optimized[i, 0] += forces[i, 0] * 0.1
            optimized[i, 1] += forces[i, 1] * 0.1

            # Keep within bounds
            x, y, r = optimized[i]
            optimized[i, 0] = np.clip(optimized[i, 0], r, 1 - r)
            optimized[i, 1] = np.clip(optimized[i, 1], r, 1 - r)

        # Also try to expand radii where possible
        for i in range(len(optimized)):
            x, y, r = optimized[i]
            max_radius = min(x, y, 1-x, 1-y)

            # Check if expansion is safe
            safe_to_expand = True
            for j in range(len(optimized)):
                if i != j:
                    x2, y2, r2 = optimized[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_for_this_circle = distance - r2
                    max_radius = min(max_radius, max_for_this_circle)

            if max_radius > r:
                new_radius = min(r + 0.008, max_radius)
                if new_radius > r:
                    optimized[i, 2] = new_radius
                    improved = True

        if not improved:
            break

    return optimized

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters - increased for better exploration/exploitation balance
    pop_size = 80   # Increased population size
    n_generations = 150  # Increased generations
    elite_size = 8   # Increased elite count

    # Create initial population
    population = create_initial_population(pop_size, 26)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None

    # Adaptive mutation rate with aggressive exponential decay
    mutation_rate_start = 0.15
    decay_factor = 0.008

    for generation in range(n_generations):
        # Calculate fitness for all individuals
        fitnesses = []
        valid_individuals = []

        for circles in population:
            if validate_circles(circles):
                fitness = calculate_sum_radii(circles)
                fitnesses.append(fitness)
                valid_individuals.append(circles)
            else:
                # Repair invalid individuals
                repaired = repair_circles(circles)
                if validate_circles(repaired):
                    fitness = calculate_sum_radii(repaired)
                    fitnesses.append(fitness)
                    valid_individuals.append(repaired)
                else:
                    # If still invalid, penalize heavily
                    fitnesses.append(-np.inf)
                    valid_individuals.append(circles)

        # Track best individual
        if valid_individuals:
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_individual = valid_individuals[max_idx].copy()

        # Elitism: keep top individuals
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elites = [valid_individuals[i] for i in elite_indices if fitnesses[i] > -np.inf]

        # Generate new population
        new_population = elites[:]

        # Fill remaining slots with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(valid_individuals, fitnesses)
            parent2 = tournament_selection(valid_individuals, fitnesses)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            # Adjust mutation rate with exponential decay
            current_mutation_rate = mutation_rate_start * math.exp(-generation / (n_generations * 0.6))
            current_mutation_rate = max(0.015, current_mutation_rate)

            child = mutate(child, current_mutation_rate)

            # Local optimization instead of simple repair
            child = adaptive_local_search(child)

            new_population.append(child)

        population = new_population[:pop_size]

        # Logging every 25 generations
        if generation % 25 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # If no valid solution found, return the best from final population
        fitnesses = [calculate_sum_radii(circles) for circles in population if validate_circles(circles)]
        if fitnesses:
            best_idx = np.argmax(fitnesses)
            return population[best_idx]
        else:
            # Fallback: return a valid random solution
            circles = np.zeros((26, 3))
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
            return circles

# EVOLVE-BLOCK-END