# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
import random
from typing import List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    Uses efficient spatial indexing for overlap checking.
    """
    n = len(circles)

    # Check containment constraints
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
    """Create initial population using hybrid grid-Voronoi initialization"""
    population = []

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Choose initialization strategy: grid-based or Voronoi-based
        strategy = np.random.choice(['grid', 'voronoi', 'mixed'], p=[0.4, 0.4, 0.2])

        if strategy == 'grid':
            # Grid-based initialization (existing approach)
            strategies = [
                (2, 2),   # 4 circles
                (3, 3),   # 9 circles
                (4, 4),   # 16 circles
                (5, 5),   # 25 circles
            ]

            # Select strategy
            rows, cols = strategies[np.random.randint(0, len(strategies))]
            actual_rows = min(rows, n_circles // cols + (1 if n_circles % cols else 0))
            actual_cols = min(cols, (n_circles + actual_rows - 1) // actual_rows)  # Ceiling division

            spacing_x = 1.0 / (actual_cols + 1)
            spacing_y = 1.0 / (actual_rows + 1)

            # Place circles in grid pattern
            idx = 0
            for i in range(actual_rows):
                for j in range(actual_cols):
                    if idx >= n_circles:
                        break

                    # Position with slight randomness
                    x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                    y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)

                    # Ensure within bounds
                    x = np.clip(x, 0.01, 0.99)
                    y = np.clip(y, 0.01, 0.99)

                    # Initial radius based on proximity to edges
                    min_dist_to_edge = min(x, 1-x, y, 1-y)
                    r = min(0.08, min_dist_to_edge * np.random.uniform(0.6, 0.9))

                    circles[idx] = [x, y, r]
                    idx += 1

                if idx >= n_circles:
                    break

        elif strategy == 'voronoi':
            # Voronoi-based initialization
            # Generate random points first
            points = np.random.rand(n_circles + 10, 2)  # Extra points to handle edge cases

            # Add boundary points to improve edge coverage
            boundary_points = []
            for _ in range(10):
                boundary_points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
            points = np.vstack([points, boundary_points])

            # Generate Voronoi diagram
            try:
                vor = Voronoi(points)
                # Use Voronoi cell centers for circle positions
                centers = vor.points[vor.point_region[:-1]]  # Remove infinite regions

                # Take first n_circles centers that are inside unit square
                valid_centers = []
                for center in centers:
                    if 0.01 <= center[0] <= 0.99 and 0.01 <= center[1] <= 0.99:
                        valid_centers.append(center)
                        if len(valid_centers) >= n_circles:
                            break

                # Fill circles array with Voronoi-based positions
                idx = 0
                for center in valid_centers:
                    if idx >= n_circles:
                        break
                    x, y = center
                    # Initial radius based on proximity to edges
                    min_dist_to_edge = min(x, 1-x, y, 1-y)
                    r = min(0.08, min_dist_to_edge * np.random.uniform(0.6, 0.9))
                    circles[idx] = [x, y, r]
                    idx += 1

                # Fill remaining circles if needed
                for i in range(idx, n_circles):
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    min_dist_to_edge = min(x, 1-x, y, 1-y)
                    r = min(0.08, min_dist_to_edge * np.random.uniform(0.5, 0.8))
                    circles[i] = [x, y, r]

            except:
                # Fallback to grid if Voronoi fails
                strategies = [(3, 3), (4, 4)]
                rows, cols = strategies[np.random.randint(0, len(strategies))]
                spacing_x = 1.0 / (cols + 1)
                spacing_y = 1.0 / (rows + 1)

                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if idx >= n_circles:
                            break
                        x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                        y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                        x = np.clip(x, 0.01, 0.99)
                        y = np.clip(y, 0.01, 0.99)
                        min_dist_to_edge = min(x, 1-x, y, 1-y)
                        r = min(0.08, min_dist_to_edge * np.random.uniform(0.6, 0.9))
                        circles[idx] = [x, y, r]
                        idx += 1

                for i in range(idx, n_circles):
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    min_dist_to_edge = min(x, 1-x, y, 1-y)
                    r = min(0.08, min_dist_to_edge * np.random.uniform(0.5, 0.8))
                    circles[i] = [x, y, r]

        else:  # mixed
            # Mix grid and random for variety
            # First, fill with grid pattern
            strategies = [(3, 3), (4, 4)]
            rows, cols = strategies[np.random.randint(0, len(strategies))]
            spacing_x = 1.0 / (cols + 1)
            spacing_y = 1.0 / (rows + 1)

            idx = 0
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x = (j + 1) * spacing_x + np.random.uniform(-spacing_x/6, spacing_x/6)
                    y = (i + 1) * spacing_y + np.random.uniform(-spacing_y/6, spacing_y/6)
                    x = np.clip(x, 0.01, 0.99)
                    y = np.clip(y, 0.01, 0.99)
                    min_dist_to_edge = min(x, 1-x, y, 1-y)
                    r = min(0.08, min_dist_to_edge * np.random.uniform(0.6, 0.9))
                    circles[idx] = [x, y, r]
                    idx += 1

            # Fill remaining with Voronoi-like random distribution (but not full Voronoi)
            for i in range(idx, n_circles):
                # Prefer edges and corners for better potential radii
                if np.random.rand() < 0.3:
                    # Corner/edge placement
                    corner_x = np.random.choice([0.1, 0.9]) + np.random.uniform(-0.05, 0.05)
                    corner_y = np.random.choice([0.1, 0.9]) + np.random.uniform(-0.05, 0.05)
                    x, y = corner_x, corner_y
                else:
                    # Regular random placement
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                r = min(0.07, min_dist_to_edge * np.random.uniform(0.5, 0.8))
                circles[i] = [x, y, r]

        # Phase 3: Pre-optimization - try to slightly improve initial configuration
        for _ in range(10):
            improved = False
            for i in range(n_circles):
                # Try to increase radius safely
                original_r = circles[i, 2]
                potential_r = original_r * 1.05

                # Check if we can increase the radius
                can_increase = True
                for j in range(n_circles):
                    if i != j:
                        distance = np.sqrt((circles[i, 0] - circles[j, 0])**2 +
                                         (circles[i, 1] - circles[j, 1])**2)
                        if distance < (potential_r + circles[j, 2]):
                            can_increase = False
                            break

                if can_increase:
                    # Check boundary constraints
                    min_edge_dist = min(circles[i, 0], 1-circles[i, 0],
                                      circles[i, 1], 1-circles[i, 1])
                    if potential_r <= min_edge_dist * 0.95:
                        circles[i, 2] = potential_r
                        improved = True

            if not improved:
                break

        # Phase 4: Add diversity with small perturbations
        for i in range(n_circles):
            if np.random.rand() < 0.5:  # 50% chance to perturb
                circles[i, 0] += np.random.uniform(-0.003, 0.003)
                circles[i, 1] += np.random.uniform(-0.003, 0.003)
                circles[i, 2] += np.random.uniform(-0.001, 0.001)

                # Ensure bounds remain valid
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])

        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 5) -> np.ndarray:
    """Select parent using tournament selection with adaptive size"""
    # Use adaptive tournament size that decreases with generations
    actual_tournament_size = max(3, tournament_size - int(len(population) / 15))
    tournament_indices = np.random.choice(len(population), actual_tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Adaptive crossover that considers proximity and fitness"""
    child = parent1.copy()

    # Calculate pairwise distances for adaptive crossover
    points1 = parent1[:, :2]
    points2 = parent2[:, :2]

    # For each circle, determine crossover probability based on proximity
    for i in range(len(child)):
        # Check how close this circle is to others in both parents
        d1 = np.min([np.sqrt((parent1[i, 0] - parent1[j, 0])**2 + (parent1[i, 1] - parent1[j, 1])**2)
                     for j in range(len(parent1)) if j != i] + [100])  # Dummy large value
        d2 = np.min([np.sqrt((parent2[i, 0] - parent2[j, 0])**2 + (parent2[i, 1] - parent2[j, 1])**2)
                     for j in range(len(parent2)) if j != i] + [100])

        # Average distance to nearest neighbor
        avg_dist = (d1 + d2) / 2

        # Higher probability for distant circles (more independent), lower for close ones
        crossover_prob = max(0.3, 0.8 - avg_dist * 0.5)

        # Apply crossover with adaptive probability
        if np.random.rand() < crossover_prob:
            # Swap genes with 50% probability each
            if np.random.rand() < 0.5:
                child[i, 0] = parent2[i, 0]
            if np.random.rand() < 0.5:
                child[i, 1] = parent2[i, 1]
            if np.random.rand() < 0.5:
                child[i, 2] = parent2[i, 2]

    return child

def mutate(circles: np.ndarray, generation: int, max_generations: int,
           base_mutation_rate: float = 0.15) -> np.ndarray:
    """Apply mutation with adaptive rate and enhanced strategies"""
    # Three-phase adaptive mutation rate
    if generation < max_generations * 0.4:
        # Phase 1: High exploration
        mutation_rate = base_mutation_rate
    elif generation < max_generations * 0.7:
        # Phase 2: Balanced refinement
        mutation_rate = base_mutation_rate * 0.5
    else:
        # Phase 3: Exploitation
        mutation_rate = base_mutation_rate * 0.1

    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius with different intensities based on phase
            if np.random.rand() < 0.6:  # 60% chance to mutate position
                # Larger scale early, smaller later
                scale_factor = 0.05 * (1 - generation/max_generations) + 0.005
                mutated[i, 0] += np.random.normal(0, scale_factor)
                mutated[i, 1] += np.random.normal(0, scale_factor)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius (40% chance)
                # Use different scale based on phase
                scale_factor = 0.02 * (1 - generation/max_generations) + 0.001
                mutated[i, 2] += np.random.normal(0, scale_factor)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Improved repair with more robust overlap resolution"""
    repaired = circles.copy()

    # First ensure all circles are within bounds
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        # Keep within bounds
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])  # Ensure positive radius

    # Then resolve overlaps using iterative repulsion with early termination and better strategy
    points = repaired[:, :2]
    tree = cKDTree(points)

    # Try to resolve overlaps iteratively with limited attempts and early exit
    for _ in range(60):  # Increase iterations for better resolution
        any_changes = False
        # Process circles in random order for better results
        circle_order = list(range(len(repaired)))
        np.random.shuffle(circle_order)

        for i in circle_order:
            x1, y1, r1 = repaired[i]

            # Find nearby circles more efficiently
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        # Apply stronger repulsion initially, then decrease
                        repulsion_strength = 0.6 * (1 - _ / 60)  # Decrease over iterations

                        # Repel circles apart with physics-inspired force
                        if distance > 0.001:  # Avoid division by zero
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance

                            # Move them apart with varying force
                            move_amount = (min_distance - distance) * repulsion_strength * 1.2
                            repaired[i, 0] += dx * move_amount
                            repaired[i, 1] += dy * move_amount
                            repaired[j, 0] -= dx * move_amount
                            repaired[j, 1] -= dy * move_amount

                            # Clamp to bounds
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)

                        any_changes = True

        if not any_changes:
            break

    return repaired

def local_refinement(circles: np.ndarray, max_iterations: int = 200) -> np.ndarray:
    """
    Enhanced local refinement with multiple optimization strategies
    """
    refined = circles.copy()
    n_circles = len(refined)

    # Strategy 1: Greedy radius expansion with backtracking
    improved = True
    iteration = 0

    while improved and iteration < 100:  # More iterations for better convergence
        improved = False
        iteration += 1

        # Try to expand each circle's radius systematically
        for i in range(n_circles):
            original_r = refined[i, 2]

            # Compute maximum possible radius
            x, y, _ = refined[i]
            max_possible_r = min(x, 1-x, y, 1-y)

            # Try increasing radius in steps - optimized steps
            steps = [0.015, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0001]

            for step in steps:
                test_r = min(original_r + step, max_possible_r * 0.98)

                if test_r > original_r + 1e-6:
                    # Test the new configuration
                    temp_circles = refined.copy()
                    temp_circles[i, 2] = test_r

                    # Check validity with entire population
                    if validate_circles(temp_circles):
                        refined = temp_circles
                        improved = True
                        break

    # Strategy 2: Position adjustment to enable larger radii with systematic search
    for _ in range(60):  # More iterations for position optimization
        # Try to improve each circle by adjusting its position
        improved_local = False
        # Process circles in shuffled order
        indices = list(range(n_circles))
        np.random.shuffle(indices)

        for i in indices:
            x, y, r = refined[i]

            # Store best improvement found
            best_x, best_y, best_r = x, y, r
            best_radius = r
            best_valid = False

            # Test various small adjustments
            adjustments = [
                (0, 0, 0),
                (0.002, 0, 0),
                (-0.002, 0, 0),
                (0, 0.002, 0),
                (0, -0.002, 0),
                (0.001, 0.001, 0),
                (-0.001, -0.001, 0),
                (0.001, -0.001, 0),
                (-0.001, 0.001, 0),
                (0.003, 0.003, 0),
                (-0.003, -0.003, 0),
                (0.002, 0.001, 0),
                (0.001, 0.002, 0),
                (-0.002, -0.001, 0),
                (-0.001, -0.002, 0),
            ]

            for dx, dy, dr in adjustments:
                test_x = max(0.001, min(0.999, x + dx))
                test_y = max(0.001, min(0.999, y + dy))
                test_r = max(0.001, min(r + dr,
                                      min(test_x, test_y, 1-test_x, 1-test_y) * 0.99))

                # Test new configuration
                temp_circles = refined.copy()
                temp_circles[i] = [test_x, test_y, test_r]

                if validate_circles(temp_circles) and test_r > best_radius:
                    best_x, best_y, best_r = test_x, test_y, test_r
                    best_radius = test_r
                    best_valid = True

            # Update if we found a significantly better configuration
            if best_valid:
                refined[i] = [best_x, best_y, best_r]
                improved_local = True

        if not improved_local:
            break

    # Strategy 3: Final comprehensive repair
    # Perform final validation and repair if needed
    final_repaired = repair_circles(refined)

    # Double-check that the result is valid
    if validate_circles(final_repaired):
        return final_repaired
    else:
        # If still invalid, return original with minimal fixes
        result = circles.copy()
        for i in range(n_circles):
            result[i, 0] = np.clip(result[i, 0], result[i, 2], 1 - result[i, 2])
            result[i, 1] = np.clip(result[i, 1], result[i, 2], 1 - result[i, 2])
            result[i, 2] = max(0.001, result[i, 2])
        return result

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters - tuned for better performance
    pop_size = 100  # Increased population size for better exploration
    n_generations = 180  # More generations for deeper exploration
    elite_size = 12  # More elites for better preservation of good solutions
    tournament_size = 8  # Larger tournaments to increase selection pressure

    # Create initial population
    population = create_initial_population(pop_size, 26)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None

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
            parent1 = tournament_selection(valid_individuals, fitnesses, tournament_size)
            parent2 = tournament_selection(valid_individuals, fitnesses, tournament_size)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate(child, generation, n_generations)

            # Repair
            child = repair_circles(child)

            new_population.append(child)

        population = new_population[:pop_size]

    # Return the best solution found with final refinement
    if best_individual is not None:
        # Apply local refinement to the best solution
        refined_solution = local_refinement(best_individual)
        if validate_circles(refined_solution):
            return refined_solution
        else:
            return best_individual
    else:
        # If no valid solution found, return the best from final population
        fitnesses = [calculate_sum_radii(circles) for circles in population if validate_circles(circles)]
        if fitnesses:
            best_idx = np.argmax(fitnesses)
            # Apply local refinement to the best candidate from final population
            refined_solution = local_refinement(population[best_idx])
            if validate_circles(refined_solution):
                return refined_solution
            else:
                return population[best_idx]
        else:
            # Fallback: return a valid random solution
            circles = np.zeros((26, 3))
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
            return circles

# EVOLVE-BLOCK-END