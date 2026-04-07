# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def compute_violation_count(circles: np.ndarray) -> int:
    """Count total overlap violations for a configuration"""
    n = len(circles)
    violation_count = 0

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            violation_count += 1

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
                    violation_count += 1

    return violation_count

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
    """Create initial population using advanced multi-scale grid-based initialization"""
    population = []

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Multi-scale grid initialization with different approaches
        grid_sizes = [3, 4, 5]  # Different grid sizes to explore
        grid_size = grid_sizes[np.random.randint(0, len(grid_sizes))]

        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        # Place circles on a grid with better randomness
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break

                # Position in grid cell with more controlled randomness
                x = (i + 1) * spacing_x + np.random.uniform(-spacing_x/8, spacing_x/8)
                y = (j + 1) * spacing_y + np.random.uniform(-spacing_y/8, spacing_y/8)

                # Ensure within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)

                # Assign initial radius based on proximity to edges with variation
                min_dist_to_edge = min(x, 1-x, y, 1-y)
                r = min(0.08, min_dist_to_edge * np.random.uniform(0.7, 0.9))

                circles[idx] = [x, y, r]
                idx += 1

            if idx >= n_circles:
                break

        # Fill remaining circles with random but informed placement
        for i in range(idx, n_circles):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            r = min(0.08, min_dist_to_edge * np.random.uniform(0.5, 0.8))
            circles[i] = [x, y, r]

        # Improve initial configuration by trying to increase radii using a smarter approach
        improved = True
        attempts = 0
        while improved and attempts < 15:
            improved = False
            attempts += 1
            for i in range(n_circles):
                # Try to increase radius safely
                original_r = circles[i, 2]
                potential_r = original_r * 1.05  # Slightly larger increase

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

        # Add slight random perturbations to improve diversity
        for i in range(n_circles):
            if np.random.rand() < 0.4:  # 40% chance to perturb
                circles[i, 0] += np.random.uniform(-0.005, 0.005)
                circles[i, 1] += np.random.uniform(-0.005, 0.005)
                circles[i, 2] += np.random.uniform(-0.002, 0.002)

                # Ensure bounds remain valid
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])

        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 5) -> np.ndarray:
    """Select parent using tournament selection with adaptive size and constraint awareness"""
    # Use adaptive tournament size that decreases with generations
    actual_tournament_size = max(3, tournament_size - int(len(population) / 10))

    # Sample tournament participants and select based on both fitness and constraint quality
    tournament_indices = np.random.choice(len(population), actual_tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

    # Calculate constraint quality scores (lower violation count is better)
    tournament_violations = [compute_violation_count(population[i]) for i in tournament_indices]

    # Combine fitness and constraint quality into a single score for selection
    # We want to favor individuals that are both fit AND have few violations
    combined_scores = []
    for i, (fitness, violations) in enumerate(zip(tournament_fitnesses, tournament_violations)):
        # Normalize violations to avoid dominance by extremely poor individuals
        normalized_violations = min(violations, 50) / 50.0
        # The constraint penalty is subtracted from fitness (the lower violations, the higher score)
        combined_score = fitness - (normalized_violations * max(0, fitness) * 0.1)
        combined_scores.append(combined_score)

    # Select the individual with highest combined score
    winner_index = tournament_indices[np.argmax(combined_scores)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Uniform crossover between two parents with enhanced mixing"""
    child = parent1.copy()

    # Apply crossover with variable probability based on circle properties
    for i in range(len(child)):
        # Higher probability for position genes, lower for radius
        pos_prob = 0.7
        rad_prob = 0.4

        # Apply crossover for position
        if np.random.rand() < pos_prob:
            child[i, 0] = parent2[i, 0]
        if np.random.rand() < pos_prob:
            child[i, 1] = parent2[i, 1]

        # Apply crossover for radius
        if np.random.rand() < rad_prob:
            child[i, 2] = parent2[i, 2]

    return child

def mutate(circles: np.ndarray, generation: int, max_generations: int,
           base_mutation_rate: float = 0.15) -> np.ndarray:
    """Apply mutation with adaptive rate and enhanced strategies"""
    # Exponential decay mutation rate
    mutation_rate = base_mutation_rate * (0.1**(generation/max_generations))

    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius with different intensities
            if np.random.rand() < 0.5:  # Mutate position
                # Larger mutations early, smaller later
                scale = 0.04 * (1 - generation/max_generations) + 0.005
                mutated[i, 0] += np.random.normal(0, scale)
                mutated[i, 1] += np.random.normal(0, scale)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                # Use smaller scale for radius mutations
                scale = 0.02 * (1 - generation/max_generations) + 0.001
                mutated[i, 2] += np.random.normal(0, scale)
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
    for _ in range(50):  # Increase iterations for better resolution
        any_changes = False
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]

            # Find nearby circles more efficiently
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        # Repel circles apart with smarter movement
                        if distance > 0.001:  # Avoid division by zero
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance

                            # Move them apart proportionally to the overlap - increased force
                            move_amount = (min_distance - distance) * 0.8
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

def local_refinement(circles: np.ndarray, max_iterations: int = 150) -> np.ndarray:
    """
    Enhanced local refinement with multiple optimization strategies focused on high-violation areas
    """
    refined = circles.copy()
    n_circles = len(refined)

    # Strategy 1: Greedy radius expansion with backtracking
    improved = True
    iteration = 0

    while improved and iteration < 80:  # More iterations for better convergence
        improved = False
        iteration += 1

        # Try to expand each circle's radius systematically
        for i in range(n_circles):
            original_r = refined[i, 2]

            # Compute maximum possible radius
            x, y, _ = refined[i]
            max_possible_r = min(x, 1-x, y, 1-y)

            # Try increasing radius in steps - more fine-grained steps
            steps = [0.01, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0001]

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

    # Strategy 2: Targeted position adjustment focused on high-violation areas
    # First, identify problematic circles
    violation_counts = [0] * n_circles
    points = refined[:, :2]
    tree = cKDTree(points)

    # Count violations per circle
    for i in range(n_circles):
        x1, y1, r1 = refined[i]
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = refined[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2

                if distance_sq < min_distance_sq:
                    violation_counts[i] += 1

    # Sort circles by violation count (descending) to prioritize fixing high-violation areas
    sorted_indices = sorted(range(n_circles), key=lambda i: violation_counts[i], reverse=True)

    for _ in range(50):  # More iterations for position optimization
        # Try to improve each circle by adjusting its position, prioritizing high-violation ones
        improved_local = False
        # Process circles in violation order
        for i in sorted_indices:
            x, y, r = refined[i]

            # Try several small position adjustments to find better spots
            best_x, best_y, best_r = x, y, r
            best_radius = r
            best_improvement = 0

            # Test various small adjustments with more combinations
            adjustments = [
                (0, 0, 0),
                (0.003, 0, 0),
                (-0.003, 0, 0),
                (0, 0.003, 0),
                (0, -0.003, 0),
                (0.002, 0.002, 0),
                (-0.002, -0.002, 0),
                (0.002, -0.002, 0),
                (-0.002, 0.002, 0),
                (0.001, 0.001, 0),
                (-0.001, -0.001, 0)
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
                    best_improvement = test_r - r

            # Update if we found a significantly better configuration
            if best_improvement > 1e-6:
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
    pop_size = 80  # Increased population size for better exploration
    n_generations = 150  # More generations for deeper exploration
    elite_size = 10  # More elites for better preservation of good solutions
    tournament_size = 7  # Larger tournaments to increase selection pressure

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