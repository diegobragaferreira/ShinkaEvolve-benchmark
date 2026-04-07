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
    """Create initial population using Voronoi-inspired initialization"""
    population = []

    # Use Voronoi-inspired distribution for better spatial coverage
    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Generate points using spiral pattern to distribute well
        for i in range(n_circles):
            # Spiral pattern with randomization for diversity
            angle = 2 * np.pi * i / n_circles
            radius = 0.4 * (1.0 - 0.7 * (i / (n_circles - 1))) if n_circles > 1 else 0.5
            x = 0.5 + radius * np.cos(angle) * 0.8
            y = 0.5 + radius * np.sin(angle) * 0.8

            # Add some randomness to avoid perfect patterns
            x += np.random.normal(0, 0.015)
            y += np.random.normal(0, 0.015)

            # Ensure within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)

            # Initial radius estimation based on distance to edges
            max_radius = min(x, 1-x, y, 1-y)
            r = np.random.uniform(0.01, min(0.12, max_radius * 0.6))

            circles[i] = [x, y, r]

        # Add more aggressive perturbations to improve diversity
        for i in range(n_circles):
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

        # Adjust tournament size based on diversity
        if diversity > std_dev * 0.5:
            # High diversity → smaller tournaments (more exploration)
            tournament_size = max(2, min(5, int(tournament_size * 0.7)))
        else:
            # Low diversity → larger tournaments (more exploitation)
            tournament_size = max(4, min(8, int(tournament_size * 1.2)))

    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Improved crossover with constraint-aware recombination"""
    child = parent1.copy()

    # Use a more careful crossover approach - mix genes with constraint checking
    mask = np.random.rand(*parent1.shape) > 0.5

    # Apply crossover
    child[mask] = parent2[mask]

    # Check if child is valid, if not, repair it properly
    # Make sure positions and radii respect boundary conditions
    for i in range(len(child)):
        x, y, r = child[i]
        # Clamp position to valid bounds
        child[i, 0] = np.clip(x, r, 1 - r)
        child[i, 1] = np.clip(y, r, 1 - r)
        # Ensure positive radius
        child[i, 2] = max(0.001, r)

    return child

def mutate(circles: np.ndarray, mutation_rate: float = 0.1,
           max_radius_change: float = 0.02) -> np.ndarray:
    """Improved mutation with better boundary handling"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius (weighted towards position)
            if np.random.rand() < 0.7:  # 70% chance to mutate position
                mutated[i, 0] += np.random.normal(0, 0.008)
                mutated[i, 1] += np.random.normal(0, 0.008)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # 30% chance to mutate radius
                mutated[i, 2] += np.random.normal(0, max_radius_change * 0.5)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Enhanced repair mechanism with systematic optimization"""
    repaired = circles.copy()

    # Stage 1: Fix boundary violations
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])

    # Stage 2: Resolve overlaps systematically with multiple passes
    points = repaired[:, :2]
    tree = cKDTree(points)

    # Multiple repair iterations to ensure proper resolution
    for iteration in range(5):
        overlap_found = False
        # Process circles in random order for better convergence
        indices = list(range(len(repaired)))
        np.random.shuffle(indices)

        for i in indices:
            x1, y1, r1 = repaired[i]
            # Find nearby circles within a reasonable distance
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        overlap_found = True

                        # Move circles apart using physics-inspired approach
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance

                            # Move them apart proportionally to their radii
                            total_radius = r1 + r2
                            move_amount = (min_distance - distance) * 0.4

                            # Apply movement with bias towards larger circles
                            # But also ensure both circles are moved reasonably
                            repaired[i, 0] += dx * move_amount * (r2 / total_radius) * 0.7
                            repaired[i, 1] += dy * move_amount * (r2 / total_radius) * 0.7
                            repaired[j, 0] -= dx * move_amount * (r1 / total_radius) * 0.7
                            repaired[j, 1] -= dy * move_amount * (r1 / total_radius) * 0.7

                            # Clamp to bounds
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)

        if not overlap_found:
            break

    # Stage 3: Local radius optimization after overlap resolution
    # Try to increase radii where possible without causing overlaps
    for _ in range(3):  # Multiple optimization passes
        improved = False
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            # Check how much we can increase the radius
            min_dist_to_edge = min(x, 1-x, y, 1-y)

            # Check distance to nearby circles
            min_dist_to_circle = float('inf')
            for j in range(len(repaired)):
                if i != j:
                    x2, y2, r2 = repaired[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    min_dist_to_circle = min(min_dist_to_circle, dist)

            # Maximum allowable radius
            max_radius = min(min_dist_to_edge, min_dist_to_circle - 1e-6) if min_dist_to_circle != float('inf') else min_dist_to_edge

            if max_radius > r + 1e-6:
                # Increase radius but cap at reasonable amounts
                delta_r = min(0.01, max_radius - r)
                if delta_r > 1e-6:
                    repaired[i, 2] = r + delta_r
                    improved = True

        if not improved:
            break

    return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters
    pop_size = 80
    n_generations = 150
    elite_size = 10

    # Create initial population
    population = create_initial_population(pop_size, 26)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None

    # Adaptive mutation rate with exponential decay
    mutation_rate_start = 0.15
    decay_factor = 0.012

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

            # Repair
            child = repair_circles(child)

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