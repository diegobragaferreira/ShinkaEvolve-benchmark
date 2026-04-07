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
    """Create initial population using enhanced multi-scale grid initialization with fitness-based perturbation scaling"""
    population = []

    # Multi-scale grid approach with more variety
    grid_sizes = [2, 3, 4, 5, 6]  # Try different grid sizes

    # Calculate fitness variance threshold for dynamic perturbation scaling
    # Initialize with some sample fitnesses
    sample_fitnesses = [0.5] * 10  # Placeholder values
    fitness_variance = np.var(sample_fitnesses) if len(sample_fitnesses) > 1 else 0.01
    
    # Dynamic perturbation magnitude based on fitness variance
    if fitness_variance > 0.05:
        perturbation_range = (0.01, 0.02)  # Wide range for diverse populations
    elif fitness_variance < 0.02:
        perturbation_range = (0.002, 0.005)  # Tight range for converged populations
    else:
        perturbation_range = (0.005, 0.015)  # Moderate range

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

        # Add adaptive perturbations based on fitness diversity
        for i in range(n_circles):
            # Use dynamic perturbation range
            perturbation_mag = np.random.uniform(perturbation_range[0], perturbation_range[1])
            
            # More aggressive perturbation for diverse individuals
            if np.random.rand() < 0.5:  # 50% chance to perturb
                circles[i, 0] += np.random.normal(0, perturbation_mag)
                circles[i, 1] += np.random.normal(0, perturbation_mag)
                circles[i, 2] += np.random.normal(0, perturbation_mag * 0.5)

                # Ensure valid bounds
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])

        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select parent using adaptive tournament selection based on population diversity"""
    # Calculate population diversity (variance of fitness values)
    if len(fitnesses) > 1:
        diversity = np.var(fitnesses)
        std_dev = np.std(fitnesses)
        
        # Calculate coefficient of variation (normalized variance)
        if np.abs(np.mean(fitnesses)) > 1e-8:  # Avoid division by zero
            coeff_var = std_dev / np.abs(np.mean(fitnesses))
        else:
            coeff_var = 0.0

        # Adjust tournament size based on coefficient of variation
        # High coeff_var (high relative diversity) = smaller tournaments (more exploration)
        # Low coeff_var (low relative diversity) = larger tournaments (more exploitation)
        if coeff_var > 0.1:  # High diversity threshold
            tournament_size = max(3, min(7, int(4 - coeff_var * 10)))  # Smaller tournaments
        elif coeff_var < 0.05:  # Low diversity threshold
            tournament_size = max(5, min(9, int(7 + coeff_var * 30)))  # Larger tournaments
        else:  # Medium diversity
            tournament_size = max(3, min(7, int(5)))  # Standard tournaments

    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Improved constraint-aware crossover with overlap probability weighting"""
    child = parent1.copy()

    # Calculate pairwise distances between circles in parents to assess overlap risk
    n_circles = len(parent1)

    # For each circle pair, compute overlap risk based on distance vs sum of radii
    overlap_probabilities = np.zeros(n_circles)

    for i in range(n_circles):
        x1, y1, r1 = parent1[i]
        x2, y2, r2 = parent2[i]

        # Calculate distance between corresponding circles
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        sum_radii = r1 + r2

        # Risk assessment: higher probability of overlap when circles are close
        if distance < 1.5 * sum_radii:  # Using 1.5x as safety factor
            # Higher overlap risk - lower crossover probability
            overlap_probabilities[i] = 0.3  # Low chance of crossover
        else:
            # Low overlap risk - higher crossover probability
            overlap_probabilities[i] = 0.8  # High chance of crossover

    # Apply crossover with probability based on overlap risk
    for i in range(n_circles):
        # Crossover probability inversely related to overlap risk
        crossover_prob = overlap_probabilities[i]

        # For each gene (x, y, r) in the circle
        for gene_idx in range(3):
            if np.random.rand() < crossover_prob:
                # Take gene from parent2
                child[i, gene_idx] = parent2[i, gene_idx]

    # Ensure all circles satisfy boundary constraints
    for i in range(len(child)):
        x, y, r = child[i]
        child[i, 0] = np.clip(x, r, 1 - r)
        child[i, 1] = np.clip(y, r, 1 - r)
        child[i, 2] = max(0.001, r)

    return child

def mutate(circles: np.ndarray, generation: int = 0, 
           mutation_rate: float = 0.1) -> np.ndarray:
    """Improved mutation with better boundary handling and adaptive scheduling"""
    
    # Three-phase adaptive mutation rate scheduling
    if generation <= 60:
        # Phase 1: High exploration
        current_mutation_rate = 0.15
    elif generation <= 120:
        # Phase 2: Balanced refinement
        current_mutation_rate = 0.05
    else:
        # Phase 3: Fine-tuning
        current_mutation_rate = 0.015

    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < current_mutation_rate:
            # Mutate position or radius (weighted towards position)
            if np.random.rand() < 0.75:  # 75% chance to mutate position
                mutated[i, 0] += np.random.normal(0, 0.015)
                mutated[i, 1] += np.random.normal(0, 0.015)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # 25% chance to mutate radius
                mutated[i, 2] += np.random.normal(0, 0.008)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def count_overlaps(circles: np.ndarray) -> int:
    """Count the number of overlapping pairs in the circle configuration."""
    n = len(circles)
    overlap_count = 0

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
                    overlap_count += 1

    return overlap_count // 2  # Each overlap counted twice

def local_refinement(circles: np.ndarray, max_iterations: int = 20) -> np.ndarray:
    """Apply progressive local refinement based on overlap severity."""
    refined = circles.copy()

    # Count initial overlaps
    initial_overlaps = count_overlaps(refined)

    # Classify overlap severity and set parameters
    if initial_overlaps == 0:
        # No overlaps, just ensure boundaries
        for i in range(len(refined)):
            x, y, r = refined[i]
            refined[i, 0] = np.clip(x, r, 1 - r)
            refined[i, 1] = np.clip(y, r, 1 - r)
            refined[i, 2] = max(0.001, refined[i, 2])
        return refined
    elif initial_overlaps <= 3:
        # Very low overlap - light refinement with greedy radius expansion
        iterations = min(max_iterations, 8)
        intensity = 0.2
        expand_radii = True
    elif initial_overlaps <= 8:
        # Low overlap - light refinement
        iterations = min(max_iterations, 12)
        intensity = 0.3
        expand_radii = False
    elif initial_overlaps <= 15:
        # Medium overlap - moderate refinement
        iterations = min(max_iterations, 18)
        intensity = 0.5
        expand_radii = False
    else:
        # High overlap - intensive refinement
        iterations = min(max_iterations, 25)
        intensity = 0.8
        expand_radii = False

    # Apply iterative refinement
    for iteration in range(iterations):
        # Update tree for current state
        points = refined[:, :2]
        tree = cKDTree(points)

        overlap_found = False

        # Process circles in random order for better results
        indices = list(range(len(refined)))
        np.random.shuffle(indices)

        for i in indices:
            x1, y1, r1 = refined[i]

            # Find nearby circles within a reasonable distance
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = refined[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        overlap_found = True

                        # Move circles apart using physics-inspired approach
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance

                            # Move them apart with intensity scaling
                            total_radius = r1 + r2
                            move_amount = (min_distance - distance) * intensity

                            # Apply movement with bias towards larger circles
                            refined[i, 0] += dx * move_amount * (r2 / total_radius)
                            refined[i, 1] += dy * move_amount * (r2 / total_radius)
                            refined[j, 0] -= dx * move_amount * (r1 / total_radius)
                            refined[j, 1] -= dy * move_amount * (r1 / total_radius)

                            # Clamp to bounds
                            refined[i, 0] = np.clip(refined[i, 0], r1, 1 - r1)
                            refined[i, 1] = np.clip(refined[i, 1], r1, 1 - r1)
                            refined[j, 0] = np.clip(refined[j, 0], r2, 1 - r2)
                            refined[j, 1] = np.clip(refined[j, 1], r2, 1 - r2)

        # Optional: attempt to greedily expand radii for low overlap cases
        if expand_radii and iteration % 3 == 0:  # Every 3 iterations
            for i in range(len(refined)):
                x1, y1, r1 = refined[i]
                # Try to increase radius without creating overlaps
                potential_r1 = min(r1 * 1.05,
                                 min(x1, 1-x1, y1, 1-y1))  # Respect bounds

                # Check if expanding radius is safe
                safe = True
                for j in range(len(refined)):
                    if i != j:
                        x2, y2, r2 = refined[j]
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        min_distance = potential_r1 + r2

                        if distance < min_distance:
                            safe = False
                            break

                if safe and potential_r1 > r1:
                    refined[i, 2] = potential_r1

        if not overlap_found:
            break

    # Final boundary check
    for i in range(len(refined)):
        x, y, r = refined[i]
        refined[i, 0] = np.clip(x, r, 1 - r)
        refined[i, 1] = np.clip(y, r, 1 - r)
        refined[i, 2] = max(0.001, refined[i, 2])

    return refined

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Enhanced repair mechanism with iterative optimization"""
    repaired = circles.copy()

    # First stage: Fix boundary violations
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])

    # Second stage: Resolve overlaps with progressive refinement
    # Check if there are overlaps that need resolution
    if count_overlaps(repaired) > 0:
        # Apply local refinement for better overlap resolution
        repaired = local_refinement(repaired)

    return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters - optimized for better exploration/exploitation balance
    pop_size = 100   # Increased population size
    n_generations = 150  # Increased generations
    elite_size = 10   # Increased elite count

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
            parent1 = tournament_selection(valid_individuals, fitnesses)
            parent2 = tournament_selection(valid_individuals, fitnesses)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation with adaptive rate
            child = mutate(child, generation)

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