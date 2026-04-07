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

    Args:
        circles: np.array of shape (n, 3) where each row is (x, y, r)

    Returns:
        True if all circles are valid, False otherwise
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
    """Create initial population using multi-scale grid initialization"""
    population = []

    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))

        # Multi-scale approach: use different grid sizes for different portions
        # First, place circles in multiple grid layers
        layer_sizes = [4, 9, 13]  # Different grid sizes
        layer_positions = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]  # Grid centers
        
        # Layer 1: coarse grid
        if n_circles >= 4:
            grid_size = 2
            spacing_x = 0.5 / (grid_size + 1)
            spacing_y = 0.5 / (grid_size + 1)
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= 4:
                        break
                    x = layer_positions[0][0] + (i + 1) * spacing_x - 0.25
                    y = layer_positions[0][1] + (j + 1) * spacing_y - 0.25
                    # Add randomness
                    x += np.random.uniform(-spacing_x/3, spacing_x/3)
                    y += np.random.uniform(-spacing_y/3, spacing_y/3)
                    r = np.random.uniform(0.02, 0.05)
                    # Ensure bounds
                    if 0.01 <= x <= 0.49 and 0.01 <= y <= 0.49:
                        circles[idx] = [x, y, r]
                        idx += 1
                if idx >= 4:
                    break

        # Layer 2: medium grid
        if n_circles >= 9:
            grid_size = 3
            spacing_x = 0.5 / (grid_size + 1)
            spacing_y = 0.5 / (grid_size + 1)
            idx = 4
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= 9:
                        break
                    x = layer_positions[1][0] + (i + 1) * spacing_x - 0.25
                    y = layer_positions[1][1] + (j + 1) * spacing_y - 0.25
                    # Add randomness
                    x += np.random.uniform(-spacing_x/3, spacing_x/3)
                    y += np.random.uniform(-spacing_y/3, spacing_y/3)
                    r = np.random.uniform(0.015, 0.04)
                    # Ensure bounds
                    if 0.26 <= x <= 0.74 and 0.26 <= y <= 0.74:
                        circles[idx] = [x, y, r]
                        idx += 1
                if idx >= 9:
                    break

        # Layer 3: fine grid
        if n_circles >= 13:
            grid_size = 3
            spacing_x = 0.5 / (grid_size + 1)
            spacing_y = 0.5 / (grid_size + 1)
            idx = 9
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= 13:
                        break
                    x = layer_positions[2][0] + (i + 1) * spacing_x - 0.25
                    y = layer_positions[2][1] + (j + 1) * spacing_y - 0.25
                    # Add randomness
                    x += np.random.uniform(-spacing_x/3, spacing_x/3)
                    y += np.random.uniform(-spacing_y/3, spacing_y/3)
                    r = np.random.uniform(0.01, 0.03)
                    # Ensure bounds
                    if 0.51 <= x <= 0.99 and 0.51 <= y <= 0.99:
                        circles[idx] = [x, y, r]
                        idx += 1
                if idx >= 13:
                    break

        # Fill remaining circles randomly while ensuring they're valid
        idx = 13
        while idx < n_circles:
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            r = np.random.uniform(0.005, 0.02)
            
            # Ensure valid bounds
            max_r = min(x, 1-x, y, 1-y)
            r = min(r, max_r * 0.9)
            
            if r > 0.001:
                circles[idx] = [x, y, r]
                idx += 1

        # Add some random perturbations to improve diversity
        for i in range(n_circles):
            if np.random.rand() < 0.4:  # 40% chance to perturb
                circles[i, 0] += np.random.normal(0, 0.01)
                circles[i, 1] += np.random.normal(0, 0.01)
                circles[i, 2] += np.random.normal(0, 0.003)

                # Ensure valid bounds
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])

        population.append(circles)

    return population

def compute_diversity(population: List[np.ndarray]) -> float:
    """Compute population diversity based on radius variation"""
    if len(population) < 2:
        return 0.0
    
    all_radii = np.concatenate([circles[:, 2] for circles in population])
    return np.std(all_radii)

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select parent using tournament selection with diversity adjustment"""
    # Increase tournament size if diversity is high
    if len(population) > 10:
        diversity = compute_diversity(population)
        if diversity > 0.05:
            tournament_size = min(6, tournament_size + 1)
    
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Constraint-aware crossover that favors valid configurations"""
    child = parent1.copy()
    
    # 70% chance of uniform crossover
    if np.random.random() < 0.7:
        # Apply crossover with 50% probability for each gene
        mask = np.random.rand(*parent1.shape) > 0.5
        child[mask] = parent2[mask]
    else:
        # 30% chance of blending - take weighted average of similar elements
        alpha = np.random.random()
        for i in range(len(child)):
            child[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]

    return child

def mutate(circles: np.ndarray, mutation_rate: float = 0.1,
           max_radius_change: float = 0.02) -> np.ndarray:
    """Apply mutation to a circle configuration with enhanced strategy"""
    mutated = circles.copy()

    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # 70% chance to mutate position, 30% to mutate radius
            if np.random.rand() < 0.7:  # Mutate position
                # Larger mutation for better exploration
                mutated[i, 0] += np.random.normal(0, 0.02)
                mutated[i, 1] += np.random.normal(0, 0.02)

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # Mutate radius
                mutated[i, 2] += np.random.normal(0, max_radius_change * 0.5)
                mutated[i, 2] = max(0.001, mutated[i, 2])

    return mutated

def repair_circles(circles: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """Repair invalid circle configurations with iterative overlap resolution"""
    repaired = circles.copy()

    # First ensure all circles are within bounds
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        # Keep within bounds
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])  # Ensure positive radius

    # Resolve overlaps using iterative approach
    for iteration in range(max_iterations):
        points = repaired[:, :2]
        tree = cKDTree(points)
        any_changes = False

        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]

            # Find nearby circles
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))

            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2

                    if distance < min_distance:
                        # Repel circles apart
                        if distance > 0.001:  # Avoid division by zero
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance

                            # Move them apart
                            move_amount = (min_distance - distance) * 0.7
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

def local_optimization(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Local optimization that tries to increase radii while maintaining constraints"""
    optimized = circles.copy()
    
    # Try to increase radii systematically
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase each radius
        for i in range(len(optimized)):
            x, y, r = optimized[i]
            
            # Try to increase radius up to boundary
            max_possible_r = min(x, 1-x, y, 1-y)
            
            # Only try to increase if we have room
            if r < max_possible_r * 0.99:
                # Test a larger radius
                test_r = min(r + 0.001, max_possible_r * 0.99)
                
                # Check if it causes overlap
                valid = True
                for j in range(len(optimized)):
                    if i != j:
                        x2, y2, r2 = optimized[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < test_r + r2:
                            valid = False
                            break
                
                if valid:
                    optimized[i, 2] = test_r
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
    # Algorithm parameters
    pop_size = 60
    n_generations = 100
    elite_size = 8
    
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

        # Compute adaptive mutation rate with exponential decay
        # Start at 0.1, decay to 0.005 over 80 generations
        mutation_rate = 0.1 * (0.005/0.1)**(generation/80.0)
        mutation_rate = max(0.005, mutation_rate)  # Minimum rate
        
        # Fill remaining slots with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(valid_individuals, fitnesses)
            parent2 = tournament_selection(valid_individuals, fitnesses)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutate(child, mutation_rate)

            # Repair
            child = repair_circles(child)

            new_population.append(child)

        population = new_population[:pop_size]

    # Apply final local optimization to the best individual
    if best_individual is not None:
        best_individual = local_optimization(best_individual)
        
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