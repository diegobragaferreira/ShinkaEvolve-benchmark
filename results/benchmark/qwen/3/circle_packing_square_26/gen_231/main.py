# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 500
TOURNAMENT_SIZE = 7
MUTATION_RATE_START = 0.3
MUTATION_RATE_END = 0.01
CROSSOVER_PROB = 0.9
ELITISM_COUNT = 8
BOUNDARY_MARGIN = 0.01
SPATIAL_INDEXING_THRESHOLD = 50
BOUNDARY_PENALTY_BASE = 1000.0
OVERLAP_PENALTY_BASE = 10000.0

def poisson_disk_sampling(n_points: int, min_distance: float = 0.1) -> List[Tuple[float, float]]:
    """Generate points using Poisson disk sampling for better uniformity."""
    points = []
    active_list = []

    # Start with a random point
    points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
    active_list.append(0)

    while len(points) < n_points:
        if not active_list:
            break

        # Pick a random active point
        idx = random.choice(active_list)
        x, y = points[idx]

        # Try to generate a new point
        found = False
        for _ in range(30):  # Limit attempts
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_distance, 2 * min_distance)

            new_x = x + radius * math.cos(angle)
            new_y = y + radius * math.sin(angle)

            # Check bounds
            if new_x < 0.05 or new_x > 0.95 or new_y < 0.05 or new_y > 0.95:
                continue

            # Check distance to existing points
            too_close = False
            for px, py in points:
                dist = math.sqrt((new_x - px)**2 + (new_y - py)**2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                points.append((new_x, new_y))
                active_list.append(len(points) - 1)
                found = True
                break

        if not found:
            active_list.remove(idx)

    # If we didn't get enough points, fill with random ones
    while len(points) < n_points:
        points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))

    return points[:n_points]

def initialize_population(n: int, population_size: int) -> np.ndarray:
    """Initialize population with enhanced Voronoi and structured grid approach."""
    population = []

    # Generate points using Poisson disk sampling for better distribution
    sample_points = poisson_disk_sampling(n, 0.15)

    # Create structured grid for systematic placement
    grid_size = max(4, int(np.ceil(np.sqrt(n))))
    structured_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(structured_positions) < n:
                x = 0.1 + (i / (grid_size - 1)) * 0.8
                y = 0.1 + (j / (grid_size - 1)) * 0.8
                structured_positions.append((x, y))

    # Create multiple populations with variation
    for _ in range(population_size):
        circles = np.zeros((n, 3))

        # Distribute circles using combined approaches
        for i in range(n):
            # Use structured positions with perturbation
            if i < len(structured_positions):
                x_base, y_base = structured_positions[i]
                # Add systematic perturbation
                perturbation_x = random.uniform(-0.02, 0.02) + 0.01 * (i % 3 - 1)
                perturbation_y = random.uniform(-0.02, 0.02) + 0.01 * (i % 2 - 1)
                x = max(0.01, min(0.99, x_base + perturbation_x))
                y = max(0.01, min(0.99, y_base + perturbation_y))
            else:
                # Random placement
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)

            circles[i] = [x, y, 0.05]  # Base radius

        # Refine circles with better radius assignment
        for i in range(n):
            x, y, r = circles[i]
            margin = min(x, y, 1 - x, 1 - y)
            base_radius = min(0.15, margin / 2.0)
            # Use structured approach for better radius distribution
            if base_radius > 0.01:
                radius_variation = random.uniform(0.8, 1.2)
                circles[i, 2] = max(0.01, base_radius * radius_variation)
            else:
                circles[i, 2] = random.uniform(0.01, 0.1)

        # Ensure circles don't overlap initially
        circles = resolve_initial_overlaps(circles)
        population.append(circles)

    return np.array(population)

def resolve_initial_overlaps(circles: np.ndarray) -> np.ndarray:
    """Resolve overlaps in initial configuration using force-based approach."""
    resolved = circles.copy()

    # Iteratively resolve overlaps with better constraints
    for _ in range(10):
        changed = False
        if len(resolved) > SPATIAL_INDEXING_THRESHOLD:
            # Use efficient KDTree for large populations
            tree = cKDTree(resolved[:, :2])
            pairs = tree.query_pairs(0.001)
            
            for i, j in pairs:
                xi, yi, ri = resolved[i]
                xj, yj, rj = resolved[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)

                if dist < (ri + rj - 1e-6):
                    # Move circles apart using force-based approach
                    dx = xj - xi
                    dy = yj - yi
                    distance = max(1e-6, dist)
                    
                    # Normalize
                    dx /= distance
                    dy /= distance
                    
                    # Move based on inverse radius ratio
                    move_amount = (ri + rj - dist) * 0.5
                    
                    # Apply movement in opposite directions
                    resolved[i, 0] -= dx * move_amount * 0.4
                    resolved[i, 1] -= dy * move_amount * 0.4
                    resolved[j, 0] += dx * move_amount * 0.4
                    resolved[j, 1] += dy * move_amount * 0.4
                    changed = True
        else:
            # Use grid-based approach for small populations
            grid = get_grid_cells(resolved, 20)

            for (gx, gy), indices in grid.items():
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        xi, yi, ri = resolved[idx1]
                        xj, yj, rj = resolved[idx2]
                        dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)

                        if dist < (ri + rj - 1e-6):
                            # Move circles apart using force-based approach
                            dx = xj - xi
                            dy = yj - yi
                            distance = max(1e-6, dist)
                            
                            # Normalize
                            dx /= distance
                            dy /= distance
                            
                            # Move based on inverse radius ratio
                            move_amount = (ri + rj - dist) * 0.5
                            
                            # Apply movement in opposite directions
                            resolved[idx1, 0] -= dx * move_amount * 0.4
                            resolved[idx1, 1] -= dy * move_amount * 0.4
                            resolved[idx2, 0] += dx * move_amount * 0.4
                            resolved[idx2, 1] += dy * move_amount * 0.4
                            changed = True

        # Ensure bounds
        for i in range(len(resolved)):
            x, y, r = resolved[i]
            # Clamp to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            resolved[i] = [x, y, r]

        if not changed:
            break

    return resolved

def get_grid_cells(circles: np.ndarray, grid_size: int = 20) -> dict:
    """Create a spatial grid for fast neighbor lookups."""
    grid = {}
    cell_size = 1.0 / grid_size

    for i, (x, y, r) in enumerate(circles):
        # Determine which grid cells this circle might occupy
        min_x_cell = max(0, int((x - r) / cell_size))
        max_x_cell = min(grid_size - 1, int((x + r) / cell_size))
        min_y_cell = max(0, int((y - r) / cell_size))
        max_y_cell = min(grid_size - 1, int((y + r) / cell_size))

        for gx in range(min_x_cell, max_x_cell + 1):
            for gy in range(min_y_cell, max_y_cell + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)

    return grid

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration is valid (no overlaps, fully contained)"""
    n_circles = len(circles)
    
    # Check boundary containment
    for i in range(n_circles):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps using KDTree for efficiency
    points = circles[:, :2]
    if n_circles > SPATIAL_INDEXING_THRESHOLD:
        tree = cKDTree(points)
        pairs = tree.query_pairs(2 * np.max(circles[:, 2]), p=np.inf)
        for i, j in pairs:
            if i != j:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
    else:
        # For smaller populations, use direct computation
        for i in range(n_circles):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n_circles):
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
    
    return True

def calculate_penalty(circles: np.ndarray, generation: int = 0, total_generations: int = 100) -> float:
    """Compute penalty based on constraint violations with progressive scaling."""
    penalty = 0.0
    
    # Dynamic penalty scaling factor
    penalty_scale = 1.0 + (generation / total_generations) * 5.0

    # Check containment violations with scaled penalties
    for x, y, r in circles:
        # Boundary violations
        if x - r < 0:
            penalty += (abs(x - r) ** 2) * BOUNDARY_PENALTY_BASE * penalty_scale
        elif x + r > 1:
            penalty += (abs(x + r - 1) ** 2) * BOUNDARY_PENALTY_BASE * penalty_scale
        if y - r < 0:
            penalty += (abs(y - r) ** 2) * BOUNDARY_PENALTY_BASE * penalty_scale
        elif y + r > 1:
            penalty += (abs(y + r - 1) ** 2) * BOUNDARY_PENALTY_BASE * penalty_scale

    # Check overlap violations with scaled penalties
    if not is_valid_configuration(circles):
        penalty += OVERLAP_PENALTY_BASE * penalty_scale

    return penalty

def evaluate_fitness(circles: np.ndarray, generation: int = 0, total_generations: int = 100) -> float:
    """Evaluate fitness of a circle configuration."""
    # If invalid, heavily penalize
    if not is_valid_configuration(circles):
        penalty = calculate_penalty(circles, generation, total_generations)
        return -penalty

    # Otherwise, return total radius
    total_radius = np.sum(circles[:, 2])
    return total_radius

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive rates."""
    mutated = circles.copy()

    # Adaptive mutation rate using exponential decay
    progress = generation / total_generations
    mutation_rate = MUTATION_RATE_START * (1 - progress)**2 + MUTATION_RATE_END * progress

    n = len(mutated)

    # Mutate some circles
    for i in range(n):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate with higher probability for position
            choices = [0, 0, 1, 1, 2]  # Position mutations more likely
            choice = random.choice(choices)

            if choice == 0:  # X coordinate
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, 0.025)))
            elif choice == 1:  # Y coordinate
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, 0.025)))
            else:  # Radius
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.02)))

    # Ensure valid configuration after mutation
    return enforce_constraints(mutated)

def enforce_constraints(circles: np.ndarray) -> np.ndarray:
    """Enforce constraints on circle positions and radii."""
    result = circles.copy()

    # Adjust positions and radii to satisfy bounds
    for i in range(len(result)):
        x, y, r = result[i]

        # Ensure circle fits in the unit square
        r = min(r, x, 1-x, y, 1-y)
        r = max(0.001, min(0.49, r))

        # Clamp coordinates to valid range
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))

        result[i] = [x, y, r]

    return result

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two parent configurations."""
    if random.random() > CROSSOVER_PROB:
        # Return one of the parents randomly
        return parent1.copy() if random.random() < 0.5 else parent2.copy()

    n = len(parent1)
    child = np.zeros_like(parent1)

    # Multi-point crossover with strategic point selection
    crossover_points = sorted(random.sample(range(1, n), min(4, n-1)))

    # Alternate between parents for segments
    last_point = 0
    use_parent1 = True
    for point in crossover_points:
        if use_parent1:
            child[last_point:point, :] = parent1[last_point:point, :]
        else:
            child[last_point:point, :] = parent2[last_point:point, :]
        last_point = point
        use_parent1 = not use_parent1

    # Handle final segment
    if use_parent1:
        child[last_point:, :] = parent1[last_point:, :]
    else:
        child[last_point:, :] = parent2[last_point:, :]

    # Apply local refinement to ensure validity
    refined_child = refine_configuration(child)

    return refined_child

def refine_configuration(circles: np.ndarray) -> np.ndarray:
    """Refine configuration to remove overlaps and correct constraints."""
    refined = circles.copy()

    # Force-based refinement with better overlap resolution
    for iteration in range(10):
        resolved = False

        if len(refined) > SPATIAL_INDEXING_THRESHOLD:
            # Use efficient KDTree for large populations
            tree = cKDTree(refined[:, :2])
            pairs = tree.query_pairs(0.001)

            for i, j in pairs:
                xi, yi, ri = refined[i]
                xj, yj, rj = refined[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)

                if dist < (ri + rj - 1e-6):
                    # Resolve overlap by moving circles apart with force-based approach
                    dx = xj - xi
                    dy = yj - yi
                    distance = max(1e-6, dist)

                    # Normalize direction vector
                    dx /= distance
                    dy /= distance

                    # Move circles apart based on their relative sizes and distances
                    move_amount = (ri + rj - dist) * 0.5

                    # Scale by inverse radii to balance movement
                    scale_factor = min(1.0, ri / (ri + rj + 1e-6))
                    refined[i, 0] -= dx * move_amount * scale_factor * 0.3
                    refined[i, 1] -= dy * move_amount * scale_factor * 0.3
                    refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.3
                    refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.3
                    resolved = True
        else:
            # Use grid-based approach for small populations
            grid = get_grid_cells(refined, 20)

            # Check for overlaps and resolve them
            for (gx, gy), indices in grid.items():
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        xi, yi, ri = refined[idx1]
                        xj, yj, rj = refined[idx2]
                        dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)

                        if dist < (ri + rj - 1e-6):
                            # Resolve overlap by moving circles apart with force-based approach
                            dx = xj - xi
                            dy = yj - yi
                            distance = max(1e-6, dist)

                            # Normalize direction vector
                            dx /= distance
                            dy /= distance

                            # Move circles apart based on their relative sizes and distances
                            move_amount = (ri + rj - dist) * 0.5

                            # Scale by inverse radii to balance movement
                            scale_factor = min(1.0, ri / (ri + rj + 1e-6))
                            refined[idx1, 0] -= dx * move_amount * scale_factor * 0.3
                            refined[idx1, 1] -= dy * move_amount * scale_factor * 0.3
                            refined[idx2, 0] += dx * move_amount * (1 - scale_factor) * 0.3
                            refined[idx2, 1] += dy * move_amount * (1 - scale_factor) * 0.3
                            resolved = True

        # Enforce bounds
        for i in range(len(refined)):
            x, y, r = refined[i]
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]

        # Early stopping if no changes made
        if not resolved:
            break

    return refined

def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    """Select parent using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def evolve_circles(n_circles: int = 26, generations: int = GENERATIONS) -> np.ndarray:
    """Main evolutionary algorithm with improved strategies"""
    # Initialize population
    population = initialize_population(n_circles, POPULATION_SIZE)
    
    best_fitness_history = []
    
    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(ind, gen, generations) for ind in population]
        
        # Track best fitness
        best_fitness = max(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
        elites = population[elite_indices].copy()
        
        # Create new population
        new_population = []
        
        # Add elites first
        new_population.extend(elites)
        
        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitness_scores, TOURNAMENT_SIZE)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation with adaptive strategy
            child = mutate(child, gen, generations)
            
            new_population.append(child)
        
        # Trim to exact population size
        population = np.array(new_population[:POPULATION_SIZE])
        
        # Early stopping when improvement plateaus
        if len(best_fitness_history) > 20:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-20]
            if recent_improvement < 0.0005:
                break
    
    # Final local optimization on best individuals
    final_fitness_scores = [evaluate_fitness(ind, generations, generations) for ind in population]
    best_idx = np.argmax(final_fitness_scores)
    
    # Apply final refinement
    refined_solution = refine_configuration(population[best_idx])
    
    return refined_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    try:
        circles = evolve_circles(n_circles=26, generations=GENERATIONS)
        return circles
    except Exception as e:
        # Fallback to simple initialization if evolution fails
        print(f"Evolution failed: {e}")
        circles = np.zeros((26, 3))
        # Simple grid initialization
        grid_size = int(np.ceil(np.sqrt(26)))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                r = 0.02
                circles[count] = [x, y, r]
                count += 1
        return circles

# EVOLVE-BLOCK-END