# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math

# Global constants for optimization
POP_SIZE = 50
GENERATIONS = 1000
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.25
MUTATION_RATE_END = 0.01
CROSSOVER_PROB = 0.9
VALIDITY_THRESHOLD = 1e-6
INITIAL_RADIUS_FACTOR = 0.3

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

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
    """Initialize population with improved Voronoi-based distribution using Poisson disk sampling."""
    population = []
    
    # Generate better distributed points using Poisson disk sampling
    sample_points = poisson_disk_sampling(n, 0.15)
    
    # Create multiple populations with variation
    for _ in range(population_size):
        circles = np.zeros((n, 3))
        
        # Distribute circles using the sample points
        for i in range(min(n, len(sample_points))):
            x_base, y_base = sample_points[i]
            
            # Add jitter for diversity
            x = max(0.01, min(0.99, x_base + random.uniform(-0.03, 0.03)))
            y = max(0.01, min(0.99, y_base + random.uniform(-0.03, 0.03)))
            
            # Initial radius - start with moderately large values
            circles[i] = [x, y, INITIAL_RADIUS_FACTOR]
        
        # Fill remaining circles
        for i in range(len(sample_points), n):
            # Place remaining circles more randomly but still with some structure
            if random.random() < 0.4:
                # Near an existing circle
                idx = random.randint(0, min(i-1, len(sample_points)-1))
                x_base, y_base = sample_points[idx]
                x = max(0.01, min(0.99, x_base + random.uniform(-0.08, 0.08)))
                y = max(0.01, min(0.99, y_base + random.uniform(-0.08, 0.08)))
            else:
                # Completely random
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
            
            circles[i] = [x, y, INITIAL_RADIUS_FACTOR * 0.5]
        
        # Ensure circles don't overlap initially
        circles = resolve_initial_overlaps(circles)
        population.append(circles)
    
    return np.array(population)

def resolve_initial_overlaps(circles: np.ndarray) -> np.ndarray:
    """Resolve overlaps in initial configuration using force-based approach."""
    resolved = circles.copy()
    
    # Iteratively resolve overlaps
    for _ in range(10):
        changed = False
        # Use scipy spatial tree for efficient neighbor checking
        try:
            tree = cKDTree(resolved[:, :2])
            pairs = tree.query_pairs(r=2.0, output_type='ndarray')
            
            for i, j in pairs:
                if i < j:  # Avoid duplicates
                    xi, yi, ri = resolved[i]
                    xj, yj, rj = resolved[j]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    
                    if dist < (ri + rj - VALIDITY_THRESHOLD):
                        # Move circles apart
                        dx = xj - xi
                        dy = yj - yi
                        distance = max(VALIDITY_THRESHOLD, dist)
                        
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
        except Exception:
            # Fallback to brute force if spatial indexing fails
            for i in range(len(resolved)):
                for j in range(i+1, len(resolved)):
                    xi, yi, ri = resolved[i]
                    xj, yj, rj = resolved[j]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    
                    if dist < (ri + rj - VALIDITY_THRESHOLD):
                        # Move circles apart
                        dx = xj - xi
                        dy = yj - yi
                        distance = max(VALIDITY_THRESHOLD, dist)
                        
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

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained in the unit square."""
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap_efficient(circles: np.ndarray) -> bool:
    """Check if any circles overlap using spatial tree for improved efficiency."""
    if len(circles) <= 1:
        return False
        
    try:
        # Use spatial tree for O(n log n) overlap detection
        tree = cKDTree(circles[:, :2])
        # Query pairs within sum of radii distance
        pairs = tree.query_pairs(r=2.0, output_type='ndarray')
        
        for i, j in pairs:
            if i < j:  # Avoid duplicate checks
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2 - VALIDITY_THRESHOLD):
                    return True
    except Exception:
        # Fallback to brute force
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2 - VALIDITY_THRESHOLD):
                    return True
    
    return False

def is_valid(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping."""
    # Check boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints
    return not check_overlap_efficient(circles)

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def evaluate_fitness(circles: np.ndarray, generation: int = 0, total_generations: int = 1000) -> float:
    """Evaluate fitness of a solution, higher is better."""
    if not is_valid(circles):
        # Apply penalty for constraint violations with progressive scaling
        penalty = 0
        penalty_scale = 1.0 + (generation / total_generations) * 5.0

        # Boundary penalty with stronger weighting
        boundary_violations = 0
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0:
                boundary_violations += (r - x)**2 * 10000 * penalty_scale
            if x + r > 1:
                boundary_violations += (x + r - 1)**2 * 10000 * penalty_scale
            if y - r < 0:
                boundary_violations += (r - y)**2 * 10000 * penalty_scale
            if y + r > 1:
                boundary_violations += (y + r - 1)**2 * 10000 * penalty_scale

        penalty += boundary_violations

        # Overlap penalty with stronger weighting
        overlap_penalty = 0
        # Use spatial tree for overlap checking
        try:
            tree = cKDTree(circles[:, :2])
            pairs = tree.query_pairs(r=2.0, output_type='ndarray')
            
            for i, j in pairs:
                if i < j:  # Avoid duplicate checks
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < (r1 + r2 - VALIDITY_THRESHOLD):
                        overlap_penalty += (r1 + r2 - dist)**2 * 100000 * penalty_scale
        except Exception:
            # Fallback to brute force
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < (r1 + r2 - VALIDITY_THRESHOLD):
                        overlap_penalty += (r1 + r2 - dist)**2 * 100000 * penalty_scale

        penalty += overlap_penalty

        return -penalty - 1000000

    return calculate_sum_radii(circles)

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive dual strategy."""
    mutated = circles.copy()

    # Adaptive mutation rate
    mutation_rate_start = 0.25
    mutation_rate_end = 0.01
    mutation_rate = mutation_rate_start + (mutation_rate_end - mutation_rate_start) * \
                   (generation / total_generations)
    
    # Dual mutation strategy based on generation phase
    if generation < total_generations * 0.3:
        # Early phase: large mutations for exploration
        mutation_strength = 0.05
        pos_mutate_weight = 0.6  # Higher chance for position changes
        rad_mutate_weight = 0.1
    elif generation < total_generations * 0.7:
        # Middle phase: balanced mutations
        mutation_strength = 0.02
        pos_mutate_weight = 0.4
        rad_mutate_weight = 0.4
    else:
        # Late phase: small mutations for exploitation
        mutation_strength = 0.01
        pos_mutate_weight = 0.2
        rad_mutate_weight = 0.8  # Higher chance for radius fine-tuning
    
    n = len(mutated)
    
    # Mutate some circles
    for i in range(n):
        if random.random() < mutation_rate:
            # Choose mutation type with weighted probabilities
            choice = random.choices([0, 1, 2], weights=[pos_mutate_weight, pos_mutate_weight, rad_mutate_weight])[0]
            
            if choice == 0:  # X coordinate
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, mutation_strength)))
            elif choice == 1:  # Y coordinate
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, mutation_strength)))
            else:  # Radius
                # Apply bounded mutation to radius using log-normal distribution
                log_factor = random.gauss(0, 0.15)
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] * math.exp(log_factor)))

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
    
    # Single point crossover
    crossover_point = random.randint(1, n-1)
    
    for i in range(n):
        if i < crossover_point:
            child[i] = parent1[i].copy()
        else:
            child[i] = parent2[i].copy()
    
    # Apply local refinement to ensure validity
    refined_child = refine_configuration(child)
    
    return refined_child

def refine_configuration(circles: np.ndarray) -> np.ndarray:
    """Refine configuration to remove overlaps and correct constraints."""
    refined = circles.copy()
    
    # Force-based refinement with better overlap resolution
    for iteration in range(8):
        # Use spatial tree for overlap detection  
        try:
            tree = cKDTree(refined[:, :2])
            pairs = tree.query_pairs(r=2.0, output_type='ndarray')
            
            resolved = False
            for i, j in pairs:
                if i < j:  # Avoid duplicate checks  
                    xi, yi, ri = refined[i]
                    xj, yj, rj = refined[j]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    
                    if dist < (ri + rj - VALIDITY_THRESHOLD):
                        # Resolve overlap by moving circles apart with force-based approach
                        dx = xj - xi
                        dy = yj - yi
                        distance = max(VALIDITY_THRESHOLD, dist)
                        
                        # Normalize direction vector
                        dx /= distance
                        dy /= distance
                        
                        # Move circles apart based on their relative sizes and distances
                        move_amount = (ri + rj - dist) * 0.5
                        
                        # Scale by inverse radii to balance movement
                        scale_factor = min(1.0, ri / (ri + rj + 0.001))
                        refined[i, 0] -= dx * move_amount * scale_factor * 0.3
                        refined[i, 1] -= dy * move_amount * scale_factor * 0.3
                        refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.3
                        refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.3
                        resolved = True
        except Exception:
            # Fallback to brute force
            resolved = False
            for i in range(len(refined)):
                for j in range(i+1, len(refined)):
                    xi, yi, ri = refined[i]
                    xj, yj, rj = refined[j]
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    
                    if dist < (ri + rj - VALIDITY_THRESHOLD):
                        # Resolve overlap by moving circles apart with force-based approach
                        dx = xj - xi
                        dy = yj - yi
                        distance = max(VALIDITY_THRESHOLD, dist)
                        
                        # Normalize direction vector
                        dx /= distance
                        dy /= distance
                        
                        # Move circles apart based on their relative sizes and distances
                        move_amount = (ri + rj - dist) * 0.5
                        
                        # Scale by inverse radii to balance movement
                        scale_factor = min(1.0, ri / (ri + rj + 0.001))
                        refined[i, 0] -= dx * move_amount * scale_factor * 0.3
                        refined[i, 1] -= dy * move_amount * scale_factor * 0.3
                        refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.3
                        refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.3
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

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize population
    population = initialize_population(n, POP_SIZE)
    
    # Evaluate initial population
    fitnesses = [evaluate_fitness(individual, 0, GENERATIONS) for individual in population]
    
    # Evolution loop
    for gen in range(GENERATIONS):
        # Selection, crossover, and mutation
        new_population = []
        
        # Keep best individuals (elitism)
        elite_count = POP_SIZE // 10
        sorted_indices = np.argsort(fitnesses)[::-1][:elite_count]
        for idx in sorted_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < POP_SIZE:
            # Tournament selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, gen, GENERATIONS)
            
            # Apply local refinement to ensure validity
            child = refine_configuration(child)
            
            new_population.append(child)
        
        # Evaluate new population
        population = np.array(new_population)
        fitnesses = [evaluate_fitness(individual, gen, GENERATIONS) for individual in population]
        
        # Print progress
        best_fitness = max(fitnesses)
        if gen % 100 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness}")
    
    # Return the best individual
    best_index = np.argmax(fitnesses)
    best_solution = population[best_index]
    
    return best_solution

# EVOLVE-BLOCK-END