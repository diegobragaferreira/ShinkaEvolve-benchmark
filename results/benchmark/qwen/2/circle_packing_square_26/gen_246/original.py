# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 250
TOURNAMENT_SIZE = 4
MUTATION_RATE = 0.15
ELITISM_COUNT = 8
MAX_ATTEMPTS = 500

def validate_circle_placement(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and don't overlap"""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    for i in range(n):
        x, y, r = circles[i]
        # Find nearby circles (within 2*r distance)
        indices = tree.query_ball_point([x, y], 2*r)
        for j in indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate sum of all radii"""
    return np.sum(circles[:, 2])

def create_grid_initialization(n_circles: int, grid_levels: List[Tuple[int, int]] = None) -> np.ndarray:
    """Create initial circle configuration using multi-scale grid placement"""
    circles = np.zeros((n_circles, 3))
    
    if grid_levels is None:
        # Define multiple grid resolutions
        grid_levels = [(3, 3), (4, 4), (5, 5), (6, 6)]
    
    # Try different grid levels to find a good starting point
    success = False
    max_attempts = 20
    
    for attempt in range(max_attempts):
        # Select random grid level
        rows, cols = random.choice(grid_levels)
        
        if rows * cols < n_circles:
            continue
            
        # Generate grid positions
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n_circles:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                positions.append((x, y))
        
        if len(positions) < n_circles:
            continue
            
        # Assign circles with some randomness and proper radii
        for i in range(n_circles):
            x, y = positions[i]
            
            # Add random perturbation based on grid spacing
            spacing_x = 1.0 / cols
            spacing_y = 1.0 / rows
            perturbation = random.uniform(0.1, 0.3)
            x += (random.random() - 0.5) * spacing_x * perturbation
            y += (random.random() - 0.5) * spacing_y * perturbation
            
            # Limit x, y to valid range
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Calculate initial radius based on proximity to boundaries
            max_radius = min(x, 1-x, y, 1-y)
            # Start with a relatively small radius
            r = min(0.04, max_radius * random.uniform(0.4, 0.7))
            
            circles[i] = [x, y, r]
        
        # If valid, try local improvement
        if validate_circle_placement(circles):
            success = True
            break
    
    # If no valid initialization found, fallback to random
    if not success:
        for i in range(n_circles):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            max_radius = min(x, 1-x, y, 1-y)
            r = min(0.05, max_radius * random.uniform(0.3, 0.7))
            circles[i] = [x, y, r]
    
    return circles

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population with improved initialization strategy"""
    population = []
    
    for _ in range(pop_size):
        # Use grid-based initialization
        circles = create_grid_initialization(n_circles)
        
        # Apply local optimization to improve the placement
        circles = local_improvement(circles)
        
        # Ensure validity through repair if needed
        if not validate_circle_placement(circles):
            circles = repair_constraints(circles)
            
        population.append(circles)
    
    return population

def local_improvement(circles: np.ndarray) -> np.ndarray:
    """Apply smart local improvement to increase radii while satisfying constraints"""
    n = len(circles)
    circles_copy = circles.copy()

    # Perform multiple rounds of optimization
    for round_num in range(30):
        improved = False
        for i in range(n):
            x, y, r = circles_copy[i]
            
            # Calculate max possible radius at current position
            max_r = min(x, 1-x, y, 1-y)
            
            if max_r <= r:
                continue
                
            # Try to increase radius
            new_r = min(r + 0.01, max_r)
            
            # Check if we can actually increase it without violating constraints
            valid = True
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles_copy[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < new_r + r2:
                        valid = False
                        break
            
            if valid and new_r > r:
                circles_copy[i, 2] = new_r
                improved = True
                
        # If no improvements were made in this round, stop early
        if not improved:
            break

    return circles_copy

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int) -> np.ndarray:
    """Select individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def constraint_aware_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover that respects constraints and produces valid offspring"""
    n = len(parent1)
    child = np.zeros_like(parent1)

    # Determine which circles are safe to copy from each parent based on overlap risk
    safe_from_parent1 = np.ones(n, dtype=bool)
    safe_from_parent2 = np.ones(n, dtype=bool)
    
    # Analyze overlap risks between parents
    points1 = parent1[:, :2]
    points2 = parent2[:, :2]
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    for i in range(n):
        x1, y1, r1 = parent1[i]
        x2, y2, r2 = parent2[i]
        
        # Consider if swapping circles would cause immediate overlap
        # Use conservative estimate with small safety margin
        if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < 1.5*(r1 + r2):
            # If too close, decide carefully
            dist1 = np.min([np.sqrt((x1 - px)**2 + (y1 - py)**2) for j, (px, py, _) in enumerate(parent2) if j != i])
            dist2 = np.min([np.sqrt((x2 - px)**2 + (y2 - py)**2) for j, (px, py, _) in enumerate(parent1) if j != i])
            
            # Prefer keeping the circle that's farther from others
            if dist1 > dist2:
                safe_from_parent2[i] = False
            else:
                safe_from_parent1[i] = False
    
    # Now perform crossover safely
    for i in range(n):
        # Decide from which parent to take this circle
        if safe_from_parent1[i] and safe_from_parent2[i]:
            # Both safe, choose randomly
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        elif safe_from_parent1[i]:
            child[i] = parent1[i]
        elif safe_from_parent2[i]:
            child[i] = parent2[i]
        else:
            # Neither safe, pick one conservatively
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
    
    return child

def mutate(individual: np.ndarray) -> np.ndarray:
    """Apply smart mutation that respects spatial constraints"""
    mutated = individual.copy()
    n = len(mutated)
    
    for i in range(n):
        if random.random() < MUTATION_RATE:
            x, y, r = mutated[i]
            
            # Determine how much we can modify based on current position
            max_x_shift = min(x, 1-x) * 0.5
            max_y_shift = min(y, 1-y) * 0.5
            max_radius_change = min(r, 0.5-r) * 0.8
            
            # Choose type of mutation
            mutation_type = random.choices(['position', 'radius'], weights=[0.7, 0.3])[0]
            
            if mutation_type == 'position':
                # Mutate position with bounded random shift
                dx = (random.random() - 0.5) * max_x_shift * 2
                dy = (random.random() - 0.5) * max_y_shift * 2
                x += dx
                y += dy
                
                # Clip to valid bounds
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                mutated[i, 0] = x
                mutated[i, 1] = y
            else:
                # Mutate radius
                dr = (random.random() - 0.5) * max_radius_change * 2
                r += dr
                r = max(0.001, min(0.5, r))  # Keep reasonable bounds
                mutated[i, 2] = r
    
    # Apply constraint repair
    repaired = repair_constraints(mutated)
    return repaired

def repair_constraints(circles: np.ndarray) -> np.ndarray:
    """Repair constraint violations using a sophisticated approach"""
    repaired = circles.copy()
    n = len(repaired)
    
    # First pass: enforce boundary constraints
    for i in range(n):
        x, y, r = repaired[i]
        r = max(0.001, r)
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        repaired[i] = [x, y, r]
    
    # Second pass: resolve overlaps with iterative adjustment
    for iter_count in range(100):
        any_changes = False
        
        # Try to resolve overlaps systematically
        for i in range(n):
            x, y, r = repaired[i]
            
            # Check all other circles for conflicts
            for j in range(n):
                if i != j:
                    x2, y2, r2 = repaired[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    
                    if dist < r + r2:
                        # Push this circle away from the conflicting one
                        dx = x2 - x
                        dy = y2 - y
                        total_dist = max(1e-6, dist)
                        
                        # Calculate required separation
                        separation_needed = (r + r2) - dist
                        
                        # Normalize displacement vector
                        dx_norm = dx / total_dist
                        dy_norm = dy / total_dist
                        
                        # Determine movement magnitude (smaller for tight spaces)
                        movement_factor = min(1.0, separation_needed * 0.5)
                        move_distance = separation_needed * movement_factor
                        
                        # Apply movement
                        x -= dx_norm * move_distance
                        y -= dy_norm * move_distance
                        
                        # Ensure still within bounds
                        x = np.clip(x, r, 1-r)
                        y = np.clip(y, r, 1-r)
                        
                        repaired[i] = [x, y, r]
                        any_changes = True
        
        # Stop early if no changes
        if not any_changes:
            break
    
    return repaired

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    n_circles = 26

    # Create initial population
    population = create_initial_population(POPULATION_SIZE, n_circles)

    best_solution = None
    best_fitness = -np.inf

    for generation in range(GENERATIONS):
        # Evaluate fitness of each individual
        fitnesses = []
        for individual in population:
            if validate_circle_placement(individual):
                fitness = calculate_sum_radii(individual)
                fitnesses.append(fitness)
            else:
                # Invalid solutions get very low fitness
                fitnesses.append(-1000000)

        # Track best solution so far
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        elites = [population[i].copy() for i in elite_indices]

        # Create new population
        new_population = elites.copy()

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            # Crossover (constraint-aware)
            child = constraint_aware_crossover(parent1, parent2)

            # Mutation
            child = mutate(child)

            # Add to new population
            new_population.append(child)

        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to final population if no valid solution was found
        return population[0]

# EVOLVE-BLOCK-END