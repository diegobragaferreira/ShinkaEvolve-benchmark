# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 150  # Increased population size for better exploration
GENERATIONS = 300      # More generations for better convergence
INITIAL_MUTATION_RATE = 0.15  # Higher initial mutation for exploration
FINAL_MUTATION_RATE = 0.02    # Lower final mutation for exploitation
ELITISM_COUNT = 10     # More elites for better preservation of good solutions
TOURNAMENT_SIZE = 5    # Larger tournaments for stronger selection pressure
MAX_ATTEMPTS = 1000    # More attempts for initialization
REPAIR_ITERATIONS = 20 # More repair iterations for better constraint satisfaction
LOCAL_OPTIMIZATION_ITERATIONS = 50  # More local optimization rounds

def validate_circle_placement(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and don't overlap"""
    n = len(circles)

    # Check containment constraints efficiently
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    # Process each circle
    for i in range(n):
        x, y, r = circles[i]
        # Find nearby circles (within 2*r distance) - more efficient than full pairwise comparison
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

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population with improved hexagonal grid initialization"""
    population = []

    # Hexagonal grid initialization for better spatial distribution
    def generate_hex_grid():
        # Calculate grid parameters for 26 circles  
        # Using 5 rows and 6 columns for optimal hexagonal packing
        rows = 5
        cols = 6
        grid = []
        
        # Hexagonal packing parameters
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        # Generate hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(grid) >= n_circles:
                    break
                # Offset odd rows for hexagonal packing
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 0.5 + x_offset) * spacing_x
                y = (i + 0.5) * spacing_y
                grid.append((x, y))
        
        return grid[:n_circles]
    
    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))
        
        # Generate initial hexagonal grid
        grid_positions = generate_hex_grid()
        
        # Place circles with controlled randomness and proper radii
        for i, (x, y) in enumerate(grid_positions):
            # Add moderate random variation to positions
            x += (random.random() - 0.5) * 0.04
            y += (random.random() - 0.5) * 0.04
            
            # Calculate appropriate initial radius based on proximity to boundaries
            max_radius = min(x, 1-x, y, 1-y) * 0.8
            # Use better distribution for initial radii
            r = max(0.01, min(max_radius, random.uniform(0.03, 0.09)))
            
            circles[i] = [x, y, r]
        
        # Apply local optimization to improve initial placement
        circles = local_improvement(circles)
        
        # If still invalid, try alternative initialization
        if not validate_circle_placement(circles):
            # Alternative: grid-based initialization with more careful spacing
            circles = np.zeros((n_circles, 3))
            grid_size = max(1, int(np.ceil(np.sqrt(n_circles))))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= n_circles:
                        break
                    x = (i + 1) * spacing_x
                    y = (j + 1) * spacing_y
                    # Initial radius with better distribution
                    r = min(spacing_x, spacing_y) * random.uniform(0.2, 0.4)
                    # Add more controlled randomness
                    r = max(0.005, r * random.uniform(0.7, 1.3))
                    x = max(r, min(1-r, x + random.uniform(-spacing_x*0.15, spacing_x*0.15)))
                    y = max(r, min(1-r, y + random.uniform(-spacing_y*0.15, spacing_y*0.15)))
                    circles[idx] = [x, y, r]
                    idx += 1
            
            # Fill remaining with intelligent random placement  
            for i in range(idx, n_circles):
                max_attempts = 100
                placed = False
                attempts = 0
                
                while not placed and attempts < max_attempts:
                    x = np.random.triangular(0.05, 0.5, 0.95)
                    y = np.random.triangular(0.05, 0.5, 0.95)
                    r = np.random.loguniform(0.005, 0.15)
                    
                    valid_placement = True
                    if r <= x <= 1 - r and r <= y <= 1 - r:
                        for j in range(i):
                            existing_x, existing_y, existing_r = circles[j]
                            distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                            if distance < r + existing_r:
                                valid_placement = False
                                break
                    else:
                        valid_placement = False
                    
                    if valid_placement:
                        circles[i] = [x, y, r]
                        placed = True
                    attempts += 1
                
                if not placed:
                    x = 0.5 + np.random.normal(0, 0.1)
                    y = 0.5 + np.random.normal(0, 0.1)
                    r = 0.01
                    x = max(r, min(1-r, x))
                    y = max(r, min(1-r, y))
                    circles[i] = [x, y, r]
        
        population.append(circles)

    return population

def local_improvement(circles: np.ndarray) -> np.ndarray:
    """Apply advanced local optimization to increase radii and improve placement"""
    n = len(circles)
    circles_copy = circles.copy()
    
    # Multiple optimization passes with diminishing returns
    for pass_num in range(LOCAL_OPTIMIZATION_ITERATIONS):
        improvement_count = 0
        
        # Try to expand radii first
        for i in range(n):
            x, y, r = circles_copy[i]
            
            # Calculate maximum possible radius
            max_r = min(x, 1-x, y, 1-y)
            
            # Try to increase radius while respecting constraints
            if max_r > r:
                # Binary search for maximum radius that doesn't cause overlap
                left, right = r, max_r
                best_r = r
                
                # Limit binary search iterations to avoid slow down
                for _ in range(10):
                    mid = (left + right) / 2
                    valid = True
                    
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if dist < mid + r2:
                                valid = False
                                break
                    
                    if valid:
                        best_r = mid
                        left = mid
                    else:
                        right = mid
                
                if best_r > r:
                    circles_copy[i, 2] = best_r
                    improvement_count += 1
        
        # Position adjustment to resolve overlaps
        for i in range(n):
            x, y, r = circles_copy[i]
            
            # Collect overlapping circles
            overlapping = []
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles_copy[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < r + r2:
                        overlapping.append((j, x2, y2, r2, dist))
            
            # Adjust position to resolve overlaps
            if overlapping:
                total_dx, total_dy = 0, 0
                for j, x2, y2, r2, dist in overlapping:
                    if dist > 0:
                        dx = x - x2
                        dy = y - y2
                        # Push away from overlapped circle
                        push_force = (r + r2 - dist) / dist * 0.1
                        total_dx += dx * push_force * 0.1
                        total_dy += dy * push_force * 0.1
                
                # Apply adjustment
                new_x = x + total_dx
                new_y = y + total_dy
                
                # Keep within bounds
                new_x = np.clip(new_x, r, 1-r)
                new_y = np.clip(new_y, r, 1-r)
                
                # Only update if there's a meaningful change
                if abs(new_x - x) > 1e-6 or abs(new_y - y) > 1e-6:
                    circles_copy[i, 0] = new_x
                    circles_copy[i, 1] = new_y
                    improvement_count += 1

        # Stop early if no significant improvements
        if improvement_count == 0:
            break

    return circles_copy

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int) -> np.ndarray:
    """Select individual using tournament selection with adaptive size"""
    # Use larger tournament size for better selection pressure
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform uniform crossover between two parents with enhanced mixing"""
    n = len(parent1)
    child = np.zeros_like(parent1)

    # Perform crossover with bias towards preserving good features
    for i in range(n):
        # 60% chance to inherit from parent1, 40% from parent2
        if random.random() < 0.6:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    # Post-crossover validation and repair
    return repair_constraints(child)

def mutate(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Apply enhanced mutation to an individual"""
    mutated = individual.copy()
    n = len(mutated)

    # Apply mutations to each circle
    for i in range(n):
        if random.random() < mutation_rate:
            # Mutate either position or radius with balanced probabilities
            if random.random() < 0.6:  # 60% chance to mutate position
                # Mutate position with larger perturbations for better exploration
                mutated[i, 0] += (random.random() - 0.5) * 0.12
                mutated[i, 1] += (random.random() - 0.5) * 0.12

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1], 0.01, 0.99)
            else:
                # Mutate radius with log-normal to ensure positivity and control
                x, y, r = mutated[i]
                # Log-normal mutation - keeps radius positive and allows larger changes
                old_r = r
                r = np.exp(np.log(old_r) + np.random.normal(0, 0.25))
                # Keep positive with minimum
                r = max(0.001, r)
                mutated[i] = [x, y, r]
        else:
            # Small probability of fine-tuning even without direct mutation
            if random.random() < 0.02:
                # Slight adjustment to radius for fine-tuning
                x, y, r = mutated[i]
                r = max(0.001, r * random.uniform(0.98, 1.02))
                mutated[i] = [x, y, r]

    # Repair any constraint violations
    repaired = repair_constraints(mutated)
    return repaired

def repair_constraints(circles: np.ndarray) -> np.ndarray:
    """Improved constraint repair with better handling of edge cases"""
    repaired = circles.copy()
    n = len(repaired)

    # Phase 1: Fix boundary violations
    for i in range(n):
        x, y, r = repaired[i]
        # Ensure radius is positive
        r = max(0.001, r)
        # Keep within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        repaired[i] = [x, y, r]

    # Phase 2: Resolve overlaps with iterative adjustment
    for iteration in range(REPAIR_ITERATIONS):
        any_changes = False
        
        # Collect all overlaps once for efficiency
        overlaps = []
        for i in range(n):
            x, y, r = repaired[i]
            for j in range(i+1, n):
                x2, y2, r2 = repaired[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                min_distance = r + r2
                
                if distance < min_distance:
                    overlaps.append((i, j, x, y, r, x2, y2, r2, distance, min_distance))
        
        # Resolve overlaps in batch
        for i, j, x, y, r, x2, y2, r2, distance, min_distance in overlaps:
            # Calculate displacement vector
            dx = x2 - x
            dy = y2 - y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                # Move both circles away from each other
                factor = (min_distance - distance) / dist * 0.3
                repaired[i, 0] -= dx * factor * 0.5
                repaired[i, 1] -= dy * factor * 0.5
                repaired[j, 0] += dx * factor * 0.5
                repaired[j, 1] += dy * factor * 0.5
                any_changes = True
        
        # Keep within bounds after adjustments
        for i in range(n):
            x, y, r = repaired[i]
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]
        
        if not any_changes:
            break

    # Phase 3: Local optimization after repair
    repaired = local_improvement(repaired)
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
        # Calculate adaptive mutation rate - exponential decay
        mutation_rate = INITIAL_MUTATION_RATE * (FINAL_MUTATION_RATE / INITIAL_MUTATION_RATE) ** (generation / GENERATIONS)
        if mutation_rate < FINAL_MUTATION_RATE:
            mutation_rate = FINAL_MUTATION_RATE

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

            # Crossover
            child = uniform_crossover(parent1, parent2)

            # Mutation
            child = mutate(child, mutation_rate)

            # Add to new population
            new_population.append(child)

        population = new_population[:POPULATION_SIZE]

        # Print progress every 50 generations
        if generation % 50 == 0:
            avg_fitness = np.mean([f for f in fitnesses if f > -1000000])
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, Avg fitness = {avg_fitness:.6f}")

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to final population if no valid solution was found
        return population[0]

# EVOLVE-BLOCK-END