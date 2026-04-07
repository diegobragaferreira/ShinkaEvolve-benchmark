# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 150
TOURNAMENT_SIZE = 3
INITIAL_MUTATION_RATE = 0.1
FINAL_MUTATION_RATE = 0.01
ELITISM_COUNT = 5
MAX_ATTEMPTS = 500
REPAIR_ITERATIONS = 15
LOCAL_OPTIMIZATION_ITERATIONS = 30

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

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population with improved hexagonal grid initialization"""
    population = []

    # Hexagonal grid initialization for better spatial distribution
    def generate_hex_grid():
        # Calculate grid parameters for 26 circles
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
                # Offset odd rows
                x_offset = 0 if i % 2 == 0 else spacing_x / 2
                x = (j + 0.5 + x_offset) * spacing_x
                y = (i + 0.5) * spacing_y
                grid.append((x, y))
        
        return grid[:n_circles]
    
    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))
        
        # Generate initial hexagonal grid
        grid_positions = generate_hex_grid()
        
        # Place circles with some randomness and proper radii
        for i, (x, y) in enumerate(grid_positions):
            # Add small random variation to positions
            x += (random.random() - 0.5) * 0.03
            y += (random.random() - 0.5) * 0.03
            
            # Set initial radius based on proximity to boundaries
            max_radius = min(x, 1-x, y, 1-y) * 0.8
            r = max(0.01, min(max_radius, random.uniform(0.02, 0.08)))
            
            circles[i] = [x, y, r]
        
        # Apply local optimization to improve placement
        circles = local_improvement(circles)
        
        # If still invalid, try random initialization
        if not validate_circle_placement(circles):
            circles = np.zeros((n_circles, 3))
            for i in range(n_circles):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                max_radius = min(x, 1-x, y, 1-y)
                r = max(0.01, min(0.1, max_radius * random.uniform(0.5, 0.8)))
                circles[i] = [x, y, r]
        
        population.append(circles)

    return population

def local_improvement(circles: np.ndarray) -> np.ndarray:
    """Apply local optimization to increase radii and improve placement"""
    n = len(circles)
    circles_copy = circles.copy()
    
    # Multiple optimization passes
    for pass_num in range(LOCAL_OPTIMIZATION_ITERATIONS):
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
                
                circles_copy[i, 2] = best_r
        
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
                        push_force = (r + r2 - dist) / dist
                        total_dx += dx * push_force * 0.1
                        total_dy += dy * push_force * 0.1
                
                # Apply adjustment
                new_x = x + total_dx
                new_y = y + total_dy
                
                # Keep within bounds
                new_x = np.clip(new_x, r, 1-r)
                new_y = np.clip(new_y, r, 1-r)
                
                circles_copy[i, 0] = new_x
                circles_copy[i, 1] = new_y

    return circles_copy

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int) -> np.ndarray:
    """Select individual using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform uniform crossover between two parents"""
    n = len(parent1)
    child = np.zeros_like(parent1)

    for i in range(n):
        # Each element has 50% chance of coming from parent1 or parent2
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    return child

def mutate(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Apply mutation to an individual"""
    mutated = individual.copy()
    n = len(mutated)

    for i in range(n):
        if random.random() < mutation_rate:
            # Mutate either position or radius (with bias toward position)
            if random.random() < 0.7:  # 70% chance to mutate position
                # Mutate position
                mutated[i, 0] += (random.random() - 0.5) * 0.08
                mutated[i, 1] += (random.random() - 0.5) * 0.08

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1], 0.01, 0.99)
            else:
                # Mutate radius
                mutated[i, 2] += (random.random() - 0.5) * 0.04

                # Ensure positive radius
                mutated[i, 2] = max(0.001, mutated[i, 2])

    # Repair any constraint violations
    repaired = repair_constraints(mutated)
    return repaired

def repair_constraints(circles: np.ndarray) -> np.ndarray:
    """Repair any constraint violations with improved method"""
    repaired = circles.copy()
    n = len(repaired)

    # Phase 1: Fix boundary violations
    for i in range(n):
        x, y, r = repaired[i]
        r = max(0.001, r)
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        repaired[i] = [x, y, r]

    # Phase 2: Resolve overlaps with iterative adjustment
    for iteration in range(REPAIR_ITERATIONS):
        any_changes = False
        
        for i in range(n):
            x, y, r = repaired[i]
            original_x, original_y, original_r = x, y, r
            
            # Check all overlaps
            for j in range(n):
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    min_distance = r + r2
                    
                    if distance < min_distance:
                        # Calculate displacement vector
                        dx = x2 - x
                        dy = y2 - y
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist > 0:
                            # Move away from overlapping circle
                            factor = (min_distance - distance) / dist * 0.3
                            x += dx * factor
                            y += dy * factor
                            any_changes = True
            
            # Keep within bounds
            r = max(0.001, r)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            # Update if changed
            if x != original_x or y != original_y:
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
        # Calculate adaptive mutation rate
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

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to final population if no valid solution was found
        return population[0]

# EVOLVE-BLOCK-END