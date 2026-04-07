# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 150
GENERATIONS = 300
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.15
ELITISM_COUNT = 10
MAX_ATTEMPTS = 500
LOCAL_IMPROVEMENT_ITERATIONS = 100

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

def create_hexagonal_grid(n_circles: int) -> np.ndarray:
    """Create initial configuration using hexagonal packing pattern"""
    circles = np.zeros((n_circles, 3))
    
    # Calculate grid dimensions
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))
    
    # Hexagonal packing parameters
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    hex_height = spacing_y * np.sqrt(3) / 2
    
    # Place circles in hexagonal pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
                
            # Offset every other row
            x_offset = (i % 2) * spacing_x / 2
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Add small randomness to avoid perfect grid
            x += (random.random() - 0.5) * spacing_x * 0.3
            y += (random.random() - 0.5) * spacing_y * 0.3
            
            # Clamp to bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Set initial radius based on available space
            max_radius = min(x, 1-x, y, 1-y) * 0.4
            r = np.clip(max_radius, 0.01, 0.2)
            
            circles[idx] = [x, y, r]
            idx += 1
            
    return circles

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population with improved initialization strategies"""
    population = []

    for _ in range(pop_size):
        # Try hexagonal grid initialization first
        circles = create_hexagonal_grid(n_circles)
        
        # Apply local improvement to enhance initial configuration
        improved = local_improvement(circles)
        
        # Validate and repair if needed
        if validate_circle_placement(improved):
            circles = improved
        else:
            # Fallback to random initialization with clustering
            circles = np.zeros((n_circles, 3))
            cluster_centers = []
            
            # Generate cluster centers
            for _ in range(5):
                cluster_centers.append([
                    random.uniform(0.1, 0.9),
                    random.uniform(0.1, 0.9)
                ])
            
            # Distribute circles around clusters
            for i in range(n_circles):
                center = cluster_centers[i % len(cluster_centers)]
                x = center[0] + (random.random() - 0.5) * 0.3
                y = center[1] + (random.random() - 0.5) * 0.3
                
                # Clamp to bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                # Set radius
                max_radius = min(x, 1-x, y, 1-y) * 0.3
                r = np.clip(max_radius, 0.01, 0.15)
                
                circles[i] = [x, y, r]
                
            # Apply local improvement
            improved = local_improvement(circles)
            if validate_circle_placement(improved):
                circles = improved

        population.append(circles)

    return population

def local_improvement(circles: np.ndarray) -> np.ndarray:
    """Apply advanced local improvement to maximize radii while maintaining constraints"""
    n = len(circles)
    circles_copy = circles.copy()
    
    # Use a more sophisticated approach with multiple phases
    for phase in range(3):
        for _ in range(LOCAL_IMPROVEMENT_ITERATIONS // 3):
            # Randomly select circle to improve
            i = random.randint(0, n-1)
            x, y, r = circles_copy[i]
            
            # Try to increase radius while maintaining constraints
            max_r = min(x, 1-x, y, 1-y)
            if max_r > r:
                new_r = min(r + 0.005, max_r)
                
                # Check overlap with all other circles
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles_copy[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < new_r + r2:
                            valid = False
                            break
                
                if valid and new_r > r:
                    circles_copy[i, 2] = new_r
                    
            # Small position adjustments to improve packing
            if random.random() < 0.3:
                dx = (random.random() - 0.5) * 0.02
                dy = (random.random() - 0.5) * 0.02
                new_x, new_y = x + dx, y + dy
                
                # Clamp to bounds
                new_x = np.clip(new_x, r, 1-r)
                new_y = np.clip(new_y, r, 1-r)
                
                # Check if this improves the configuration
                old_dist = 0
                new_dist = 0
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles_copy[j]
                        old_dist += max(0, r + r2 - np.sqrt((x - x2)**2 + (y - y2)**2))
                        new_dist += max(0, r + r2 - np.sqrt((new_x - x2)**2 + (new_y - y2)**2))
                        
                # If movement helps or doesn't hurt too much, accept it
                if new_dist <= old_dist or random.random() < 0.1:
                    circles_copy[i, 0] = new_x
                    circles_copy[i, 1] = new_y

    return circles_copy

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int) -> np.ndarray:
    """Select individual using tournament selection with dynamic size"""
    tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform uniform crossover with adaptive probability"""
    n = len(parent1)
    child = np.zeros_like(parent1)

    for i in range(n):
        # Use higher probability for position genes to preserve good locations
        if random.random() < 0.6:  # 60% chance for position
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]

    return child

def mutate(individual: np.ndarray) -> np.ndarray:
    """Apply adaptive mutation to an individual"""
    mutated = individual.copy()
    n = len(mutated)

    # Adaptive mutation rate based on how close we are to optimal
    adaptive_rate = MUTATION_RATE * (1 - min(0.9, calculate_sum_radii(mutated) / 15.0))

    for i in range(n):
        if random.random() < adaptive_rate:
            # Mutate either position or radius with preference for position
            if random.random() < 0.7:  # 70% chance to mutate position
                # Mutate position
                mutated[i, 0] += (random.random() - 0.5) * 0.08
                mutated[i, 1] += (random.random() - 0.5) * 0.08

                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1], 0.01, 0.99)
            else:
                # Mutate radius
                mutated[i, 2] += (random.random() - 0.5) * 0.03

                # Ensure positive radius
                mutated[i, 2] = max(0.001, mutated[i, 2])

    # Repair any constraint violations
    repaired = repair_constraints(mutated)
    return repaired

def repair_constraints(circles: np.ndarray) -> np.ndarray:
    """Advanced constraint repair mechanism"""
    repaired = circles.copy()
    n = len(repaired)

    # First ensure all circles are within bounds
    for i in range(n):
        x, y, r = repaired[i]
        r = max(0.001, r)
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        repaired[i] = [x, y, r]

    # Then resolve overlaps through iterative adjustment
    for iteration in range(50):
        any_changes = False
        for i in range(n):
            x, y, r = repaired[i]
            
            # Check for overlaps and adjust
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
                            # Push apart
                            factor = (min_distance - distance) / dist * 0.3
                            x -= dx * factor
                            y -= dy * factor
                            any_changes = True

            # Keep within bounds
            r = max(0.001, r)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]

        # Early termination if no significant changes
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

            # Crossover
            child = uniform_crossover(parent1, parent2)

            # Mutation
            child = mutate(child)

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