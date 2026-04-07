# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import math
from scipy.optimize import minimize
from typing import Tuple, List

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Problem parameters
    N_CIRCLES = 26
    POP_SIZE = 100
    N_GEN = 200
    MUT_PB_START = 0.15
    MUT_PB_END = 0.015
    CROSSOVER_PB = 0.8
    ELITE_COUNT = 10

    def create_initial_population(size: int) -> List[np.ndarray]:
        """Create initial population with improved multi-scale initialization"""
        population = []
        
        for _ in range(size):
            circles = np.zeros((N_CIRCLES, 3))
            
            # Strategy 1: Place key strategic positions
            strategic_positions = [
                (0.5, 0.5),      # center
                (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),  # corners
                (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5),  # midpoints
            ]
            
            placed_positions = set()
            count = 0
            
            # Place circles in strategic positions first
            for pos in strategic_positions:
                if count >= N_CIRCLES:
                    break
                x, y = pos
                # Ensure we don't go out of bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))

                # Skip if already placed
                if (round(x, 3), round(y, 3)) not in placed_positions:
                    # Calculate max possible radius
                    min_dist_to_bound = min(x, 1-x, y, 1-y)
                    r = min(0.12, min_dist_to_bound/2)
                    # Add some variance
                    r *= random.uniform(0.85, 1.15)
                    r = max(0.005, min(0.15, r))
                    circles[count] = [x, y, r]
                    placed_positions.add((round(x, 3), round(y, 3)))
                    count += 1

            # Strategy 2: Fill remaining with grid-based approach
            if count < N_CIRCLES:
                rows_cols = int(math.ceil(math.sqrt(N_CIRCLES - count)))
                if rows_cols < 1:
                    rows_cols = 1

                spacing_x = 1.0 / (rows_cols + 1)
                spacing_y = 1.0 / (rows_cols + 1)

                for i in range(rows_cols):
                    if count >= N_CIRCLES:
                        break
                    for j in range(rows_cols):
                        if count >= N_CIRCLES:
                            break
                        x = (i + 1) * spacing_x
                        y = (j + 1) * spacing_y

                        # Ensure boundary constraints
                        x = max(0.05, min(0.95, x))
                        y = max(0.05, min(0.95, y))

                        # Check if position already occupied
                        if (round(x, 3), round(y, 3)) not in placed_positions:
                            min_dist_to_bound = min(x, 1-x, y, 1-y)
                            r = min(0.08, min_dist_to_bound/2)
                            # Add variance
                            r *= random.uniform(0.75, 1.2)
                            r = max(0.005, min(0.15, r))
                            circles[count] = [x, y, r]
                            placed_positions.add((round(x, 3), round(y, 3)))
                            count += 1

            # Strategy 3: Fill any remaining positions with random placement
            for i in range(count, N_CIRCLES):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = random.uniform(0.005, 0.12)
                circles[i] = [x, y, r]
            
            # Apply pre-evolution refinement to maximize initial radii
            circles = maximize_radii(circles)
            population.append(circles)
        
        return population

    def validate_circles(circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and non-overlapping"""
        n = len(circles)
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False

        # Check overlap constraints efficiently using KDTree
        if n <= 1:
            return True
            
        points = circles[:, :2]
        tree = cKDTree(points)
        
        for i in range(n):
            x1, y1, r1 = circles[i]
            # Find nearby circles (within 2*(r1+r2) range)
            neighbors = tree.query_ball_point([x1, y1], 2*(r1 + 0.01))
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if distance < (r1 + r2):
                        return False
        return True

    def calculate_sum_radii(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def maximize_radii(circles: np.ndarray) -> np.ndarray:
        """Pre-evolution local optimization to maximize radii"""
        circles_copy = circles.copy()
        n = len(circles_copy)
        
        # Try to increase radii systematically
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            
            for i in range(n):
                x, y, r = circles_copy[i]
                
                # Calculate maximum possible radius at current position
                max_r = min(x, 1-x, y, 1-y)
                
                if max_r > r + 1e-6:
                    # Try to increase radius as much as possible
                    test_r = min(r + 0.002, max_r)
                    
                    # Check if this new radius works with all other circles
                    valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles_copy[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < test_r + r2:
                                valid = False
                                break
                    
                    if valid:
                        circles_copy[i, 2] = test_r
                        improved = True
                        
        return circles_copy

    def evaluate_fitness(circles: np.ndarray) -> float:
        """Evaluate fitness - maximize sum of radii with strong penalties for constraint violations"""
        if not validate_circles(circles):
            # Severe penalty for constraint violations
            return -1000000.0
            
        # Return sum of radii as primary objective
        return calculate_sum_radii(circles)

    def mutate_individual(individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply adaptive mutation to an individual"""
        mutated = individual.copy()
        n = len(mutated)
        
        # Dynamic mutation rate
        mutation_rate = MUT_PB_START * (MUT_PB_END/MUT_PB_START) ** (generation/max_generations)
        
        for i in range(n):
            if random.random() < mutation_rate:
                # Mutate either position or radius
                if random.random() < 0.5:  # Mutate position
                    mutated[i, 0] += (random.random() - 0.5) * 0.05
                    mutated[i, 1] += (random.random() - 0.5) * 0.05
                    
                    # Keep within bounds
                    mutated[i, 0] = np.clip(mutated[i, 0], 0.01, 0.99)
                    mutated[i, 1] = np.clip(mutated[i, 1], 0.01, 0.99)
                else:  # Mutate radius
                    mutated[i, 2] += (random.random() - 0.5) * 0.03
                    
                    # Ensure positive radius
                    mutated[i, 2] = max(0.001, mutated[i, 2])
        
        return mutated

    def crossover_parents(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover with overlap-aware repair"""
        child = np.zeros_like(parent1)
        n = len(parent1)
        
        # Uniform crossover
        for i in range(n):
            if random.random() < 0.5:
                child[i] = parent1[i].copy()
            else:
                child[i] = parent2[i].copy()
        
        # Apply repair to fix constraints
        child = repair_constraints(child)
        return child

    def repair_constraints(circles: np.ndarray) -> np.ndarray:
        """Repair any constraint violations"""
        repaired = circles.copy()
        n = len(repaired)
        
        # Ensure all circles are within bounds
        for i in range(n):
            x, y, r = repaired[i]
            r = max(0.001, r)
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            repaired[i] = [x, y, r]
        
        # Apply iterative constraint repair
        for _ in range(15):
            any_changes = False
            for i in range(n):
                x, y, r = repaired[i]
                # Check overlaps and adjust if needed
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = repaired[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        min_distance = r + r2
                        if distance < min_distance:
                            # Move circle away from overlapping one
                            dx = x2 - x
                            dy = y2 - y
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > 0:
                                factor = (min_distance - distance) / dist * 0.05
                                x += dx * factor
                                y += dy * factor
                                any_changes = True
                
                # Keep within bounds
                r = max(0.001, r)
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                repaired[i] = [x, y, r]
            
            if not any_changes:
                break
        
        return repaired

    def tournament_selection(population: List[np.ndarray], fitnesses: List[float], 
                           tournament_size: int = 5) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    # Create initial population
    population = create_initial_population(POP_SIZE)

    best_solution = None
    best_fitness = -np.inf

    # Evolution loop
    for generation in range(N_GEN):
        # Evaluate fitness of each individual
        fitnesses = []
        for individual in population:
            fitness = evaluate_fitness(individual)
            fitnesses.append(fitness)
        
        # Track best solution so far
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitnesses)[-ELITE_COUNT:]
        elites = [population[i].copy() for i in elite_indices]

        # Create new population
        new_population = elites.copy()

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POP_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            if random.random() < CROSSOVER_PB:
                child = crossover_parents(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            child = mutate_individual(child, generation, N_GEN)

            # Add to new population
            new_population.append(child)

        # Trim population to exact size
        population = new_population[:POP_SIZE]

    # Apply advanced refinement to best solution
    if best_solution is not None:
        refined_solution = refine_solution(best_solution)
        return refined_solution
    else:
        # Fallback to final population if no valid solution was found
        return population[0]

def refine_solution(circles: np.ndarray) -> np.ndarray:
    """Apply advanced refinement to improve final solution"""
    result = circles.copy()
    
    # Phase 1: Local optimization to maximize radii
    improved = True
    iterations = 0
    while improved and iterations < 100:
        improved = False
        iterations += 1
        
        for i in range(len(result)):
            x, y, r = result[i]
            
            # Calculate maximum possible radius at current position
            max_r = min(x, 1-x, y, 1-y)
            
            if max_r > r + 1e-6:
                # Try to increase radius as much as possible
                test_r = min(r + 0.005, max_r)
                
                # Check if this new radius works with all other circles
                valid = True
                for j in range(len(result)):
                    if i != j:
                        x2, y2, r2 = result[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < test_r + r2:
                            valid = False
                            break
                
                if valid:
                    result[i, 2] = test_r
                    improved = True
    
    # Phase 2: Position adjustment to minimize overlaps
    tree = cKDTree(result[:, :2])
    
    for _ in range(30):
        any_changes = False
        for i in range(len(result)):
            x1, y1, r1 = result[i]
            
            # Find nearby circles
            nearby = tree.query_ball_point([x1, y1], 2*(r1 + 0.01))
            
            for j in nearby:
                if i != j:
                    x2, y2, r2 = result[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if distance < (r1 + r2):
                        # Move circles apart along the line connecting their centers
                        overlap = (r1 + r2) - distance
                        
                        # Move circles apart
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 0:
                            factor = overlap / dist * 0.1
                            result[i, 0] -= dx * factor
                            result[i, 1] -= dy * factor
                            result[j, 0] += dx * factor
                            result[j, 1] += dy * factor
                            any_changes = True
        
        if not any_changes:
            break

    # Final cleanup and validation
    for i in range(len(result)):
        x, y, r = result[i]
        # Ensure radius is positive
        r = max(0.001, r)
        # Keep within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        result[i] = [x, y, r]
        
    return result

# EVOLVE-BLOCK-END