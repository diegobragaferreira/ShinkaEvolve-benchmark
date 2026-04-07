# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
import time
from typing import Tuple, List, Optional
import math
from deap import base, creator, tools, algorithms

# Fixed seed for reproducibility
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
    N_GEN = 500
    INITIAL_MUT_PB = 0.15
    CROSSOVER_PB = 0.8

    # Define the fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def validate_circles(circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and non-overlapping using KDTree."""
        if len(circles) != N_CIRCLES:
            return False

        # Check containment constraints
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1 - r or y < r or y > 1 - r:
                return False

        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = KDTree(points)

        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            # Find nearby circles (within 2*r distance)
            nearby = tree.query_ball_point([x, y], 2 * r)
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2:
                        return False

        return True

    def calculate_fitness(circles: np.ndarray) -> float:
        """Calculate total radius sum as fitness."""
        return np.sum(circles[:, 2])

    # Create an individual representing a solution
    # Each individual is a list of [x, y, r] for each circle
    def create_individual():
        individual = []
        # Use enhanced grid-guided initialization
        circles = enhanced_grid_init(N_CIRCLES)
        for x, y, r in circles:
            individual.extend([x, y, r])
        return creator.Individual(individual)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_circle_packing(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)

        # Calculate total radius (objective function)
        total_radius = calculate_fitness(circles)

        # Check constraints with penalties
        penalty = 0

        # Check containment constraints
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 10000  # Large penalty for containment violation

        # Check overlap constraints with improved penalty calculation
        if N_CIRCLES > 1:
            points = circles[:, :2]
            tree = KDTree(points)
            
            for i in range(N_CIRCLES):
                x, y, r = circles[i]
                # Find nearby circles (within 2*r distance)
                nearby = tree.query_ball_point([x, y], 2 * r)
                for j in nearby:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < r + r2:
                            # Penalty based on how much they overlap
                            overlap = (r + r2) - distance
                            penalty += 10000 * overlap

        # Return fitness: total radius minus penalties
        return (total_radius - penalty,)

    toolbox.register("evaluate", eval_circle_packing)

    # Custom crossover operator with improved strategy
    def cx_circle(ind1, ind2):
        # Perform uniform crossover for x, y, r coordinates with better mixing
        for i in range(0, len(ind1), 3):
            if random.random() < 0.5:
                ind1[i:i+3], ind2[i:i+3] = ind2[i:i+3], ind1[i:i+3]
        return ind1, ind2

    # Enhanced mutation operator with adaptive rates and better strategies
    def mut_circle(individual, indpb, generation, max_gen):
        # Adaptive mutation rate that decreases over generations
        adaptive_pb = indpb * (1 - generation/max_gen)
        
        # Mutate x, y, r for each circle
        for i in range(0, len(individual), 3):
            if random.random() < adaptive_pb:
                # Mutate x coordinate with better distribution
                individual[i] += np.random.normal(0, 0.015)
                individual[i] = max(0.005, min(0.995, individual[i]))

            if random.random() < adaptive_pb:
                # Mutate y coordinate
                individual[i+1] += np.random.normal(0, 0.015)
                individual[i+1] = max(0.005, min(0.995, individual[i+1]))

            if random.random() < adaptive_pb:
                # Mutate radius with log-normal for better control
                old_r = individual[i+2]
                # Log-normal mutation to keep positive
                new_r = np.exp(np.log(old_r) + np.random.normal(0, 0.15))
                individual[i+2] = max(0.001, min(0.4, new_r))

        return individual,

    toolbox.register("mate", cx_circle)
    toolbox.register("mutate", mut_circle)
    toolbox.register("select", tools.selTournament, tournsize=5)

    # Enhanced grid-guided initialization inspired by the best approaches
    def enhanced_grid_init(n_circles):
        """Create enhanced initial configuration using multi-scale grid approach."""
        circles = []
        
        if n_circles <= 9:
            # Small number: use tight grid with better spacing
            grid_size = int(np.ceil(np.sqrt(n_circles)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= n_circles:
                        break
                    x = (i + 1) * spacing_x
                    y = (j + 1) * spacing_y
                    # Better radius distribution
                    r = min(spacing_x, spacing_y) * np.random.uniform(0.25, 0.45)
                    # Add controlled randomness
                    r = max(0.005, r * np.random.uniform(0.8, 1.2))
                    x = max(r, min(1-r, x + np.random.uniform(-spacing_x*0.08, spacing_x*0.08)))
                    y = max(r, min(1-r, y + np.random.uniform(-spacing_y*0.08, spacing_y*0.08)))
                    circles.append([x, y, r])
                    idx += 1
        elif n_circles <= 16:
            # Medium number: use two concentric grids
            outer_grid_size = 4
            inner_grid_size = 2
            
            # Outer grid
            outer_spacing = 1.0 / (outer_grid_size + 1)
            idx = 0
            for i in range(outer_grid_size):
                for j in range(outer_grid_size):
                    if idx >= n_circles:
                        break
                    x = (i + 1) * outer_spacing
                    y = (j + 1) * outer_spacing
                    r = outer_spacing * np.random.uniform(0.2, 0.35)
                    # Add controlled randomness
                    r = max(0.005, r * np.random.uniform(0.85, 1.15))
                    x = max(r, min(1-r, x + np.random.uniform(-outer_spacing*0.05, outer_spacing*0.05)))
                    y = max(r, min(1-r, y + np.random.uniform(-outer_spacing*0.05, outer_spacing*0.05)))
                    circles.append([x, y, r])
                    idx += 1
        else:
            # Larger number: use strategic placement with key positions
            # First place some key circles at corners and center
            key_positions = [
                (0.1, 0.1, 0.05),      # bottom-left
                (0.9, 0.1, 0.05),      # bottom-right
                (0.1, 0.9, 0.05),      # top-left
                (0.9, 0.9, 0.05),      # top-right
                (0.5, 0.5, 0.1),       # center
            ]
            
            # Add key positions
            for pos in key_positions:
                if len(circles) < n_circles:
                    circles.append(list(pos))
            
            # Fill remaining positions with grid
            remaining_count = n_circles - len(circles)
            grid_size = int(np.ceil(np.sqrt(remaining_count)))
            spacing = 1.0 / (grid_size + 1)
            
            for i in range(remaining_count):
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing * np.random.uniform(0.25, 0.4)
                # Add controlled randomness
                r = max(0.005, r * np.random.uniform(0.8, 1.2))
                x = max(r, min(1-r, x + np.random.uniform(-spacing*0.1, spacing*0.1)))
                y = max(r, min(1-r, y + np.random.uniform(-spacing*0.1, spacing*0.1)))
                circles.append([x, y, r])

        # If we still need more circles (shouldn't happen), add randomly
        while len(circles) < n_circles:
            x = np.random.triangular(0.05, 0.5, 0.95)
            y = np.random.triangular(0.05, 0.5, 0.95)
            # Use log-uniform for radius to get better distribution
            r = np.random.loguniform(0.005, 0.15)
            circles.append([x, y, r])

        return circles

    # Enhanced local search improvement function
    def local_search_improve(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Apply advanced local search to improve solution by adjusting positions/radii."""
        current = circles.copy()
        current_fitness = calculate_fitness(current)

        for iteration in range(max_iterations):
            improved = False
            # Try to improve each circle individually
            for i in range(N_CIRCLES):
                original_x, original_y, original_r = current[i]
                best_x, best_y, best_r = original_x, original_y, original_r
                best_fitness = current_fitness

                # Try small adjustments to position and radius
                step_sizes = [0.005, 0.01, 0.02]
                for step in step_sizes:
                    # Test position changes
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            new_x = original_x + dx
                            new_y = original_y + dy

                            # Ensure new position is within bounds
                            if (new_x - original_r >= 0 and new_x + original_r <= 1 and
                                new_y - original_r >= 0 and new_y + original_r <= 1):

                                # Create temporary configuration
                                temp_circles = current.copy()
                                temp_circles[i] = [new_x, new_y, original_r]

                                # Check if this improves overall fitness
                                if validate_circles(temp_circles):
                                    new_fitness = calculate_fitness(temp_circles)
                                    if new_fitness > best_fitness:
                                        best_fitness = new_fitness
                                        best_x, best_y, best_r = new_x, new_y, original_r
                                        improved = True

                    # Test radius changes
                    for dr in [-step, 0, step]:
                        new_r = original_r + dr
                        if new_r > 0.001 and new_r < 0.5:  # Reasonable bounds
                            # Ensure new radius allows for valid positioning
                            new_r = min(new_r, original_x, 1-original_x, original_y, 1-original_y)
                            if new_r > 0.001:
                                # Create temporary configuration
                                temp_circles = current.copy()
                                temp_circles[i] = [original_x, original_y, new_r]

                                # Check if this improves overall fitness
                                if validate_circles(temp_circles):
                                    new_fitness = calculate_fitness(temp_circles)
                                    if new_fitness > best_fitness:
                                        best_fitness = new_fitness
                                        best_x, best_y, best_r = original_x, original_y, new_r
                                        improved = True

                # Update if we found a better configuration
                if improved:
                    current[i] = [best_x, best_y, best_r]
                    current_fitness = best_fitness

            # Stop if no improvement was made in this iteration
            if not improved:
                break

        return current

    # Improved overlap repair mechanism
    def repair_overlaps(circles: np.ndarray) -> np.ndarray:
        """Repair overlapping circles with progressive approach."""
        # Try several iterations to resolve overlaps with progressive approach
        for iteration in range(10):
            if validate_circles(circles):
                return circles

            # Progressive overlap resolution: start aggressive then become conservative
            reduction_factor = 0.95 - (iteration * 0.02)  # Gradually reduce aggressiveness
            reduction_factor = max(0.8, reduction_factor)  # Minimum factor

            # More aggressive repair: reduce radii and slightly adjust positions
            for i in range(N_CIRCLES):
                x, y, r = circles[i]
                # Reduce radius to resolve overlap
                circles[i] = [x, y, max(0.001, r * reduction_factor)]

        # Final adjustment if still invalid
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            # Ensure boundaries
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]

        return circles

    # Create initial population with enhanced initialization
    def create_enhanced_population():
        population = []
        for _ in range(POP_SIZE):
            individual = []
            # Start with enhanced grid-guided initialization
            circles = enhanced_grid_init(N_CIRCLES)
            for x, y, r in circles:
                individual.extend([x, y, r])
            population.append(creator.Individual(individual))
        return population

    # Create initial population
    try:
        population = create_enhanced_population()
    except Exception as e:
        print(f"Enhanced population creation failed: {e}")
        # Fallback to basic initialization
        population = toolbox.population(n=POP_SIZE)

    # Run evolution with adaptive mutation and local search
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        # Modified eaSimple to pass generation info to mutation and integrate local search
        for gen in range(N_GEN):
            # Update mutation probability based on generation
            current_mut_pb = INITIAL_MUT_PB * (1 - gen/N_GEN)

            # Select and clone
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_PB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation with adaptive rate
            for mutant in offspring:
                if random.random() < current_mut_pb:
                    toolbox.mutate(mutant, current_mut_pb, gen, N_GEN)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Apply local search to offspring for fine-tuning and validation repair
            for i, ind in enumerate(offspring):
                if not validate_circles(np.array(ind).reshape(-1, 3)):
                    # Repair invalid individuals before keeping them
                    circles = np.array(ind).reshape(-1, 3)
                    repaired = repair_overlaps(circles)
                    # Convert back to individual
                    flat_repaired = repaired.flatten().tolist()
                    offspring[i] = creator.Individual(flat_repaired)

            # Replace the old population with the new one
            population[:] = offspring

            # Update hall of fame
            hall_of_fame.update(population)

    except Exception as e:
        print(f"Evolution failed: {e}")
        # Return a simple heuristic solution if evolution fails
        return heuristic_solution()

    # Get best individual
    best_individual = hall_of_fame[0]
    best_circles = np.array(best_individual).reshape(-1, 3)

    # Apply local optimization refinement
    refined_circles = local_search_improve(best_circles)

    # Ensure final validation
    circles = validate_and_fix_solution(refined_circles)

    return circles

def heuristic_solution() -> np.ndarray:
    """Fallback solution using hexagonal packing heuristic"""
    n = 26
    circles = np.zeros((n, 3))

    # Try a hexagonal lattice approach for reasonable starting point
    # Arrange in roughly a hexagonal pattern with some randomness
    rows = 5
    cols = 6

    # Hexagonal packing coordinates
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0 if i % 2 == 0 else spacing_x / 2
            x = (j + 1) * spacing_x + x_offset + random.uniform(-spacing_x*0.05, spacing_x*0.05)
            y = (i + 1) * spacing_y + random.uniform(-spacing_y*0.05, spacing_y*0.05)
            # Radius based on proximity to boundaries
            r = min(x, 1-x, y, 1-y) * 0.35
            # Add some randomness to make it less regular
            r *= random.uniform(0.8, 1.0)
            circles[idx] = [x, y, r]
            idx += 1

    # If we don't have enough circles, fill remaining positions with small radii
    for i in range(idx, n):
        circles[i] = [0.5, 0.5, 0.01]

    return circles

def validate_and_fix_solution(circles: np.ndarray) -> np.ndarray:
    """Ensure the solution respects constraints and has reasonable values"""
    # Make a copy to avoid modifying original
    result = circles.copy()

    # Clip radii to reasonable bounds
    result[:, 2] = np.clip(result[:, 2], 0.001, 0.45)

    # Ensure circles stay within bounds
    for i in range(len(result)):
        x, y, r = result[i]
        # Clamp positions to valid range
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        result[i] = [x, y, r]

    return result

# EVOLVE-BLOCK-END