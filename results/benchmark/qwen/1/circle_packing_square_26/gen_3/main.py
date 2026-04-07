# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
import random
import math
from typing import Tuple

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 26
    MAX_RADIUS = 0.5
    
    def check_collision(circles, new_circle, index):
        """Check if new circle collides with existing circles"""
        x, y, r = new_circle
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return True
            
        # Use KDTree for efficient neighbor search
        if len(circles) > 0:
            tree = cKDTree(circles[:, :2])
            neighbors = tree.query_ball_point([x, y], r + 0.001)
            for i in neighbors:
                if i != index:
                    ox, oy, oradius = circles[i]
                    distance = math.sqrt((x - ox)**2 + (y - oy)**2)
                    if distance < r + oradius:
                        return True
        return False
    
    def create_initial_population(size: int):
        """Create initial population with valid configurations"""
        population = []
        for _ in range(size):
            circles = []
            # Place circles greedily
            while len(circles) < N_CIRCLES:
                # Try to place a circle with random position and radius
                max_attempts = 1000
                placed = False
                attempts = 0
                
                while not placed and attempts < max_attempts:
                    # Randomly generate center and radius
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    
                    # Start with maximum possible radius and decrease until valid
                    r = min(MAX_RADIUS, 0.1)
                    while r > 0.001:
                        test_circle = [x, y, r]
                        if not check_collision(np.array(circles), test_circle, len(circles)):
                            circles.append(test_circle)
                            placed = True
                            break
                        r *= 0.9  # Decrease radius
                    attempts += 1
                    
                # If couldn't place more circles, try a different approach
                if not placed:
                    # Force placement with minimal radius
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    r = 0.01
                    circles.append([x, y, r])
            
            # Ensure we have exactly 26 circles
            if len(circles) < N_CIRCLES:
                # Fill remaining slots with small circles
                for i in range(N_CIRCLES - len(circles)):
                    x = np.random.uniform(0.01, 0.99)
                    y = np.random.uniform(0.01, 0.99)
                    r = 0.01
                    circles.append([x, y, r])
            elif len(circles) > N_CIRCLES:
                circles = circles[:N_CIRCLES]
                
            population.append(np.array(circles))
        return population
    
    def evaluate(individual):
        """Evaluate fitness (negative sum of radii)"""
        total_radius = sum(circle[2] for circle in individual)
        return (-total_radius,)
    
    def mutate_individual(individual, indpb=0.1):
        """Mutate an individual by adjusting positions and radii"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < indpb:
                # Mutate position and radius
                x, y, r = mutated[i]
                
                # Mutate position slightly
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                
                # Ensure within bounds
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                
                # Mutate radius
                r *= np.random.normal(1, 0.1)
                r = max(0.001, min(r, 0.5))
                
                mutated[i] = [x, y, r]
        
        # Repair invalid configurations
        valid_circles = []
        for i, circle in enumerate(mutated):
            x, y, r = circle
            
            # Boundary check
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            # Collision check with already placed circles
            valid = True
            for other_circle in valid_circles:
                ox, oy, oradius = other_circle
                distance = math.sqrt((x - ox)**2 + (y - oy)**2)
                if distance < r + oradius:
                    valid = False
                    break
            
            if valid:
                valid_circles.append([x, y, r])
            else:
                # Revert to original if collision
                valid_circles.append([x, y, r])
        
        # Pad or truncate to ensure exactly N_CIRCLES
        while len(valid_circles) < N_CIRCLES:
            valid_circles.append([0.5, 0.5, 0.01])
        valid_circles = valid_circles[:N_CIRCLES]
        
        return (np.array(valid_circles),)
    
    def cx_two_point(ind1, ind2):
        """Two-point crossover"""
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            # Swap parts between individuals
            for i in range(cxpoint1, cxpoint2):
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    # Create toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("population", create_initial_population, size=20)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", cx_two_point)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population = toolbox.population()
    
    # Evolution parameters
    CXPB = 0.5   # Crossover probability
    MUTPB = 0.2  # Mutation probability
    NGEN = 50    # Number of generations
    
    # Main evolution loop
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        population[:] = offspring
    
    # Find best solution
    best_individual = tools.selBest(population, 1)[0]
    
    # Final local optimization using simulated annealing-like approach
    def local_improve(individual):
        """Apply local improvement to the solution"""
        current = individual.copy()
        best = current.copy()
        best_fitness = evaluate(best)[0]
        
        # Try small adjustments
        for iteration in range(1000):
            # Create perturbation
            new_individual = current.copy()
            idx = random.randint(0, len(new_individual)-1)
            
            # Small random changes
            x, y, r = new_individual[idx]
            x += np.random.normal(0, 0.005)
            y += np.random.normal(0, 0.005)
            r *= np.random.normal(1, 0.05)
            
            # Clamp values
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            r = max(0.001, min(r, 0.5))
            
            new_individual[idx] = [x, y, r]
            
            # Check validity and fitness
            if not check_collision(np.concatenate([new_individual[:idx], new_individual[idx+1:]]), 
                                  [x, y, r], idx):
                new_fitness = evaluate(new_individual)[0]
                if new_fitness < best_fitness:  # Since we're minimizing negative
                    best = new_individual.copy()
                    best_fitness = new_fitness
                    
        return best
    
    # Run final local optimization
    final_solution = local_improve(best_individual)
    
    # Convert to the required format
    result = np.array(final_solution)
    return result

# EVOLVE-BLOCK-END
