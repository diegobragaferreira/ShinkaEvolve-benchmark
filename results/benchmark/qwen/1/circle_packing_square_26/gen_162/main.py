# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Global constants
POPULATION_SIZE = 80
GENERATIONS = 600
ELITISM_COUNT = 10
MAX_ATTEMPTS = 50
INITIAL_MUTATION_RATE = 0.2
ADAPTIVE_MUTATION_BASE = 0.1
MIN_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.4

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses evolutionary algorithm with Voronoi-based initialization and constraint-aware operators.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n_circles = 26
    max_radius = 0.5
    min_radius = 0.001
    
    # Enhanced Voronoi-based initialization
    def generate_voronoi_initial():
        # Create initial points using a combination of grid and random distribution
        points = []
        grid_size = 6
        spacing = 1.0 / (grid_size + 1)
        
        # Grid points with slight randomness
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 1) * spacing + np.random.uniform(-spacing/8, spacing/8)
                y = (j + 1) * spacing + np.random.uniform(-spacing/8, spacing/8)
                points.append([x, y])
        
        points = np.array(points[:n_circles])
        
        # Create Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to random points if Voronoi fails
            points = np.random.rand(n_circles, 2)
            vor = Voronoi(points)
        
        # Create circles based on Voronoi cells
        circles = []
        for i in range(n_circles):
            if i < len(vor.points):
                x, y = vor.points[i]
                
                # Estimate radius based on minimum distance to neighbors
                min_dist = float('inf')
                for j in range(n_circles):
                    if i != j:
                        dist = np.sqrt((vor.points[i, 0] - vor.points[j, 0])**2 +
                                     (vor.points[i, 1] - vor.points[j, 1])**2)
                        min_dist = min(min_dist, dist)
                
                # Safe radius is half the minimum neighbor distance or 0.2, whichever is smaller
                estimated_radius = min(0.2, min_dist / 4.0) if min_dist < float('inf') else 0.1
                
                # Ensure reasonable bounds
                radius = max(min_radius, min(max_radius, estimated_radius))
                
                # Clamp to unit square
                x = max(radius, min(1-radius, x))
                y = max(radius, min(1-radius, y))
                
                circles.append([x, y, radius])
            else:
                # Fall back to simple random
                x = np.random.uniform(min_radius, 1 - min_radius)
                y = np.random.uniform(min_radius, 1 - min_radius)
                r = np.random.uniform(min_radius, max_radius)
                circles.append([x, y, r])
        
        return np.array(circles)
    
    # Constraint checking function using efficient KDTree
    def is_valid_configuration(circles_array):
        # Check containment constraints
        for x, y, r in circles_array:
            if not (min_radius <= r <= max_radius and 
                   min_radius <= x <= 1 - min_radius and 
                   min_radius <= y <= 1 - min_radius):
                return False
        
        # Check overlap constraints efficiently using KDTree
        try:
            tree = cKDTree(circles_array[:, :2])
            # Query pairs within double the max radius to reduce unnecessary checks
            pairs = tree.query_pairs(2 * max_radius)
            
            for i, j in pairs:
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist_squared = (x1 - x2)**2 + (y1 - y2)**2
                radius_sum = r1 + r2
                
                if dist_squared < radius_sum**2:
                    return False
        except:
            # Fallback to brute force if KDTree fails
            n = len(circles_array)
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles_array[i]
                    x2, y2, r2 = circles_array[j]
                    dist_squared = (x1 - x2)**2 + (y1 - y2)**2
                    radius_sum = r1 + r2
                    
                    if dist_squared < radius_sum**2:
                        return False
                        
        return True
    
    # Fitness function with penalty for constraint violations
    def evaluate(individual):
        circles_array = np.array(individual).reshape(-1, 3)
        if not is_valid_configuration(circles_array):
            return (0,)  # Invalid configuration gets zero fitness
        total_radius = np.sum(circles_array[:, 2])
        return (total_radius,)
    
    # Constraint-aware crossover
    def crossover(parent1, parent2):
        child = parent1.copy()
        n = len(child)
        
        # Uniform crossover
        for i in range(n):
            if random.random() < 0.5:
                child[i] = parent2[i]
        
        # Repair child to maintain validity
        repair_individual(child)
        return child
    
    # Constraint-aware mutation
    def mutate(individual, mutation_rate):
        mutated = individual.copy()
        n = len(mutated)
        
        # Dynamic mutation rate based on diversity
        dynamic_mutation_rate = max(MIN_MUTATION_RATE, 
                                  min(MAX_MUTATION_RATE, 
                                      mutation_rate * (1.0 + np.random.normal(0, 0.1))))
        
        for i in range(n):
            if random.random() < dynamic_mutation_rate:
                # Mutate x, y, and r components
                gene_idx = i % 3
                
                if gene_idx == 0:  # x coordinate
                    mutated[i] = mutated[i] + np.random.normal(0, 0.01)
                    mutated[i] = np.clip(mutated[i], 
                                       mutated[i+1], 1 - mutated[i+1])  # bound by y and r
                elif gene_idx == 1:  # y coordinate
                    mutated[i] = mutated[i] + np.random.normal(0, 0.01)
                    mutated[i] = np.clip(mutated[i], 
                                       mutated[i-1], 1 - mutated[i-1])  # bound by x and r
                else:  # radius
                    # Mutate radius with multiplicative factor
                    mutated[i] = mutated[i] * np.random.uniform(0.8, 1.2)
                    mutated[i] = np.clip(mutated[i], min_radius, max_radius)
        
        # Repair to ensure validity
        repair_individual(mutated)
        return mutated
    
    # Repair individual to ensure it satisfies constraints
    def repair_individual(individual):
        circles = individual.reshape(-1, 3)
        n = len(circles)
        
        # Ensure all circles are within bounds
        for i in range(n):
            x, y, r = circles[i]
            circles[i][0] = np.clip(x, r, 1 - r)
            circles[i][1] = np.clip(y, r, 1 - r)
            circles[i][2] = np.clip(r, min_radius, max_radius)
        
        # Resolve overlaps using simple iterative approach
        for _ in range(20):  # Limit repair iterations
            any_improvement = False
            for i in range(n):
                x1, y1, r1 = circles[i]
                for j in range(i+1, n):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if dist < r1 + r2:
                        # Move circles apart along the line connecting centers
                        dx = x2 - x1
                        dy = y2 - y1
                        length = np.sqrt(dx*dx + dy*dy)
                        
                        if length > 0:
                            # Normalize and push apart
                            dx /= length
                            dy /= length
                            separation = (r1 + r2) - dist
                            
                            # Move both circles away from each other
                            circles[i][0] -= dx * separation * 0.5
                            circles[i][1] -= dy * separation * 0.5
                            circles[j][0] += dx * separation * 0.5
                            circles[j][1] += dy * separation * 0.5
                            
                            # Clamp to bounds again
                            circles[i][0] = np.clip(circles[i][0], circles[i][2], 1 - circles[i][2])
                            circles[i][1] = np.clip(circles[i][1], circles[i][2], 1 - circles[i][2])
                            circles[j][0] = np.clip(circles[j][0], circles[j][2], 1 - circles[j][2])
                            circles[j][1] = np.clip(circles[j][1], circles[j][2], 1 - circles[j][2])
                            
                            any_improvement = True
            
            if not any_improvement:
                break
    
    # Selection with tournament and elitism
    def select(population, fitnesses):
        # Elitism: keep the best individuals
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        selected = [population[i] for i in elite_indices]
        
        # Tournament selection for the rest
        tournament_size = 5
        remaining_slots = POPULATION_SIZE - ELITISM_COUNT
        
        for _ in range(remaining_slots):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_index])
        
        return selected
    
    # Local refinement function
    def refine_radii(circles):
        """Try to increase radii of circles while maintaining constraints"""
        circles = circles.copy()
        max_iter = 30
        
        for _ in range(max_iter):
            improved = False
            # Try each circle for radius increase
            for i in range(len(circles)):
                old_r = circles[i][2]
                # Try increasing radius by 2%
                new_r = min(old_r * 1.02, max_radius)
                
                # Check if we can increase radius without violating constraints
                temp_circles = circles.copy()
                temp_circles[i][2] = new_r
                
                # Check containment
                x, y, r = temp_circles[i]
                if not (min_radius <= r <= 1 - min_radius and 
                       min_radius <= x <= 1 - min_radius and 
                       min_radius <= y <= 1 - min_radius):
                    continue
                    
                # Check overlap with others using KDTree
                valid = True
                try:
                    tree = cKDTree(temp_circles[:, :2])
                    neighbors = tree.query_ball_point([x, y], 2 * max_radius)
                    for j in neighbors:
                        if i != j:
                            x1, y1, r1 = temp_circles[i]
                            x2, y2, r2 = temp_circles[j]
                            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if dist < r1 + r2:
                                valid = False
                                break
                except:
                    # Fallback to brute force
                    for j in range(len(temp_circles)):
                        if i != j:
                            x1, y1, r1 = temp_circles[i]
                            x2, y2, r2 = temp_circles[j]
                            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if dist < r1 + r2:
                                valid = False
                                break
                
                if valid:
                    circles[i][2] = new_r
                    improved = True
            
            if not improved:
                break
                
        return circles
    
    # Main evolutionary algorithm
    best_total_radius = 0
    best_solution = None
    
    for attempt in range(MAX_ATTEMPTS):
        # Generate initial population
        population = []
        for _ in range(POPULATION_SIZE):
            # Use Voronoi initialization
            initial_circles = generate_voronoi_initial()
            individual = initial_circles.flatten().tolist()
            population.append(individual)
        
        # Evolutionary cycle
        for generation in range(GENERATIONS):
            # Evaluate fitness
            fitnesses = [evaluate(individual)[0] for individual in population]
            
            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_total_radius:
                best_total_radius = fitnesses[max_fitness_idx]
                best_solution = np.array(population[max_fitness_idx]).reshape(-1, 3).copy()
            
            # Selection
            selected = select(population, fitnesses)
            
            # Create new population
            new_population = selected.copy()  # Elitism
            
            # Crossover and mutation
            while len(new_population) < POPULATION_SIZE:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                child = crossover(parent1, parent2)
                child = mutate(child, INITIAL_MUTATION_RATE)
                
                new_population.append(child)
            
            population = new_population[:POPULATION_SIZE]
        
        # Final refinement of best solution
        if best_solution is not None:
            refined = refine_radii(best_solution)
            refined_total = np.sum(refined[:, 2])
            if refined_total > best_total_radius:
                best_total_radius = refined_total
                best_solution = refined
    
    # Fallback if nothing was found
    if best_solution is None:
        # Generate a simple grid-based solution
        circles = np.zeros((n_circles, 3))
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
                r = min(spacing_x, spacing_y) * 0.3
                circles[idx] = [x, y, r]
                idx += 1
        
        # Fill remaining circles with random positions
        for i in range(idx, n_circles):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            circles[i] = [x, y, r]
        
        return circles
    
    return best_solution

# EVOLVE-BLOCK-END