# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
from collections import defaultdict
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    MAX_EVALS = 10000
    POPSIZE = 50
    GEN_COUNT = 200
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    def eval_individual(individual):
        """Evaluate fitness of individual (negative sum of radii for maximization)"""
        # Extract circles from individual (x, y, r) triplets
        circles = []
        for i in range(n):
            x, y, r = individual[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return (float('inf'),)  # Invalid - penalize heavily
            
            circles.append((x, y, r))
        
        # Check overlaps
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                if dist_sq < (r1+r2)**2:
                    return (float('inf'),)  # Overlapping - penalize heavily
        
        # Return negative sum of radii (minimize to maximize sum)
        total_radius = sum(circle[2] for circle in circles)
        return (-total_radius,)
    
    def create_initial_population():
        """Create initial population using Voronoi-based approach"""
        pop = []
        for _ in range(POPSIZE):
            # Try multiple initialization methods
            try:
                # Method 1: Voronoi-inspired initialization
                circles = voronoi_init(n)
                if circles is not None:
                    # Flatten to individual genes (x, y, r)
                    individual = []
                    for x, y, r in circles:
                        individual.extend([x, y, r])
                    pop.append(individual)
                    continue
                    
            except:
                pass
                
            # Method 2: Grid fallback
            try:
                circles = grid_init(n)
                individual = []
                for x, y, r in circles:
                    individual.extend([x, y, r])
                pop.append(individual)
                continue
            except:
                pass
                
            # Method 3: Random initialization (fallback)
            try:
                circles = random_init(n)
                individual = []
                for x, y, r in circles:
                    individual.extend([x, y, r])
                pop.append(individual)
                continue
            except:
                # If all methods fail, use dummy individual
                individual = [random.random(), random.random(), random.random() for _ in range(n*3)]
                pop.append(individual)
                
        return pop
    
    def voronoi_init(n):
        """Voronoi-inspired initialization to spread circles"""
        circles = []
        # Generate candidate points using regular grid
        grid_size = int(np.ceil(np.sqrt(n)))
        candidates = []
        spacing = 0.15  # Spacing to avoid boundary issues
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = spacing + (i * (1 - 2*spacing) / (grid_size - 1)) if grid_size > 1 else 0.5
                y = spacing + (j * (1 - 2*spacing) / (grid_size - 1)) if grid_size > 1 else 0.5
                candidates.append((x, y))
                
        # Place circles ensuring they don't overlap with each other
        for i, (x, y) in enumerate(candidates[:n]):
            max_radius = min(x, 1-x, y, 1-y)
            
            # Try to find a reasonably large radius
            radius = max_radius * 0.4
            if radius <= 0:
                radius = 0.05
                
            # Check for overlaps with already placed circles
            valid = True
            for prev_x, prev_y, prev_r in circles:
                dist_sq = (x - prev_x)**2 + (y - prev_y)**2
                if dist_sq < (radius + prev_r)**2:
                    valid = False
                    break
                    
            if valid:
                circles.append((x, y, radius))
                if len(circles) == n:
                    break
                    
        if len(circles) < n:
            # Fill remaining with random positions
            for i in range(len(circles), n):
                circles.append(random_init_circle())
                
        return circles
    
    def grid_init(n):
        """Grid-based initialization"""
        circles = []
        rows = cols = int(np.ceil(np.sqrt(n)))
        padding = 0.1
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = padding + (j * (1 - 2*padding) / (cols - 1)) if cols > 1 else 0.5
                y = padding + (i * (1 - 2*padding) / (rows - 1)) if rows > 1 else 0.5
                radius = min(x, 1-x, y, 1-y) * 0.25
                if radius > 0:
                    circles.append((x, y, radius))
                    
        return circles[:n]
    
    def random_init(n):
        """Random initialization with basic constraints"""
        circles = []
        max_attempts = 1000
        for _ in range(max_attempts):
            if len(circles) >= n:
                break
            circle = random_init_circle()
            if circle is not None:
                circles.append(circle)
                
        # Fill any missing circles
        while len(circles) < n:
            circles.append(random_init_circle())
            
        return circles[:n]
    
    def random_init_circle():
        """Generate one random valid circle"""
        max_radius = 0.4
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        radius = random.uniform(0.01, max_radius)
        return (x, y, radius)
    
    def mutate_individual(individual):
        """Custom mutation operator"""
        # Convert individual to list of (x,y,r) tuples
        circles = [(individual[i], individual[i+1], individual[i+2]) for i in range(0, len(individual), 3)]
        
        # Apply mutation to some circles
        mutated_circles = []
        for x, y, r in circles:
            if random.random() < 0.3:  # 30% chance to mutate
                # Mutate radius with Gaussian noise
                new_r = r + random.gauss(0, 0.02)
                new_r = max(0.001, min(0.4, new_r))
                
                # Mutate position with Gaussian noise
                new_x = x + random.gauss(0, 0.01)
                new_y = y + random.gauss(0, 0.01)
                
                # Ensure position stays within bounds
                new_x = max(r, min(1-r, new_x))
                new_y = max(r, min(1-r, new_y))
                
                mutated_circles.append((new_x, new_y, new_r))
            else:
                mutated_circles.append((x, y, r))
        
        # Flatten back to individual
        flattened = []
        for x, y, r in mutated_circles:
            flattened.extend([x, y, r])
        return tuple(flattened),
    
    def crossover_individuals(ind1, ind2):
        """Custom crossover operator"""
        # Create offspring by combining parts from both parents
        child1, child2 = [], []
        
        for i in range(0, len(ind1), 3):
            # 50% chance to take from parent1, 50% from parent2
            if random.random() < 0.5:
                child1.extend([ind1[i], ind1[i+1], ind1[i+2]])
                child2.extend([ind2[i], ind2[i+1], ind2[i+2]])  
            else:
                child1.extend([ind2[i], ind2[i+1], ind2[i+2]])
                child2.extend([ind1[i], ind1[i+1], ind1[i+2]])
        
        return tuple(child1), tuple(child2)
    
    def repair_individual(individual):
        """Repair invalid individuals"""
        # Ensure circles are within bounds and non-overlapping
        circles = [(individual[i], individual[i+1], individual[i+2]) for i in range(0, len(individual), 3)]
        
        # Adjust positions and radii to satisfy constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Ensure circle is within bounds
            r = min(r, x, 1-x, y, 1-y)
            
            # Clamp position to be within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[i] = (x, y, r)
            
        # Resolve overlaps iteratively
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    
                    # If overlapping
                    if dist_sq < (r1+r2)**2:
                        # Move circles apart
                        if dist_sq > 0:
                            dist = dist_sq**0.5
                            overlap = (r1 + r2) - dist
                            # Normalize direction vector
                            dx_norm = dx / dist if dist > 0 else 0
                            dy_norm = dy / dist if dist > 0 else 0
                            
                            # Adjust positions
                            move_factor = 0.5
                            circles[i] = (x1 + dx_norm * overlap * move_factor, 
                                        y1 + dy_norm * overlap * move_factor, r1)
                            circles[j] = (x2 - dx_norm * overlap * move_factor, 
                                        y2 - dy_norm * overlap * move_factor, r2)
                            changed = True
                        else:
                            # Handle identical positions
                            circles[i] = (x1 + 0.001, y1, r1)
                            circles[j] = (x2 - 0.001, y2, r2)
                            changed = True
            iterations += 1
        
        # Flatten back to individual
        flattened = []
        for x, y, r in circles:
            flattened.extend([x, y, r])
        return tuple(flattened)
    
    # Setup DEAP framework
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, random.random, n*3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Custom operators
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population = create_initial_population()
    
    # Run evolutionary algorithm
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        population, logbook = algorithms.eaSimple(population, toolbox, 
                                                cxpb=0.7, mutpb=0.2, 
                                                ngen=GEN_COUNT, 
                                                stats=stats, 
                                                halloffame=hof, 
                                                verbose=False)
    except Exception as e:
        # Fallback to simple optimization loop
        best_ind = None
        best_fitness = float('inf')
        for gen in range(GEN_COUNT):
            # Evaluate current population
            for ind in population:
                fit = eval_individual(ind)[0]
                if fit < best_fitness:
                    best_fitness = fit
                    best_ind = list(ind)
                    
            # Selection and reproduction
            selected = tools.selTournament(population, k=len(population), tournsize=3)
            offspring = []
            for i in range(0, len(selected), 2):
                if i+1 < len(selected):
                    child1, child2 = toolbox.mate(selected[i], selected[i+1])
                    child1 = toolbox.mutate(child1)[0]
                    child2 = toolbox.mutate(child2)[0]
                    offspring.extend([child1, child2])
                else:
                    child = toolbox.mutate(selected[i])[0]
                    offspring.append(child)
            
            # Repair all offspring
            for i in range(len(offspring)):
                offspring[i] = repair_individual(offspring[i])
                
            population = offspring
            
            # Early stopping if improvement is minimal
            if gen > 10 and abs(best_fitness) < 0.1:
                break
    
    # Get best individual
    if len(hof) > 0:
        best_ind = hof[0]
    else:
        # Fallback to best from population
        best_ind = min(population, key=lambda x: eval_individual(x)[0])
    
    # Convert final result to circles array
    circles = []
    for i in range(0, len(best_ind), 3):
        circles.append([best_ind[i], best_ind[i+1], best_ind[i+2]])
    
    # Ensure we have exactly n circles
    circles = circles[:n]
    while len(circles) < n:
        circles.append([0.5, 0.5, 0.01])  # Add dummy circles if needed
    
    return np.array(circles)

# EVOLVE-BLOCK-END
