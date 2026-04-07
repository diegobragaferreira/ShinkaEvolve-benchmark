# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from deap import base, creator, tools, algorithms
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Parameters
    n_circles = 32
    grid_size = 10  # Grid resolution for spatial indexing
    max_radius = 0.5
    min_radius = 0.001
    
    # Create spatial grid for fast collision detection
    class SpatialGrid:
        def __init__(self, grid_size):
            self.grid_size = grid_size
            self.grid = {}
            
        def _get_cell(self, x, y):
            return (int(x * self.grid_size), int(y * self.grid_size))
        
        def add_circle(self, x, y, r):
            cell = self._get_cell(x, y)
            if cell not in self.grid:
                self.grid[cell] = []
            self.grid[cell].append((x, y, r))
            
        def get_neighbors(self, x, y, r):
            neighbors = []
            center_cell = self._get_cell(x, y)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (center_cell[0] + dx, center_cell[1] + dy)
                    if cell in self.grid:
                        neighbors.extend(self.grid[cell])
            return neighbors
            
        def clear(self):
            self.grid.clear()
    
    def is_valid_placement(x, y, r, existing_circles, spatial_grid):
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
            
        # Get nearby circles using spatial grid
        neighbors = spatial_grid.get_neighbors(x, y, r)
        
        # Check collisions with nearby circles
        for nx, ny, nr in neighbors:
            dist_sq = (x - nx)**2 + (y - ny)**2
            if dist_sq < (r + nr)**2:
                return False
                
        return True
    
    def evaluate_individual(individual):
        # Format individual: [x1, y1, r1, x2, y2, r2, ...]
        circles = []
        spatial_grid = SpatialGrid(grid_size)
        
        total_radius = 0
        for i in range(0, len(individual), 3):
            x, y, r = individual[i:i+3]
            if is_valid_placement(x, y, r, circles, spatial_grid):
                circles.append((x, y, r))
                spatial_grid.add_circle(x, y, r)
                total_radius += r
            else:
                return (0,)  # Invalid configuration
        
        # Only count valid circles
        return (total_radius,)
    
    # Initialize population with good starting points
    def create_initial_population(size):
        population = []
        
        # Try several strategies
        for _ in range(size):
            individual = []
            circles_placed = 0
            spatial_grid = SpatialGrid(grid_size)
            
            # Strategy 1: Place circles near corners
            if circles_placed < n_circles:
                # Fill corners with large circles
                corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
                for cx, cy in corners[:min(4, n_circles)]:
                    r = min(0.1, max_radius)
                    individual.extend([cx, cy, r])
                    circles_placed += 1
                    spatial_grid.add_circle(cx, cy, r)
            
            # Strategy 2: Fill remaining space with random placement
            while circles_placed < n_circles:
                x = random.uniform(0.01, 0.99)
                y = random.uniform(0.01, 0.99)
                r = random.uniform(min_radius, max_radius)
                
                if is_valid_placement(x, y, r, [], spatial_grid):
                    individual.extend([x, y, r])
                    circles_placed += 1
                    spatial_grid.add_circle(x, y, r)
                elif circles_placed == 0 and random.random() < 0.1:  # Occasionally accept invalid ones for diversity
                    individual.extend([x, y, r])  # Even if invalid, still include
                    
            population.append(individual)
            
        return population
    
    # DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.01, 0.99)
    toolbox.register("attr_radius", random.uniform, min_radius, max_radius)
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     [toolbox.attr_float, toolbox.attr_float, toolbox.attr_radius], n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Custom crossover and mutation with better control
    def cx_two_point(ind1, ind2):
        size = min(len(ind1), len(ind2)) // 3
        if size < 2:
            return ind1, ind2
            
        cxpoint1 = random.randint(1, size-1) * 3
        cxpoint2 = random.randint(cxpoint1, size-1) * 3
        
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
        return ind1, ind2
    
    def mut_gaussian(individual, mu=0, sigma=0.02, indpb=0.1):
        for i in range(0, len(individual), 3):
            if random.random() < indpb:
                # Mutate x
                individual[i] = max(0.01, min(0.99, individual[i] + random.gauss(mu, sigma)))
            if random.random() < indpb:
                # Mutate y
                individual[i+1] = max(0.01, min(0.99, individual[i+1] + random.gauss(mu, sigma)))
            if random.random() < indpb:
                # Mutate r
                individual[i+2] = max(min_radius, min(max_radius, individual[i+2] + random.gauss(mu, sigma/2)))
        return individual,
    
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", cx_two_point)
    toolbox.register("mutate", mut_gaussian)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop = create_initial_population(100)
    
    # Evolution parameters
    CXPB = 0.7
    MUTPB = 0.3
    NGEN = 50
    
    # Run evolution
    for gen in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
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
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        pop[:] = offspring
    
    # Find best individual
    best_ind = tools.selBest(pop, 1)[0]
    
    # Convert back to circles format
    circles = []
    for i in range(0, len(best_ind), 3):
        circles.append([best_ind[i], best_ind[i+1], best_ind[i+2]])
    
    return np.array(circles)

# EVOLVE-BLOCK-END
