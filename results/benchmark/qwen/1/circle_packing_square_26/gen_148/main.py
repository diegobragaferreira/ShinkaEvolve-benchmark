# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from deap import base, creator, tools, algorithms
import time

# Core validation functions
def _validate_circle_placement(circles: np.ndarray) -> bool:
    """Validate that circles are within bounds and don't overlap."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check non-overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)

    # Find all pairs within distance 2*r (minimum separation needed to avoid overlap)
    pairs = tree.query_pairs(2 * min(circles[:, 2]), output_type='ndarray')

    for i, j in pairs:
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance_sq = (x1 - x2)**2 + (y1 - y2)**2
        min_distance_sq = (r1 + r2)**2
        if distance_sq < min_distance_sq:
            return False

    return True

def _evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as negative sum of radii (since we want to maximize)."""
    if not _validate_circle_placement(circles):
        return -float('inf')  # Invalid configuration gets very low fitness
    return float(np.sum(circles[:, 2]))

# Enhanced initialization strategies
def _generate_voronoi_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """Generate initial circle positions using a Voronoi-inspired spreading mechanism."""
    np.random.seed(seed)

    # Create a grid of candidate positions
    grid_size = max(3, int(np.ceil(np.sqrt(n_circles))))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)

    # Generate all grid points
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append([x, y])

    # If we have more circles than grid points, add some random points
    if len(grid_points) < n_circles:
        extra_points = n_circles - len(grid_points)
        for _ in range(extra_points):
            grid_points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

    # Shuffle the points to avoid systematic bias
    random.shuffle(grid_points)

    # Take the first n_circles points
    points = np.array(grid_points[:n_circles])

    # Initialize circles with small radii
    circles = np.zeros((n_circles, 3))
    circles[:, 0] = points[:, 0]  # x coordinates
    circles[:, 1] = points[:, 1]  # y coordinates
    circles[:, 2] = 0.01         # initial small radii

    return circles

def _generate_grid_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """Generate initial circle positions using a grid layout."""
    np.random.seed(seed)
    
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing_x = 0.9 / grid_size
    spacing_y = 0.9 / grid_size
    
    circles = np.zeros((n_circles, 3))
    
    for i in range(n_circles):
        row = i // grid_size
        col = i % grid_size
        x = 0.05 + (col + 0.5) * spacing_x
        y = 0.05 + (row + 0.5) * spacing_y
        r = 0.01  # Initial small radius
        
        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = r

    return circles

def _generate_spiral_initialization(n_circles: int, seed: int = 42) -> np.ndarray:
    """Generate initial circle positions using a spiral pattern."""
    np.random.seed(seed)
    
    circles = np.zeros((n_circles, 3))
    
    # Spiral parameters
    a = 0.05  # spiral parameter
    b = 0.05  # spiral parameter
    
    for i in range(n_circles):
        angle = 2 * np.pi * i / n_circles * 5  # spiral with 5 turns
        radius = a + b * angle
        radius = min(radius, 0.45)  # cap at reasonable value
        
        x = 0.5 + radius * np.cos(angle) * 0.4
        y = 0.5 + radius * np.sin(angle) * 0.4
        
        # Clip to valid range
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        
        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = 0.01  # small initial radius

    return circles

def _greedy_fallback(n_circles: int) -> np.ndarray:
    """Fallback method to generate a feasible configuration."""
    # Simple greedy approach: place circles in order of decreasing radius
    circles = np.zeros((n_circles, 3))

    # Start with small radii and gradually increase
    # Place in a way that they don't overlap initially
    positions = []
    radii = []

    # Try to place circles greedily by spacing them out
    placed = 0
    radius = 0.05
    while placed < n_circles and radius > 0.005:
        # Try placing circles in a spiral pattern or grid
        attempt = 0
        while attempt < 100 and placed < n_circles:
            # Place in grid-like fashion
            rows = int(np.sqrt(n_circles)) + 1
            cols = n_circles // rows + 1

            for i in range(rows):
                for j in range(cols):
                    if placed >= n_circles:
                        break
                    x = 0.1 + j * 0.8 / cols
                    y = 0.1 + i * 0.8 / rows

                    # Check if this position is valid
                    valid = True
                    for pos, rad in zip(positions, radii):
                        dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                        if dist_sq < (rad + radius)**2:
                            valid = False
                            break

                    if valid:
                        positions.append([x, y])
                        radii.append(radius)
                        placed += 1
            attempt += 1

        radius *= 0.9  # Decrease radius slightly

    # Fill remaining circles
    while placed < n_circles:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        positions.append([x, y])
        radii.append(0.01)
        placed += 1

    circles[:, 0] = [pos[0] for pos in positions]
    circles[:, 1] = [pos[1] for pos in positions]
    circles[:, 2] = radii

    return circles

# Advanced optimization components
def _initialize_population(n_circles: int, n_pop: int, seed: int = 42) -> list:
    """Initialize population with multiple strategies."""
    np.random.seed(seed)
    population = []
    
    # Add multiple initialization strategies
    # Voronoi initialization
    voronoi_init = _generate_voronoi_initialization(n_circles, seed)
    population.append(voronoi_init.flatten().tolist())
    
    # Grid initialization
    grid_init = _generate_grid_initialization(n_circles, seed + 1)
    population.append(grid_init.flatten().tolist())
    
    # Spiral initialization
    spiral_init = _generate_spiral_initialization(n_circles, seed + 2)
    population.append(spiral_init.flatten().tolist())
    
    # Add some random initializations
    for i in range(n_pop - 3):
        individual = []
        for j in range(n_circles):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.005, 0.45)
            individual.extend([x, y, r])
        population.append(individual)
    
    return population

def _adaptive_mutation(individual, generation: int = 0, max_gen: int = 100):
    """Apply adaptive mutation with dynamic rate based on generation."""
    mutation_rate = 0.15 - (generation / max_gen) * 0.10  # Decrease from 0.15 to 0.05
    mutation_rate = max(mutation_rate, 0.05)  # Min bound
    
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            if i % 3 == 0:  # x coordinate
                individual[i] = np.clip(individual[i] + np.random.normal(0, 0.02), 0.05, 0.95)
            elif i % 3 == 1:  # y coordinate
                individual[i] = np.clip(individual[i] + np.random.normal(0, 0.02), 0.05, 0.95)
            else:  # radius
                individual[i] = np.clip(individual[i] + np.random.normal(0, 0.01), 0.005, 0.45)
    return individual,

def _constraint_aware_local_search(circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """
    Apply constraint-aware local search to improve the solution.
    """
    improved_circles = circles.copy()
    n = len(improved_circles)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try to increase radii while maintaining constraints
        for i in range(n):
            x, y, r = improved_circles[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x, 1-x, y, 1-y)
            
            # Find neighboring circles to check constraints
            neighbors = []
            for j in range(n):
                if i != j:
                    x2, y2, r2 = improved_circles[j]
                    dist_sq = (x - x2)**2 + (y - y2)**2
                    min_dist_sq = (r + r2)**2
                    neighbors.append((j, dist_sq, min_dist_sq))
            
            # Check if we can safely increase radius
            safe_increase = True
            for j, dist_sq, min_dist_sq in neighbors:
                if dist_sq < (r + 0.01)**2:
                    safe_increase = False
                    break
            
            if safe_increase and r < max_radius - 0.001:
                new_radius = min(max_radius, r + 0.005)
                improved_circles[i, 2] = new_radius
                improved = True
        
        # If no improvement in radii, try position adjustments
        if not improved:
            for i in range(n):
                x, y, r = improved_circles[i]
                
                # Try small movements in 8 directions
                movements = [(-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                           (0, -0.005),              (0, 0.005),
                           (0.005, -0.005), (0.005, 0), (0.005, 0.005)]
                
                best_x, best_y = x, y
                best_score = -float('inf')
                
                for dx, dy in movements:
                    new_x, new_y = x + dx, y + dy
                    
                    # Check bounds
                    if new_x - r < 0 or new_x + r > 1 or new_y - r < 0 or new_y + r > 1:
                        continue
                        
                    # Check overlap with neighbors
                    overlap_penalty = 0
                    valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = improved_circles[j]
                            dist_sq = (new_x - x2)**2 + (new_y - y2)**2
                            min_dist_sq = (r + r2)**2
                            if dist_sq < min_dist_sq:
                                overlap_penalty += (min_dist_sq - dist_sq) * 1000
                                valid = False
                    
                    if valid:
                        # Score based on overlap reduction and radius maximization
                        score = -overlap_penalty + r
                        if score > best_score:
                            best_score = score
                            best_x, best_y = new_x, new_y
                
                # Apply the best movement if it helps
                if best_x != x or best_y != y:
                    improved_circles[i, 0] = best_x
                    improved_circles[i, 1] = best_y
                    improved = True
        
        # If no improvement made, exit loop
        if not improved:
            break
    
    return improved_circles

def _evolutionary_pipeline(n_circles: int, max_evaluations: int, seed: int = 42) -> np.ndarray:
    """Execute the complete evolutionary optimization pipeline."""
    np.random.seed(seed)

    # Create individual and population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define bounds for each dimension: [x, y, r] for each circle
    # x, y in [0.05, 0.95], r in [0.005, 0.45]
    def create_individual():
        individual = []
        for i in range(n_circles):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.005, 0.45)
            individual.extend([x, y, r])
        return creator.Individual(individual)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    def evaluate(individual):
        circles = np.array(individual).reshape(-1, 3)
        return _evaluate_fitness(circles),

    toolbox.register("evaluate", evaluate)

    # Genetic operators with adaptive mutation
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", _adaptive_mutation)
    toolbox.register("select", tools.selTournament, tournsize=5)

    # Create initial population using hybrid approach
    pop = _initialize_population(n_circles, 50, seed)

    # Convert to DEAP individuals
    deap_pop = []
    for ind_list in pop:
        deap_ind = creator.Individual(ind_list)
        deap_pop.append(deap_ind)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, deap_pop))
    for ind, fit in zip(deap_pop, fitnesses):
        ind.fitness.values = fit

    # Evolution parameters
    cxpb = 0.7      # crossover probability
    ngen = max_evaluations // 50  # number of generations

    # Begin evolution with adaptive parameters
    for gen in range(ngen):
        # Select the next generation individuals
        offspring = toolbox.select(deap_pop, len(deap_pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply adaptive mutation
        for mutant in offspring:
            toolbox.mutate(mutant, gen)
            del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the new generation
        deap_pop[:] = offspring

    # Get the best individual
    best_ind = tools.selBest(deap_pop, 1)[0]
    circles = np.array(best_ind).reshape(-1, 3)

    return circles

def _refinement_stage(circles: np.ndarray) -> np.ndarray:
    """Apply final refinement to improve solution quality."""
    # Apply constraint-aware local search
    refined_circles = _constraint_aware_local_search(circles, max_iterations=200)
    
    # Additional post-processing if needed
    return refined_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    max_evaluations = 5000
    seed = 42

    # Stage 1: Evolutionary optimization
    try:
        circles = _evolutionary_pipeline(n, max_evaluations, seed)
        
        # Stage 2: Refinement
        circles = _refinement_stage(circles)
        
        # Stage 3: Validation and fallback
        if not _validate_circle_placement(circles):
            # Try greedy fallback
            circles = _greedy_fallback(n)
            
            if not _validate_circle_placement(circles):
                # Last resort: Voronoi initialization
                circles = _generate_voronoi_initialization(n, seed)
                
    except Exception as e:
        # Fallback if anything goes wrong
        circles = _generate_voronoi_initialization(n, seed)
        circles = _constraint_aware_local_search(circles, max_iterations=100)
    
    return circles

# EVOLVE-BLOCK-END