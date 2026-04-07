# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import cdist
from deap import base, creator, tools, algorithms
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_hexagonal_grid(n_circles, square_size=1.0):
    """Generate initial circle positions using a hexagonal grid pattern"""
    # Calculate how many rows and columns we need
    rows = int(math.sqrt(n_circles / (math.sqrt(3)/2)))
    cols = int(n_circles / rows) + 1

    # Ensure we have enough space
    if rows * cols < n_circles:
        rows += 1

    # Calculate spacing based on desired circle count
    max_radius = 0.1
    spacing_x = 2 * max_radius
    spacing_y = 2 * max_radius * math.sqrt(3) / 2

    # Adjust spacing to fit within square
    while spacing_x * cols > square_size or spacing_y * rows > square_size:
        max_radius *= 0.9
        spacing_x = 2 * max_radius
        spacing_y = 2 * max_radius * math.sqrt(3) / 2

    # Generate positions
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n_circles:
                break
            x = spacing_x * j + max_radius
            if i % 2 == 1:  # Offset every other row
                x += spacing_x / 2
            y = spacing_y * i + max_radius
            if x <= square_size - max_radius and y <= square_size - max_radius:
                positions.append([x, y])
        if len(positions) >= n_circles:
            break

    # If we don't have enough points, fill with random points
    while len(positions) < n_circles:
        positions.append([np.random.uniform(max_radius, square_size - max_radius),
                         np.random.uniform(max_radius, square_size - max_radius)])

    # Generate initial radii (they'll be optimized later)
    radii = [max_radius] * min(len(positions), n_circles)

    # Fill with remaining circles if needed
    if len(positions) < n_circles:
        for _ in range(n_circles - len(positions)):
            radii.append(max_radius)

    return np.array(positions[:n_circles]), radii[:n_circles]

def validate_solution(circles, square_size=1.0):
    """Validate that all circles are within bounds and non-overlapping"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > square_size or y - r < 0 or y + r > square_size:
            return False
    
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
    
    return True

def evaluate_fitness(individual, square_size=1.0):
    """Evaluate fitness of an individual (circles) - higher is better"""
    # Convert individual to circles array
    circles = np.array(individual).reshape(-1, 3)
    
    # Check if solution is valid
    if not validate_solution(circles, square_size):
        return (0,)  # Invalid solution gets very low fitness
    
    # Return negative sum of radii (since we want to maximize sum of radii)
    total_radius = np.sum(circles[:, 2])
    return (-total_radius,)

def mutate_individual(individual, indpb=0.1, square_size=1.0):
    """Custom mutation for circle positioning"""
    for i in range(len(individual)):
        if random.random() < indpb:
            # Mutate either position or radius
            if i % 3 == 0:  # Mutate x coordinate
                individual[i] = max(0.001, min(square_size - 0.001, individual[i] + random.gauss(0, 0.02)))
            elif i % 3 == 1:  # Mutate y coordinate
                individual[i] = max(0.001, min(square_size - 0.001, individual[i] + random.gauss(0, 0.02)))
            else:  # Mutate radius
                individual[i] = max(0.001, min(0.5, individual[i] + random.gauss(0, 0.01)))
    return individual,

def crossover_individuals(ind1, ind2, crossover_rate=0.8):
    """Custom crossover that maintains spatial relationships"""
    if random.random() < crossover_rate:
        # Simple uniform crossover
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def optimize_radii(circles, square_size=1.0, max_iterations=100):
    """Perform local optimization to increase radii while maintaining constraints"""
    n = len(circles)
    circles = circles.copy()
    
    # Create a copy to work with
    temp_circles = circles.copy()
    
    for _ in range(max_iterations):
        improved = False
        # Try to increase each circle's radius
        for i in range(n):
            current_r = temp_circles[i, 2]
            
            # Binary search for maximum possible radius
            left, right = current_r, min(0.5, min(temp_circles[:, 0] - current_r, temp_circles[:, 1] - current_r,
                                                 square_size - temp_circles[:, 0] - current_r,
                                                 square_size - temp_circles[:, 1] - current_r))
            
            if right > left:
                # Simple greedy approach: try to increase radius
                new_r = min(right, current_r * 1.1)  # Slightly increase radius
                
                # Check if we can make it larger
                temp_circles[i, 2] = new_r
                
                # Validate the change
                if validate_solution(temp_circles, square_size):
                    circles[i, 2] = new_r
                    improved = True
                else:
                    temp_circles[i, 2] = current_r
        if not improved:
            break
            
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    square_size = 1.0
    
    # Initialize population using hexagonal grid
    positions, radii = generate_hexagonal_grid(n, square_size)
    
    # Create initial population
    population = []
    for _ in range(50):  # Larger population for better diversity
        individual = []
        for i in range(n):
            # Add some random noise to initial positions
            x = max(0.001, min(square_size - 0.001, positions[i][0] + np.random.normal(0, 0.01)))
            y = max(0.001, min(square_size - 0.001, positions[i][1] + np.random.normal(0, 0.01)))
            r = max(0.001, min(0.5, radii[i] + np.random.normal(0, 0.005)))
            individual.extend([x, y, r])
        population.append(individual)
    
    # Set up evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [np.random.uniform(0.001, square_size-0.001) if i % 3 < 2 else np.random.uniform(0.001, 0.5) for i in range(n*3)])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_fitness, square_size=square_size)
    
    # Register crossover and mutation operators
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual, indpb=0.15, square_size=square_size)  # Higher initial mutation rate
    toolbox.register("select", tools.selTournament, tournsize=5)  # Tournament selection
    
    # Run the evolutionary algorithm
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run evolution for more generations with decreasing mutation rate
    for gen in range(1000):  # More generations
        # Adapt mutation rate
        current_mutation_rate = max(0.02, 0.15 * (1 - gen/1000))
        toolbox.unregister("mutate")
        toolbox.register("mutate", mutate_individual, indpb=current_mutation_rate, square_size=square_size)
        
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=current_mutation_rate, 
                                         ngen=1, stats=stats, halloffame=hof, verbose=False)
    
    # Get the best individual
    best_individual = hof[0]
    circles = np.array(best_individual).reshape(-1, 3)
    
    # Apply local optimization to maximize radii
    circles = optimize_radii(circles, square_size)
    
    # Final validation
    if not validate_solution(circles, square_size):
        # Fall back to initial hexagonal configuration if validation fails
        positions, radii = generate_hexagonal_grid(n, square_size)
        circles = np.column_stack([positions, radii])
    
    return circles

# EVOLVE-BLOCK-END