# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from deap import base, creator, tools, algorithms
import time
from itertools import combinations

# Global constants for optimization
POPULATION_SIZE = 80
GENERATIONS = 80
TOURNAMENT_SIZE = 3
INITIAL_MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.7
ADAPTIVE_DECAY = 0.98
MIN_MUTATION_RATE = 0.02
MAX_RADIUS = 0.3
BOUNDARY_PENALTY_WEIGHT = 1000
OVERLAP_PENALTY_WEIGHT = 100

def check_containment_with_penalty(circles):
    """Check containment with penalty scoring"""
    penalty = 0
    for x, y, r in circles:
        if x - r < 0:
            penalty += BOUNDARY_PENALTY_WEIGHT * abs(x - r)
        if x + r > 1:
            penalty += BOUNDARY_PENALTY_WEIGHT * abs(x + r - 1)
        if y - r < 0:
            penalty += BOUNDARY_PENALTY_WEIGHT * abs(y - r)
        if y + r > 1:
            penalty += BOUNDARY_PENALTY_WEIGHT * abs(y + r - 1)
    return penalty

def check_overlap_with_penalty(circles):
    """Check overlap with penalty scoring"""
    penalty = 0
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                # Penalty proportional to overlap amount
                overlap = (r1 + r2) - distance
                penalty += OVERLAP_PENALTY_WEIGHT * overlap
    return penalty

def calculate_objective(circles):
    """Calculate the sum of radii"""
    return sum(circle[2] for circle in circles)

def calculate_fitness(circles):
    """Calculate fitness with penalties"""
    objective = calculate_objective(circles)
    
    # Calculate penalties
    containment_penalty = check_containment_with_penalty(circles)
    overlap_penalty = check_overlap_with_penalty(circles)
    
    # Return fitness (negative because DEAP minimizes by default)
    # We want to maximize objective minus penalties
    total_penalty = containment_penalty + overlap_penalty
    return -(objective - total_penalty)

def generate_voronoi_candidates(n, num_candidates=500):
    """Generate candidate positions based on Voronoi-like distribution"""
    # Start with corners and edges
    candidates = [
        (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
    ]
    
    # Add random points
    for _ in range(num_candidates - len(candidates)):
        candidates.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        
    # Generate Voronoi vertices by sampling points and finding nearest neighbors
    sampled_points = []
    for _ in range(200):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        sampled_points.append((x, y))
    
    # Weighted sampling towards corners and edges
    weighted_candidates = []
    for i, (x, y) in enumerate(candidates):
        # Higher weight for corners and edges
        weight = 2 if (abs(x-0.5) < 0.4 and abs(y-0.5) < 0.4) else 4
        for _ in range(weight):
            weighted_candidates.append((x, y))
    
    # Add more points in corners and edges
    for _ in range(100):
        # Bias towards corners and edges
        if random.random() < 0.6:
            x = random.choice([0.05, 0.95]) if random.random() < 0.5 else random.uniform(0.05, 0.95)
            y = random.choice([0.05, 0.95]) if random.random() < 0.5 else random.uniform(0.05, 0.95)
        else:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
        weighted_candidates.append((x, y))
        
    return weighted_candidates[:num_candidates]

def place_largest_circles_first(candidates, n):
    """Place largest possible circles greedily in given candidates"""
    circles = []
    
    # Sort candidates by priority (edges/corners first, then randomness)
    def candidate_priority(candidate):
        x, y = candidate
        distance_to_edge = min(x, 1-x, y, 1-y)
        # Favor candidates closer to edges/corners
        return -distance_to_edge
    
    sorted_candidates = sorted(candidates, key=candidate_priority)
    
    # Try to place circles starting with the largest ones
    for i, (x, y) in enumerate(sorted_candidates):
        if len(circles) >= n:
            break
            
        # Calculate maximum possible radius at this position
        max_radius = min(x, 1-x, y, 1-y)
        if max_radius <= 0:
            continue
            
        # Check if we can place a circle here with a reasonable size
        test_circle = [x, y, max_radius]
        temp_circles = circles + [test_circle]
        
        # Check constraints
        if check_containment_with_penalty(temp_circles) == 0 and check_overlap_with_penalty(temp_circles) == 0:
            circles.append([x, y, max_radius])
        elif len(circles) < 8:  # Allow small circles in tight spaces initially
            circles.append([x, y, min(max_radius, 0.05)])
    
    # Fill remaining slots with small circles
    while len(circles) < n:
        found = False
        for _ in range(100):  # Try 100 times
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            max_radius = min(x, 1-x, y, 1-y)
            
            if max_radius <= 0:
                continue
                
            test_circle = [x, y, max_radius]
            temp_circles = circles + [test_circle]
            
            if check_containment_with_penalty(temp_circles) == 0 and check_overlap_with_penalty(temp_circles) == 0:
                circles.append([x, y, max_radius])
                found = True
                break
                
        if not found:
            # Last resort: place a tiny circle
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles.append([x, y, 0.01])
    
    return circles[:n]

def create_initial_placement(n):
    """Create initial placement using Voronoi-inspired approach"""
    # Generate Voronoi-like candidates
    candidates = generate_voronoi_candidates(n)
    
    # Place circles greedily
    circles = place_largest_circles_first(candidates, n)
    
    # Ensure all circles fit properly
    for i in range(len(circles)):
        x, y, r = circles[i]
        circles[i] = [x, y, min(r, MAX_RADIUS)]
        
    return circles

def evaluate_individual(individual):
    """Evaluate fitness of an individual solution"""
    # Convert flat array to circles
    circles = np.array(individual).reshape(-1, 3)

    # Calculate fitness with penalties
    fitness = calculate_fitness(circles)
    
    # Return as tuple (DEAP requires tuple return)
    return (fitness,)

def mutate_individual(individual, mutation_rate):
    """Mutate an individual solution with adaptive rate"""
    # Randomly change some circles' positions and/or radii
    for i in range(0, len(individual), 3):
        if random.random() < mutation_rate:
            # Mutate position or radius
            if random.random() < 0.7:  # 70% chance to mutate position
                # Mutate x coordinate
                individual[i] += random.gauss(0, 0.015)
                individual[i] = max(0.01, min(0.99, individual[i]))

                # Mutate y coordinate
                individual[i+1] += random.gauss(0, 0.015)
                individual[i+1] = max(0.01, min(0.99, individual[i+1]))
            else:  # 30% chance to mutate radius
                individual[i+2] += random.gauss(0, 0.008)
                individual[i+2] = max(0.001, min(MAX_RADIUS, individual[i+2]))

    return individual,

def crossover_individuals(ind1, ind2):
    """Crossover two individuals"""
    # Simple uniform crossover
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2

def calculate_diversity(population):
    """Calculate population diversity based on Euclidean distance"""
    if len(population) < 2:
        return 0.0
        
    total_distance = 0
    count = 0
    
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            dist = np.linalg.norm(np.array(population[i]) - np.array(population[j]))
            total_distance += dist
            count += 1
            
    return total_distance / count if count > 0 else 0.0

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create initial population
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize fitness
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize the toolbox
    def init_individual():
        circles = create_initial_placement(32)
        # Flatten the circles into a single list
        flat = []
        for x, y, r in circles:
            flat.extend([x, y, r])
        return creator.Individual(flat)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation and operators
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    # Create initial population
    population = toolbox.population(n=POPULATION_SIZE)

    # Run evolution with adaptive mutation rate
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Track best individual
    best_fitness = float('inf')
    best_individual = None
    
    # Run the evolutionary algorithm with adaptive mutation
    current_mutation_rate = INITIAL_MUTATION_RATE
    
    for gen in range(GENERATIONS):
        # Evolve one generation
        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=CROSSOVER_RATE,
            mutpb=current_mutation_rate,
            ngen=1,
            stats=stats,
            halloffame=hof,
            verbose=False
        )
        
        # Update best individual
        if len(hof) > 0 and hof[0].fitness.values[0] < best_fitness:
            best_fitness = hof[0].fitness.values[0]
            best_individual = hof[0]
            
        # Adaptive mutation rate decay
        current_mutation_rate = max(MIN_MUTATION_RATE, current_mutation_rate * ADAPTIVE_DECAY)
        
        # Print generation info for debugging
        if gen % 10 == 0:
            print(f"Generation {gen}: Best fitness = {-best_fitness}")

    # Get the best individual
    if best_individual is not None:
        circles = np.array(best_individual).reshape(-1, 3)
    else:
        # Fallback to creating a new initial placement
        circles = np.array(create_initial_placement(32))
        
    # Final constraint check and refinement
    if check_containment_with_penalty(circles) != 0 or check_overlap_with_penalty(circles) != 0:
        # Try to refine the solution
        refined_circles = create_initial_placement(32)
        circles = np.array(refined_circles)
        
    return circles

# EVOLVE-BLOCK-END