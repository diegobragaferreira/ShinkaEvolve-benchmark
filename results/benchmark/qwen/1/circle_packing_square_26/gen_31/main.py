# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time

# Global constants for the optimization
INITIAL_POPULATION_SIZE = 150
MAX_GENERATIONS = 600
TOURNAMENT_SIZE = 5
INITIAL_MUTATION_RATE = 0.15
ELITISM_COUNT = 8
MIN_MUTATION_RATE = 0.02
BOUNDARY_MARGIN = 0.01

def is_valid_configuration(circles):
    """Check if a configuration of circles is valid (no overlaps, fully contained)."""
    n = len(circles)

    # Check containment constraints (with margin for numerical stability)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r + BOUNDARY_MARGIN or x > 1-r - BOUNDARY_MARGIN or y < r + BOUNDARY_MARGIN or y > 1-r - BOUNDARY_MARGIN:
            return False

    # Check overlap constraints with early termination
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_squared = (x1-x2)**2 + (y1-y2)**2
            min_distance_squared = (r1 + r2)**2
            if distance_squared < min_distance_squared:
                return False

    return True

def evaluate_fitness(circles):
    """Evaluate fitness as the sum of radii."""
    return np.sum(circles[:, 2])

def create_voronoi_initialization(n_circles):
    """Create initial configuration using Voronoi-based spreading."""
    # Generate random points
    points = np.random.rand(n_circles, 2)
    
    # Add boundary points to ensure good coverage
    boundary_points = []
    for _ in range(10):
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            boundary_points.append([np.random.rand(), 1.0])
        elif side == 1:  # Bottom
            boundary_points.append([np.random.rand(), 0.0])
        elif side == 2:  # Left
            boundary_points.append([0.0, np.random.rand()])
        else:  # Right
            boundary_points.append([1.0, np.random.rand()])
    
    points = np.vstack([points, boundary_points])
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)
        # Use Voronoi cell centers as initial circle positions
        centroids = vor.points[vor.point_region[:-1]]  # Exclude infinite region
        
        # Limit to number of circles needed
        selected_centroids = centroids[:n_circles]
        
        # Create circles with initial radii
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x, y = selected_centroids[i]
            # Initial radius based on proximity to neighbors
            distances = np.sqrt(np.sum((selected_centroids - [x, y])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance
            if len(distances) > 0:
                avg_distance = np.min(distances) * 0.4
                radius = min(avg_distance, 0.2)
            else:
                radius = 0.1
                
            # Ensure it's within bounds
            radius = min(radius, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                        y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
            
            circles[i] = [x, y, max(radius, 0.001)]
            
        return circles
    except:
        # Fallback to random initialization if Voronoi fails
        return generate_random_initialization(n_circles)

def generate_random_initialization(n_circles):
    """Generate random initial configuration."""
    circles = np.zeros((n_circles, 3))
    
    # Try multiple attempts to place valid circles
    max_attempts = 2000
    for attempt in range(max_attempts):
        success = True
        circles = np.zeros((n_circles, 3))
        
        for i in range(n_circles):
            # Try to place circle without overlap
            placed = False
            inner_attempts = 0
            max_inner = 100
            
            while not placed and inner_attempts < max_inner:
                x = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                y = np.random.uniform(BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
                
                # Try different radii sizes - biased towards smaller radii to avoid conflicts
                r = np.random.uniform(0.001, 0.15)
                
                # Check if it fits with existing circles
                valid = True
                for j in range(i):
                    prev_x, prev_y, prev_r = circles[j]
                    distance_squared = (x - prev_x)**2 + (y - prev_y)**2
                    min_distance_squared = (r + prev_r)**2
                    
                    if distance_squared < min_distance_squared:
                        valid = False
                        break
                
                # Check containment
                if valid and (r > x or r > 1-x or r > y or r > 1-y):
                    valid = False
                
                if valid:
                    circles[i] = [x, y, r]
                    placed = True
                else:
                    inner_attempts += 1
                    
            if not placed:
                success = False
                break
                
        if success:
            return circles
            
    # If we couldn't place circles, use a simpler method
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / grid_size
    r = spacing * 0.3
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n_circles:
                x = (j + 0.5) * spacing
                y = (i + 0.5) * spacing
                # Adjust for boundary constraints
                x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                circles[count] = [x, y, r]
                count += 1
    
    return circles

def initialize_population(pop_size, n_circles):
    """Create an initial population of valid circle configurations."""
    population = []
    
    # Use Voronoi-based initialization for first few individuals
    for i in range(min(30, pop_size)):
        circles = create_voronoi_initialization(n_circles)
        if is_valid_configuration(circles):
            population.append(circles)
    
    # Fill up with random initializations
    while len(population) < pop_size:
        circles = generate_random_initialization(n_circles)
        if is_valid_configuration(circles):
            population.append(circles)
    
    return population

def tournament_selection(population, fitnesses, tournament_size):
    """Select an individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def crossover(parent1, parent2):
    """Perform crossover between two parent configurations."""
    if np.random.random() > 0.8:  # Lower crossover rate for better preservation
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    crossover_point = np.random.randint(1, n)

    child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

    # Ensure children are valid
    child1 = enforce_boundaries(child1)
    child2 = enforce_boundaries(child2)
    
    return child1, child2

def enforce_boundaries(circles):
    """Ensure circles respect boundary constraints."""
    result = circles.copy()
    for i in range(len(result)):
        x, y, r = result[i]
        # Clip position to stay within boundaries
        x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
        y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
        result[i] = [x, y, r]
    return result

def mutate(circles, mutation_rate=0.1):
    """Mutate a circle configuration with improved strategy."""
    mutated = circles.copy()

    # Mutate each circle with some probability
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Decide what to mutate with bias towards position
            mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])

            if mutation_type == 'position':
                # Slightly perturb position with bounded adjustment
                dx = np.random.normal(0, 0.015)
                dy = np.random.normal(0, 0.015)
                
                mutated[i][0] = np.clip(mutated[i][0] + dx,
                                       mutated[i][2] + BOUNDARY_MARGIN, 1 - mutated[i][2] - BOUNDARY_MARGIN)
                mutated[i][1] = np.clip(mutated[i][1] + dy,
                                       mutated[i][2] + BOUNDARY_MARGIN, 1 - mutated[i][2] - BOUNDARY_MARGIN)
            else:
                # Mutate radius with careful bounds
                dr = np.random.normal(0, 0.01)
                new_radius = mutated[i][2] + dr
                # Ensure radius stays positive and reasonable
                mutated[i][2] = np.clip(new_radius, 0.001, min(0.4, 1 - mutated[i][0], mutated[i][0], 
                                                              1 - mutated[i][1], mutated[i][1]))

    return mutated

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility

    n = 26
    best_solution = None
    best_fitness = -np.inf

    # Initialize population
    population = initialize_population(INITIAL_POPULATION_SIZE, n)

    # Remove invalid solutions
    valid_population = [ind for ind in population if is_valid_configuration(ind)]
    if not valid_population:
        # Fallback to simple initialization
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < n:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    circles[count] = [x, y, r]
                    count += 1
        return circles

    population = valid_population

    start_time = time.time()
    
    for generation in range(MAX_GENERATIONS):
        # Calculate adaptive mutation rate
        # Decrease over time to reduce exploration and increase exploitation
        adaptive_mutation_rate = MAX_GENERATIONS * INITIAL_MUTATION_RATE / (MAX_GENERATIONS + generation * 2)
        adaptive_mutation_rate = max(adaptive_mutation_rate, MIN_MUTATION_RATE)

        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(ind) for ind in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < INITIAL_POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, adaptive_mutation_rate)
            child2 = mutate(child2, adaptive_mutation_rate)

            # Ensure validity of children
            if is_valid_configuration(child1):
                new_population.append(child1)
            if len(new_population) < INITIAL_POPULATION_SIZE and is_valid_configuration(child2):
                new_population.append(child2)

        population = new_population[:INITIAL_POPULATION_SIZE]

        # Early stopping check
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to simple initialization if no good solution was found
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < n:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    circles[count] = [x, y, r]
                    count += 1
        return circles


# EVOLVE-BLOCK-END