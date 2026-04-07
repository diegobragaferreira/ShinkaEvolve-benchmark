# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_constraints(circles):
    """Check if all circles satisfy containment and non-overlap constraints using vectorized operations."""
    n = len(circles)
    
    # Check containment constraints
    if np.any(circles[:, 0] - circles[:, 2] < 0) or np.any(circles[:, 0] + circles[:, 2] > 1) or \
       np.any(circles[:, 1] - circles[:, 2] < 0) or np.any(circles[:, 1] + circles[:, 2] > 1):
        return False

    # Check non-overlap constraints using vectorized operations
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Calculate pairwise distances between circle centers
    distances = cdist(positions, positions)
    
    # Check if any circles overlap
    # Create upper triangular matrix to avoid double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    distances_masked = distances[mask]
    radii_sum = (radii[:, None] + radii[None, :])[mask]
    
    if np.any(distances_masked < radii_sum):
        return False
    
    return True

def evaluate_fitness(circles):
    """Evaluate fitness as the sum of all radii."""
    return np.sum(circles[:, 2])

def create_initial_individual():
    """Create a single individual using a more intelligent initialization approach."""
    circles = np.zeros((26, 3))
    
    # Start with a grid-like pattern and randomize slightly
    grid_size = int(np.ceil(np.sqrt(26)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    # Use a larger initial radius to allow for better packing
    base_radius = 0.08
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= 26:
                break
            # Position with slight randomization to avoid perfect grid
            x = (i + 1) * spacing_x + np.random.normal(0, spacing_x * 0.05)
            y = (j + 1) * spacing_y + np.random.normal(0, spacing_y * 0.05)
            
            # Clip to valid range
            x = np.clip(x, base_radius, 1 - base_radius)
            y = np.clip(y, base_radius, 1 - base_radius)
            
            # Adjust radius to be smaller for better packing
            r = base_radius * (0.8 + np.random.random() * 0.4)
            r = np.clip(r, 0.005, 0.15)
            
            circles[count] = [x, y, r]
            count += 1
        if count >= 26:
            break
    
    # If we couldn't fill all slots, fill the rest with random valid circles
    for i in range(count, 26):
        attempts = 0
        while attempts < 100:
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            r = np.random.uniform(0.005, 0.1)
            
            # Insert temporarily and test
            circles[i] = [x, y, r]
            if check_constraints(circles):
                break
            attempts += 1
    
    return circles

def create_initial_population(pop_size, n_circles):
    """Create initial population with better starting points."""
    population = []
    
    # Create diverse initial solutions
    for _ in range(pop_size):
        individual = create_initial_individual()
        if check_constraints(individual):
            population.append(individual)
        else:
            # Fallback to a simpler approach
            individual = np.zeros((n_circles, 3))
            spacing = 1.0 / 6.0
            radius = spacing / 3.0
            count = 0
            for i in range(6):
                for j in range(6):
                    if count >= 26:
                        break
                    x = (i + 1) * spacing
                    y = (j + 1) * spacing
                    individual[count] = [x, y, radius]
                    count += 1
                if count >= 26:
                    break
            population.append(individual)
    
    return population

def mutate_individual(individual, mutation_rate=0.1):
    """Apply mutation to an individual with better constraint handling."""
    mutated = individual.copy()
    
    # Apply mutation to each circle
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Decide what to mutate
            if np.random.random() < 0.5:
                # Mutate position
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.01), 0.005, 0.995)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.01), 0.005, 0.995)
            else:
                # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.005), 0.001, 0.2)
    
    # Ensure constraints are satisfied, if not, try to fix them
    if not check_constraints(mutated):
        # Try random restart for invalid individuals
        mutated = create_initial_individual()
    
    return mutated

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    # Use uniform crossover with some bias toward better parents
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Perform crossover for each circle
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()
    
    return child1, child2

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def optimize_circles():
    """Main optimization function using evolutionary algorithm with enhancements."""
    n_circles = 26
    pop_size = 50
    generations = 100
    mutation_rate = 0.1

    # Create initial population
    population = create_initial_population(pop_size, n_circles)

    best_fitness = 0
    best_individual = None

    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()

        # Select parents
        parents = [tournament_selection(population, fitnesses) for _ in range(pop_size)]

        # Create new population through crossover and mutation
        new_population = []

        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)

        # Generate offspring
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = parents[np.random.randint(0, len(parents))]
            parent2 = parents[np.random.randint(0, len(parents))]

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)

            # Ensure children meet constraints
            if check_constraints(child1):
                new_population.append(child1)
            if len(new_population) < pop_size and check_constraints(child2):
                new_population.append(child2)

        # Trim to population size
        population = new_population[:pop_size]

    return best_individual if best_individual is not None else population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = optimize_circles()
        return circles
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to better heuristic
        circles = np.zeros((26, 3))
        
        # Create a more sophisticated grid pattern with better distribution
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x / 3.0

        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                # Add more randomness to positions to avoid regular patterns
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                # Ensure positions stay within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                # Use a somewhat larger radius than average to increase total sum
                r = np.clip(radius * (1.2 + np.random.random() * 0.4), 0.005, 0.15)
                circles[count] = [x, y, r]
                count += 1
            if count >= 26:
                break

        return circles


# EVOLVE-BLOCK-END