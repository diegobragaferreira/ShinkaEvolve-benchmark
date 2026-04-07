# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from deap import base, creator, tools, algorithms
import random
import time

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n_circles = 26
    
    # Define the fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define bounds for x, y, r
    # x, y in [0.01, 0.99] to ensure radius constraint
    # r in [0.001, 0.49] to ensure containment and reasonable values
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.001, 0.49)]
    
    def create_individual():
        """Create a random individual with valid initial configuration"""
        # Create a structured initial placement to avoid collisions
        individual = []
        # Grid-based initialization with some randomness
        rows = cols = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x = (col + 1) * spacing_x + np.random.uniform(-spacing_x/4, spacing_x/4)
            y = (row + 1) * spacing_y + np.random.uniform(-spacing_y/4, spacing_y/4)
            
            # Ensure valid coordinates
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Initial radius - small but valid
            r = 0.02 + np.random.uniform(0, 0.03)
            r = max(0.001, min(0.49, r))
            
            individual.extend([x, y, r])
        
        return creator.Individual(individual)
    
    def check_constraints(individual):
        """Check if individual satisfies all constraints"""
        # Convert to circles array for easier access
        circles = np.array(individual).reshape(-1, 3)
        
        # Check containment constraints: r <= x <= 1-r and r <= y <= 1-r
        for x, y, r in circles:
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return False
                
        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = cKDTree(points)
        
        # For each circle, find neighbors within 2*r distance (potential overlaps)
        for i, (x, y, r) in enumerate(circles):
            # Find nearby circles - only check those within 2*r distance
            nearby_indices = tree.query_ball_point([x, y], 2*r)
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance_squared = (x - x2)**2 + (y - y2)**2
                    # If circles are too close, they overlap
                    if distance_squared < (r + r2)**2:
                        return False
                        
        return True
    
    def evaluate(individual):
        """Evaluate fitness of individual - sum of radii"""
        if not check_constraints(individual):
            return (0.0,)  # Invalid individuals get zero fitness
            
        # Calculate total radius (sum of all radii)
        total_radius = sum(individual[2::3])  # Every third element starting from index 2
        
        return (total_radius,)
    
    def mutate(individual):
        """Mutation operator"""
        # Adaptive mutation rate that decreases over generations
        mutation_rate = max(0.01, 0.1 * (1 - generation_count/200.0))
        
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                if i % 3 == 0:  # x coordinate
                    individual[i] += np.random.normal(0, 0.02)
                    individual[i] = max(0.01, min(0.99, individual[i]))
                elif i % 3 == 1:  # y coordinate
                    individual[i] += np.random.normal(0, 0.02)
                    individual[i] = max(0.01, min(0.99, individual[i]))
                else:  # radius
                    individual[i] += np.random.normal(0, 0.02)
                    individual[i] = max(0.001, min(0.49, individual[i]))
        return individual,
    
    def crossover(individual1, individual2):
        """Crossover operator with repair for constraints"""
        # Uniform crossover
        for i in range(len(individual1)):
            if random.random() < 0.5:
                individual1[i], individual2[i] = individual2[i], individual1[i]
        
        # Repair individuals to ensure constraints are met
        for ind in [individual1, individual2]:
            # Adjust to ensure containment and valid radius bounds
            for i in range(0, len(ind), 3):
                x, y, r = ind[i], ind[i+1], ind[i+2]
                
                # Fix bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                r = max(0.001, min(0.49, r))
                
                # Ensure sufficient clearance from boundaries based on radius
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                
                ind[i], ind[i+1], ind[i+2] = x, y, r
                
        return individual1, individual2
    
    # Register functions in toolbox
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    
    # Select with dynamic tournament size
    def select_with_adaptation(population, k):
        # Calculate population diversity
        if len(population) < 5:
            tournament_size = 2
        else:
            fitness_values = [ind.fitness.values[0] for ind in population]
            diversity = np.std(fitness_values)
            # If diversity is high, use smaller tournament for exploration
            # If diversity is low, use larger tournament for exploitation
            tournament_size = max(2, min(5, int(3 + diversity * 10)))
        
        return tools.selTournament(population, k, tournament_size=tournament_size)
    
    toolbox.register("select", select_with_adaptation)
    
    # Algorithm parameters
    population_size = 50
    generations = 200
    generation_count = 0
    
    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Hall of fame to keep track of best individual
    hof = tools.HallOfFame(1)
    
    # Run the evolutionary algorithm
    try:
        # Run evolution for specified number of generations
        for gen in range(generations):
            generation_count = gen
            
            # Select the next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:  # Crossover probability
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            for mutant in offspring:
                if random.random() < 0.2:  # Mutation probability
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace the old population with the new one
            population[:] = offspring
            
            # Update hall of fame
            hof.update(population)
            
            # Print progress every 20 generations
            if gen % 20 == 0:
                best_fitness = max([ind.fitness.values[0] for ind in population])
                print(f"Generation {gen}: Best fitness = {best_fitness:.6f}")
                
    except Exception as e:
        print(f"Error during evolution: {e}")
    
    # Return the best individual found
    best_individual = hof[0]
    
    # Convert to desired format
    circles = np.array(best_individual).reshape(-1, 3)
    
    return circles

# EVOLVE-BLOCK-END
