# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import functools
from typing import Tuple, List

# Set fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def create_circle_packing_problem():
    """Creates the circle packing optimization problem setup"""
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Using a 2:1 ratio (width=1.33, height=0.67) for better packing efficiency
    rect_width = 1.3333333333333333
    rect_height = 0.6666666666666667
    
    # Number of circles
    n_circles = 21
    
    # Bounds for x, y, r
    x_bounds = (0, rect_width)
    y_bounds = (0, rect_height)
    r_bounds = (0, min(rect_width, rect_height) / 2)
    
    return {
        'rect_width': rect_width,
        'rect_height': rect_height,
        'n_circles': n_circles,
        'x_bounds': x_bounds,
        'y_bounds': y_bounds,
        'r_bounds': r_bounds
    }

def initialize_population(pop_size: int, params: dict) -> list:
    """Initialize population with diverse strategies"""
    population = []
    
    # Grid-based initialization for first 30% of population
    for _ in range(pop_size // 3):
        circles = []
        # Create grid layout
        rows = int(np.ceil(np.sqrt(params['n_circles'])))
        cols = rows
        spacing_x = params['rect_width'] / (cols + 1)
        spacing_y = params['rect_height'] / (rows + 1)
        
        for i in range(params['n_circles']):
            row = i // cols
            col = i % cols
            x = spacing_x * (col + 1)
            y = spacing_y * (row + 1)
            # Add small random perturbation
            x += np.random.uniform(-0.05, 0.05)
            y += np.random.uniform(-0.05, 0.05)
            # Radius based on proximity to edges
            min_dist_to_edge = min(x, params['rect_width'] - x, y, params['rect_height'] - y)
            r = min(min_dist_to_edge * 0.8, 0.2)
            circles.append([x, y, r])
        population.append(circles)
    
    # Random initialization for remaining population
    for _ in range(pop_size - len(population)):
        circles = []
        for _ in range(params['n_circles']):
            x = np.random.uniform(params['x_bounds'][0], params['x_bounds'][1])
            y = np.random.uniform(params['y_bounds'][0], params['y_bounds'][1])
            r = np.random.uniform(params['r_bounds'][0], params['r_bounds'][1])
            circles.append([x, y, r])
        population.append(circles)
    
    return population

def check_constraints(circles: np.ndarray, params: dict) -> bool:
    """Check if all circles are within bounds and don't overlap"""
    # Check boundary constraints
    for circle in circles:
        x, y, r = circle
        if (x - r < params['x_bounds'][0] or 
            x + r > params['x_bounds'][1] or 
            y - r < params['y_bounds'][0] or 
            y + r > params['y_bounds'][1]):
            return False
    
    # Check overlap constraints using distance matrix
    positions = circles[:, :2]
    distances = cdist(positions, positions)
    np.fill_diagonal(distances, np.inf)
    
    # For each pair of circles, check if they overlap
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            distance = distances[i, j]
            if distance < (circles[i, 2] + circles[j, 2]):  # Overlap detected
                return False
                
    return True

def evaluate_fitness(circles: np.ndarray, params: dict) -> float:
    """Evaluate fitness of a circle packing configuration"""
    # If constraints violated, return very low fitness
    if not check_constraints(circles, params):
        return -1000.0
    
    # Return sum of radii as fitness
    return float(np.sum(circles[:, 2]))

@functools.lru_cache(maxsize=1000)
def cached_evaluate_fitness_cached(circles_tuple: tuple, params: dict) -> float:
    """Cached version of fitness evaluation for optimization"""
    circles = np.array(circles_tuple).reshape(-1, 3)
    return evaluate_fitness(circles, params)

def evaluate_individual(individual: list, params: dict) -> float:
    """Wrapper function for DEAP compatibility"""
    individual_array = np.array(individual)
    # Convert to tuple for caching
    individual_tuple = tuple(tuple(row) for row in individual_array)
    return cached_evaluate_fitness_cached(individual_tuple, params)

def crossover_and_mutate(individual1: list, individual2: list, 
                        params: dict, mut_rate: float = 0.1) -> Tuple[list, list]:
    """Custom crossover and mutation operators for circle packing"""
    # Simple crossover: blend two individuals
    child1, child2 = [], []
    for i in range(len(individual1)):
        p1, p2 = individual1[i], individual2[i]
        # Blend positions
        x1, y1, r1 = p1
        x2, y2, r2 = p2
        
        # Blend x coordinate
        x_blend = np.random.uniform(min(x1, x2), max(x1, x2))
        # Blend y coordinate  
        y_blend = np.random.uniform(min(y1, y2), max(y1, y2))
        # Blend radius
        r_blend = np.random.uniform(min(r1, r2), max(r1, r2))
        
        child1.append([x_blend, y_blend, r_blend])
        
        # For second child swap the blending
        x_blend2 = np.random.uniform(min(x1, x2), max(x1, x2))
        y_blend2 = np.random.uniform(min(y1, y2), max(y1, y2))
        r_blend2 = np.random.uniform(min(r1, r2), max(r1, r2))
        
        child2.append([x_blend2, y_blend2, r_blend2])
    
    # Apply mutation to children
    mutated_child1 = mutate_individual(child1, params, mut_rate)
    mutated_child2 = mutate_individual(child2, params, mut_rate)
    
    return mutated_child1, mutated_child2

def mutate_individual(individual: list, params: dict, mut_rate: float = 0.1) -> list:
    """Mutate an individual"""
    mutated = []
    for circle in individual:
        x, y, r = circle[:]
        # Apply mutation with given probability
        if np.random.random() < mut_rate:
            # Mutate x coordinate
            delta_x = np.random.normal(0, 0.05)
            x = max(params['x_bounds'][0], min(params['x_bounds'][1], x + delta_x))
            
        if np.random.random() < mut_rate:
            # Mutate y coordinate
            delta_y = np.random.normal(0, 0.05)
            y = max(params['y_bounds'][0], min(params['y_bounds'][1], y + delta_y))
            
        if np.random.random() < mut_rate:
            # Mutate radius
            delta_r = np.random.normal(0, 0.03)
            r = max(params['r_bounds'][0], min(params['r_bounds'][1], r + delta_r))
        
        mutated.append([x, y, r])
    
    return mutated

# Global parameters
params = create_circle_packing_problem()

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Initialize population
    pop_size = 50
    population = initialize_population(pop_size, params)
    
    # Evaluate initial population
    fitnesses = Parallel(n_jobs=-1)(delayed(evaluate_individual)(ind, params) for ind in population)
    
    # Set up the evolution
    for i, (ind, fit) in enumerate(zip(population, fitnesses)):
        population[i] = creator.Individual(ind)
        population[i].fitness.values = (fit,)
    
    # Evolution parameters
    n_generations = 100
    cx_prob = 0.7
    mut_prob = 0.2
    
    # Main evolutionary loop
    for generation in range(n_generations):
        # Select parents
        offspring = tools.selTournament(population, len(population), k=len(population), tournsize=3)
        
        # Apply crossover and mutation
        for i in range(0, len(offspring)-1, 2):
            if np.random.random() < cx_prob:
                offspring[i], offspring[i+1] = crossover_and_mutate(
                    list(offspring[i]), list(offspring[i+1]), params)
        
        # Mutate
        for ind in offspring:
            if np.random.random() < mut_prob:
                ind[:] = mutate_individual(list(ind), params)
        
        # Evaluate fitness of new population
        fitnesses = Parallel(n_jobs=-1)(delayed(evaluate_individual)(ind, params) for ind in offspring)
        
        # Update population with evaluated individuals
        for i, (ind, fit) in enumerate(zip(offspring, fitnesses)):
            offspring[i].fitness.values = (fit,)
        
        # Replace old population with new one
        population[:] = offspring
    
    # Find best individual
    best_ind = tools.selBest(population, 1)[0]
    
    # Convert back to numpy array
    result = np.array(best_ind)
    
    # Ensure proper shape
    if result.shape[0] != params['n_circles']:
        # Re-initialize with better random values if needed
        result = np.array(initialize_population(1, params)[0])
    
    return result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
