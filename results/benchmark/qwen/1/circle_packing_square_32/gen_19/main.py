# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools, algorithms
import math
from typing import Tuple, List
import time

# Set seeds for deterministic behavior
random.seed(42)
np.random.seed(42)

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with diverse circle configurations."""
    population = []
    
    # Create initial configurations using different strategies
    for _ in range(pop_size):
        # Strategy 1: Greedy placement with random perturbation
        circles = np.zeros((n_circles, 3))
        
        # Place circles greedily in corners and centers
        positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        center_positions = [(0.5, 0.5)]
        
        placed_count = 0
        
        # Place some circles in corners
        for x, y in positions[:min(4, n_circles)]:
            circles[placed_count] = [x, y, min(x, 1-x, y, 1-y) * 0.3]
            placed_count += 1
        
        # Place remaining circles in center
        for _ in range(n_circles - placed_count):
            if placed_count < n_circles:
                # Add small random perturbations to center
                x = 0.5 + np.random.uniform(-0.1, 0.1)
                y = 0.5 + np.random.uniform(-0.1, 0.1)
                max_radius = min(x, 1-x, y, 1-y)
                radius = max_radius * np.random.uniform(0.1, 0.4) if max_radius > 0 else 0.01
                circles[placed_count] = [x, y, radius]
                placed_count += 1
                
        # Randomize some positions slightly
        for i in range(n_circles):
            if np.random.random() < 0.3:  # 30% chance to perturb
                circles[i, 0] += np.random.normal(0, 0.02)
                circles[i, 1] += np.random.normal(0, 0.02)
                circles[i, 0] = np.clip(circles[i, 0], 0.01, 0.99)
                circles[i, 1] = np.clip(circles[i, 1], 0.01, 0.99)
        
        population.append(circles)
    
    return population

def calculate_fitness(circles: np.ndarray) -> Tuple[float, float]:
    """
    Calculate fitness for circle packing configuration.
    Returns (sum_of_radii, penalty_score)
    """
    n = len(circles)
    
    # Extract positions and radii
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Calculate sum of radii (primary objective)
    sum_radii = np.sum(radii)
    
    # Penalty for constraint violations
    penalty = 0.0
    
    # Boundary constraint penalty (negative if outside bounds)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 1000  # Large penalty for boundary violations
    
    # Overlap penalty (positive if overlapping)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:  # Circles overlap
                overlap = (r1 + r2 - distance) * 100
                penalty += overlap
    
    # Return sum of radii with penalty (lower penalty = higher fitness)
    return sum_radii, penalty

def evaluate_individual(individual: np.ndarray) -> Tuple[float,]:
    """Evaluate individual and return fitness tuple."""
    sum_radii, penalty = calculate_fitness(individual)
    # Fitness is negative sum of radii (since we maximize) plus penalty
    # This is a maximization problem, so we want to maximize sum_radii
    # We penalize invalid configurations heavily
    fitness = sum_radii - penalty
    return (fitness,)

def create_toolbox(n_circles: int) -> base.Toolbox:
    """Create DEAP toolbox for evolution."""
    # Define fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Create an individual (32 circles, each with 3 values)
    def create_individual():
        # Each circle is represented as [x, y, r]
        individual = np.zeros((n_circles, 3))
        # Initialize with random valid positions and radii
        for i in range(n_circles):
            # Random position with valid radius
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            max_radius = min(x, 1-x, y, 1-y)
            r = np.random.uniform(0.01, max_radius * 0.3) if max_radius > 0.01 else 0.01
            individual[i] = [x, y, r]
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_individual)
    
    # Register genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def mutate_individual(individual: np.ndarray, indpb: float = 0.1) -> Tuple[np.ndarray,]:
    """Mutate an individual by modifying positions and radii."""
    mutated_individual = individual.copy()
    
    for i in range(len(mutated_individual)):
        if np.random.random() < indpb:
            # Mutate position
            mutated_individual[i, 0] += np.random.normal(0, 0.01)
            mutated_individual[i, 1] += np.random.normal(0, 0.01)
            # Clamp to valid range
            mutated_individual[i, 0] = np.clip(mutated_individual[i, 0], 0.01, 0.99)
            mutated_individual[i, 1] = np.clip(mutated_individual[i, 1], 0.01, 0.99)
            
        if np.random.random() < indpb:
            # Mutate radius
            mutated_individual[i, 2] += np.random.normal(0, 0.005)
            # Ensure positive radius
            mutated_individual[i, 2] = max(0.005, mutated_individual[i, 2])
            
            # Adjust position if needed due to radius change
            x, y, r = mutated_individual[i]
            max_radius = min(x, 1-x, y, 1-y)
            if r > max_radius:
                mutated_individual[i, 2] = max_radius * 0.9  # Scale down radius
    
    return (mutated_individual,)

def optimize_circles(n_circles: int = 32, pop_size: int = 50, 
                     generations: int = 100, 
                     crossover_prob: float = 0.8,
                     mutation_prob: float = 0.2) -> np.ndarray:
    """Main optimization loop using evolutionary algorithm."""
    # Create toolbox
    toolbox = create_toolbox(n_circles)
    
    # Create initial population
    population = toolbox.population(n=pop_size)
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evolution
    population, logbook = algorithms.eaSimple(
        population, toolbox, 
        cxpb=crossover_prob, 
        mutpb=mutation_prob, 
        ngen=generations,
        stats=stats,
        verbose=False
    )
    
    # Get best individual
    best_individual = tools.selBest(population, 1)[0]
    return best_individual

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        # Run optimization
        start_time = time.time()
        circles = optimize_circles(n_circles=32, pop_size=50, generations=100)
        end_time = time.time()
        
        # Validate final result
        sum_radii, penalty = calculate_fitness(circles)
        if penalty > 100:  # High penalty indicates serious constraint violations
            # Fall back to simple initialization if optimization failed
            circles = np.zeros((32, 3))
            print("Optimization failed, using fallback initialization")
        
        # Ensure all circles are within bounds
        for i in range(32):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                # Fix boundary issues
                r = min(x, 1-x, y, 1-y)
                if r <= 0:
                    r = 0.01
                circles[i] = [x, y, r]
                
        return circles
    
    except Exception as e:
        # Fallback in case of any error
        print(f"Error in optimization: {e}")
        circles = np.zeros((32, 3))
        return circles

# EVOLVE-BLOCK-END
