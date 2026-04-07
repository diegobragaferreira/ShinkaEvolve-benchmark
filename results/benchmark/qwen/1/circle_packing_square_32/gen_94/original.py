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

def check_collision(circle1, circle2):
    """Check if two circles collide using squared distances for efficiency"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance_squared = (x1 - x2)**2 + (y1 - y2)**2
    return distance_squared < (r1 + r2)**2

def is_valid_position(circle, circles):
    """Check if a circle position is valid (within bounds and no collisions)"""
    x, y, r = circle

    # Check boundary constraints
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False

    # Check collision with existing circles
    for existing_circle in circles:
        if check_collision(circle, existing_circle):
            return False

    return True

def place_circle_greedy(circles, max_circles):
    """Place circles greedily with maximum radius"""
    new_circles = circles.copy()

    # Predefined strategic positions for initial placement
    strategic_positions = [
        (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),  # corners
        (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),  # edges
        (0.5, 0.5),  # center
    ]

    # Place initial strategic circles
    placed = 0
    for i, (x, y) in enumerate(strategic_positions[:min(9, max_circles)]):
        if placed >= max_circles:
            break
        # Try to place with maximum possible radius
        max_radius = min(x, 1-x, y, 1-y)
        new_circle = (x, y, max_radius)
        if is_valid_position(new_circle, new_circles):
            new_circles[placed] = new_circle
            placed += 1

    # Fill remaining spots with greedy approach
    while placed < max_circles:
        best_circle = None
        best_radius = 0

        # Try to place circles in multiple candidate positions
        candidates = []
        # Sample random positions in the square
        for _ in range(1000):
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            # Estimate maximum radius for this position
            max_radius = min(x, 1-x, y, 1-y)
            candidates.append((x, y, max_radius))

        # Find the best valid circle among candidates
        for x, y, max_radius in candidates:
            if max_radius <= best_radius:
                continue
            test_circle = (x, y, max_radius)
            if is_valid_position(test_circle, new_circles[:placed]):
                best_circle = test_circle
                best_radius = max_radius

        if best_circle is None:
            # If we can't find a valid circle, use a small radius and place anyway
            # This shouldn't happen often with good initialization
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            test_circle = (x, y, 0.01)
            if is_valid_position(test_circle, new_circles[:placed]):
                new_circles[placed] = test_circle
                placed += 1
            else:
                break  # Can't place more circles
        else:
            new_circles[placed] = best_circle
            placed += 1

    return new_circles

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with diverse circle configurations."""
    population = []
    
    # Create initial configurations using different strategies
    for _ in range(pop_size):
        # Strategy 1: Greedy placement with random perturbation
        circles = np.zeros((n_circles, 3))
        
        # Place circles greedily in corners and centers first
        circles = place_circle_greedy(circles, n_circles)
        
        # Add small random perturbations to improve diversity
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
            
            distance_squared = (x1 - x2)**2 + (y1 - y2)**2
            distance = math.sqrt(distance_squared)
            if distance < r1 + r2:  # Circles overlap
                overlap = (r1 + r2 - distance) * 100
                penalty += overlap
    
    # Return sum of radii with penalty (lower penalty = higher fitness)
    return sum_radii, penalty

def evaluate_individual(individual: np.ndarray) -> Tuple[float,]:
    """Evaluate individual and return fitness tuple."""
    sum_radii, penalty = calculate_fitness(individual)
    # Fitness is sum of radii minus penalty (since we maximize sum_radii)
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
        # Start with greedy initialization
        individual = np.zeros((n_circles, 3))
        individual = place_circle_greedy(individual, n_circles)
        
        # Apply small random perturbations to create initial diversity
        for i in range(n_circles):
            if np.random.random() < 0.3:  # 30% chance to perturb
                individual[i, 0] += np.random.normal(0, 0.01)
                individual[i, 1] += np.random.normal(0, 0.01)
                individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
                individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)
        
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
        # Start with greedy initialization for good baseline
        n = 32
        circles = np.zeros((n, 3))
        circles = place_circle_greedy(circles, n)
        
        # Run optimization to refine the solution
        start_time = time.time()
        optimized_circles = optimize_circles(n_circles=32, pop_size=50, generations=100)
        end_time = time.time()
        
        # Validate final result
        sum_radii, penalty = calculate_fitness(optimized_circles)
        if penalty > 100:  # High penalty indicates serious constraint violations
            # Fall back to greedy solution if optimization failed
            print("Optimization failed, using greedy solution")
            return circles
        
        # Verify that the optimized result actually has better sum of radii
        original_sum = np.sum(circles[:, 2])
        optimized_sum = np.sum(optimized_circles[:, 2])
        if optimized_sum > original_sum:
            return optimized_circles
        else:
            return circles
    
    except Exception as e:
        # Fallback in case of any error
        print(f"Error in optimization: {e}")
        # Return greedy solution as final fallback
        n = 32
        circles = np.zeros((n, 3))
        circles = place_circle_greedy(circles, n)
        return circles

# EVOLVE-BLOCK-END
