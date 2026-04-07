# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
import math

# Global constants for optimization
POP_SIZE = 100
GENERATIONS = 500
INITIAL_MUTATION_RATE = 0.15
FINAL_MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
BOUNDARY_PENALTY = 1000.0
OVERLAP_PENALTY = 10000.0

def sigmoid_decay(current_gen: int, max_generations: int, start_rate: float, end_rate: float) -> float:
    """Apply sigmoidal decay to mutation rate"""
    # Sigmoid function: 1 / (1 + e^(-k(x - x0)))
    k = 6  # Steepness of sigmoid
    x0 = max_generations / 2  # Midpoint
    rate = end_rate + (start_rate - end_rate) / (1 + math.exp(-k * (current_gen - x0)))
    return rate

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with Voronoi-based distribution and better spacing"""
    population = []
    
    # Generate initial points using a grid with some randomness
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)
    
    # Create grid points
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            if len(grid_points) < n_circles:
                grid_points.append([x, y])
    
    # Adjust number of points to match required circles
    if len(grid_points) < n_circles:
        # Add random points to reach target
        for _ in range(n_circles - len(grid_points)):
            grid_points.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])
    
    # Fill population with variations using better distribution
    for _ in range(pop_size):
        # Get a subset of points with slight variation
        individual = np.zeros((n_circles, 3))
        
        # Assign positions with better perturbation
        for i in range(n_circles):
            # Start with base point
            base_x, base_y = grid_points[i % len(grid_points)]
            
            # Perturb slightly with more controlled variation
            perturbation = random.uniform(-0.03, 0.03)
            individual[i, 0] = max(0.05, min(0.95, base_x + perturbation))
            individual[i, 1] = max(0.05, min(0.95, base_y + perturbation))
            
            # Assign random radius with better distribution
            individual[i, 2] = random.uniform(0.01, 0.12)
            
        population.append(individual)
    
    return population

def is_valid_circle(circle: np.ndarray, other_circles: np.ndarray) -> bool:
    """Check if a circle is valid (contained and not overlapping)"""
    x, y, r = circle
    
    # Check containment
    if r > x or r > y or r > (1 - x) or r > (1 - y):
        return False
    
    # Check overlap with existing circles
    for other in other_circles:
        if other[2] > 0:  # Only check non-zero radius circles
            ox, oy, oradius = other
            distance = np.sqrt((x - ox)**2 + (y - oy)**2)
            if distance < (r + oradius):
                return False
    
    return True

def calculate_penalty(circles: np.ndarray) -> float:
    """Calculate penalty based on constraint violations"""
    penalty = 0.0
    n = len(circles)
    
    # Check containment penalties - more precise measurement
    for circle in circles:
        x, y, r = circle
        # Calculate how much it violates boundaries - measure actual excess
        left_violation = max(0, r - x)
        right_violation = max(0, r - (1 - x))
        bottom_violation = max(0, r - y)
        top_violation = max(0, r - (1 - y))
        
        # Apply penalty proportional to violation
        penalty += BOUNDARY_PENALTY * (left_violation + right_violation + 
                                      bottom_violation + top_violation)
    
    # Check overlap penalties using efficient spatial indexing
    valid_circles = [c for c in circles if c[2] > 0]
    if len(valid_circles) > 1:
        # Build spatial index efficiently
        coords = np.array([c[:2] for c in valid_circles])
        tree = cKDTree(coords)
        
        # Use ball query to find neighbors within sum of radii
        # This is more efficient than pairwise checking
        for i, circle in enumerate(valid_circles):
            x, y, r = circle
            # Query neighbors within distance 2*(r_max) to avoid too many queries
            # But we'll do a more direct approach for overlaps
            neighbors = tree.query_ball_point([x, y], 2*r, p=2)
            
            for j in neighbors:
                if i != j:
                    other = valid_circles[j]
                    ox, oy, oradius = other
                    distance = np.sqrt((x - ox)**2 + (y - oy)**2)
                    min_dist = r + oradius
                    
                    if distance < min_dist:
                        # Apply penalty proportional to overlap
                        overlap = min_dist - distance
                        penalty += OVERLAP_PENALTY * overlap * 10  # Scale up overlap penalty
    
    return penalty

def evaluate_fitness(circles: np.ndarray) -> Tuple[float, float]:
    """Evaluate the fitness of a solution"""
    # Sum of radii (primary objective)
    total_radius = np.sum(circles[:, 2])
    
    # Penalty for constraint violations
    penalty = calculate_penalty(circles)
    
    # Fitness is total radius minus penalty
    fitness = total_radius - penalty
    
    return fitness, total_radius

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], 
                        tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Select an individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = [fitness_scores[i] for i in selected_indices]
    
    winner_idx = selected_indices[np.argmax(selected_fitness)]
    return population[winner_idx].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray, 
             crossover_rate: float = CROSSOVER_RATE) -> np.ndarray:
    """Perform crossover between two parents"""
    if random.random() > crossover_rate:
        return parent1.copy()
    
    n = len(parent1)
    child = np.zeros_like(parent1)
    
    # Single-point crossover on radii - better balanced approach
    crossover_point = random.randint(1, n - 1)
    
    # Copy positions from parent1 (for both halves)
    child[:crossover_point, :2] = parent1[:crossover_point, :2]
    child[crossover_point:, :2] = parent2[crossover_point:, :2]
    
    # Copy radii from parent1 (for both halves)
    child[:crossover_point, 2] = parent1[:crossover_point, 2]
    child[crossover_point:, 2] = parent2[crossover_point:, 2]
    
    # Local refinement to fix any constraint violations
    child = refine_solution(child)
    
    return child

def refine_solution(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to fix constraint violations"""
    refined = circles.copy()
    
    # Ensure containment and fix overlaps iteratively
    for i in range(len(refined)):
        x, y, r = refined[i]
        
        # Fix containment with better bounds
        r = min(r, x, y, 1 - x, 1 - y)
        r = max(0.001, r)
        refined[i, 2] = r
    
    # Perform iterative overlap removal with better separation logic
    MAX_ITER = 50
    for iteration in range(MAX_ITER):
        changed = False
        for i in range(len(refined)):
            x, y, r = refined[i]
            
            # Check overlap with all others
            for j in range(len(refined)):
                if i != j:
                    ox, oy, oradius = refined[j]
                    distance = np.sqrt((x - ox)**2 + (y - oy)**2)
                    
                    if distance < (r + oradius):
                        # Move circle away from overlapping one with better separation
                        if distance > 0.001:
                            dx = (x - ox) / distance
                            dy = (y - oy) / distance
                            
                            # Reduce radius to prevent further overlap
                            new_r = max(0.001, (r + oradius) * 0.95 - 0.001)
                            refined[i, 2] = new_r
                            
                            # Adjust position to separate
                            separation_distance = 0.001
                            refined[i, 0] = x + dx * separation_distance
                            refined[i, 1] = y + dy * separation_distance
                            
                            # Ensure containment after adjustment
                            refined[i, 0] = np.clip(refined[i, 0], refined[i, 2], 1 - refined[i, 2])
                            refined[i, 1] = np.clip(refined[i, 1], refined[i, 2], 1 - refined[i, 2])
                        
                        changed = True
                        
        if not changed:
            break
            
    return refined

def mutate(individual: np.ndarray, mutation_rate: float = INITIAL_MUTATION_RATE) -> np.ndarray:
    """Mutate an individual with adaptive rate"""
    mutated = individual.copy()
    n = len(mutated)
    
    for i in range(n):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate
            choice = random.randint(0, 2)
            
            if choice == 0:  # Mutate x position
                mutated[i, 0] = np.clip(mutated[i, 0] + random.gauss(0, 0.015), 0.05, 0.95)
            elif choice == 1:  # Mutate y position
                mutated[i, 1] = np.clip(mutated[i, 1] + random.gauss(0, 0.015), 0.05, 0.95)
            else:  # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, 0.008), 0.001, 0.15)
    
    # Local refinement after mutation
    mutated = refine_solution(mutated)
    return mutated

def evolve_population(population: List[np.ndarray], generation: int) -> Tuple[List[np.ndarray], float, float]:
    """Evolve the population for one generation"""
    # Evaluate fitness
    fitness_scores = []
    total_radii = []
    
    for individual in population:
        fitness, total_radius = evaluate_fitness(individual)
        fitness_scores.append(fitness)
        total_radii.append(total_radius)
    
    # Track best individual
    best_idx = np.argmax(fitness_scores)
    best_fitness = fitness_scores[best_idx]
    best_total_radius = total_radii[best_idx]
    
    # Create new population
    new_population = []
    
    # Elitism: keep the best individual
    new_population.append(population[best_idx].copy())
    
    # Calculate adaptive mutation rate
    current_mutation_rate = sigmoid_decay(generation, GENERATIONS, INITIAL_MUTATION_RATE, FINAL_MUTATION_RATE)
    
    # Generate rest of population
    while len(new_population) < len(population):
        # Selection
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        
        # Crossover
        child = crossover(parent1, parent2)
        
        # Mutation
        child = mutate(child, current_mutation_rate)
        
        new_population.append(child)
    
    return new_population, best_fitness, best_total_radius

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 26
    population = initialize_population(POP_SIZE, n)
    
    best_total_radius = 0.0
    best_individual = None
    
    # Evolution loop
    for generation in range(GENERATIONS):
        population, gen_fitness, gen_radius = evolve_population(population, generation)
        
        if gen_radius > best_total_radius:
            best_total_radius = gen_radius
            best_individual = population[0]  # Keep track of best individual
            
        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best radius sum = {gen_radius:.6f}")
    
    print(f"Final result: Best radius sum = {best_total_radius:.6f}")
    
    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to returning first individual if something went wrong
        return population[0]

# EVOLVE-BLOCK-END
