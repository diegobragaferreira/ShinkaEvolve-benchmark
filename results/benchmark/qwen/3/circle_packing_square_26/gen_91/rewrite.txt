# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
from collections import defaultdict

# Global constants for optimization
POP_SIZE = 50
GENERATIONS = 300
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3
BOUNDARY_PENALTY = 1000.0
OVERLAP_PENALTY = 10000.0

def generate_voronoi_initialization(n_circles: int) -> np.ndarray:
    """Generate initial configuration using Voronoi tessellation"""
    # Create initial points using a structured approach
    # Start with a grid of points and add some randomness
    grid_size = max(5, int(np.ceil(np.sqrt(n_circles * 1.5))))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)
    
    # Generate points
    points = []
    for x in x_coords:
        for y in y_coords:
            if len(points) < n_circles * 2:  # Generate extra points for Voronoi
                points.append([x, y])
    
    # Add random points for variety
    while len(points) < n_circles * 2:
        points.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])
    
    # Generate Voronoi diagram
    vor = Voronoi(points)
    
    # Select Voronoi cell centroids as circle centers
    centroids = []
    for i in range(len(vor.points)):
        if i < n_circles:
            # Use Voronoi cell centroid, but make sure it's within bounds
            centroid = vor.points[i]
            # Ensure centroid is within unit square
            x = np.clip(centroid[0], 0.05, 0.95)
            y = np.clip(centroid[1], 0.05, 0.95)
            centroids.append([x, y])
    
    # If we don't have enough centroids, add some random points
    while len(centroids) < n_circles:
        centroids.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])
    
    # Take first n_circles points
    centroids = centroids[:n_circles]
    
    # Create circles array with initial radii
    circles = np.zeros((n_circles, 3))
    for i, (x, y) in enumerate(centroids):
        circles[i, 0] = x
        circles[i, 1] = y
        # Initial radius based on proximity to neighbors
        min_dist = float('inf')
        for j, (ox, oy) in enumerate(centroids):
            if i != j:
                dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                min_dist = min(min_dist, dist)
        
        # Set initial radius to half minimum neighbor distance, capped at 0.15
        base_radius = min(0.15, min_dist / 2.0 if min_dist < float('inf') else 0.1)
        circles[i, 2] = max(0.01, base_radius)
    
    return circles

def calculate_voronoi_constraints(circles: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """Calculate Voronoi-based constraints and overlap information"""
    n = len(circles)
    penalty = 0.0
    overlaps = []
    
    # Check containment penalties
    for i, (x, y, r) in enumerate(circles):
        # Calculate how much it violates boundaries
        left_violation = max(0, r - x)
        right_violation = max(0, r - (1 - x))
        bottom_violation = max(0, r - y)
        top_violation = max(0, r - (1 - y))
        
        penalty += BOUNDARY_PENALTY * (left_violation + right_violation + 
                                      bottom_violation + top_violation)
    
    # Check overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            min_dist = r1 + r2
            
            if distance < min_dist:
                overlap = min_dist - distance
                penalty += OVERLAP_PENALTY * overlap
                overlaps.append((i, j))
    
    return penalty, overlaps

def optimize_voronoi_structure(circles: np.ndarray) -> np.ndarray:
    """Optimize the Voronoi-based structure with geometric constraints"""
    optimized = circles.copy()
    n = len(optimized)
    
    # Apply geometric constraints for better distribution
    for i in range(n):
        x, y, r = optimized[i]
        
        # Constrain to unit square with margin
        r = min(r, x, y, 1-x, 1-y)
        r = max(r, 0.001)
        optimized[i, 2] = r
        
        # Keep position within bounds
        optimized[i, 0] = np.clip(x, r, 1-r)
        optimized[i, 1] = np.clip(y, r, 1-r)
    
    return optimized

def voronoi_mutation(individual: np.ndarray, mutation_rate: float = MUTATION_RATE) -> np.ndarray:
    """Specialized mutation that preserves Voronoi structure"""
    mutated = individual.copy()
    n = len(mutated)
    
    # Apply mutation to each circle
    for i in range(n):
        if random.random() < mutation_rate:
            # Choose mutation type
            mutation_type = random.choice(['position', 'radius', 'voronoi'])
            
            if mutation_type == 'position':
                # Mutate position with focus on Voronoi-like behavior
                # Make small changes that respect Voronoi constraints
                mutated[i, 0] += random.gauss(0, 0.01)
                mutated[i, 1] += random.gauss(0, 0.01)
                # Keep within bounds
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
                
            elif mutation_type == 'radius':
                # Mutate radius
                mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, 0.008), 0.001, 0.2)
                
            elif mutation_type == 'voronoi':
                # Voronoi-based mutation: adjust position to maintain Voronoi-like spacing
                # First find the nearest points to influence the mutation
                distances = []
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((mutated[i, 0] - mutated[j, 0])**2 + 
                                     (mutated[i, 1] - mutated[j, 1])**2)
                        distances.append((dist, j))
                
                # Sort by distance
                distances.sort()
                
                # Apply small perturbation based on neighbors
                if distances:
                    nearest_idx = distances[0][1]
                    dx = mutated[i, 0] - mutated[nearest_idx, 0]
                    dy = mutated[i, 1] - mutated[nearest_idx, 1]
                    # Move in opposite direction slightly to avoid overlap
                    mutated[i, 0] -= dx * 0.005
                    mutated[i, 1] -= dy * 0.005
                    
                    # Keep within bounds
                    mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                    mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
    
    # Apply geometric constraint optimization
    mutated = optimize_voronoi_structure(mutated)
    return mutated

def voronoi_crossover(parent1: np.ndarray, parent2: np.ndarray, 
                     crossover_rate: float = CROSSOVER_RATE) -> np.ndarray:
    """Cross-over that respects Voronoi structure"""
    if random.random() > crossover_rate:
        return parent1.copy()
    
    n = len(parent1)
    child = np.zeros_like(parent1)
    
    # Use a mixed approach: Voronoi-aware crossover
    # Take position from parent1, radius from parent2 with special handling
    for i in range(n):
        # Cross-over position (x,y) 
        if random.random() < 0.5:
            child[i, :2] = parent1[i, :2]
        else:
            child[i, :2] = parent2[i, :2]
        
        # Cross-over radii with additional constraint checking
        if random.random() < 0.5:
            child[i, 2] = parent1[i, 2]
        else:
            child[i, 2] = parent2[i, 2]
        
        # Ensure child respects constraints
        x, y, r = child[i]
        r = min(r, x, y, 1-x, 1-y)
        r = max(r, 0.001)
        child[i, 2] = r
        
        # Clip positions
        child[i, 0] = np.clip(x, r, 1-r)
        child[i, 1] = np.clip(y, r, 1-r)
    
    # Final optimization
    child = optimize_voronoi_structure(child)
    return child

def evaluate_fitness(circles: np.ndarray) -> Tuple[float, float]:
    """Evaluate the fitness of a solution using Voronoi-aware approach"""
    # Sum of radii (primary objective)
    total_radius = np.sum(circles[:, 2])
    
    # Penalty for constraint violations
    penalty, _ = calculate_voronoi_constraints(circles)
    
    # Fitness is total radius minus penalty
    fitness = total_radius - penalty
    
    return fitness, total_radius

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], 
                        tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Select individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = [fitness_scores[i] for i in selected_indices]
    
    winner_idx = selected_indices[np.argmax(selected_fitness)]
    return population[winner_idx].copy()

def evolve_population(population: List[np.ndarray]) -> Tuple[List[np.ndarray], float, float]:
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
    
    # Generate rest of population with Voronoi-aware operators
    while len(new_population) < len(population):
        # Selection
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        
        # Crossover
        child = voronoi_crossover(parent1, parent2)
        
        # Mutation
        child = voronoi_mutation(child)
        
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
    population = []
    
    # Initialize with Voronoi-based approach
    for _ in range(POP_SIZE):
        individual = generate_voronoi_initialization(n)
        population.append(individual)
    
    best_total_radius = 0.0
    best_individual = None
    
    # Evolution loop with multi-stage approach
    for generation in range(GENERATIONS):
        population, gen_fitness, gen_radius = evolve_population(population)
        
        if gen_radius > best_total_radius:
            best_total_radius = gen_radius
            best_individual = population[0]  # Keep track of best individual
            
        # Progress reporting
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