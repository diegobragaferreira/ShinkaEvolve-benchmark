# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

def generate_voronoi_seeds(n_points: int, bounds: Tuple[float, float, float, float] = (0, 0, 1, 1)) -> np.ndarray:
    """Generate well-distributed seed points using Voronoi diagrams."""
    x_min, y_min, x_max, y_max = bounds
    
    # Generate random points
    points = np.random.rand(n_points, 2)
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min
    points[:, 1] = points[:, 1] * (y_max - y_min) + y_min
    
    # Add boundary points to ensure good coverage
    boundary_points = []
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            boundary_points.append([x, y])
    
    # Add some edge points
    for i in range(5):
        boundary_points.append([x_min + (x_max - x_min) * random.random(), y_min])
        boundary_points.append([x_min + (x_max - x_min) * random.random(), y_max])
        boundary_points.append([x_min, y_min + (y_max - y_min) * random.random()])
        boundary_points.append([x_max, y_min + (y_max - y_min) * random.random()])
        
    points = np.vstack([points, boundary_points])
    
    # Generate Voronoi diagram and get centroids
    vor = Voronoi(points)
    
    # Filter out points that are too close to boundaries
    valid_centroids = []
    for centroid in vor.points:
        if (centroid[0] > 0.01 and centroid[0] < 0.99 and 
            centroid[1] > 0.01 and centroid[1] < 0.99):
            valid_centroids.append(centroid)
            
    return np.array(valid_centroids[:n_points])

def check_containment(circle: np.ndarray, bounds: Tuple[float, float, float, float] = (0, 0, 1, 1)) -> bool:
    """Check if circle is fully contained within bounds."""
    x, y, r = circle
    x_min, y_min, x_max, y_max = bounds
    return (r <= x <= x_max - r and r <= y <= y_max - r)

def check_overlap(circle1: np.ndarray, circle2: np.ndarray) -> bool:
    """Check if two circles overlap."""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance_squared = (x1 - x2)**2 + (y1 - y2)**2
    return distance_squared < (r1 + r2)**2

def compute_fitness(circles: np.ndarray) -> float:
    """Compute fitness as sum of radii, penalizing violations."""
    total_radius = np.sum(circles[:, 2])
    
    # Check constraints and apply penalties
    penalty = 0.0
    
    # Check containment
    for circle in circles:
        if not check_containment(circle):
            penalty += 1000.0
            
    # Check overlaps
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            if check_overlap(circles[i], circles[j]):
                penalty += 1000.0
                
    return total_radius - penalty

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with Voronoi-based seeds."""
    population = []
    
    for _ in range(pop_size):
        # Generate Voronoi-based seed points
        seeds = generate_voronoi_seeds(n_circles)
        
        # Create circles with random radii
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x, y = seeds[i]
            # Initial radius - small enough to fit in square with room for others
            max_radius = min(x, y, 1-x, 1-y) * 0.4
            r = np.random.uniform(0.01, max_radius)
            circles[i] = [x, y, r]
            
        # Local optimization to improve initial configuration
        circles = optimize_circle_placement(circles)
        
        population.append(circles)
        
    return population

def optimize_circle_placement(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply local optimization to improve circle placement."""
    # Simple iterative improvement
    for _ in range(max_iter):
        improved = False
        for i in range(len(circles)):
            # Try small movements
            original = circles[i].copy()
            best_circle = original.copy()
            best_fitness = compute_fitness(circles)
            
            # Try different moves
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    if dx == 0 and dy == 0:
                        continue
                    circles[i][0] = original[0] + dx
                    circles[i][1] = original[1] + dy
                    
                    # Ensure circle stays within bounds
                    if check_containment(circles[i]):
                        fitness = compute_fitness(circles)
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_circle = circles[i].copy()
                            improved = True
                        
            circles[i] = best_circle
            if not improved:
                break
                
    return circles

def mutate(circles: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
    """Mutate circles with possibility of changing positions and radii."""
    mutated = circles.copy()
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position
            mutated[i][0] += np.random.normal(0, 0.01)
            mutated[i][1] += np.random.normal(0, 0.01)
            
            # Mutate radius
            mutated[i][2] += np.random.normal(0, 0.005)
            
            # Ensure radius remains positive
            mutated[i][2] = max(0.001, mutated[i][2])
            
            # Ensure circle stays within bounds
            if mutated[i][0] - mutated[i][2] < 0:
                mutated[i][0] = mutated[i][2]
            if mutated[i][0] + mutated[i][2] > 1:
                mutated[i][0] = 1 - mutated[i][2]
            if mutated[i][1] - mutated[i][2] < 0:
                mutated[i][1] = mutated[i][2]
            if mutated[i][1] + mutated[i][2] > 1:
                mutated[i][1] = 1 - mutated[i][2]
                
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Crossover two parent solutions."""
    child = parent1.copy()
    
    # Single point crossover on radii
    crossover_point = random.randint(0, len(parent1))
    child[crossover_point:, 2] = parent2[crossover_point:, 2]
    
    # Fix any violated constraints
    child = optimize_circle_placement(child)
    
    return child

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    pop_size = 50
    generations = 200
    elite_size = 5
    
    # Initialize population
    population = initialize_population(pop_size, n)
    
    # Evaluate initial population
    fitness_scores = [compute_fitness(individual) for individual in population]
    
    # Evolution loop
    for gen in range(generations):
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = population[:elite_size]
        
        # Create new population
        new_population = elite.copy()
        
        # Generate offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parent1_idx = random.randint(0, pop_size // 2)
            parent2_idx = random.randint(0, pop_size // 2)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            # Optimize the child
            child = optimize_circle_placement(child)
            
            new_population.append(child)
            
        population = new_population
        
        # Evaluate new population
        fitness_scores = [compute_fitness(individual) for individual in population]
        
        # Print progress
        if gen % 50 == 0:
            best_fitness = max(fitness_scores)
            print(f"Generation {gen}: Best fitness = {best_fitness}")
    
    # Return the best solution
    best_idx = np.argmax(fitness_scores)
    best_solution = population[best_idx]
    
    # Final optimization
    best_solution = optimize_circle_placement(best_solution, max_iter=500)
    
    return best_solution

# EVOLVE-BLOCK-END
