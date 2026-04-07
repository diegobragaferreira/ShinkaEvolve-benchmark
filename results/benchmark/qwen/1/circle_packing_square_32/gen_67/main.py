# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from collections import defaultdict
import math

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Parameters
    N_CIRCLES = 32
    MAX_ITERATIONS = 15000
    POPULATION_SIZE = 150
    TOURNAMENT_SIZE = 5
    ELITISM_COUNT = 10
    MUTATION_RATE_START = 0.3
    MUTATION_RATE_DECAY = 0.995
    GRID_RESOLUTION = 15
    
    # Initialize population
    def initialize_population(size):
        population = []
        for _ in range(size):
            circles = generate_initial_configuration()
            population.append(circles)
        return population
    
    # Generate initial configuration using Voronoi-based approach with corner bias
    def generate_initial_configuration():
        # Sample points in a way that favors corners and edges
        points = []
        # Add corner points
        corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        for x, y in corners:
            points.append((x, y))
            
        # Add edge points
        for _ in range(8):
            x = random.uniform(0.05, 0.95)
            y = random.choice([0.05, 0.95])
            points.append((x, y))
            
        for _ in range(8):
            x = random.choice([0.05, 0.95])
            y = random.uniform(0.05, 0.95)
            points.append((x, y))
        
        # Add interior points
        for _ in range(12):
            points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
        
        # Create Voronoi diagram
        vor = Voronoi(points)
        centers = vor.points
        
        # For each center, compute maximum possible radius
        circles = []
        for i, (cx, cy) in enumerate(centers[:N_CIRCLES]):
            if i >= N_CIRCLES:
                break
                
            # Calculate max radius for this circle
            min_dist_to_boundary = min(
                cx, 1-cx, cy, 1-cy
            )
            
            # Find minimum distance to other centers
            min_dist_to_other = float('inf')
            for j, (other_cx, other_cy) in enumerate(centers[:N_CIRCLES]):
                if i != j:
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    min_dist_to_other = min(min_dist_to_other, dist)
                    
            # Set radius safely
            radius = min(min_dist_to_other / 2.0, min_dist_to_boundary)
            radius = max(radius, 0.001)  # Ensure minimum radius
            
            circles.append([cx, cy, radius])
            
        # If we don't have enough circles, fill with random ones
        while len(circles) < N_CIRCLES:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            # Find minimum distance to existing circles
            min_dist = float('inf')
            for cx, cy, r in circles:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                min_dist = min(min_dist, dist)
                
            radius = min(min_dist/2.0, 1-x, x, 1-y, y)
            radius = max(radius, 0.001)
            circles.append([x, y, radius])
            
        return np.array(circles[:N_CIRCLES])
    
    # Spatial grid for fast collision detection
    class SpatialGrid:
        def __init__(self, resolution=GRID_RESOLUTION):
            self.resolution = resolution
            self.grid = defaultdict(list)
            
        def clear(self):
            self.grid.clear()
            
        def add_circle(self, x, y, r, index):
            # Get grid cell indices
            grid_x = int(x * self.resolution)
            grid_y = int(y * self.resolution)
            self.grid[(grid_x, grid_y)].append((index, x, y, r))
            
        def get_candidates(self, x, y, r):
            candidates = []
            grid_x = int(x * self.resolution)
            grid_y = int(y * self.resolution)
            
            # Check nearby cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (grid_x + dx, grid_y + dy)
                    if cell in self.grid:
                        candidates.extend(self.grid[cell])
                        
            return candidates
    
    # Calculate fitness with proper penalties
    def calculate_fitness(circles):
        total_radius = np.sum(circles[:, 2])
        
        # Apply penalties for constraint violations
        penalty = 0.0
        
        # Boundary violations
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or y - r < 0 or x + r > 1 or y + r > 1:
                penalty += 1000 * (r**2)  # Quadratic penalty for boundary violations
                
        # Overlap violations
        grid = SpatialGrid(GRID_RESOLUTION)
        for i in range(len(circles)):
            grid.add_circle(*circles[i], i)
            
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            candidates = grid.get_candidates(x1, y1, r1)
            
            for j, x2, y2, r2 in candidates:
                if i != j:
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2):
                        penalty += (r1 + r2 - distance) * 10
                        
        return total_radius - penalty
    
    # Mutate a solution
    def mutate(circles, mutation_rate):
        new_circles = circles.copy()
        for i in range(len(new_circles)):
            if random.random() < mutation_rate:
                # Randomly modify one circle
                x, y, r = new_circles[i]
                
                # Perturb position
                x += random.gauss(0, 0.01)
                y += random.gauss(0, 0.01)
                
                # Clamp to valid range
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                # Adjust radius to respect constraints
                min_dist_to_boundary = min(x, 1-x, y, 1-y)
                r = min(r + random.gauss(0, 0.005), min_dist_to_boundary)
                r = max(0.001, r)
                
                new_circles[i] = [x, y, r]
                
        return new_circles
    
    # Crossover two solutions
    def crossover(parent1, parent2):
        # Simple uniform crossover
        child = parent1.copy()
        for i in range(len(child)):
            if random.random() < 0.5:
                child[i] = parent2[i].copy()
        return child
    
    # Tournament selection
    def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
        selected_indices = random.sample(range(len(population)), k)
        best_idx = selected_indices[0]
        for idx in selected_indices[1:]:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
        return population[best_idx].copy()
    
    # Main evolutionary algorithm
    population = initialize_population(POPULATION_SIZE)
    
    best_fitness = -float('inf')
    best_individual = None
    
    # Track diversity to avoid stagnation
    last_improvement = 0
    stagnation_counter = 0
    
    for generation in range(MAX_ITERATIONS):
        # Calculate fitness for all individuals
        fitnesses = []
        for i, individual in enumerate(population):
            fitness = calculate_fitness(individual)
            fitnesses.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                last_improvement = generation
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
        # Adaptive mutation rate
        mutation_rate = MUTATION_RATE_START * (MUTATION_RATE_DECAY ** generation)
        
        # Check for stagnation and restart if needed
        if stagnation_counter > 500:
            print(f"Stagnation detected at generation {generation}, restarting...")
            population = initialize_population(POPULATION_SIZE)
            stagnation_counter = 0
            continue
            
        # Selection and reproduction
        new_population = []
        
        # Elitism - keep best individuals
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        for i in range(min(ELITISM_COUNT, len(population))):
            new_population.append(population[sorted_indices[i]].copy())
            
        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
            
        population = new_population[:POPULATION_SIZE]
        
        # Print progress
        if generation % 1000 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
            
    return best_individual

# EVOLVE-BLOCK-END
