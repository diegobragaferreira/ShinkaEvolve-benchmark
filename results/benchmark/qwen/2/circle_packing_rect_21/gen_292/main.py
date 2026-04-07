# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import math
from copy import deepcopy
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Optimized rectangle dimensions for better packing efficiency
    width, height = 1.25, 0.75
    
    # Initialize parameters
    n_circles = 21
    random.seed(42)
    np.random.seed(42)
    
    class PhysicsGuidedEvolutionOptimizer:
        def __init__(self, width, height, n_circles):
            self.width = width
            self.height = height
            self.n_circles = n_circles
            self.rect_area = width * height
            self.max_iterations = 500
            self.population_size = 60
            self.elite_size = 10
            self.mutation_rate = 0.15
            
        def initialize_population(self):
            """Initialize diverse population using multiple strategies"""
            population = []
            
            # Strategy 1: Grid-based initialization with hexagonal offset
            def grid_initialization():
                circles = np.zeros((self.n_circles, 3))
                
                # Calculate grid dimensions
                sqrt_n = math.sqrt(self.n_circles)
                cols = max(1, int(sqrt_n * 1.2))
                rows = max(1, int(self.n_circles / cols))
                
                # Ensure we have enough slots
                while cols * rows < self.n_circles:
                    cols += 1
                
                spacing_x = self.width / (cols + 1)
                spacing_y = self.height / (rows + 1)
                
                idx = 0
                for i in range(cols):
                    for j in range(rows):
                        if idx >= self.n_circles:
                            break
                        # Hexagonal offset
                        offset = (j % 2) * spacing_x * 0.5
                        x = (i + 1) * spacing_x + offset + random.uniform(-spacing_x*0.1, spacing_x*0.1)
                        y = (j + 1) * spacing_y + random.uniform(-spacing_y*0.1, spacing_y*0.1)
                        
                        # Initial radius based on spacing
                        r = min(spacing_x, spacing_y) * 0.3
                        
                        # Position-based adjustment to prevent clustering at edges
                        center_x, center_y = self.width/2, self.height/2
                        dist_to_center = math.sqrt((x-center_x)**2 + (y-center_y)**2)
                        max_dist_to_center = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                        if max_dist_to_center > 0:
                            adjustment = 1.0 + 0.3 * (1.0 - dist_to_center/max_dist_to_center)
                            r *= adjustment
                        
                        circles[idx] = [x, y, max(0.005, r)]
                        idx += 1
                    if idx >= self.n_circles:
                        break
                return circles
            
            # Strategy 2: Random initialization with boundary constraints
            def random_initialization():
                circles = np.zeros((self.n_circles, 3))
                for i in range(self.n_circles):
                    # Ensure circles fit within bounds
                    max_radius = min(self.width, self.height) * 0.2
                    x = random.uniform(max_radius, self.width - max_radius)
                    y = random.uniform(max_radius, self.height - max_radius)
                    r = random.uniform(0.01, max_radius * 0.8)
                    circles[i] = [x, y, r]
                return circles
            
            # Create diverse initial population
            for i in range(self.population_size):
                if i == 0:
                    # First individual: grid initialization
                    circles = grid_initialization()
                elif i < 15:
                    # Some individuals: grid with noise
                    circles = grid_initialization()
                    for j in range(self.n_circles):
                        circles[j, 0] += random.uniform(-0.1, 0.1) * (self.width / 10)
                        circles[j, 1] += random.uniform(-0.1, 0.1) * (self.height / 10)
                        circles[j, 2] *= random.uniform(0.9, 1.1)
                elif i < 30:
                    # Middle individuals: random
                    circles = random_initialization()
                else:
                    # Later individuals: hybrid
                    circles = grid_initialization()
                    # Add some randomization
                    for j in range(self.n_circles):
                        if random.random() < 0.3:
                            circles[j, 0] += random.uniform(-0.05, 0.05) * self.width
                            circles[j, 1] += random.uniform(-0.05, 0.05) * self.height
                            circles[j, 2] *= random.uniform(0.8, 1.2)
                
                # Ensure all circles are within bounds
                for j in range(self.n_circles):
                    x, y, r = circles[j]
                    circles[j] = [max(r, min(self.width - r, x)), 
                                 max(r, min(self.height - r, y)), 
                                 max(0.001, r)]
                
                population.append(circles)
            
            return population
        
        def calculate_fitness(self, circles):
            """Calculate fitness with constraint penalty"""
            # Objective: maximize sum of radii
            total_radius = np.sum(circles[:, 2])
            
            # Constraints: penalty for violations
            penalty = 0
            
            # Boundary penalties
            for i in range(self.n_circles):
                x, y, r = circles[i]
                if x - r < 0.001:
                    penalty += 10000 * (r - x)**2
                if x + r > self.width - 0.001:
                    penalty += 10000 * (x + r - self.width)**2
                if y - r < 0.001:
                    penalty += 10000 * (r - y)**2
                if y + r > self.height - 0.001:
                    penalty += 10000 * (y + r - self.height)**2
            
            # Overlap penalties using efficient spatial indexing
            try:
                points = circles[:, :2]
                tree = cKDTree(points)
                
                # Query pairs efficiently
                max_radius = np.max(circles[:, 2])
                if max_radius > 0:
                    pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
                    
                    for i, j in pairs:
                        if i < j:  # Only check each pair once
                            x1, y1, r1 = circles[i]
                            x2, y2, r2 = circles[j]
                            dx = x2 - x1
                            dy = y2 - y1
                            distance = math.sqrt(dx*dx + dy*dy)
                            overlap = max(0, (r1 + r2) - distance)
                            penalty += 1000 * overlap**2  # Heavier penalty for overlaps
            except:
                # Fallback to brute force if needed
                for i in range(self.n_circles):
                    for j in range(i+1, self.n_circles):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)
                        overlap = max(0, (r1 + r2) - distance)
                        penalty += 1000 * overlap**2
            
            # Balance objective and penalty
            return total_radius - penalty
        
        def get_spatial_index(self, circles):
            """Create spatial index for neighbor queries"""
            points = circles[:, :2]
            return cKDTree(points)
        
        def check_collision(self, circles, idx1, idx2):
            """Fast collision check between two circles"""
            x1, y1, r1 = circles[idx1]
            x2, y2, r2 = circles[idx2]
            dx = x2 - x1
            dy = y2 - y1
            distance = math.sqrt(dx*dx + dy*dy)
            return distance < (r1 + r2)
        
        def apply_physics_simulation(self, circles, iterations=150):
            """Apply physics simulation with adaptive parameters"""
            # Create spatial index for efficient neighbor search
            tree = self.get_spatial_index(circles)
            
            # Adaptive parameters based on optimization progress
            for iteration in range(iterations):
                # Dynamically adjust forces based on iteration
                repulsion_strength = 3.0 * (1.0 - iteration / iterations)
                attraction_strength = 0.2 * (1.0 - iteration / iterations * 0.5)
                boundary_strength = 1.0
                
                forces = np.zeros_like(circles)
                
                # Calculate repulsion forces
                for i in range(len(circles)):
                    x1, y1, r1 = circles[i]
                    neighbors = tree.query_ball_point([x1, y1], 3 * (r1 + 0.01))
                    
                    for j in neighbors:
                        if i != j:
                            x2, y2, r2 = circles[j]
                            dx = x2 - x1
                            dy = y2 - y1
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            if distance > 0.001:
                                overlap = (r1 + r2) - distance
                                if overlap > 0:
                                    force_magnitude = repulsion_strength * overlap / (distance ** 2)
                                    forces[i, 0] -= force_magnitude * dx / distance
                                    forces[i, 1] -= force_magnitude * dy / distance
                
                # Apply center attraction and boundary forces
                center_x, center_y = self.width / 2, self.height / 2
                for i in range(len(circles)):
                    x, y, r = circles[i]
                    
                    # Attract to center
                    dx = center_x - x
                    dy = center_y - y
                    forces[i, 0] += attraction_strength * dx
                    forces[i, 1] += attraction_strength * dy
                    
                    # Boundary forces
                    if x - r < 0.001:
                        forces[i, 0] += boundary_strength * (r - x)
                    if x + r > self.width - 0.001:
                        forces[i, 0] -= boundary_strength * (x + r - self.width)
                    if y - r < 0.001:
                        forces[i, 1] += boundary_strength * (r - y)
                    if y + r > self.height - 0.001:
                        forces[i, 1] -= boundary_strength * (y + r - self.height)
                
                # Apply forces
                step_size = 0.02 * (1.0 - iteration / iterations)
                for i in range(len(circles)):
                    circles[i, 0] += forces[i, 0] * step_size
                    circles[i, 1] += forces[i, 1] * step_size
                    
                    # Enforce bounds and keep radii positive
                    x, y, r = circles[i]
                    circles[i] = [
                        max(r, min(self.width - r, x)),
                        max(r, min(self.height - r, y)),
                        max(0.001, r)
                    ]
                
                # Update spatial index for next iteration if needed
                if iteration % 20 == 0:
                    tree = self.get_spatial_index(circles)
            
            return circles
        
        def mutate_individual(self, individual, generation=0):
            """Apply mutation to individual with adaptive parameters"""
            mutated = individual.copy()
            
            # Adaptive mutation rate
            adaptive_rate = self.mutation_rate * (1.0 - generation / self.max_iterations)
            
            for i in range(self.n_circles):
                if random.random() < adaptive_rate:
                    # Decide what to mutate
                    mutation_type = random.choice(['position', 'radius'])
                    
                    if mutation_type == 'position':
                        # Mutate position
                        mutate_x = random.uniform(-0.05, 0.05) * self.width
                        mutate_y = random.uniform(-0.05, 0.05) * self.height
                        
                        mutated[i, 0] = max(mutated[i, 2], 
                                          min(self.width - mutated[i, 2], 
                                              mutated[i, 0] + mutate_x))
                        mutated[i, 1] = max(mutated[i, 2], 
                                          min(self.height - mutated[i, 2], 
                                              mutated[i, 1] + mutate_y))
                    else:
                        # Mutate radius
                        mutate_r = random.uniform(-0.02, 0.02) * max(0.01, mutated[i, 2])
                        mutated[i, 2] = max(0.001, mutated[i, 2] + mutate_r)
            
            return mutated
        
        def crossover_individuals(self, parent1, parent2):
            """Perform uniform crossover between two individuals"""
            child = parent1.copy()
            mask = np.random.rand(self.n_circles) > 0.5
            
            for i in range(self.n_circles):
                if mask[i]:
                    child[i] = parent2[i].copy()
                    
            return child
        
        def optimize(self):
            """Main optimization loop using physics-guided evolution"""
            # Initialize population
            population = self.initialize_population()
            
            best_fitness = float('-inf')
            best_individual = None
            
            # Record fitness history for early stopping
            fitness_history = []
            
            for generation in range(self.max_iterations):
                # Evaluate fitness for all individuals
                fitness_scores = []
                for individual in population:
                    score = self.calculate_fitness(individual)
                    fitness_scores.append(score)
                    
                    if score > best_fitness:
                        best_fitness = score
                        best_individual = individual.copy()
                
                # Sort by fitness (descending)
                sorted_indices = np.argsort(fitness_scores)[::-1]
                population = [population[i] for i in sorted_indices]
                fitness_scores = [fitness_scores[i] for i in sorted_indices]
                
                # Track history for early stopping
                fitness_history.append(best_fitness)
                if len(fitness_history) > 10:
                    fitness_history.pop(0)
                
                # Early stopping if no improvement
                if len(fitness_history) >= 10:
                    recent_improvement = fitness_history[-1] - fitness_history[0]
                    if recent_improvement < 1e-6:
                        break
                
                # Create new population
                new_population = population[:self.elite_size]  # Keep elites
                
                # Generate offspring through crossover and mutation
                while len(new_population) < self.population_size:
                    # Tournament selection
                    parent1_idx = self.tournament_selection(population, fitness_scores, 5)
                    parent2_idx = self.tournament_selection(population, fitness_scores, 5)
                    
                    # Crossover
                    child = self.crossover_individuals(population[parent1_idx], population[parent2_idx])
                    
                    # Mutation
                    child = self.mutate_individual(child, generation)
                    
                    # Apply physics simulation to improve offspring
                    child = self.apply_physics_simulation(child, iterations=50)
                    
                    new_population.append(child)
                
                population = new_population
                
                # Periodic physics refinement
                if generation % 20 == 0:
                    for i in range(min(5, len(population))):
                        population[i] = self.apply_physics_simulation(population[i], iterations=30)
            
            # Final optimization pass
            if best_individual is not None:
                final_result = self.apply_physics_simulation(best_individual, iterations=100)
                final_fitness = self.calculate_fitness(final_result)
                
                if final_fitness > best_fitness:
                    best_individual = final_result
            
            return best_individual if best_individual is not None else population[0]
        
        def tournament_selection(self, population, fitness_scores, k):
            """Tournament selection with proper fitness ranking"""
            tournament_indices = random.sample(range(len(population)), min(k, len(population)))
            tournament_fitness = [(i, fitness_scores[i]) for i in tournament_indices]
            
            # Sort by fitness descending
            tournament_fitness.sort(key=lambda x: x[1], reverse=True)
            
            return tournament_fitness[0][0]
    
    # Run optimization
    optimizer = PhysicsGuidedEvolutionOptimizer(width, height, n_circles)
    circles = optimizer.optimize()
    
    # Final validation and adjustment
    # Ensure all circles are properly placed and validated
    for i in range(n_circles):
        x, y, r = circles[i]
        circles[i] = [
            max(r, min(width - r, x)),
            max(r, min(height - r, y)),
            max(0.001, r)
        ]
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
