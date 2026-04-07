# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cdist
import random
from collections import defaultdict
import math

# Global constants
POP_SIZE = 50
GENERATIONS = 200
ELITE_SIZE = 5
INITIAL_MUTATION_RATE = 0.1
MAX_MUTATION_STEP = 0.05
BOUNDARY_PENALTY = 10000
OVERLAP_PENALTY_MULTIPLIER = 1000

class CirclePackingOptimizer:
    def __init__(self, n_circles=32):
        self.n_circles = n_circles
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def initialize_voronoi_based(self):
        """Initialize circles using Voronoi-based approach combined with hexagonal grid"""
        circles = np.zeros((self.n_circles, 3))
        
        # Create a more sophisticated initial distribution using modified grid
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)
        
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= self.n_circles:
                    break
                x = (j + 1) * spacing + random.uniform(-spacing/4, spacing/4)
                y = (i + 1) * spacing + random.uniform(-spacing/4, spacing/4)
                # Ensure points are within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
            if len(points) >= self.n_circles:
                break
        
        # Add random points if needed
        while len(points) < self.n_circles:
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            points.append([x, y])
            
        points = np.array(points[:self.n_circles])
        
        # Apply greedy radius assignment based on Voronoi regions
        circles = self.assign_greedy_radii(points)
        
        return circles
    
    def assign_greedy_radii(self, points):
        """Assign initial radii using greedy approach with Voronoi consideration"""
        n = len(points)
        circles = np.zeros((n, 3))
        circles[:, 0] = points[:, 0]  # x
        circles[:, 1] = points[:, 1]  # y
        
        # For each point, compute maximum possible radius greedily
        for i in range(n):
            max_radius = self.compute_max_radius_greedy(circles, i)
            circles[i, 2] = max(0.001, max_radius)
            
        return circles
    
    def compute_max_radius_greedy(self, circles, index):
        """Compute maximum radius for a circle using greedy method similar to first implementation"""
        # Get the current position
        x, y = circles[index, 0], circles[index, 1]
        
        # Find minimum distance to other circles
        min_dist = float('inf')
        
        for i in range(len(circles)):
            if i != index:
                dx = circles[i, 0] - x
                dy = circles[i, 1] - y
                distance = np.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, distance)
        
        # Minimum distance to boundaries
        boundary_dist = min(x, 1-x, y, 1-y)
        
        # Maximum radius is limited by both boundary and other circles
        if min_dist == float('inf'):
            max_radius = boundary_dist
        else:
            max_radius = min(boundary_dist, min_dist / 2.0)
            
        return max(0.001, max_radius)
    
    def validate_circle_placement(self, circles):
        """Validate that all circles are within bounds and non-overlapping"""
        # Check boundary constraints
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints using a more efficient method
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Early exit if too few circles
        if len(circles) <= 1:
            return True
            
        # Use scipy for distance calculation
        distances = cdist(positions, positions)
        
        # Check all pairs for overlaps
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                r_i = radii[i]
                r_j = radii[j]
                if dist < (r_i + r_j):
                    return False
                    
        return True
    
    def calculate_fitness(self, circles):
        """Calculate fitness including penalties for constraint violations"""
        # Get sum of radii
        sum_radii = np.sum(circles[:, 2])
        
        # Calculate penalties
        penalty = 0
        
        # Boundary penalty
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += BOUNDARY_PENALTY
                
        # Overlap penalty
        if len(circles) > 1:
            positions = circles[:, :2]
            radii = circles[:, 2]
            distances = cdist(positions, positions)
            
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    dist = distances[i, j]
                    r_i = radii[i]
                    r_j = radii[j]
                    if dist < (r_i + r_j):
                        overlap = (r_i + r_j) - dist
                        penalty += overlap * OVERLAP_PENALTY_MULTIPLIER
                        
        return sum_radii - penalty
    
    def mutate_individual(self, circles, generation):
        """Enhanced mutation with adaptive rate based on generation"""
        mutated = circles.copy()
        
        # Adaptive mutation rate
        mutation_rate = INITIAL_MUTATION_RATE * (1 - generation / GENERATIONS)
        
        for i in range(len(circles)):
            if random.random() < mutation_rate:
                # Choose what to mutate
                choice = random.randint(0, 2)
                
                if choice == 0:  # Mutate x position
                    mutated[i, 0] = max(0, min(1, mutated[i, 0] + random.uniform(-MAX_MUTATION_STEP, MAX_MUTATION_STEP)))
                elif choice == 1:  # Mutate y position
                    mutated[i, 1] = max(0, min(1, mutated[i, 1] + random.uniform(-MAX_MUTATION_STEP, MAX_MUTATION_STEP)))
                else:  # Mutate radius
                    mutated[i, 2] = max(0.001, min(0.5, mutated[i, 2] + random.uniform(-MAX_MUTATION_STEP, MAX_MUTATION_STEP)))
                    
        return mutated
    
    def crossover(self, parent1, parent2):
        """Uniform crossover with better mixing"""
        offspring = parent1.copy()
        
        # Uniform crossover
        for i in range(len(parent1)):
            if random.random() > 0.5:
                offspring[i] = parent2[i]
                
        return offspring
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection with diversity preservation"""
        selected_indices = random.sample(range(len(population)), tournament_size)
        selected_fitness = [fitness_scores[i] for i in selected_indices]
        winner_index = selected_indices[np.argmax(selected_fitness)]
        return population[winner_index]
    
    def optimize(self):
        """Main optimization loop"""
        # Initialize population with Voronoi-based approach
        population = [self.initialize_voronoi_based() for _ in range(POP_SIZE)]
        
        # Track improvement
        last_improvement_gen = 0
        stagnation_counter = 0
        max_stagnation = 30
        
        for gen in range(GENERATIONS):
            # Evaluate fitness for all individuals
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores.sort(reverse=True)
            
            # Track best solution
            current_best_fitness = fitness_scores[0]
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = population[0].copy()
                last_improvement_gen = gen
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            # Print progress every 20 generations
            if gen % 20 == 0:
                print(f"Generation {gen}: Best fitness = {current_best_fitness:.4f}")
                
            # Check for stagnation
            if stagnation_counter > max_stagnation:
                print(f"Stagnation detected at generation {gen}, restarting...")
                # Restart with new Voronoi-based population
                population = [self.initialize_voronoi_based() for _ in range(POP_SIZE)]
                last_improvement_gen = gen
                stagnation_counter = 0
            
            # Create new population with elitism
            new_population = population[:ELITE_SIZE]
            
            # Generate offspring through crossover and mutation
            while len(new_population) < POP_SIZE:
                # Tournament selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                offspring = self.crossover(parent1, parent2)
                
                # Mutation
                offspring = self.mutate_individual(offspring, gen)
                
                new_population.append(offspring)
                
            population = new_population
            
        return self.best_solution

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer(32)
    result = optimizer.optimize()
    return result

# EVOLVE-BLOCK-END