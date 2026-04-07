# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time
from itertools import product
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return np.min(distances) / max_dist
    
    # Grid-based evolutionary approach
    class GridOptimizer:
        def __init__(self, num_points=16, grid_size=(8, 8), max_iterations=500):
            self.num_points = num_points
            self.grid_rows, self.grid_cols = grid_size
            self.max_iterations = max_iterations
            self.grid_cells = self.grid_rows * self.grid_cols
            
        def generate_initial_population(self, pop_size=50):
            """Generate initial population of grid assignments"""
            population = []
            for _ in range(pop_size):
                # Assign each point to a grid cell (with replacement)
                assignments = np.random.randint(0, self.grid_cells, self.num_points)
                population.append(assignments)
            return population
            
        def evaluate_individual(self, assignments):
            """Evaluate fitness of a grid assignment"""
            # Convert assignments to actual points
            points = []
            for assignment in assignments:
                # Convert grid cell index to coordinates
                row = assignment // self.grid_cols
                col = assignment % self.grid_cols
                # Position within cell (random within cell)
                x = (col + np.random.random()) / self.grid_cols
                y = (row + np.random.random()) / self.grid_rows
                points.append([x, y])
            
            points = np.array(points)
            return compute_min_max_ratio(points)
            
        def crossover(self, parent1, parent2):
            """Single-point crossover"""
            crossover_point = len(parent1) // 2
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
            
        def mutate(self, individual, mutation_rate=0.1):
            """Mutate individual"""
            mutated = individual.copy()
            for i in range(len(mutated)):
                if np.random.random() < mutation_rate:
                    mutated[i] = np.random.randint(0, self.grid_cells)
            return mutated
            
        def optimize(self):
            """Main optimization loop with grid-based evolutionary approach"""
            # Generate initial population
            population = self.generate_initial_population(50)
            fitness_history = []
            
            # Evolutionary optimization
            for generation in range(self.max_iterations):
                # Evaluate fitness
                fitness_scores = [self.evaluate_individual(individual) for individual in population]
                
                # Track best
                best_idx = np.argmax(fitness_scores)
                best_fitness = fitness_scores[best_idx]
                fitness_history.append(best_fitness)
                
                # Keep track of best solution
                if generation == 0:
                    best_assignments = population[best_idx].copy()
                    best_score = best_fitness
                else:
                    if best_fitness > best_score:
                        best_assignments = population[best_idx].copy()
                        best_score = best_fitness
                
                # Selection (tournament selection)
                tournament_size = 3
                new_population = []
                
                # Elitism: keep best
                new_population.append(population[best_idx])
                
                # Generate offspring
                while len(new_population) < len(population):
                    # Tournament selection
                    parents = []
                    for _ in range(2):
                        tournament = np.random.choice(len(population), tournament_size)
                        winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
                        parents.append(population[winner])
                    
                    # Crossover
                    child1, child2 = self.crossover(parents[0], parents[1])
                    
                    # Mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.extend([child1, child2])
                
                # Trim population if needed
                population = new_population[:len(population)]
                
                # Adaptive mutation rate
                if generation > 100 and np.std(fitness_history[-50:]) < 1e-6:
                    # Convergence detected, increase mutation rate
                    pass  # In practice, we'd increase mutation rate here
            
            # Refine best solution with local optimization
            refined_points = []
            for assignment in best_assignments:
                row = assignment // self.grid_cols
                col = assignment % self.grid_cols
                # Position in cell with small random offset
                x = (col + np.random.random() * 0.3 + 0.35) / self.grid_cols
                y = (row + np.random.random() * 0.3 + 0.35) / self.grid_rows
                refined_points.append([x, y])
            
            refined_points = np.array(refined_points)
            refined_points = np.clip(refined_points, 0, 1)
            
            # Final local optimization using differential evolution on the refined points
            def objective(x_flat):
                points = x_flat.reshape(-1, 2)
                return -compute_min_max_ratio(points)
                
            bounds = [(0, 1) for _ in range(32)]
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=200,
                    popsize=10,
                    tol=1e-8,
                    seed=42,
                    callback=None
                )
                if result.success:
                    refined_points = result.x.reshape(-1, 2)
                    refined_points = np.clip(refined_points, 0, 1)
            except:
                pass
                
            return refined_points
    
    def create_hexagonal_placement():
        """Create initial hexagonal arrangement as baseline"""
        points = []
        spacing_x = 1.0 / 3.0  # horizontal spacing
        spacing_y = spacing_x * np.sqrt(3) / 2  # vertical spacing
        
        # Create triangular lattice with 4x4 grid
        for i in range(4):
            for j in range(4):
                x = j * spacing_x
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                y = i * spacing_y
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to [0,1] x [0,1]
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
            
        # Scale to fit nicely within [0.05, 0.95] x [0.05, 0.95]
        points[:, 0] = points[:, 0] * 0.9 + 0.05
        points[:, 1] = points[:, 1] * 0.9 + 0.05
        
        return points
    
    # Multi-strategy approach with grid-based optimization as primary
    best_points = None
    best_ratio = -np.inf
    
    # Strategy 1: Grid-based evolutionary optimization (primary approach)
    try:
        grid_optimizer = GridOptimizer(num_points=16, grid_size=(6, 6), max_iterations=300)
        points = grid_optimizer.optimize()
        ratio = compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except:
        pass
    
    # Strategy 2: Hexagonal initialization followed by refinement
    if best_points is None:
        try:
            points = create_hexagonal_placement()
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except:
            pass
    
    # Strategy 3: Final refinement if needed
    if best_points is not None:
        # Attempt final local optimization
        try:
            def objective(x_flat):
                points = x_flat.reshape(-1, 2)
                return -compute_min_max_ratio(points)
                
            bounds = [(0, 1) for _ in range(32)]
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=5,
                tol=1e-8,
                seed=42,
                callback=None
            )
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                ratio = compute_min_max_ratio(refined_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
        except:
            pass
    
    # Fallback to hexagonal if all else fails
    if best_points is None:
        best_points = create_hexagonal_placement()
    
    return best_points

# EVOLVE-BLOCK-END