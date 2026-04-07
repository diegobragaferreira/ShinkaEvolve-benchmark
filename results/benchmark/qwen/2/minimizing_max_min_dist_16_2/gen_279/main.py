# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import random
import time
from typing import Tuple, List

class GridEvolutionOptimizer:
    def __init__(self, num_points: int = 16, grid_resolution: int = 20):
        self.num_points = num_points
        self.grid_resolution = grid_resolution
        self.grid_size = grid_resolution * grid_resolution
        self.grid_cells = [(i, j) for i in range(grid_resolution) for j in range(grid_resolution)]
        
        # Pre-compute all possible pairwise distances on grid
        self.grid_distances = {}
        for i, cell1 in enumerate(self.grid_cells):
            for j, cell2 in enumerate(self.grid_cells):
                if i <= j:
                    dx = cell1[0] - cell2[0]
                    dy = cell1[1] - cell2[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    self.grid_distances[(i, j)] = dist
                    self.grid_distances[(j, i)] = dist

    def compute_min_max_ratio(self, points: np.ndarray) -> Tuple[float, float, float]:
        """Compute the minimum to maximum distance ratio for given points."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0, min_dist, max_dist

        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist

    def point_to_grid_index(self, point: Tuple[float, float]) -> int:
        """Convert continuous point to discrete grid index."""
        x, y = point
        grid_x = int(x * self.grid_resolution)
        grid_y = int(y * self.grid_resolution)
        # Clamp to valid range
        grid_x = max(0, min(self.grid_resolution - 1, grid_x))
        grid_y = max(0, min(self.grid_resolution - 1, grid_y))
        return grid_x * self.grid_resolution + grid_y

    def grid_index_to_point(self, index: int) -> Tuple[float, float]:
        """Convert discrete grid index back to continuous point."""
        grid_x = index // self.grid_resolution
        grid_y = index % self.grid_resolution
        x = (grid_x + 0.5) / self.grid_resolution
        y = (grid_y + 0.5) / self.grid_resolution
        return (x, y)

    def validate_and_fix_points(self, points: np.ndarray) -> np.ndarray:
        """Ensure points are within valid bounds and not duplicate."""
        validated = points.copy()
        # Ensure within bounds
        validated = np.clip(validated, 0.001, 0.999)
        
        # Remove duplicates and re-add if needed
        unique_points = []
        for point in validated:
            # Check if point is close to any existing point
            is_duplicate = False
            for existing in unique_points:
                if np.linalg.norm(point - existing) < 1e-6:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        # If we lost points, generate new ones
        while len(unique_points) < self.num_points:
            new_point = np.random.uniform(0.001, 0.999, 2)
            unique_points.append(new_point)
        
        return np.array(unique_points[:self.num_points])

    def create_individual(self) -> np.ndarray:
        """Create a random individual (point configuration) on grid."""
        # Select random grid indices
        grid_indices = random.sample(range(self.grid_size), self.num_points)
        points = np.array([self.grid_index_to_point(idx) for idx in grid_indices])
        return self.validate_and_fix_points(points)

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two individuals."""
        # Use 50% crossover rate
        mask = np.random.random(self.num_points) > 0.5
        child = np.where(mask, parent1, parent2)
        return self.validate_and_fix_points(child)

    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Mutate individual by randomly changing some points."""
        mutated = individual.copy()
        for i in range(self.num_points):
            if random.random() < mutation_rate:
                # Change this point to a random grid location
                new_point = self.create_individual()[0]
                mutated[i] = new_point
        return mutated

    def evaluate_fitness(self, individual: np.ndarray) -> float:
        """Evaluate fitness of individual (higher is better)."""
        ratio, _, _ = self.compute_min_max_ratio(individual)
        return ratio

    def evolve_population(self, population: List[np.ndarray], 
                         elite_size: int = 2, 
                         mutation_rate: float = 0.1,
                         generations: int = 100) -> np.ndarray:
        """Evolve population using genetic algorithm."""
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [(ind, self.evaluate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep elite
            elite = [ind for ind, _ in fitness_scores[:elite_size]]
            
            # Create new population
            new_population = elite.copy()
            
            # Generate offspring through crossover and mutation
            while len(new_population) < len(population):
                # Tournament selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child, mutation_rate)
                
                new_population.append(child)
            
            population = new_population[:len(population)]
            
            # Occasionally add diverse individuals
            if gen % 20 == 0 and len(population) > 0:
                diversification = [self.create_individual() for _ in range(2)]
                population.extend(diversification)
                population = population[:len(population)]
                
        # Return best individual
        final_fitness = [(ind, self.evaluate_fitness(ind)) for ind in population]
        final_fitness.sort(key=lambda x: x[1], reverse=True)
        return final_fitness[0][0]

    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Select individual using tournament selection."""
        tournament_size = 3
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]

    def local_refinement(self, points: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Refine point configuration using local hill climbing."""
        current_points = points.copy()
        current_ratio = self.evaluate_fitness(current_points)
        
        for _ in range(max_iterations):
            # Try small random perturbations
            new_points = current_points.copy()
            idx = random.randint(0, self.num_points - 1)
            # Small perturbation
            new_points[idx] += np.random.normal(0, 0.005, 2)
            new_points[idx] = np.clip(new_points[idx], 0.001, 0.999)
            
            new_ratio = self.evaluate_fitness(new_points)
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                
        return current_points

    def optimize(self, max_time: float = 170.0) -> np.ndarray:
        """Main optimization function."""
        start_time = time.time()
        
        # Initial population
        population_size = 20
        population = [self.create_individual() for _ in range(population_size)]
        
        # Evolve for specified time
        evolved_solution = self.evolve_population(
            population, 
            elite_size=3, 
            mutation_rate=0.15,
            generations=50
        )
        
        # Local refinement
        refined_solution = self.local_refinement(evolved_solution)
        
        # Further improvement through random restarts
        for _ in range(3):
            if time.time() - start_time > max_time - 2:
                break
            random_restart = self.create_individual()
            local_result = self.local_refinement(random_restart)
            if self.evaluate_fitness(local_result) > self.evaluate_fitness(refined_solution):
                refined_solution = local_result
        
        return refined_solution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Initialize optimizer
    optimizer = GridEvolutionOptimizer(num_points=16, grid_resolution=20)
    
    # Optimize
    points = optimizer.optimize(max_time=170.0)
    
    return points

# EVOLVE-BLOCK-END