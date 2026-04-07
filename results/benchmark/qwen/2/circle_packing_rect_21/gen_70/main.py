# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
import math
from joblib import Parallel, delayed

# Global constants for optimization
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.0
RECT_HEIGHT = 1.0
NUM_CIRCLES = 21
POPULATION_SIZE = 150
MAX_GENERATIONS = 1000
ELITE_SIZE = 15
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ADAPTIVE_MUTATION_START = 0.2
ADAPTIVE_MUTATION_END = 0.05
SEED = 42

class CirclePacker:
    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.rect_area = width * height
        
        # Initialize random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

    def is_valid_circle(self, x: float, y: float, r: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and 
                r <= x <= self.width - r and
                r <= y <= self.height - r)

    def check_overlap_fast(self, circles: np.ndarray, tree: cKDTree = None) -> int:
        """Fast overlap checking using spatial indexing"""
        if tree is None:
            points = circles[:, :2]
            tree = cKDTree(points)
        
        # Find all pairs within 2 * max_radius
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
        
        violations = 0
        for i, j in pairs:
            if i < j:  # Avoid duplicate checks
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                radius_sum = r1 + r2
                if dist_sq < radius_sum * radius_sum:
                    violations += 1
        return violations

    def calculate_total_radius_sum(self, circles: np.ndarray) -> float:
        """Calculate sum of all circle radii"""
        return np.sum(circles[:, 2])

    def evaluate_fitness(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations

        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = self.calculate_total_radius_sum(circles)
        
        # Count constraint violations
        violations = 0
        
        # Check boundary violations
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                violations += 1000

        # Check overlap violations using optimized spatial indexing
        violations += self.check_overlap_fast(circles)

        # Return fitness with penalty
        penalty_weight = 10000.0
        return total_radius - (penalty_weight * violations), violations

    def generate_initial_solution_hexagonal(self) -> np.ndarray:
        """Generate initial solution using hexagonal packing pattern."""
        circles = np.zeros((self.num_circles, 3))
        
        # Use hexagonal packing for better initial distribution
        rows = int(np.ceil(np.sqrt(self.num_circles)))
        cols = int(np.ceil(self.num_circles / rows))
        
        # Hexagonal packing parameters
        cell_width = self.width / cols
        cell_height = self.height / rows
        hex_radius = min(cell_width, cell_height) * 0.3
        
        # Place circles in a hexagonal pattern
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.num_circles:
                    break
                    
                # Hexagonal offset
                offset = (i % 2) * (cell_width / 2)
                x = (j + 0.5) * cell_width + offset + (random.random() - 0.5) * cell_width * 0.2
                y = (i + 0.5) * cell_height + (random.random() - 0.5) * cell_height * 0.2
                
                # Ensure valid bounds
                x = max(hex_radius, min(self.width - hex_radius, x))
                y = max(hex_radius, min(self.height - hex_radius, y))
                
                # Initial radius based on available space
                min_radius = min(x, self.width - x, y, self.height - y)
                radius = min(min_radius * 0.5, hex_radius)
                
                circles[idx] = [x, y, radius]
                idx += 1
                
        # Refine using local optimization to ensure no overlaps
        circles = self.refine_positions(circles)
        
        return circles

    def generate_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate initial population of circle configurations"""
        population = []
        
        # Generate diverse initial solutions
        for _ in range(pop_size):
            circles = self.generate_initial_solution_hexagonal()
            population.append(circles)
            
        return population

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Select individual using tournament selection with adaptive size"""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitness = [(i, fitness_scores[i][0]) for i in tournament_indices]

        # Sort by fitness (descending)
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)

        return population[tournament_fitness[0][0]].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover between two parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # For each circle, randomly inherit from either parent
        mask = np.random.rand(self.num_circles) > 0.5

        for i in range(self.num_circles):
            if mask[i]:
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()

        return child1, child2

    def mutate(self, individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply mutation to an individual with adaptive mutation rate"""
        # Adaptive mutation rate: decrease over generations
        adaptive_rate = ADAPTIVE_MUTATION_START - (ADAPTIVE_MUTATION_START - ADAPTIVE_MUTATION_END) * \
                       (generation / max_generations)
        
        mutated = individual.copy()

        for i in range(self.num_circles):
            if np.random.rand() < adaptive_rate:
                # Mutate either center position or radius
                if np.random.rand() < 0.6:  # 60% chance to mutate position
                    # Mutate position with more substantial changes
                    mutated[i, 0] += (random.random() - 0.5) * 0.15 * self.width
                    mutated[i, 1] += (random.random() - 0.5) * 0.15 * self.height
                    
                    # Keep within bounds
                    mutated[i, 0] = max(0.001, min(self.width - 0.001, mutated[i, 0]))
                    mutated[i, 1] = max(0.001, min(self.height - 0.001, mutated[i, 1]))
                else:
                    # Mutate radius with more aggressive changes
                    mutated[i, 2] *= random.uniform(0.6, 1.4)
                    
                    # Ensure positive radius
                    mutated[i, 2] = max(0.001, mutated[i, 2])

        return mutated

    def refine_positions(self, circles: np.ndarray) -> np.ndarray:
        """Refine positions to ensure no overlaps and respect boundaries."""
        # Use a more sophisticated iterative improvement approach
        for iter_count in range(200):  # Limited iterations for performance
            updated = False
            # Try to increase radii while maintaining no overlaps
            for i in range(len(circles)):
                x, y, r = circles[i]
                
                # Calculate maximum possible radius at this location
                max_r = min(x, self.width - x, y, self.height - y)
                
                # Check for overlap with other circles
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x2 - x)**2 + (y2 - y)**2)
                        
                        # Can't get closer than sum of radii
                        if distance < (r + r2):
                            max_r = min(max_r, distance - r2)
                
                # Increase radius if beneficial and safe
                if max_r > r and max_r > 0.001:
                    # Use a more conservative increase to maintain stability
                    new_r = min(max_r, r * 1.02)
                    if new_r > r + 0.0001:
                        circles[i][2] = new_r
                        updated = True
            
            if not updated:
                break
                
        return circles

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm"""
        start_time = time.time()

        # Generate initial population
        population = self.generate_initial_population(POPULATION_SIZE)

        best_solution = None
        best_fitness = float('-inf')
        fitness_history = []

        # Track convergence
        stagnant_count = 0
        prev_best = float('-inf')

        for generation in range(MAX_GENERATIONS):
            # Evaluate fitness for entire population (parallelized for performance)
            def evaluate_individual(individual):
                return self.evaluate_fitness(individual)
            
            fitness_results = Parallel(n_jobs=-1)(delayed(evaluate_individual)(individual) 
                                                for individual in population)
            fitness_scores = [result[0] for result in fitness_results]
            
            # Update best solution
            max_fitness_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[max_fitness_idx]
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[max_fitness_idx].copy()
                stagnant_count = 0  # Reset stagnation counter
            else:
                stagnant_count += 1
                
            fitness_history.append(current_best_fitness)
            
            # Print progress every 50 generations
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {current_best_fitness:.6f}")

            # Early stopping if converged
            if stagnant_count > 100:
                print(f"Early stopping at generation {generation} due to no improvement")
                break

            # Create new population through selection, crossover, and mutation
            new_population = []

            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-ELITE_SIZE:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = self.tournament_selection(population, [(fs, 0) for fs in fitness_scores])
                parent2 = self.tournament_selection(population, [(fs, 0) for fs in fitness_scores])

                # Crossover
                if random.random() < CROSSOVER_RATE:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self.mutate(child1, generation, MAX_GENERATIONS)
                child2 = self.mutate(child2, generation, MAX_GENERATIONS)
                
                # Refine positions to handle constraint violations
                child1 = self.refine_positions(child1)
                child2 = self.refine_positions(child2)

                new_population.extend([child1, child2])
                
            population = new_population[:POPULATION_SIZE]

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness achieved: {best_fitness:.6f}")

        return best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create packer instance with rectangle dimensions
    packer = CirclePacker(width=1.0, height=1.0, num_circles=21)

    # Run optimization
    circles = packer.optimize()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
