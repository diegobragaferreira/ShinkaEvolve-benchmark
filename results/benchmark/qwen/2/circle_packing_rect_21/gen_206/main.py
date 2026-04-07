# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import random
from typing import Tuple, List
import time
import warnings
from math import sqrt, pi

# Global constants
RECT_PERIMETER = 4.0
NUM_CIRCLES = 21
POPULATION_SIZE = 100
MAX_GENERATIONS = 300
INITIAL_MUTATION_RATE = 0.25
FINAL_MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 3
SEED = 42

class HexagonEvolvePacker:
    def __init__(self, num_circles: int = NUM_CIRCLES):
        self.num_circles = num_circles
        self.hex_radius = 1.0  # Base radius for hexagonal packing
        
        # Initialize random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

    def hexagonal_lattice_points(self, width: float, height: float) -> np.ndarray:
        """Generate hexagonal lattice points covering the rectangle"""
        # Calculate hexagon parameters for optimal packing
        hex_width = self.hex_radius * 2
        hex_height = self.hex_radius * sqrt(3)
        
        # Number of hexagons needed to cover the area
        num_cols = int(ceil(width / hex_width)) + 2
        num_rows = int(ceil(height / hex_height)) + 2
        
        points = []
        for i in range(num_rows):
            for j in range(num_cols):
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * (hex_width / 2)
                x = x_offset + j * hex_width + self.hex_radius
                y = i * hex_height + self.hex_radius
                
                # Only add points within bounds
                if 0 <= x <= width and 0 <= y <= height:
                    points.append([x, y])
                    
        return np.array(points)

    def is_valid_position(self, x: float, y: float, r: float, width: float, height: float) -> bool:
        """Check if circle center is within bounds"""
        return (r <= x <= width - r and
                r <= y <= height - r)

    def is_valid_circle(self, x: float, y: float, r: float, width: float, height: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and
                self.is_valid_position(x, y, r, width, height))

    def check_overlap(self, circles: np.ndarray, idx1: int, idx2: int) -> bool:
        """Check if two circles overlap using Euclidean distance"""
        x1, y1, r1 = circles[idx1]
        x2, y2, r2 = circles[idx2]

        # Calculate squared distance to avoid sqrt computation
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        radius_sum = r1 + r2
        return dist_sq < radius_sum * radius_sum

    def efficient_overlap_check(self, circles: np.ndarray, tree: cKDTree = None) -> int:
        """Efficiently check all overlaps using spatial indexing"""
        violations = 0

        if tree is None:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

        # Get max radius to determine search radius
        max_radius = np.max(circles[:, 2])

        # Query pairs efficiently
        try:
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

            for i, j in pairs:
                if self.check_overlap(circles, i, j):
                    violations += 1
        except Exception:
            # Fallback to brute force if spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap(circles, i, j):
                        violations += 1

        return violations

    def calculate_total_radius_sum(self, circles: np.ndarray) -> float:
        """Calculate sum of all circle radii"""
        return np.sum(circles[:, 2])

    def calculate_fitness(self, circles: np.ndarray, width: float, height: float) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations

        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = self.calculate_total_radius_sum(circles)
        violations = 0

        # Check boundary violations
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r, width, height):
                violations += 100  # Heavy penalty for boundary violations

        # Check overlap violations using optimized spatial indexing
        violations += self.efficient_overlap_check(circles)

        # Return negative penalty (since we want to maximize) plus positive radius sum
        # Adjust penalty weight for better balance
        penalty_weight = 1000.0
        return total_radius - (penalty_weight * violations), violations

    def generate_hexagonal_initial_population(self, width: float, height: float, pop_size: int) -> List[np.ndarray]:
        """Generate initial population using hexagonal lattice approach"""
        population = []
        
        # Get hexagonal lattice points
        lattice_points = self.hexagonal_lattice_points(width, height)
        
        for _ in range(pop_size):
            circles = np.zeros((self.num_circles, 3))
            
            # Sample points from hexagonal lattice for circle centers
            selected_points = np.random.choice(len(lattice_points), self.num_circles, replace=False)
            
            for i, point_idx in enumerate(selected_points):
                x, y = lattice_points[point_idx]
                
                # Assign appropriate radius based on proximity to center and space available
                center_x, center_y = width / 2, height / 2
                dist_to_center = sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = sqrt((width/2)**2 + (height/2)**2)
                
                # Radius inversely proportional to distance from center (favor center)
                base_radius = 0.05 + 0.15 * (1.0 - dist_to_center / max_dist)
                
                # Ensure radius fits within bounds
                max_r = min(x, width - x, y, height - y)
                r = min(base_radius, max_r * 0.8)
                
                circles[i] = [x, y, r]
            
            population.append(circles)
        
        return population

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Select individual using tournament selection"""
        tournament_indices = np.random.choice(len(population), TOURNAMENT_SIZE)
        tournament_fitness = [(i, fitness_scores[i][0]) for i in tournament_indices]

        # Sort by fitness (descending)
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)

        return population[tournament_fitness[0][0]].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two parents"""
        child = parent1.copy()

        # For each circle, randomly inherit from either parent
        mask = np.random.rand(self.num_circles) > 0.5

        for i in range(self.num_circles):
            if mask[i]:
                child[i] = parent2[i].copy()

        return child

    def mutate(self, individual: np.ndarray, generation: int, max_generations: int, 
              width: float, height: float) -> np.ndarray:
        """Apply mutation to an individual with adaptive mutation rate"""
        # Adaptive mutation rate: decrease over generations
        adaptive_rate = INITIAL_MUTATION_RATE - (INITIAL_MUTATION_RATE - FINAL_MUTATION_RATE) * \
                       (generation / max_generations)

        mutated = individual.copy()

        for i in range(self.num_circles):
            if np.random.rand() < adaptive_rate:
                # Mutate either center position or radius
                if np.random.rand() < 0.5:
                    # Mutate position
                    delta_x = np.random.uniform(-0.1, 0.1)
                    delta_y = np.random.uniform(-0.1, 0.1)
                    mutated[i, 0] = np.clip(mutated[i, 0] + delta_x, 0.001, width - 0.001)
                    mutated[i, 1] = np.clip(mutated[i, 1] + delta_y, 0.001, height - 0.001)
                else:
                    # Mutate radius
                    delta_r = np.random.uniform(-0.03, 0.03)
                    mutated[i, 2] = np.clip(mutated[i, 2] + delta_r, 0.001, 0.2)

        return mutated

    def optimize(self) -> Tuple[np.ndarray, float, float]:
        """Main optimization loop using evolutionary algorithm with rectangle optimization"""
        # First, optimize rectangle dimensions to get the best aspect ratio
        def rectangle_objective(params):
            width, height = params
            if width + height != 2.0:  # Constraint: perimeter = 4 (so width + height = 2)
                return float('inf')
            
            # Generate population with these dimensions
            population = self.generate_hexagonal_initial_population(width, height, POPULATION_SIZE)
            
            # Run optimization for this configuration
            best_solution = None
            best_fitness = float('-inf')
            
            for generation in range(100):  # Short optimization for dimension tuning
                # Evaluate fitness for entire population
                fitness_scores = []
                for individual in population:
                    fitness, violations = self.calculate_fitness(individual, width, height)
                    fitness_scores.append((fitness, violations))

                # Track best solution in this generation
                gen_best_idx = np.argmax([f[0] for f in fitness_scores])
                gen_best_fitness = fitness_scores[gen_best_idx][0]

                if gen_best_fitness > best_fitness:
                    best_fitness = gen_best_fitness
                    best_solution = population[gen_best_idx].copy()

                # Create new population through selection, crossover, and mutation
                new_population = []
                new_population.append(best_solution.copy())

                while len(new_population) < POPULATION_SIZE:
                    parent1 = self.tournament_selection(population, fitness_scores)
                    parent2 = self.tournament_selection(population, fitness_scores)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child, generation, 100, width, height)
                    new_population.append(child)

                population = new_population

            return -best_fitness  # Negative because we minimize

        # Optimize rectangle dimensions
        bounds = [(0.5, 1.5), (0.5, 1.5)]  # Reasonable bounds for width and height
        result = differential_evolution(rectangle_objective, bounds, maxiter=20, popsize=10, seed=SEED)
        best_width, best_height = result.x
        
        # Ensure perimeter constraint
        if best_width + best_height != 2.0:
            best_height = 2.0 - best_width

        print(f"Optimized rectangle: width={best_width:.3f}, height={best_height:.3f}")
        
        # Now run full optimization with the optimized rectangle dimensions
        population = self.generate_hexagonal_initial_population(best_width, best_height, POPULATION_SIZE)
        
        best_solution = None
        best_fitness = float('-inf')
        fitness_history = []

        for generation in range(MAX_GENERATIONS):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, violations = self.calculate_fitness(individual, best_width, best_height)
                fitness_scores.append((fitness, violations))

            # Track best solution in this generation
            gen_best_idx = np.argmax([f[0] for f in fitness_scores])
            gen_best_fitness = fitness_scores[gen_best_idx][0]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()

            fitness_history.append(gen_best_fitness)

            # Print progress every 50 generations
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {gen_best_fitness:.6f}")

            # Create new population through selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, generation, MAX_GENERATIONS, best_width, best_height)
                new_population.append(child)

            population = new_population

            # Early stopping if converged
            if len(fitness_history) >= 15:
                recent_avg = np.mean(fitness_history[-15:])
                prev_avg = np.mean(fitness_history[-30:-15])
                if abs(recent_avg - prev_avg) < 1e-6:
                    print(f"Converged at generation {generation}")
                    break

        return best_solution, best_width, best_height

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create packer instance
    packer = HexagonEvolvePacker(num_circles=21)

    # Run optimization
    circles, width, height = packer.optimize()

    # Final refinement - try to improve individual radii
    for _ in range(50):
        improvement_made = False
        for i in range(21):
            original_circle = circles[i].copy()
            x, y, r = original_circle
            
            # Try to slightly increase radius while ensuring constraints
            new_r = min(r * 1.01, 0.25)
            test_circle = [x, y, new_r]
            
            # Check if valid and doesn't overlap
            if (test_circle[2] <= test_circle[0] <= width - test_circle[0] and 
                test_circle[2] <= test_circle[1] <= height - test_circle[1]):
                # Check overlap with others
                temp_circles = np.delete(circles, i, axis=0)
                overlap = False
                for cx, cy, cr in temp_circles:
                    dx = x - cx
                    dy = y - cy
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < (r + cr)**2:
                        overlap = True
                        break
                        
                if not overlap:
                    circles[i] = test_circle
                    improvement_made = True
                    
        if not improvement_made:
            break

    return circles

# Helper function to ceil
def ceil(x):
    return int(x) if x == int(x) else int(x) + 1

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")