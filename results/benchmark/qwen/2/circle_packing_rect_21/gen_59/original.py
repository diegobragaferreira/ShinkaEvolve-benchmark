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

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.0  # Default rectangle dimensions (width=1, height=1)
RECT_HEIGHT = 1.0
NUM_CIRCLES = 21
POPULATION_SIZE = 50
MAX_GENERATIONS = 200
INITIAL_MUTATION_RATE = 0.15
FINAL_MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 3
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

    def is_valid_position(self, x: float, y: float, r: float) -> bool:
        """Check if circle center is within bounds"""
        return (r <= x <= self.width - r and
                r <= y <= self.height - r)

    def is_valid_circle(self, x: float, y: float, r: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and
                self.is_valid_position(x, y, r))

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

    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, int]:
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
                violations += 100  # Heavy penalty for boundary violations

        # Check overlap violations using optimized spatial indexing
        violations += self.efficient_overlap_check(circles)

        # Return negative penalty (since we want to maximize) plus positive radius sum
        # Adjust penalty weight for better balance
        penalty_weight = 1000.0
        return total_radius - (penalty_weight * violations), violations

    def generate_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate initial population of circle configurations"""
        population = []

        for _ in range(pop_size):
            circles = np.zeros((self.num_circles, 3))

            # Generate initial configuration using a more systematic approach
            # First, try to place circles using a grid-like pattern to ensure feasibility
            # Then add some randomness

            # Create initial grid-based configuration
            grid_size = int(np.ceil(np.sqrt(self.num_circles)))
            spacing_x = self.width / (grid_size + 1)
            spacing_y = self.height / (grid_size + 1)

            for i in range(self.num_circles):
                # Grid position
                row = i // grid_size
                col = i % grid_size

                # Base position
                base_x = (col + 1) * spacing_x
                base_y = (row + 1) * spacing_y

                # Add small random perturbation
                perturbation = 0.1 * min(spacing_x, spacing_y)
                x = np.clip(base_x + np.random.uniform(-perturbation, perturbation),
                           0.01, self.width - 0.01)
                y = np.clip(base_y + np.random.uniform(-perturbation, perturbation),
                           0.01, self.height - 0.01)

                # Initial radius based on available space
                max_r = min(x, self.width - x, y, self.height - y)
                r = np.random.uniform(0.01, min(max_r * 0.8, 0.15))

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

    def mutate(self, individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
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
                    mutated[i, 0] = np.random.uniform(
                        max(0.001, mutated[i, 0] - 0.05),
                        min(self.width - 0.001, mutated[i, 0] + 0.05)
                    )
                    mutated[i, 1] = np.random.uniform(
                        max(0.001, mutated[i, 1] - 0.05),
                        min(self.height - 0.001, mutated[i, 1] + 0.05)
                    )
                else:
                    # Mutate radius
                    mutated[i, 2] = np.random.uniform(
                        max(0.001, mutated[i, 2] - 0.02),
                        min(0.2, mutated[i, 2] + 0.02)
                    )

        return mutated

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm"""
        start_time = time.time()

        # Generate initial population
        population = self.generate_initial_population(POPULATION_SIZE)

        best_solution = None
        best_fitness = float('-inf')

        # Track convergence
        fitness_history = []

        for generation in range(MAX_GENERATIONS):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, violations = self.calculate_fitness(individual)
                fitness_scores.append((fitness, violations))

            # Track best solution in this generation
            gen_best_idx = np.argmax([f[0] for f in fitness_scores])
            gen_best_fitness = fitness_scores[gen_best_idx][0]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()

            fitness_history.append(gen_best_fitness)

            # Print progress every 20 generations
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {gen_best_fitness:.6f}")

            # Create new population through selection, crossover, and mutation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation with adaptive rate
                child = self.mutate(child, generation, MAX_GENERATIONS)

                new_population.append(child)

            population = new_population

            # Early stopping if converged
            if len(fitness_history) >= 10:
                recent_avg = np.mean(fitness_history[-10:])
                prev_avg = np.mean(fitness_history[-20:-10])
                if abs(recent_avg - prev_avg) < 1e-6:
                    print(f"Converged at generation {generation}")
                    break

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