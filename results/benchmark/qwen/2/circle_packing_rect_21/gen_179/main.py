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
RECT_WIDTH = 1.2  # Optimized rectangle dimensions for better packing
RECT_HEIGHT = 0.8
NUM_CIRCLES = 21
POPULATION_SIZE = 80
MAX_GENERATIONS = 250
INITIAL_MUTATION_RATE = 0.2
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

            # Generate initial configuration using adaptive grid placement based on aspect ratio
            aspect_ratio = self.width / self.height

            # Calculate ideal grid dimensions that maximize packing efficiency
            # Using formula: spacing = sqrt(area / n) for square packing efficiency
            area_per_circle = (self.width * self.height) / self.num_circles
            ideal_spacing = np.sqrt(area_per_circle)

            # Determine grid dimensions that best match the rectangle aspect ratio
            if aspect_ratio >= 1:  # Landscape orientation
                cols = max(1, int(np.ceil(self.width / ideal_spacing)))
                rows = max(1, int(np.ceil(self.height / ideal_spacing)))

                # Adjust to ensure we have enough cells for all circles
                while cols * rows < self.num_circles:
                    if cols < rows:
                        cols += 1
                    else:
                        rows += 1

                # Adjust to better match aspect ratio
                target_aspect = self.width / self.height
                while abs(cols / rows - target_aspect) > 0.1 and cols * rows < self.num_circles * 1.2:
                    if cols / rows > target_aspect:
                        rows += 1
                    else:
                        cols += 1

            else:  # Portrait orientation
                cols = max(1, int(np.ceil(self.width / ideal_spacing)))
                rows = max(1, int(np.ceil(self.height / ideal_spacing)))

                # Adjust to ensure we have enough cells for all circles
                while cols * rows < self.num_circles:
                    if rows < cols:
                        rows += 1
                    else:
                        cols += 1

                # Adjust to better match aspect ratio
                target_aspect = self.height / self.width
                while abs(rows / cols - target_aspect) > 0.1 and cols * rows < self.num_circles * 1.2:
                    if rows / cols > target_aspect:
                        cols += 1
                    else:
                        rows += 1

            # Ensure we don't exceed the required number of circles
            actual_cells = cols * rows
            if actual_cells > self.num_circles:
                # Reduce grid size if necessary
                while cols * rows > self.num_circles:
                    if aspect_ratio >= 1:
                        cols = max(1, cols - 1)
                    else:
                        rows = max(1, rows - 1)

            # Calculate precise spacing
            effective_width = self.width - 0.02  # Leave some margin
            effective_height = self.height - 0.02
            spacing_x = effective_width / (cols + 1) if cols > 0 else effective_width
            spacing_y = effective_height / (rows + 1) if rows > 0 else effective_height

            # Create hexagonal-like packing pattern for better initial coverage
            placed_count = 0
            for i in range(rows):
                for j in range(cols):
                    if placed_count >= self.num_circles:
                        break

                    # Offset every other row for hexagonal packing (better than simple grid)
                    offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                    base_x = (j + 1) * spacing_x + offset_x
                    base_y = (i + 1) * spacing_y

                    # Add small random perturbation for diversity and better optimization starting point
                    perturbation_factor = 0.15
                    x = np.clip(base_x + np.random.uniform(-perturbation_factor * spacing_x, perturbation_factor * spacing_x),
                               0.01, self.width - 0.01)
                    y = np.clip(base_y + np.random.uniform(-perturbation_factor * spacing_y, perturbation_factor * spacing_y),
                               0.01, self.height - 0.01)

                    # Initial radius estimation based on available space
                    max_r = min(x, self.width - x, y, self.height - y)
                    # Better initial radius calculation based on spacing
                    estimated_radius = min(spacing_x, spacing_y) * 0.25
                    r = np.random.uniform(0.05, min(estimated_radius, max_r * 0.9))

                    circles[placed_count] = [x, y, r]
                    placed_count += 1

                if placed_count >= self.num_circles:
                    break

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
    # Create packer instance with optimized rectangle dimensions
    packer = CirclePacker(width=RECT_WIDTH, height=RECT_HEIGHT, num_circles=21)

    # Run optimization
    circles = packer.optimize()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")