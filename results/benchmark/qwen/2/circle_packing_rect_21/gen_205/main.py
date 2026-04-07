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
from dataclasses import dataclass

# Global constants
RECT_PERIMETER = 4.0
NUM_CIRCLES = 21
POPULATION_SIZE = 50
MAX_GENERATIONS = 200
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
SEED = 42
# Try multiple aspect ratios to find optimal packing
ASPECT_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0]

@dataclass
class Circle:
    x: float
    y: float
    r: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.r])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Circle':
        return cls(arr[0], arr[1], arr[2])

class ConstraintValidator:
    """Validates circle configurations against geometric constraints."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

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

    def validate_all_constraints(self, circles: np.ndarray) -> Tuple[int, List[int]]:
        """
        Validate all constraints and return violation count and indices

        Returns:
            Tuple of (violation_count, list_of_violating_pairs)
        """
        violations = 0
        violating_pairs = []

        # Check boundary violations
        for i in range(len(circles)):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                violations += 100  # Heavy penalty for boundary violations

        # Check overlap violations using spatial indexing for efficiency
        try:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

            # Find neighbors within 2*max_radius distance (optimization)
            max_radius = np.max(circles[:, 2])
            if max_radius > 0:
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

                for i, j in pairs:
                    if self.check_overlap(circles, i, j):
                        violations += 1
                        violating_pairs.append((i, j))

        except Exception as e:
            warnings.warn(f"Error in overlap checking: {e}")
            # Fallback to brute force when spatial indexing fails
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    if self.check_overlap(circles, i, j):
                        violations += 1
                        violating_pairs.append((i, j))

        return violations, violating_pairs

class FitnessEvaluator:
    """Evaluates fitness of circle configurations."""

    def __init__(self, validator: ConstraintValidator):
        self.validator = validator

    def evaluate(self, circles: np.ndarray) -> Tuple[float, int]:
        """
        Calculate fitness: sum of radii with penalty for constraint violations
        Uses adaptive penalty based on violation severity

        Returns:
            Tuple of (fitness_score, number_of_violations)
        """
        total_radius = np.sum(circles[:, 2])

        # Validate constraints
        violations, violating_pairs = self.validator.validate_all_constraints(circles)

        # Apply adaptive penalty: more severe penalty for boundary violations
        # and progressively harsher penalties for overlapping violations
        penalty = 0
        if violations > 0:
            # Separate boundary and overlap violations
            boundary_violations = sum(1 for v in violating_pairs if v[0] < 0 or v[1] < 0)  # Simplified counting
            overlap_violations = len(violating_pairs) - boundary_violations

            # Boundary violations get higher penalty (heavier penalty per violation)
            penalty += boundary_violations * 1000.0

            # Overlap violations get moderate penalty (lighter penalty per violation)
            penalty += overlap_violations * 500.0

            # Additional penalty based on how severely constraints are violated
            # This encourages movement away from boundary violations
            violation_severity = 0
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Measure how far inside bounds the circle is
                left_dist = x - r
                right_dist = (self.validator.width - x) - r
                bottom_dist = y - r
                top_dist = (self.validator.height - y) - r

                # The more negative these distances, the more severe the boundary violation
                violation_severity += max(0, -left_dist) + max(0, -right_dist) + \
                                    max(0, -bottom_dist) + max(0, -top_dist)

            penalty += violation_severity * 100.0

        # Return fitness score (higher is better)
        return total_radius - penalty, violations

class Initializer:
    """Generates initial configurations for circle packing."""

    def __init__(self, width: float, height: float, num_circles: int):
        self.width = width
        self.height = height
        self.num_circles = num_circles

    def generate_adaptive_grid(self) -> np.ndarray:
        """Create initial configuration using adaptive grid placement"""
        circles = np.zeros((self.num_circles, 3))

        # Calculate grid dimensions based on number of circles
        rows = int(np.ceil(np.sqrt(self.num_circles)))
        cols = int(np.ceil(self.num_circles / rows))

        # Ensure we don't exceed the actual number of circles
        actual_circles = rows * cols
        if actual_circles > self.num_circles:
            rows = int(np.ceil(self.num_circles / cols))

        # Calculate spacing
        margin = 0.05  # Leave some margin around edges
        cell_width = (self.width - 2 * margin) / cols
        cell_height = (self.height - 2 * margin) / rows

        # Use minimum of cell dimensions as base for radius
        base_radius = min(cell_width, cell_height) * 0.4

        # Place circles using grid pattern
        circle_idx = 0
        for i in range(rows):
            for j in range(cols):
                if circle_idx >= self.num_circles:
                    break

                # Position circle at center of grid cell with slight randomization
                x = margin + (j + 0.5) * cell_width + np.random.uniform(-0.1 * cell_width, 0.1 * cell_width)
                y = margin + (i + 0.5) * cell_height + np.random.uniform(-0.1 * cell_height, 0.1 * cell_height)

                # Radius slightly randomized around base value
                r = base_radius * (0.8 + np.random.uniform(0, 0.4))

                # Ensure circle fits within bounds
                x = np.clip(x, r, self.width - r)
                y = np.clip(y, r, self.height - r)

                circles[circle_idx] = [x, y, r]
                circle_idx += 1

            if circle_idx >= self.num_circles:
                break

        # Ensure minimum radius and fix any constraint violations
        for i in range(self.num_circles):
            if circles[i, 2] < 0.001:
                circles[i, 2] = 0.001

        return circles

class EvolutionaryOptimizer:
    """Performs evolutionary optimization of circle configurations."""

    def __init__(self,
                 width: float,
                 height: float,
                 num_circles: int,
                 population_size: int = POPULATION_SIZE,
                 max_generations: int = MAX_GENERATIONS,
                 mutation_rate: float = MUTATION_RATE):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate

        # Initialize components
        self.validator = ConstraintValidator(width, height)
        self.evaluator = FitnessEvaluator(self.validator)
        self.initializer = Initializer(width, height, num_circles)

        # Set random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

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

    def mutate(self, individual: np.ndarray, current_radius_sum: float) -> np.ndarray:
        """Apply mutation to an individual with adaptive parameters"""
        mutated = individual.copy()

        # Adaptive mutation based on solution quality
        adaptive_mutation_rate = self.mutation_rate * (1.0 + 0.1 * np.log(current_radius_sum + 1))

        for i in range(self.num_circles):
            if np.random.rand() < adaptive_mutation_rate:
                # Mutate either center position or radius
                if np.random.rand() < 0.5:
                    # Mutate position - use adaptive step size
                    step_x = 0.02 * (1.0 + 0.01 * current_radius_sum)
                    step_y = 0.02 * (1.0 + 0.01 * current_radius_sum)
                    mutated[i, 0] = np.random.uniform(
                        max(0.001, mutated[i, 0] - step_x),
                        min(self.width - 0.001, mutated[i, 0] + step_x)
                    )
                    mutated[i, 1] = np.random.uniform(
                        max(0.001, mutated[i, 1] - step_y),
                        min(self.height - 0.001, mutated[i, 1] + step_y)
                    )
                else:
                    # Mutate radius with adaptive range
                    step_r = 0.01 * (1.0 + 0.005 * current_radius_sum)
                    mutated[i, 2] = np.random.uniform(
                        max(0.001, mutated[i, 2] - step_r),
                        min(0.2, mutated[i, 2] + step_r)
                    )

        return mutated

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm"""
        start_time = time.time()

        # Generate initial population
        population = []
        for _ in range(self.population_size):
            circles = self.initializer.generate_adaptive_grid()
            population.append(circles)

        best_solution = None
        best_fitness = float('-inf')
        fitness_history = []

        for generation in range(self.max_generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness, violations = self.evaluator.evaluate(individual)
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
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                current_radius_sum = np.sum(child[:, 2])
                child = self.mutate(child, current_radius_sum)

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

class CirclePacker:
    """Main circle packing class orchestrating the optimization process."""

    def __init__(self, width: float = 1.0, height: float = 1.0,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles

    def optimize_with_multiple_aspect_ratios(self) -> np.ndarray:
        """Try multiple aspect ratios and return the best solution."""
        best_solution = None
        best_radius_sum = -float('inf')

        print(f"Testing {len(ASPECT_RATIOS)} different aspect ratios...")

        for i, aspect_ratio in enumerate(ASPECT_RATIOS):
            # Calculate width and height based on perimeter constraint
            # width + height = 2, so if width/height = aspect_ratio, then:
            # width = aspect_ratio * height, and width + height = 2
            # Therefore: aspect_ratio * height + height = 2 => height = 2 / (1 + aspect_ratio)
            height = 2.0 / (1.0 + aspect_ratio)
            width = aspect_ratio * height

            print(f"Testing aspect ratio {aspect_ratio}: width={width:.3f}, height={height:.3f}")

            # Create optimizer with current dimensions
            optimizer = EvolutionaryOptimizer(width, height, self.num_circles)

            # Run optimization
            solution = optimizer.optimize()

            # Check if this solution is better
            radius_sum = np.sum(solution[:, 2])
            if radius_sum > best_radius_sum:
                best_radius_sum = radius_sum
                best_solution = solution.copy()
                print(f"New best found with aspect ratio {aspect_ratio}: radius sum = {radius_sum:.6f}")

        return best_solution

    def optimize(self) -> np.ndarray:
        """Run the optimization process and return the best solution."""
        return self.optimize_with_multiple_aspect_ratios()

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