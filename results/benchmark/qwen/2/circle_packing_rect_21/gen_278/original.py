# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import random
from typing import Tuple, List, Optional
import time
import warnings
from dataclasses import dataclass
from enum import Enum

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.3333333333333333  # 2/3
RECT_HEIGHT = 0.6666666666666666  # 1/3
NUM_CIRCLES = 21
SEED = 42

class OptimizationPhase(Enum):
    INITIALIZATION = "initialization"
    LOCAL_REFINEMENT = "local_refinement"
    EVOLUTIONARY = "evolutionary"

@dataclass
class OptimizationResult:
    circles: np.ndarray
    fitness: float
    phase: OptimizationPhase
    time_taken: float

class CircleInitializer:
    """Handles initialization of circle configurations"""

    @staticmethod
    def hexagonal_initialize(width: float, height: float, n_circles: int) -> np.ndarray:
        """Initialize circles using hexagonal packing for better density"""
        circles = np.zeros((n_circles, 3))

        # Calculate optimal grid dimensions
        rows = int(np.ceil(np.sqrt(n_circles * 1.5)))
        cols = int(np.ceil(n_circles / rows))

        # Ensure minimum grid size
        rows = max(rows, 3)
        cols = max(cols, 3)

        # Hexagonal grid spacing
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)

        # Hexagon packing factor
        hex_spacing_x = spacing_x * 0.75
        hex_spacing_y = spacing_y * 0.866  # sqrt(3)/2

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * (hex_spacing_x / 2)
                x = hex_spacing_x * j + x_offset + hex_spacing_x
                y = hex_spacing_y * i + hex_spacing_y

                # Ensure within bounds with safety margin
                x = max(hex_spacing_x, min(width - hex_spacing_x, x))
                y = max(hex_spacing_y, min(height - hex_spacing_y, y))

                # Initialize with a reasonable starting radius
                r = min(hex_spacing_x, hex_spacing_y) * 0.3

                circles[idx] = [x, y, r]
                idx += 1
                if idx >= n_circles:
                    break

        # Fill remaining slots with strategic random positions
        if idx < n_circles:
            for i in range(idx, n_circles):
                x = np.random.uniform(hex_spacing_x * 1.2, width - hex_spacing_x * 1.2)
                y = np.random.uniform(hex_spacing_y * 1.2, height - hex_spacing_y * 1.2)
                r = min(hex_spacing_x, hex_spacing_y) * 0.25
                circles[i] = [x, y, r]

        return circles

    @staticmethod
    def random_initialize(width: float, height: float, n_circles: int) -> np.ndarray:
        """Generate random initial configuration"""
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            # Random position
            x = np.random.uniform(0.01, width - 0.01)
            y = np.random.uniform(0.01, height - 0.01)

            # Random radius
            r = np.random.uniform(0.01, min(0.2, width/10, height/10))

            circles[i] = [x, y, r]
        return circles

class CircleValidator:
    """Handles validation of circle configurations"""

    @staticmethod
    def is_valid_position(x: float, y: float, r: float, width: float, height: float) -> bool:
        """Check if circle center is within bounds"""
        return (r <= x <= width - r and r <= y <= height - r)

    @staticmethod
    def is_valid_circle(x: float, y: float, r: float, width: float, height: float) -> bool:
        """Check if circle is valid (within bounds and positive radius)"""
        return (0 < r and CircleValidator.is_valid_position(x, y, r, width, height))

    @staticmethod
    def check_overlap(circles: np.ndarray, idx1: int, idx2: int) -> bool:
        """Check if two circles overlap using Euclidean distance"""
        x1, y1, r1 = circles[idx1]
        x2, y2, r2 = circles[idx2]

        # Calculate squared distance to avoid sqrt computation
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        radius_sum = r1 + r2
        return dist_sq < radius_sum * radius_sum

class ConstraintManager:
    """Manages constraint enforcement and penalty calculation"""

    def __init__(self, width: float, height: float, num_circles: int):
        self.width = width
        self.height = height
        self.num_circles = num_circles

    def calculate_violations(self, circles: np.ndarray) -> Tuple[int, int]:
        """Calculate number of boundary and overlap violations"""
        boundary_violations = 0
        overlap_violations = 0

        # Check boundary violations
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not CircleValidator.is_valid_circle(x, y, r, self.width, self.height):
                boundary_violations += 1

        # Check overlap violations using efficient spatial indexing
        try:
            points = circles[:, :2]
            tree = cKDTree(points)
            max_radius = np.max(circles[:, 2])
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

            for i, j in pairs:
                if CircleValidator.check_overlap(circles, i, j):
                    overlap_violations += 1

        except Exception:
            # Fallback to brute force for edge cases
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if CircleValidator.check_overlap(circles, i, j):
                        overlap_violations += 1

        return boundary_violations, overlap_violations

    def is_feasible(self, circles: np.ndarray) -> bool:
        """Check if configuration satisfies all constraints"""
        boundary_violations, overlap_violations = self.calculate_violations(circles)
        return boundary_violations == 0 and overlap_violations == 0

    def calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate fitness (sum of radii) for feasible configurations"""
        if self.is_feasible(circles):
            # Feasible solution - return sum of radii
            return np.sum(circles[:, 2])
        else:
            # Infeasible solution - return negative penalty
            boundary_violations, overlap_violations = self.calculate_violations(circles)
            # Heavy penalty for constraint violations
            penalty = (boundary_violations * 1000 + overlap_violations * 500)
            return np.sum(circles[:, 2]) - penalty

class LocalOptimizer:
    """Handles local optimization of individual circles"""

    @staticmethod
    def optimize_single_circle(circles: np.ndarray, idx: int, width: float, height: float,
                              constraint_manager: ConstraintManager) -> bool:
        """Try to improve a single circle's position and radius"""
        old_x, old_y, old_r = circles[idx]
        best_x, best_y, best_r = old_x, old_y, old_r

        # Calculate maximum possible radius at current position
        max_radius = min(old_x, width - old_x, old_y, height - old_y)
        if max_radius <= old_r:
            return False  # Cannot improve radius

        # Try different radius values within feasible range
        radius_attempts = np.linspace(old_r, min(max_radius, 0.3), 15)
        improved = False

        for test_r in radius_attempts:
            # Try nearby positions with small perturbations
            for _ in range(15):
                offset_x = np.random.uniform(-0.02, 0.02)
                offset_y = np.random.uniform(-0.02, 0.02)

                test_x = old_x + offset_x
                test_y = old_y + offset_y

                # Keep within bounds
                test_x = np.clip(test_x, test_r, width - test_r)
                test_y = np.clip(test_y, test_r, height - test_r)

                # Check if this new configuration is valid
                if CircleValidator.is_valid_circle(test_x, test_y, test_r, width, height):
                    # Check overlap with all other circles
                    valid = True
                    for i in range(len(circles)):
                        if i != idx:
                            x2, y2, r2 = circles[i]
                            dx = test_x - x2
                            dy = test_y - y2
                            dist_sq = dx*dx + dy*dy
                            if dist_sq < (test_r + r2) * (test_r + r2):
                                valid = False
                                break

                    if valid and test_r > best_r:
                        best_r = test_r
                        best_x = test_x
                        best_y = test_y
                        improved = True

        # Update if improved
        if improved:
            circles[idx] = [best_x, best_y, best_r]

        return improved

class MultiPhaseOptimizer:
    """Main optimizer implementing multi-phase approach"""

    def __init__(self, width: float = RECT_WIDTH, height: float = RECT_HEIGHT,
                 num_circles: int = NUM_CIRCLES):
        self.width = width
        self.height = height
        self.num_circles = num_circles
        self.constraint_manager = ConstraintManager(width, height, num_circles)

        # Initialize random seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

    def phase_initialization(self) -> np.ndarray:
        """Phase 1: Hexagonal grid-based initialization"""
        circles = CircleInitializer.hexagonal_initialize(self.width, self.height, self.num_circles)
        return circles

    def phase_local_refinement(self, circles: np.ndarray) -> np.ndarray:
        """Phase 2: Local optimization and refinement"""
        # Perform multiple iterations of local optimization
        for iteration in range(100):
            improved = False
            # Shuffle indices for better exploration
            indices = list(range(self.num_circles))
            np.random.shuffle(indices)

            for i in indices:
                if LocalOptimizer.optimize_single_circle(circles, i, self.width, self.height,
                                                       self.constraint_manager):
                    improved = True

            if not improved:
                break

        # Final constraint correction
        for i in range(self.num_circles):
            x, y, r = circles[i]
            # Correct boundary violations
            circles[i, 0] = np.clip(x, r, self.width - r)
            circles[i, 1] = np.clip(y, r, self.height - r)

        return circles

    def phase_evolutionary(self, initial_circles: np.ndarray) -> np.ndarray:
        """Phase 3: Evolutionary optimization with careful constraint handling"""
        # Population-based evolutionary algorithm
        population_size = 50
        max_generations = 200
        elite_size = 10

        # Generate initial population
        population = [initial_circles.copy()]
        for _ in range(population_size - 1):
            # Create perturbed version of initial solution
            mutant = initial_circles.copy()
            for i in range(self.num_circles):
                if np.random.random() < 0.3:  # 30% mutation rate
                    # Add small random perturbations
                    mutant[i, 0] += np.random.uniform(-0.03, 0.03)
                    mutant[i, 1] += np.random.uniform(-0.03, 0.03)
                    mutant[i, 2] += np.random.uniform(-0.01, 0.01)

                    # Keep within bounds
                    mutant[i, 0] = np.clip(mutant[i, 0], mutant[i, 2], self.width - mutant[i, 2])
                    mutant[i, 1] = np.clip(mutant[i, 1], mutant[i, 2], self.height - mutant[i, 2])
                    mutant[i, 2] = np.clip(mutant[i, 2], 0.001, 0.3)

            population.append(mutant)

        # Evolutionary loop
        for generation in range(max_generations):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                fitness = self.constraint_manager.calculate_fitness(individual)
                fitness_scores.append(fitness)

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            # Keep elite individuals
            new_population = population[:elite_size]

            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 5
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]

                # Select second parent
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent2_idx = tournament_indices[np.argmax(tournament_fitness)]

                # Crossover (uniform)
                child = population[parent1_idx].copy()
                for i in range(self.num_circles):
                    if np.random.random() < 0.5:
                        child[i] = population[parent2_idx][i].copy()

                # Mutation
                for i in range(self.num_circles):
                    if np.random.random() < 0.2:  # 20% mutation rate
                        child[i, 0] += np.random.uniform(-0.015, 0.015)
                        child[i, 1] += np.random.uniform(-0.015, 0.015)
                        child[i, 2] += np.random.uniform(-0.008, 0.008)

                        # Keep within bounds
                        child[i, 0] = np.clip(child[i, 0], child[i, 2], self.width - child[i, 2])
                        child[i, 1] = np.clip(child[i, 1], child[i, 2], self.height - child[i, 2])
                        child[i, 2] = np.clip(child[i, 2], 0.001, 0.3)

                new_population.append(child)

            population = new_population[:population_size]

        # Return best solution from final population
        final_fitness = [self.constraint_manager.calculate_fitness(ind) for ind in population]
        best_idx = np.argmax(final_fitness)
        return population[best_idx]

    def optimize(self) -> OptimizationResult:
        """Execute complete optimization pipeline"""
        start_time = time.time()

        # Phase 1: Initialization
        circles = self.phase_initialization()

        # Phase 2: Local refinement
        circles = self.phase_local_refinement(circles)

        # Phase 3: Evolutionary optimization
        circles = self.phase_evolutionary(circles)

        # Final fitness calculation
        fitness = self.constraint_manager.calculate_fitness(circles)

        end_time = time.time()
        time_taken = end_time - start_time

        return OptimizationResult(
            circles=circles,
            fitness=fitness,
            phase=OptimizationPhase.EVOLUTIONARY,
            time_taken=time_taken
        )

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create optimizer instance
    optimizer = MultiPhaseOptimizer(width=RECT_WIDTH, height=RECT_HEIGHT, num_circles=NUM_CIRCLES)

    # Run optimization
    result = optimizer.optimize()

    return result.circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")