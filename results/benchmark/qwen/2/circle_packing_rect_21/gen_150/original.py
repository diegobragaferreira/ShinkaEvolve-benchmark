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
RECT_WIDTH = 1.0  # Default rectangle dimensions (width=1, height=1)
RECT_HEIGHT = 1.0
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
    def grid_initialize(width: float, height: float, n_circles: int) -> np.ndarray:
        """Initialize circles using grid-based approach with adaptive spacing"""
        circles = np.zeros((n_circles, 3))

        # Calculate optimal grid dimensions
        grid_size = max(1, int(np.ceil(np.sqrt(n_circles))))
        cols = grid_size
        rows = grid_size

        # Adjust grid based on rectangle aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 1:  # Wide rectangle
            cols = max(1, int(np.ceil(np.sqrt(n_circles * aspect_ratio))))
            rows = max(1, int(np.ceil(n_circles / cols)))
        else:  # Tall rectangle
            rows = max(1, int(np.ceil(np.sqrt(n_circles / aspect_ratio))))
            cols = max(1, int(np.ceil(n_circles / rows)))

        # Ensure sufficient space
        cols = max(cols, 1)
        rows = max(rows, 1)

        # Create grid with random perturbations
        cell_width = width / cols
        cell_height = height / rows

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                # Center of cell with random perturbation
                x = (j + 0.5) * cell_width + random.uniform(-cell_width*0.1, cell_width*0.1)
                y = (i + 0.5) * cell_height + random.uniform(-cell_height*0.1, cell_height*0.1)

                # Keep within bounds
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))

                # Initial radius based on available space
                max_radius = min(x, width-x, y, height-y) * 0.4
                r = max(0.01, min(max_radius, random.uniform(0.02, 0.1)))

                circles[idx] = [x, y, r]
                idx += 1

        return circles

    @staticmethod
    def random_initialize(width: float, height: float, n_circles: int) -> np.ndarray:
        """Generate random initial configuration"""
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            # Random position
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)

            # Random radius
            r = random.uniform(0.01, min(0.2, width/10, height/10))

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

        # Check overlap violations
        try:
            # Use spatial indexing for efficiency
            points = circles[:, :2]
            tree = cKDTree(points)
            max_radius = np.max(circles[:, 2])
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

            for i, j in pairs:
                if CircleValidator.check_overlap(circles, i, j):
                    overlap_violations += 1

        except Exception:
            # Fallback to brute force
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if CircleValidator.check_overlap(circles, i, j):
                        overlap_violations += 1

        return boundary_violations, overlap_violations

    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, int]:
        """Calculate fitness with penalty for constraint violations"""
        total_radius = np.sum(circles[:, 2])

        # Count constraint violations
        boundary_violations, overlap_violations = self.calculate_violations(circles)

        # Penalty calculation
        penalty = (boundary_violations * 1000 + overlap_violations * 500)

        # Return fitness (higher is better) - negative penalty + positive radius sum
        return total_radius - penalty, boundary_violations + overlap_violations

class LocalOptimizer:
    """Handles local optimization of individual circles"""

    @staticmethod
    def optimize_single_circle(circles: np.ndarray, idx: int, width: float, height: float) -> bool:
        """Try to improve a single circle's position and radius"""
        old_x, old_y, old_r = circles[idx]
        best_x, best_y, best_r = old_x, old_y, old_r

        # Attempt to increase radius to maximum possible value
        max_radius = min(old_x, width - old_x, old_y, height - old_y)
        radius_search_space = np.linspace(old_r, min(max_radius, 0.3), 10)

        improved = False

        for test_r in radius_search_space:
            # Try nearby positions (small random offsets)
            for _ in range(10):
                offset_x = random.uniform(-0.03, 0.03)
                offset_y = random.uniform(-0.03, 0.03)

                test_x = old_x + offset_x
                test_y = old_y + offset_y

                # Keep within bounds
                test_x = max(test_r, min(width - test_r, test_x))
                test_y = max(test_r, min(height - test_r, test_y))

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
        """Phase 1: Grid-based initialization"""
        circles = CircleInitializer.grid_initialize(self.width, self.height, self.num_circles)
        return circles

    def phase_local_refinement(self, circles: np.ndarray) -> np.ndarray:
        """Phase 2: Local optimization and refinement"""
        # Perform multiple iterations of local optimization
        for _ in range(50):
            improved = False
            for i in range(self.num_circles):
                if LocalOptimizer.optimize_single_circle(circles, i, self.width, self.height):
                    improved = True

            if not improved:
                break

        # Apply final constraint validation and correction
        for i in range(self.num_circles):
            x, y, r = circles[i]
            # Correct boundary violations
            if x - r < 0:
                circles[i, 0] = r
            elif x + r > self.width:
                circles[i, 0] = self.width - r
            if y - r < 0:
                circles[i, 1] = r
            elif y + r > self.height:
                circles[i, 1] = self.height - r

        return circles

    def phase_evolutionary(self, initial_circles: np.ndarray) -> np.ndarray:
        """Phase 3: Evolutionary algorithm for further improvement"""
        # Simplified evolutionary approach with focused search
        population_size = 30
        max_generations = 100
        elite_size = 5

        # Generate initial population
        population = [initial_circles.copy()]
        for _ in range(population_size - 1):
            # Perturb initial solution
            mutant = initial_circles.copy()
            for i in range(self.num_circles):
                if random.random() < 0.3:  # 30% mutation rate
                    # Random perturbation
                    mutant[i, 0] += random.uniform(-0.02, 0.02)
                    mutant[i, 1] += random.uniform(-0.02, 0.02)
                    mutant[i, 2] += random.uniform(-0.01, 0.01)

                    # Keep within bounds
                    mutant[i, 0] = max(mutant[i, 2], min(self.width - mutant[i, 2], mutant[i, 0]))
                    mutant[i, 1] = max(mutant[i, 2], min(self.height - mutant[i, 2], mutant[i, 1]))
                    mutant[i, 2] = max(0.001, min(0.3, mutant[i, 2]))

            population.append(mutant)

        # Evolutionary loop
        for generation in range(max_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness, _ = self.constraint_manager.calculate_fitness(individual)
                fitness_scores.append(fitness)

            # Selection (tournament)
            selected = []
            for _ in range(population_size):
                tournament_indices = random.sample(range(len(population)), 3)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())

            # Elitism: keep best
            best_idx = np.argmax(fitness_scores)
            elite = [population[best_idx].copy()]

            # Create new population
            new_population = elite.copy()
            while len(new_population) < population_size:
                # Crossover (uniform)
                parent1 = selected[random.randint(0, len(selected)-1)]
                parent2 = selected[random.randint(0, len(selected)-1)]

                child = parent1.copy()
                for i in range(self.num_circles):
                    if random.random() < 0.5:
                        child[i] = parent2[i].copy()

                # Mutation
                for i in range(self.num_circles):
                    if random.random() < 0.2:  # 20% mutation rate
                        child[i, 0] += random.uniform(-0.01, 0.01)
                        child[i, 1] += random.uniform(-0.01, 0.01)
                        child[i, 2] += random.uniform(-0.005, 0.005)

                        # Keep within bounds
                        child[i, 0] = max(child[i, 2], min(self.width - child[i, 2], child[i, 0]))
                        child[i, 1] = max(child[i, 2], min(self.height - child[i, 2], child[i, 1]))
                        child[i, 2] = max(0.001, min(0.3, child[i, 2]))

                new_population.append(child)

            population = new_population[:population_size]

        # Return best solution from final population
        final_fitness = [self.constraint_manager.calculate_fitness(ind)[0] for ind in population]
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
        fitness, violations = self.constraint_manager.calculate_fitness(circles)

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
    optimizer = MultiPhaseOptimizer(width=1.0, height=1.0, num_circles=21)

    # Run optimization
    result = optimizer.optimize()

    return result.circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")