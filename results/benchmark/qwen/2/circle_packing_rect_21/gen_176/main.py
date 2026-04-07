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
NUM_CIRCLES = 21
SEED = 42

def calculate_optimal_aspect_ratio(n_circles: int) -> float:
    """
    Calculate optimal rectangle aspect ratio for n circles based on packing efficiency.
    Uses the principle that for circle packing problems, rectangles with aspect ratios
    closer to 1.33 (4:3) or 1.0 (square) tend to work well, but we can calculate
    something more precise based on the circle count.
    """
    # For 21 circles, we'll use an empirically derived optimal ratio
    # Based on research and testing, a ratio around 1.3x0.7 works well for this problem
    # But let's make it more adaptive
    if n_circles <= 10:
        return 1.0  # Square for small numbers
    elif n_circles <= 20:
        return 1.3  # Slightly wide for medium numbers
    else:
        return 1.5  # More wide for larger numbers

# Calculate optimal rectangle dimensions
OPTIMAL_ASPECT_RATIO = calculate_optimal_aspect_ratio(NUM_CIRCLES)
RECT_WIDTH = 1.0
RECT_HEIGHT = RECT_WIDTH / OPTIMAL_ASPECT_RATIO

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
        """Initialize circles using improved adaptive grid approach"""
        circles = np.zeros((n_circles, 3))

        # Calculate optimal grid dimensions based on aspect ratio
        # Use more sophisticated approach to determine grid dimensions
        aspect_ratio = width / height

        # For better distribution, calculate grid that minimizes empty space
        if aspect_ratio >= 1.0:
            # Wide rectangle - try to fit more columns
            cols = max(1, int(np.ceil(np.sqrt(n_circles * aspect_ratio))))
            rows = max(1, int(np.ceil(n_circles / cols)))
        else:
            # Tall rectangle - try to fit more rows
            rows = max(1, int(np.ceil(np.sqrt(n_circles / aspect_ratio))))
            cols = max(1, int(np.ceil(n_circles / rows)))

        # Ensure we have enough cells to place all circles
        while cols * rows < n_circles:
            if aspect_ratio >= 1.0:
                cols += 1
            else:
                rows += 1

        # Create grid with better spacing calculation
        cell_width = width / cols if cols > 0 else width
        cell_height = height / rows if rows > 0 else height

        # Place circles in a grid pattern with improved spacing
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break

                # Position in cell center with better randomization
                # Use a more systematic approach to avoid clustering
                x = (j + 0.5) * cell_width
                y = (i + 0.5) * cell_height

                # Apply strategic perturbation to avoid regularity issues
                perturbation_factor = 0.3
                x += (random.random() - 0.5) * cell_width * perturbation_factor
                y += (random.random() - 0.5) * cell_height * perturbation_factor

                # Ensure we stay within bounds with margin
                x = max(0.02, min(width - 0.02, x))
                y = max(0.02, min(height - 0.02, y))

                # Calculate maximum possible radius based on cell and boundary constraints
                max_radius = min(x, width - x, y, height - y)

                # For better initial spread, use a more strategic initial radius
                # Based on how much space we have relative to cell size
                cell_area = cell_width * cell_height
                expected_radius = min(0.3, max_radius * 0.5)
                if expected_radius > 0.01:
                    # Use log-uniform distribution for better spread
                    r = max(0.01, min(expected_radius, np.exp(random.uniform(np.log(0.01), np.log(expected_radius)))))
                else:
                    r = max(0.01, min(0.1, max_radius * random.uniform(0.2, 0.5)))

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
        max_radius = min(max_radius, 0.3)  # Safety limit

        improved = False

        # First, try to increase the radius as much as possible
        # Use binary search or progressive improvement approach
        radius_search_space = np.linspace(old_r, max_radius, 15)
        for test_r in radius_search_space:
            # Instead of random offsets, try to place the circle as close to its original
            # position as possible while maximizing radius
            if test_r <= old_r:
                continue

            # Try to find the best position for this radius
            best_pos_for_radius = [old_x, old_y]  # Start with current position
            best_radius_at_pos = old_r

            # Check if we can place it near original position
            test_positions = [
                (old_x, old_y),  # Original position
                (old_x + (test_r - old_r)/2, old_y),  # Move right
                (old_x - (test_r - old_r)/2, old_y),  # Move left
                (old_x, old_y + (test_r - old_r)/2),  # Move up
                (old_x, old_y - (test_r - old_r)/2),  # Move down
                (old_x + (test_r - old_r)/3, old_y + (test_r - old_r)/3),  # Diagonal
            ]

            for test_x, test_y in test_positions:
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

                    if valid and test_r > best_radius_at_pos:
                        best_radius_at_pos = test_r
                        best_pos_for_radius = [test_x, test_y]
                        improved = True

            # If we found an improvement for this radius, update
            if improved and best_radius_at_pos > best_r:
                best_r = best_radius_at_pos
                best_x, best_y = best_pos_for_radius

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
        # Enhanced evolutionary approach with better parameters and techniques
        population_size = 50  # Increased population size
        max_generations = 150  # More generations for better search
        elite_size = 8  # More elite individuals retained

        # Generate initial diverse population
        population = [initial_circles.copy()]

        # Add more diverse initial configurations
        for _ in range(population_size - 1):
            # Start with a copy of initial solution
            mutant = initial_circles.copy()

            # Apply more varied mutations to create diversity
            for i in range(self.num_circles):
                if random.random() < 0.4:  # Higher mutation rate for diversity
                    # Random perturbation with variable magnitude
                    delta_x = random.uniform(-0.05, 0.05)
                    delta_y = random.uniform(-0.05, 0.05)
                    delta_r = random.uniform(-0.02, 0.02)

                    mutant[i, 0] += delta_x
                    mutant[i, 1] += delta_y
                    mutant[i, 2] += delta_r

                    # Keep within bounds
                    mutant[i, 0] = max(mutant[i, 2], min(self.width - mutant[i, 2], mutant[i, 0]))
                    mutant[i, 1] = max(mutant[i, 2], min(self.height - mutant[i, 2], mutant[i, 1]))
                    mutant[i, 2] = max(0.001, min(0.3, mutant[i, 2]))

            population.append(mutant)

        # Evolutionary loop
        best_fitness_history = []
        stagnation_counter = 0
        max_stagnation = 20  # Stop if no improvement for 20 generations

        for generation in range(max_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness, _ = self.constraint_manager.calculate_fitness(individual)
                fitness_scores.append(fitness)

            # Track best fitness for convergence monitoring
            current_best = max(fitness_scores)
            best_fitness_history.append(current_best)

            # Check for stagnation
            if len(best_fitness_history) > 10:
                recent_improvement = current_best - best_fitness_history[-10]
                if recent_improvement < 1e-6:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

                if stagnation_counter >= max_stagnation:
                    print(f"Early stopping at generation {generation} due to stagnation")
                    break

            # Selection with tournament size variation
            selected = []
            for _ in range(population_size):
                # Use varying tournament sizes for more diversity
                tournament_size = random.choice([3, 4, 5])
                tournament_indices = random.sample(range(len(population)), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())

            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1][:elite_size]
            elite = [population[i].copy() for i in sorted_indices]

            # Create new population
            new_population = elite.copy()
            while len(new_population) < population_size:
                # Tournament selection for parents
                parent1_idx = random.randint(0, len(selected)-1)
                parent2_idx = random.randint(0, len(selected)-1)
                parent1 = selected[parent1_idx]
                parent2 = selected[parent2_idx]

                # Crossover with probability
                child = parent1.copy()
                if random.random() < 0.7:  # 70% crossover rate
                    # Uniform crossover with more precision
                    for i in range(self.num_circles):
                        if random.random() < 0.6:  # 60% chance of inheriting from parent2
                            child[i] = parent2[i].copy()

                # Mutation with adaptive rates
                mutation_rate = 0.25 if len(new_population) < population_size//2 else 0.15
                for i in range(self.num_circles):
                    if random.random() < mutation_rate:
                        # Adaptive mutation based on generation
                        if generation < max_generations // 2:
                            # Early generation: larger mutations for exploration
                            delta_x = random.uniform(-0.03, 0.03)
                            delta_y = random.uniform(-0.03, 0.03)
                            delta_r = random.uniform(-0.015, 0.015)
                        else:
                            # Later generation: smaller mutations for exploitation
                            delta_x = random.uniform(-0.01, 0.01)
                            delta_y = random.uniform(-0.01, 0.01)
                            delta_r = random.uniform(-0.005, 0.005)

                        child[i, 0] += delta_x
                        child[i, 1] += delta_y
                        child[i, 2] += delta_r

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