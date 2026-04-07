# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time
import warnings

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.2  # Optimized rectangle dimensions
RECT_HEIGHT = 0.8
NUM_CIRCLES = 21
POPULATION_SIZE = 120
MAX_GENERATIONS = 150
INITIAL_MUTATION_RATE = 0.3
TOURNAMENT_SIZE = 5
SEED = 42
LOCAL_OPTIMIZATION_ITERATIONS = 150

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
                violations += 1000  # Heavy penalty for boundary violations

        # Check overlap violations using spatial indexing for efficiency
        try:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

            # Find neighbors within 2*max_radius distance (optimization)
            max_radius = np.max(circles[:, 2])
            pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

            for i, j in pairs:
                if self.check_overlap(circles, i, j):
                    violations += 1

        except Exception as e:
            warnings.warn(f"Error in overlap checking: {e}")
            # Fallback to brute force when spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap(circles, i, j):
                        violations += 1

        # Return negative penalty (since we want to maximize) plus positive radius sum
        # Adjust penalty weight for better balance
        penalty_weight = 10000.0
        return total_radius - (penalty_weight * violations), violations

    def generate_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate initial population with improved initialization strategies"""
        population = []

        # Strategy: Advanced multi-phase initialization
        # Phase 1: Grid-based placement
        grid_rows = int(np.ceil(np.sqrt(self.num_circles * 1.2)))
        grid_cols = int(np.ceil(self.num_circles / grid_rows))

        # Ensure sufficient grid size
        while grid_rows * grid_cols < self.num_circles:
            grid_rows += 1

        # Calculate precise spacing
        cell_width = self.width / (grid_cols + 1)
        cell_height = self.height / (grid_rows + 1)

        # Factor to control randomness in positioning
        randomness_factor = 0.3

        for _ in range(pop_size):
            circles = np.zeros((self.num_circles, 3))

            # Phase 1: Grid-based placement with randomness
            idx = 0
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if idx >= self.num_circles:
                        break
                    # Position with controlled randomization around grid point
                    x = (j + 1) * cell_width + np.random.uniform(-cell_width * randomness_factor,
                                                                 cell_width * randomness_factor)
                    y = (i + 1) * cell_height + np.random.uniform(-cell_height * randomness_factor,
                                                                  cell_height * randomness_factor)
                    # Radius with variance that scales with cell size
                    base_radius = min(cell_width, cell_height) * 0.35
                    r = base_radius + np.random.uniform(-base_radius * 0.15, base_radius * 0.15)
                    r = max(0.01, min(0.2, r))  # Keep in reasonable bounds

                    # Clamp to valid bounds
                    x = max(r, min(self.width - r, x))
                    y = max(r, min(self.height - r, y))

                    circles[idx] = [x, y, r]
                    idx += 1

            # Phase 2: Fill remaining slots strategically
            for i in range(idx, self.num_circles):
                # Try to place in regions with more space
                attempts = 0
                while attempts < 100:  # Prevent infinite loops
                    x = np.random.uniform(0.05, self.width - 0.05)
                    y = np.random.uniform(0.05, self.height - 0.05)
                    # Try to estimate good radius based on available space
                    min_dist_to_edge = min(x, self.width - x, y, self.height - y)
                    r = min(0.15, min_dist_to_edge * 0.4)
                    r = max(0.01, r)

                    # Check if position is valid
                    if self.is_valid_circle(x, y, r):
                        circles[i] = [x, y, r]
                        break
                    attempts += 1

                # If couldn't find a valid spot, use default
                if attempts >= 100:
                    x = np.random.uniform(0.05, self.width - 0.05)
                    y = np.random.uniform(0.05, self.height - 0.05)
                    r = np.random.uniform(0.01, 0.15)
                    circles[i] = [x, y, r]

            population.append(circles)

        return population

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Select individual using tournament selection with fitness scaling"""
        # Scale fitness to avoid numerical issues
        scaled_fitness = [f[0] + 10000 for f in fitness_scores]  # Add offset to ensure all positive
        tournament_indices = np.random.choice(len(population), TOURNAMENT_SIZE)

        # Select based on scaled fitness
        tournament_fitness = [(i, scaled_fitness[i]) for i in tournament_indices]
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)

        return population[tournament_fitness[0][0]].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two parents with better trait preservation"""
        child = parent1.copy()

        # For each circle, randomly inherit from either parent with preference for better parent
        # Use better parent's traits more often
        parent1_fitness, _ = self.calculate_fitness(parent1)
        parent2_fitness, _ = self.calculate_fitness(parent2)

        # Choose crossover probability based on parent fitness
        best_parent = parent1 if parent1_fitness >= parent2_fitness else parent2
        worst_parent = parent2 if parent1_fitness >= parent2_fitness else parent1

        # More inheritance from better parent
        mask = np.random.rand(self.num_circles) > 0.3  # 70% chance from better parent

        for i in range(self.num_circles):
            if mask[i]:
                child[i] = best_parent[i].copy()
            else:
                child[i] = worst_parent[i].copy()

        return child

    def mutate(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Apply mutation to an individual with adaptive step sizes and better balance"""
        mutated = individual.copy()

        for i in range(self.num_circles):
            if np.random.rand() < mutation_rate:
                # Mutate either center position or radius with balanced probability
                if np.random.rand() < 0.6:  # 60% chance to mutate position (more important)
                    # Mutate position with adaptive step size
                    step = 0.15 * (1.0 - mutation_rate)  # Larger steps initially
                    mutated[i, 0] = np.random.uniform(
                        max(0.001, mutated[i, 0] - step),
                        min(self.width - 0.001, mutated[i, 0] + step)
                    )
                    mutated[i, 1] = np.random.uniform(
                        max(0.001, mutated[i, 1] - step),
                        min(self.height - 0.001, mutated[i, 1] + step)
                    )
                else:  # 40% chance to mutate radius
                    # Mutate radius with adaptive step size
                    step = 0.05 * (1.0 - mutation_rate)
                    mutated[i, 2] = np.random.uniform(
                        max(0.001, mutated[i, 2] - step),
                        min(0.2, mutated[i, 2] + step)
                    )

        return mutated

    def local_optimization(self, circles: np.ndarray) -> np.ndarray:
        """Apply comprehensive local optimization to improve solution quality"""
        # Create a copy to avoid modifying original
        optimized_circles = circles.copy()

        # More aggressive optimization with multiple passes and alternating strategies
        for iteration in range(LOCAL_OPTIMIZATION_ITERATIONS):
            improved = False

            # Alternate between optimizing radius and position
            # Even iterations: optimize radius only
            # Odd iterations: optimize position only

            if iteration % 2 == 0:
                # Optimize radius for each circle
                for i in range(self.num_circles):
                    # Store current state
                    current_x, current_y, current_r = optimized_circles[i]

                    # First, try to maximize radius
                    def radius_objective(r):
                        temp_circles = optimized_circles.copy()
                        temp_circles[i, 2] = r[0]

                        # Check constraint validity
                        if not self.is_valid_circle(temp_circles[i, 0], temp_circles[i, 1], r[0]):
                            return 1e10  # Penalty for invalid configuration

                        # Check overlap constraints
                        violations = 0
                        for j in range(self.num_circles):
                            if i != j and self.check_overlap(temp_circles, i, j):
                                violations += 1

                        # Return negative radius (we minimize to maximize) plus penalty
                        return -r[0] + 1000 * violations

                    # Try to maximize this circle's radius
                    bounds = [(1e-6, min(self.width/2, self.height/2, current_r * 2))]
                    new_r = current_r
                    try:
                        result = minimize(radius_objective, [current_r], bounds=bounds, method='L-BFGS-B')
                        if result.success:
                            new_r = max(1e-6, result.x[0])
                    except:
                        pass  # If optimization fails, keep current value

                    # Update radius if better
                    if new_r > current_r:
                        optimized_circles[i, 2] = new_r
                        improved = True
            else:
                # Optimize position for each circle
                for i in range(self.num_circles):
                    # Store current state
                    current_x, current_y, current_r = optimized_circles[i]

                    # Try to adjust position to avoid overlaps or improve constraint satisfaction
                    def position_objective(pos):
                        temp_circles = optimized_circles.copy()
                        temp_circles[i, :2] = pos

                        # Check constraint validity
                        if not self.is_valid_circle(temp_circles[i, 0], temp_circles[i, 1], current_r):
                            return 1e10

                        # Check overlap constraints
                        violations = 0
                        for j in range(self.num_circles):
                            if i != j and self.check_overlap(temp_circles, i, j):
                                violations += 1

                        # Simple penalty based on overlap violations
                        return 1000 * violations

                    # Try to improve position
                    bounds = [(current_x - 0.05, current_x + 0.05),
                              (current_y - 0.05, current_y + 0.05)]
                    new_pos = [current_x, current_y]

                    try:
                        result = minimize(position_objective, [current_x, current_y],
                                        bounds=bounds, method='L-BFGS-B')
                        if result.success:
                            new_pos = result.x
                            # Clamp to valid bounds
                            new_pos[0] = max(current_r, min(self.width - current_r, new_pos[0]))
                            new_pos[1] = max(current_r, min(self.height - current_r, new_pos[1]))

                            # Only update if it results in less constraint violation
                            temp_circles = optimized_circles.copy()
                            temp_circles[i, :2] = new_pos

                            # Check if this move reduces violations
                            old_violations = 0
                            new_violations = 0

                            for j in range(self.num_circles):
                                if i != j:
                                    if self.check_overlap(optimized_circles, i, j):
                                        old_violations += 1
                                    if self.check_overlap(temp_circles, i, j):
                                        new_violations += 1

                            if new_violations <= old_violations:
                                optimized_circles[i, :2] = new_pos
                                improved = True

                    except:
                        pass  # If optimization fails, keep current value

            # Early exit if no improvement
            if not improved:
                break

        return optimized_circles

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm with enhancements"""
        start_time = time.time()

        # Generate initial population with better initialization
        population = self.generate_initial_population(POPULATION_SIZE)

        best_solution = None
        best_fitness = float('-inf')

        # Track convergence
        fitness_history = []
        stagnant_generations = 0
        max_stagnant_generations = 30

        for generation in range(MAX_GENERATIONS):
            # Adaptive mutation rate that decreases more aggressively
            adaptive_mutation_rate = INITIAL_MUTATION_RATE * (1.0 - generation / MAX_GENERATIONS)
            adaptive_mutation_rate = max(0.05, adaptive_mutation_rate)  # Minimum mutation rate

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
                stagnant_generations = 0  # Reset stagnation counter
            else:
                stagnant_generations += 1

            fitness_history.append(gen_best_fitness)

            # Print progress every 15 generations
            if generation % 15 == 0:
                print(f"Generation {generation}: Best fitness = {gen_best_fitness:.6f}, "
                      f"Stagnant: {stagnant_generations}")

            # Apply local optimization to best solution periodically
            if generation % 5 == 0 and best_solution is not None:
                improved_solution = self.local_optimization(best_solution)
                improved_fitness, _ = self.calculate_fitness(improved_solution)
                if improved_fitness > best_fitness:
                    best_solution = improved_solution
                    best_fitness = improved_fitness

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

                # Mutation
                child = self.mutate(child, adaptive_mutation_rate)

                new_population.append(child)

            population = new_population

            # Early stopping if converged
            if stagnant_generations >= max_stagnant_generations:
                print(f"Converged at generation {generation}")
                break

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness achieved: {best_fitness:.6f}")

        # Final local optimization on best solution
        if best_solution is not None:
            final_solution = self.local_optimization(best_solution)
            final_fitness, _ = self.calculate_fitness(final_solution)
            if final_fitness > best_fitness:
                print("Final local optimization improved solution")
                best_solution = final_solution

        return best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Create packer instance with rectangle dimensions
    # Use optimized aspect ratio for better circle packing
    packer = CirclePacker(width=RECT_WIDTH, height=RECT_HEIGHT, num_circles=NUM_CIRCLES)

    # Run optimization
    circles = packer.optimize()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")