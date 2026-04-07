# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import time
from typing import Tuple, List

# Global constants for optimization
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.3  # Optimized width for better packing
RECT_HEIGHT = 0.7  # Optimized height for better packing
NUM_CIRCLES = 21
POPULATION_SIZE = 150  # Increased population size for better exploration
MAX_GENERATIONS = 600  # More generations for better optimization
INITIAL_MUTATION_RATE = 0.30  # Higher initial mutation rate for better exploration
FINAL_MUTATION_RATE = 0.02  # Lower final mutation rate for exploitation
TOURNAMENT_SIZE = 9  # Larger tournament size for stronger selection pressure
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

        # Query pairs efficiently with increased safety margin
        try:
            # Using a safety margin for better reliability
            search_radius = 2.5 * max_radius
            pairs = tree.query_pairs(search_radius, output_type='ndarray')

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
        # Adjust penalty weight for better balance - make it adaptive based on violations
        penalty_weight = 1000.0 + (violations * 150.0)  # Dynamic penalty based on constraint violations
        return total_radius - (penalty_weight * violations), violations

    def generate_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate initial population of circle configurations with adaptive grid"""
        population = []

        for _ in range(pop_size):
            circles = np.zeros((self.num_circles, 3))

            # Generate initial configuration using adaptive grid-based approach
            # Calculate optimal grid dimensions based on aspect ratio and circle count
            aspect_ratio = self.width / self.height

            # More sophisticated adaptive grid calculation inspired by successful approaches
            # This attempts to create a grid that closely matches the aspect ratio and circle count
            if aspect_ratio >= 1.2:  # Landscape orientation with clear preference for width
                # Increase column count for landscape orientation
                cols = int(np.ceil(np.sqrt(self.num_circles * aspect_ratio * 1.3)))
                rows = int(np.ceil(self.num_circles / cols))
            elif aspect_ratio <= 0.8:  # Portrait orientation with clear preference for height
                # Increase row count for portrait orientation
                rows = int(np.ceil(np.sqrt(self.num_circles / aspect_ratio * 1.3)))
                cols = int(np.ceil(self.num_circles / rows))
            else:  # Balanced aspect ratio
                cols = int(np.ceil(np.sqrt(self.num_circles * aspect_ratio)))
                rows = int(np.ceil(self.num_circles / cols))

            # Ensure we have enough cells and adjust for better fitting
            while cols * rows < self.num_circles:
                if aspect_ratio >= 1.2:  # Prefer more columns
                    cols += 1
                elif aspect_ratio <= 0.8:  # Prefer more rows
                    rows += 1
                else:  # Balanced case
                    if cols > rows:
                        rows += 1
                    else:
                        cols += 1

            # Calculate spacing with better consideration of aspect ratio
            spacing_x = self.width / (cols + 1.5) if cols > 0 else self.width
            spacing_y = self.height / (rows + 1.5) if rows > 0 else self.height

            # Create more sophisticated hexagonal packing pattern with better spacing
            placed_count = 0
            for i in range(rows):
                for j in range(cols):
                    if placed_count >= self.num_circles:
                        break

                    # Offset every other row for hexagonal packing with better spacing
                    offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                    base_x = (j + 1) * spacing_x + offset_x
                    base_y = (i + 1) * spacing_y

                    # Add small random perturbation for diversity and better optimization starting point
                    perturbation_factor = min(0.12, 0.25 * min(spacing_x, spacing_y))
                    x = np.clip(base_x + np.random.uniform(-perturbation_factor, perturbation_factor),
                               0.01, self.width - 0.01)
                    y = np.clip(base_y + np.random.uniform(-perturbation_factor, perturbation_factor),
                               0.01, self.height - 0.01)

                    # Initial radius estimation with better heuristics
                    max_r = min(x, self.width - x, y, self.height - y)
                    estimated_radius = min(0.18, max_r * 0.7)  # Conservative estimate for 21 circles
                    r = np.random.uniform(estimated_radius * 0.6, estimated_radius * 1.2)

                    circles[placed_count] = [x, y, r]
                    placed_count += 1

                if placed_count >= self.num_circles:
                    break

            # Ensure all circles are valid
            for i in range(self.num_circles):
                if not self.is_valid_circle(circles[i, 0], circles[i, 1], circles[i, 2]):
                    # Reinitialize invalid circles with better constraints
                    x = np.random.uniform(0.01, self.width - 0.01)
                    y = np.random.uniform(0.01, self.height - 0.01)
                    max_r = min(x, self.width - x, y, self.height - y)
                    r = min(0.15, max_r * 0.6)
                    circles[i] = [x, y, r]

            population.append(circles)

        return population

    def tournament_selection(self, population: List[np.ndarray],
                           fitness_scores: List[Tuple[float, int]]) -> np.ndarray:
        """Select individual using tournament selection with larger tournament size"""
        # Select multiple candidates and pick the best
        tournament_indices = np.random.choice(len(population), TOURNAMENT_SIZE)
        tournament_fitness = [(i, fitness_scores[i][0]) for i in tournament_indices]

        # Sort by fitness (descending)
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)

        return population[tournament_fitness[0][0]].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover with better parent fitness awareness"""
        child = parent1.copy()

        # Enhanced crossover with parent fitness consideration
        for i in range(self.num_circles):
            # Probability based on parent fitness - better parent gets more influence
            parent1_fitness = np.sum(parent1[:, 2])  # Simplified fitness approximation
            parent2_fitness = np.sum(parent2[:, 2])  # Simplified fitness approximation

            # Weighted choice based on relative fitness quality
            if parent1_fitness > parent2_fitness:
                prob_parent1 = 0.7  # Stronger preference for better parent
            elif parent2_fitness > parent1_fitness:
                prob_parent1 = 0.3
            else:
                prob_parent1 = 0.5

            if np.random.rand() < prob_parent1:
                child[i] = parent1[i].copy()
            else:
                child[i] = parent2[i].copy()

        return child

    def mutate(self, individual: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply mutation to an individual with adaptive mutation rate and smarter mutation"""
        # Adaptive mutation rate: decrease over generations
        adaptive_rate = INITIAL_MUTATION_RATE - (INITIAL_MUTATION_RATE - FINAL_MUTATION_RATE) * \
                       (generation / max_generations)

        mutated = individual.copy()

        # Mutate each circle with adaptive strategies
        for i in range(self.num_circles):
            if np.random.rand() < adaptive_rate:
                # Mutate either center position or radius with different probabilities
                if np.random.rand() < 0.65:  # 65% chance to mutate position (more impactful)
                    # Mutate position with boundary-aware perturbations
                    x = mutated[i, 0]
                    y = mutated[i, 1]
                    r = mutated[i, 2]

                    # Calculate adaptive mutation ranges based on available space
                    max_dx = min(0.15, x - r, self.width - x - r)
                    max_dy = min(0.15, y - r, self.height - y - r)

                    # Apply mutation with adaptive range
                    mutated[i, 0] = np.clip(
                        x + np.random.uniform(-max_dx, max_dx),
                        r + 0.001, self.width - r - 0.001
                    )
                    mutated[i, 1] = np.clip(
                        y + np.random.uniform(-max_dy, max_dy),
                        r + 0.001, self.height - r - 0.001
                    )
                else:  # 35% chance to mutate radius (smaller impact)
                    # Mutate radius with adaptive range
                    old_r = mutated[i, 2]
                    # Radius change depends on current radius value, bounded by max possible
                    max_dr = min(0.08, 0.25 - old_r)  # Don't let it grow too large
                    mutated[i, 2] = np.clip(
                        old_r + np.random.uniform(-max_dr, max_dr),
                        0.001, 0.25
                    )

        return mutated

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm with improvements"""
        start_time = time.time()

        # Generate initial population
        population = self.generate_initial_population(POPULATION_SIZE)

        best_solution = None
        best_fitness = float('-inf')

        # Track convergence
        fitness_history = []
        stagnation_counter = 0
        max_stagnation = 60  # Maximum generations without improvement before early stopping

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
                stagnation_counter = 0  # Reset counter on improvement
            else:
                stagnation_counter += 1

            fitness_history.append(gen_best_fitness)

            # Print progress every 40 generations
            if generation % 40 == 0:
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
            if stagnation_counter >= max_stagnation:
                print(f"Early stopped at generation {generation} due to stagnation")
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
    packer = CirclePacker(width=1.3, height=0.7, num_circles=21)

    # Run optimization
    circles = packer.optimize()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")