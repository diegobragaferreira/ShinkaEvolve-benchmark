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
MUTATION_RATE = 0.1
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

    def find_optimal_rectangle_dimensions(self) -> Tuple[float, float]:
        """
        Find optimal rectangle dimensions for packing circles given perimeter constraint.
        Perimeter = 4 means width + height = 2.
        """
        best_width = 1.0
        best_height = 1.0
        best_radius_sum = 0.0

        # Test various aspect ratios from 0.5 to 2.0 (more skewed rectangles)
        aspect_ratios = np.linspace(0.5, 2.0, 20)

        for ratio in aspect_ratios:
            width = 1.0  # Start with width = 1
            height = width / ratio

            # Ensure perimeter constraint is satisfied
            if width + height > 2.0:
                # Scale down to fit perimeter constraint
                scale = 2.0 / (width + height)
                width *= scale
                height *= scale

            # Test this dimension setup by creating a temporary packer and running optimization
            temp_packer = CirclePacker(width=width, height=height, num_circles=self.num_circles)
            try:
                # Use a quick optimization approach to get a rough estimate
                circles = temp_packer._generate_grid_initialization()
                # Just evaluate the configuration quickly - don't do full EA
                total_radius = np.sum(circles[:, 2])
                if total_radius > best_radius_sum:
                    best_radius_sum = total_radius
                    best_width = width
                    best_height = height
            except:
                continue

        return best_width, best_height

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

        # Check boundary violations - vectorized for efficiency
        radii = circles[:, 2]
        positions = circles[:, :2]

        # Vectorized boundary checks
        boundary_violations = (
            (radii > positions[:, 0]) |
            (radii > self.width - positions[:, 0]) |
            (radii > positions[:, 1]) |
            (radii > self.height - positions[:, 1])
        )
        violations += np.sum(boundary_violations.astype(int)) * 100

        # Check overlap violations using spatial indexing for efficiency
        try:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

            # Find neighbors within 2*max_radius distance (more precise threshold)
            max_radius = np.max(circles[:, 2])
            if max_radius > 0:
                # Use a slightly more conservative distance threshold
                pairs = tree.query_pairs(1.5 * max_radius, output_type='ndarray')

                for i, j in pairs:
                    if i < j:  # Only check each pair once
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
        penalty_weight = 1000.0
        return total_radius - (penalty_weight * violations), violations

    def generate_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate initial population of circle configurations"""
        population = []

        # First, determine optimal rectangle dimensions for better initial placement
        optimal_width, optimal_height = self.find_optimal_rectangle_dimensions()
        self.width = optimal_width
        self.height = optimal_height

        for _ in range(pop_size):
            # Use adaptive grid initialization for better starting point
            circles = self._generate_grid_initialization()
            population.append(circles)

        return population

    def _generate_grid_initialization(self) -> np.ndarray:
        """Create initial configuration using adaptive grid placement optimized for the rectangle dimensions"""
        circles = np.zeros((self.num_circles, 3))

        # Calculate optimal grid dimensions based on rectangle aspect ratio and circle count
        # For 21 circles, we want to find the best grid layout that maximizes packing efficiency
        aspect_ratio = self.width / self.height

        # Try different grid layouts and pick the one that best utilizes space
        best_layout_score = 0
        best_rows = 1
        best_cols = self.num_circles

        # Test various grid configurations
        for rows in range(1, int(np.sqrt(self.num_circles)) + 5):
            cols = int(np.ceil(self.num_circles / rows))
            # Score based on how well the grid fits the aspect ratio
            grid_aspect = (cols * self.width) / (rows * self.height)
            aspect_score = 1.0 / (1.0 + abs(grid_aspect - aspect_ratio))
            layout_score = aspect_score * (cols * rows)  # Prefer complete layouts

            if layout_score > best_layout_score:
                best_layout_score = layout_score
                best_rows = rows
                best_cols = cols

        rows, cols = best_rows, best_cols
        actual_circles = rows * cols
        if actual_circles > self.num_circles:
            rows = int(np.ceil(self.num_circles / cols))

        # Calculate spacing with better margin consideration
        margin = 0.03  # Reduced margin for tighter packing
        if cols > 0:
            cell_width = (self.width - 2 * margin) / cols
        else:
            cell_width = self.width / 2
        if rows > 0:
            cell_height = (self.height - 2 * margin) / rows
        else:
            cell_height = self.height / 2

        # Use minimum of cell dimensions as base for radius, with better scaling factor
        base_radius = min(cell_width, cell_height) * 0.35  # Slightly smaller base radius for better packing

        # Place circles using grid pattern with more sophisticated positioning
        circle_idx = 0
        for i in range(rows):
            for j in range(cols):
                if circle_idx >= self.num_circles:
                    break

                # Position circle at center of grid cell with strategic randomization
                x = margin + (j + 0.5) * cell_width + np.random.uniform(-0.05 * cell_width, 0.05 * cell_width)
                y = margin + (i + 0.5) * cell_height + np.random.uniform(-0.05 * cell_height, 0.05 * cell_height)

                # Radius based on cell size with randomized variation
                r = base_radius * (0.85 + np.random.uniform(0, 0.3))  # More generous variation range

                # Ensure circle fits within bounds with tighter constraints
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

    def mutate(self, individual: np.ndarray, mutation_rate: float = MUTATION_RATE,
               generation: int = 0, max_generations: int = MAX_GENERATIONS) -> np.ndarray:
        """Apply mutation to an individual with adaptive strategy"""
        mutated = individual.copy()

        # Adaptive mutation rate that decreases over time
        adaptive_mutation_rate = mutation_rate * (1.0 - generation / max_generations)

        for i in range(self.num_circles):
            if np.random.rand() < adaptive_mutation_rate:
                # Mutate either center position or radius with probability based on generation
                if np.random.rand() < 0.5:
                    # Mutate position with adaptive step size
                    step_size = 0.05 * (1.0 - generation / max_generations)  # Decrease with generation
                    mutated[i, 0] = np.random.uniform(
                        max(0.001, mutated[i, 0] - step_size),
                        min(self.width - 0.001, mutated[i, 0] + step_size)
                    )
                    mutated[i, 1] = np.random.uniform(
                        max(0.001, mutated[i, 1] - step_size),
                        min(self.height - 0.001, mutated[i, 1] + step_size)
                    )
                else:
                    # Mutate radius with adaptive range based on current radius and position constraints
                    current_radius = mutated[i, 2]
                    # Mutate by ±20% of current radius, but stay within reasonable bounds
                    delta = current_radius * 0.2
                    min_radius = 0.001
                    max_radius = min(self.width, self.height) * 0.3  # Reasonable upper bound
                    mutated[i, 2] = np.random.uniform(
                        max(min_radius, current_radius - delta),
                        min(max_radius, current_radius + delta)
                    )

        return mutated

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm"""
        start_time = time.time()

        # Increase population size and generations for better exploration
        adaptive_pop_size = max(POPULATION_SIZE, 80)
        adaptive_max_generations = max(MAX_GENERATIONS, 300)

        # Generate initial population
        population = self.generate_initial_population(adaptive_pop_size)

        best_solution = None
        best_fitness = float('-inf')

        # Track convergence
        fitness_history = []
        stagnation_counter = 0
        stagnation_threshold = 20

        for generation in range(adaptive_max_generations):
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
                stagnation_counter = 0  # Reset stagnation counter on improvement
            else:
                stagnation_counter += 1

            fitness_history.append(gen_best_fitness)

            # Print progress every 30 generations
            if generation % 30 == 0:
                print(f"Generation {generation}: Best fitness = {gen_best_fitness:.6f}")

            # Create new population through selection, crossover, and mutation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < adaptive_pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation with generation-aware parameters
                child = self.mutate(child, generation=generation, max_generations=adaptive_max_generations)

                new_population.append(child)

            population = new_population

            # Early stopping if converged or stagnated
            if len(fitness_history) >= 15:
                recent_avg = np.mean(fitness_history[-15:])
                prev_avg = np.mean(fitness_history[-30:-15])
                if abs(recent_avg - prev_avg) < 1e-6 or stagnation_counter > stagnation_threshold:
                    print(f"Converged or stagnated at generation {generation}")
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