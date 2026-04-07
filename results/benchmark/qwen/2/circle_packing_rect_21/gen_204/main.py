# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import time
import warnings
import math

# Global constants
RECT_PERIMETER = 4.0
RECT_WIDTH = 1.0  # Default rectangle dimensions (width=1, height=1)
RECT_HEIGHT = 1.0
NUM_CIRCLES = 21
POPULATION_SIZE = 100
MAX_GENERATIONS = 150
MUTATION_RATE = 0.05
TOURNAMENT_SIZE = 3
SEED = 42
ASPECT_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0]

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

        # Count constraint violations and calculate violation severity
        violations = 0
        total_violation_severity = 0.0

        # Check boundary violations
        for i in range(self.num_circles):
            x, y, r = circles[i]
            if not self.is_valid_circle(x, y, r):
                violations += 1
                # Calculate how far the circle violates boundaries
                boundary_violation = 0.0
                if r > x:
                    boundary_violation += (r - x) ** 2
                if r > self.width - x:
                    boundary_violation += (r - (self.width - x)) ** 2
                if r > y:
                    boundary_violation += (r - y) ** 2
                if r > self.height - y:
                    boundary_violation += (r - (self.height - y)) ** 2
                total_violation_severity += np.sqrt(boundary_violation)

        # Check overlap violations using spatial indexing for efficiency
        try:
            # Build KDTree for fast neighbor search
            points = circles[:, :2]  # Only x,y coordinates
            tree = cKDTree(points)

            # Find neighbors within 2*max_radius distance (optimization)
            max_radius = np.max(circles[:, 2])
            if max_radius > 0:
                # Query pairs with distance threshold and avoid redundant checking
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

                # Process pairs to detect overlaps efficiently
                for i, j in pairs:
                    if i < j and self.check_overlap(circles, i, j):
                        violations += 1
                        # Calculate overlap severity (how much they overlap)
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        dist_sq = dx*dx + dy*dy
                        overlap = max(0, r1 + r2 - np.sqrt(dist_sq))
                        total_violation_severity += overlap

        except Exception as e:
            warnings.warn(f"Error in overlap checking: {e}")
            # Fallback to brute force when spatial indexing fails
            for i in range(self.num_circles):
                for j in range(i+1, self.num_circles):
                    if self.check_overlap(circles, i, j):
                        violations += 1
                        # Calculate overlap severity for fallback
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        dist_sq = dx*dx + dy*dy
                        overlap = max(0, r1 + r2 - np.sqrt(dist_sq))
                        total_violation_severity += overlap

        # Return fitness score - higher is better
        # Adaptive penalty weight based on solution quality and constraint violation severity
        base_penalty = 1000.0
        violation_penalty = 500.0 * max(0, total_radius - 1.0)
        severity_penalty = 100.0 * total_violation_severity
        penalty_weight = base_penalty + violation_penalty + severity_penalty
        return total_radius - (penalty_weight * violations), violations

    def generate_initial_population(self, pop_size: int) -> List[np.ndarray]:
        """Generate initial population of circle configurations"""
        population = []

        for _ in range(pop_size):
            # Use adaptive grid initialization for better starting point
            circles = self._generate_optimized_initialization()
            population.append(circles)

        return population

    def _generate_optimized_initialization(self) -> np.ndarray:
        """Create initial configuration using optimized grid placement"""
        circles = np.zeros((self.num_circles, 3))

        # Calculate optimal grid dimensions based on rectangle aspect ratio and circle count
        aspect_ratio = self.width / self.height

        # Try different grid layouts and pick the one that best utilizes space
        best_layout_score = 0
        best_rows = 1
        best_cols = self.num_circles

        # Test various grid configurations with more systematic approach
        max_rows = int(np.sqrt(self.num_circles)) + 5
        for rows in range(1, min(max_rows, self.num_circles + 1)):
            cols = int(np.ceil(self.num_circles / rows))
            if cols <= 0:
                continue
            # Score based on how well the grid fits the aspect ratio and is complete
            grid_aspect = (cols * self.width) / (rows * self.height)
            aspect_score = 1.0 / (1.0 + abs(grid_aspect - aspect_ratio))
            layout_score = aspect_score * min(cols * rows, self.num_circles)  # Prefer complete layouts

            if layout_score > best_layout_score:
                best_layout_score = layout_score
                best_rows = rows
                best_cols = cols

        rows, cols = best_rows, best_cols
        actual_circles = rows * cols
        if actual_circles > self.num_circles:
            # Adjust rows to fit exactly the required number of circles
            rows = int(np.ceil(self.num_circles / cols))

        # Calculate spacing with better margin consideration
        margin = 0.02  # Even smaller margin for tighter packing
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
                r = base_radius * (0.85 + np.random.uniform(0, 0.3))

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

    def mutate(self, individual: np.ndarray, generation: int = 0) -> np.ndarray:
        """Apply mutation to an individual with adaptive parameters"""
        mutated = individual.copy()

        # Base mutation rate with adaptive scaling
        current_radius_sum = self.calculate_total_radius_sum(mutated)
        # Dynamic mutation rate that decreases over generations but with better scaling
        adaptive_mutation_rate = MUTATION_RATE * (1.0 - 0.003 * generation)
        adaptive_mutation_rate = max(0.01, adaptive_mutation_rate)  # Minimum mutation rate

        for i in range(self.num_circles):
            if np.random.rand() < adaptive_mutation_rate:
                # Mutate either center position or radius with preference for position
                if np.random.rand() < 0.7:  # 70% chance of position mutation
                    # Mutate position with adaptive step size
                    step_x = 0.03 * (1.0 + 0.007 * current_radius_sum)
                    step_y = 0.03 * (1.0 + 0.007 * current_radius_sum)
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
                    step_r = 0.02 * (1.0 + 0.005 * current_radius_sum)
                    mutated[i, 2] = np.random.uniform(
                        max(0.001, mutated[i, 2] - step_r),
                        min(0.2, mutated[i, 2] + step_r)
                    )

        return mutated

    def local_refinement(self, circles: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """Apply local refinement to improve solution quality"""
        refined = circles.copy()

        # Iteratively try to increase radii while respecting constraints
        for iter_num in range(max_iterations):
            improved = False

            # Try to increase each circle's radius
            for i in range(self.num_circles):
                original_circle = refined[i].copy()
                x, y, r = original_circle

                # Try to increase radius
                new_r = min(r * 1.015, 0.2)  # Even smaller increase factor
                test_circle = [x, y, new_r]

                # Check if it's still valid
                if not self.is_valid_circle(x, y, new_r):
                    continue

                # Check if it causes overlaps - more efficient approach
                # Create a temporary array for efficient overlap checking
                temp_circles = np.vstack([refined[:i], test_circle, refined[i+1:]])
                overlap_found = False

                # Use KDTree for faster overlap checking
                try:
                    points = temp_circles[:, :2]
                    tree = cKDTree(points)
                    pairs = tree.query_pairs(2 * max(0.2, new_r), output_type='ndarray')
                    for p, q in pairs:
                        if p < len(temp_circles) and q < len(temp_circles) and p != q:
                            if self.check_overlap(temp_circles, p, q):
                                overlap_found = True
                                break
                except:
                    # Fallback to direct checking
                    for j in range(len(temp_circles)):
                        if j != i:
                            if self.check_overlap(temp_circles, i, j):
                                overlap_found = True
                                break

                if not overlap_found:
                    refined[i] = test_circle
                    improved = True

            # If no improvements were made, stop early
            if not improved:
                break

        return refined

    def optimize(self) -> np.ndarray:
        """Main optimization loop using evolutionary algorithm with two phases"""
        start_time = time.time()

        # Phase 1: Coarse-grained evolution with relaxed constraints
        print("Starting Phase 1: Coarse evolution...")
        population = self.generate_initial_population(POPULATION_SIZE)

        best_solution = None
        best_fitness = float('-inf')
        fitness_history = []

        for generation in range(MAX_GENERATIONS // 2):  # Half the generations for phase 1
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
                print(f"Phase 1 Gen {generation}: Best fitness = {gen_best_fitness:.6f}")

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
                child = self.mutate(child, generation)

                new_population.append(child)

            population = new_population

        # Early stopping if converged
        if len(fitness_history) >= 10:
            recent_avg = np.mean(fitness_history[-10:])
            prev_avg = np.mean(fitness_history[-20:-10])
            if abs(recent_avg - prev_avg) < 1e-6:
                print(f"Phase 1 converged at generation {generation}")

        # Phase 2: Fine-grained refinement with stricter constraints
        print("Starting Phase 2: Fine-tuning...")
        # Refine the best solution from phase 1
        refined_solution = self.local_refinement(best_solution.copy(), 200)

        # Additional refinement through focused mutation and optimization
        final_solution = refined_solution.copy()

        # Try iterative improvement with more targeted approach
        for i in range(100):  # More rounds of focused updates
            # Try to improve individual circles
            improved = False
            for j in range(self.num_circles):
                original_circle = final_solution[j].copy()
                x, y, r = original_circle

                # Try to slightly increase radius
                new_r = min(r * 1.008, 0.2)
                test_circle = [x, y, new_r]

                # Check validity
                if not self.is_valid_circle(x, y, new_r):
                    continue

                # Check overlap with others using efficient method
                temp_circles = np.vstack([final_solution[:j], test_circle, final_solution[j+1:]])
                overlap_found = False

                # Use KDTree for efficient overlap checking
                try:
                    points = temp_circles[:, :2]
                    tree = cKDTree(points)
                    pairs = tree.query_pairs(2 * max(0.2, new_r), output_type='ndarray')
                    for p, q in pairs:
                        if p < len(temp_circles) and q < len(temp_circles) and p != q:
                            if self.check_overlap(temp_circles, p, q):
                                overlap_found = True
                                break
                except:
                    # Fallback to direct checking
                    for k in range(len(temp_circles)):
                        if k != j and self.check_overlap(temp_circles, j, k):
                            overlap_found = True
                            break

                if not overlap_found:
                    final_solution[j] = test_circle
                    improved = True

            if not improved:
                break

        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Final fitness achieved: {self.calculate_fitness(final_solution)[0]:.6f}")

        return final_solution

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

            # Create packer with current dimensions
            packer = CirclePacker(width=width, height=height, num_circles=self.num_circles)

            # Run optimization
            solution = packer.optimize()

            # Check if this solution is better
            radius_sum = self.calculate_total_radius_sum(solution)
            if radius_sum > best_radius_sum:
                best_radius_sum = radius_sum
                best_solution = solution.copy()
                print(f"New best found with aspect ratio {aspect_ratio}: radius sum = {radius_sum:.6f}")

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
    circles = packer.optimize_with_multiple_aspect_ratios()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")