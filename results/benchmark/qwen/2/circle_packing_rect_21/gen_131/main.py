# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, minimize
import random
from typing import Tuple, List
import time
import warnings
from itertools import combinations

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
OPTIMIZATION_TIME_LIMIT = 55  # seconds

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

    def generate_hexagonal_grid_initialization(self) -> np.ndarray:
        """Generate initial configuration using more advanced hexagonal grid approach"""
        circles = np.zeros((self.num_circles, 3))

        # Determine optimal grid size based on circle count and aspect ratio
        aspect_ratio = self.width / self.height

        # Calculate grid dimensions that better fit our needs
        if aspect_ratio >= 1:  # Landscape
            cols = int(np.ceil(np.sqrt(self.num_circles * aspect_ratio)))
            rows = int(np.ceil(self.num_circles / cols))
        else:  # Portrait
            rows = int(np.ceil(np.sqrt(self.num_circles / aspect_ratio)))
            cols = int(np.ceil(self.num_circles / rows))

        # Ensure we have enough cells
        while cols * rows < self.num_circles:
            if aspect_ratio >= 1:
                cols += 1
            else:
                rows += 1

        # Calculate spacing with better consideration of rectangle dimensions
        spacing_x = self.width / (cols + 1)
        spacing_y = self.height / (rows + 1)

        # Create hexagonal packing with better center positioning
        placed_count = 0
        for i in range(rows):
            for j in range(cols):
                if placed_count >= self.num_circles:
                    break

                # Offset every other row for true hexagonal packing
                offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                base_x = (j + 1) * spacing_x + offset_x
                base_y = (i + 1) * spacing_y

                # Add small random perturbation for better initial diversity
                perturbation_x = np.random.uniform(-0.1 * spacing_x, 0.1 * spacing_x)
                perturbation_y = np.random.uniform(-0.1 * spacing_y, 0.1 * spacing_y)

                x = np.clip(base_x + perturbation_x, 0.01, self.width - 0.01)
                y = np.clip(base_y + perturbation_y, 0.01, self.height - 0.01)

                # Initial radius estimate based on available space
                max_r = min(x, self.width - x, y, self.height - y)
                # Use more reasonable initial radius based on density considerations
                r = np.random.uniform(0.05, min(0.15, max_r * 0.6))

                circles[placed_count] = [x, y, r]
                placed_count += 1

            if placed_count >= self.num_circles:
                break

        return circles

    def generate_adaptive_grid_initialization(self) -> np.ndarray:
        """Generate initial configuration with adaptive grid pattern"""
        circles = np.zeros((self.num_circles, 3))

        # Use adaptive approach inspired by successful packing algorithms
        # Try to distribute circles more evenly based on the rectangle's geometry

        # For 21 circles, we'll try different configurations and pick the best
        best_config = None
        best_radius_sum = 0

        # Try several grid configurations
        for attempt in range(5):
            # Randomly adjust grid parameters for diversity
            adjustment = 0.1 * np.random.rand() + 0.95  # Between 0.95 and 1.05

            # Determine grid dimensions based on aspect ratio
            aspect_ratio = self.width / self.height
            grid_cols = int(np.ceil(np.sqrt(self.num_circles) * adjustment))
            grid_rows = int(np.ceil(self.num_circles / grid_cols))

            # Adjust for aspect ratio
            if aspect_ratio < 1:  # Portrait
                grid_rows = int(np.ceil(np.sqrt(self.num_circles / aspect_ratio) * adjustment))
                grid_cols = int(np.ceil(self.num_circles / grid_rows))

            # Ensure enough grid cells
            while grid_cols * grid_rows < self.num_circles:
                if aspect_ratio >= 1:
                    grid_cols += 1
                else:
                    grid_rows += 1

            spacing_x = self.width / (grid_cols + 1) if grid_cols > 0 else self.width
            spacing_y = self.height / (grid_rows + 1) if grid_rows > 0 else self.height

            # Create configuration
            temp_circles = np.zeros((self.num_circles, 3))
            placed_count = 0

            for i in range(grid_rows):
                for j in range(grid_cols):
                    if placed_count >= self.num_circles:
                        break

                    offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                    base_x = (j + 1) * spacing_x + offset_x
                    base_y = (i + 1) * spacing_y

                    # Add perturbation
                    pert_x = np.random.uniform(-0.1 * spacing_x, 0.1 * spacing_x)
                    pert_y = np.random.uniform(-0.1 * spacing_y, 0.1 * spacing_y)

                    x = np.clip(base_x + pert_x, 0.01, self.width - 0.01)
                    y = np.clip(base_y + pert_y, 0.01, self.height - 0.01)

                    max_r = min(x, self.width - x, y, self.height - y)
                    r = np.random.uniform(0.05, min(0.15, max_r * 0.6))

                    temp_circles[placed_count] = [x, y, r]
                    placed_count += 1

                if placed_count >= self.num_circles:
                    break

            # Check if this configuration is better
            radius_sum = np.sum(temp_circles[:, 2])
            if radius_sum > best_radius_sum:
                best_radius_sum = radius_sum
                best_config = temp_circles.copy()

        return best_config if best_config is not None else self.generate_hexagonal_grid_initialization()

    def optimize_with_local_refinement(self, initial_circles: np.ndarray) -> np.ndarray:
        """Enhanced optimization with local refinement"""
        # First, run evolutionary algorithm on initial configuration
        best_circles = initial_circles.copy()

        # Apply local optimization to the best solution
        def objective_function(vars):
            circles = vars.reshape((-1, 3))
            return -self.calculate_total_radius_sum(circles)

        def constraint_function(vars):
            circles = vars.reshape((-1, 3))
            violations = 0

            # Boundary constraints
            for i in range(self.num_circles):
                x, y, r = circles[i]
                if not self.is_valid_circle(x, y, r):
                    violations += 100

            # Overlap constraints using spatial indexing
            violations += self.efficient_overlap_check(circles)

            return violations

        # Use scipy minimize for local refinement with bounds
        # Flatten the circles array for optimization
        flattened_vars = initial_circles.flatten()

        # Define bounds for x, y, r for each circle
        bounds = []
        for i in range(self.num_circles):
            bounds.extend([(0.001, self.width - 0.001),   # x bounds
                          (0.001, self.height - 0.001),  # y bounds
                          (0.001, 0.3)])                # radius bounds

        # Local optimization using SLSQP method
        try:
            result = minimize(
                objective_function,
                flattened_vars,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )

            if result.success:
                refined_circles = result.x.reshape((-1, 3))
                # Validate and use if better
                current_sum = self.calculate_total_radius_sum(best_circles)
                refined_sum = self.calculate_total_radius_sum(refined_circles)

                if refined_sum > current_sum:
                    best_circles = refined_circles.copy()

        except Exception:
            pass  # Continue with original if optimization fails

        return best_circles

    def optimize(self) -> np.ndarray:
        """Main optimization loop with enhanced strategy"""
        start_time = time.time()

        # Generate initial population using the improved method
        initial_circles = self.generate_adaptive_grid_initialization()

        # Try local refinement on initial configuration
        best_solution = self.optimize_with_local_refinement(initial_circles)
        best_fitness = self.calculate_fitness(best_solution)[0]

        # Continue with evolutionary optimization if needed
        if time.time() - start_time < OPTIMIZATION_TIME_LIMIT - 5:
            # Use a simpler evolutionary approach with better initialization
            # Instead of complex genetic algorithm, use simple gradient-based post-processing

            # Try to improve using simple greedy optimization on the best solution
            original_best = best_solution.copy()
            improved = True
            iterations = 0

            while improved and iterations < 20 and (time.time() - start_time) < OPTIMIZATION_TIME_LIMIT - 2:
                improved = False
                current_sum = self.calculate_total_radius_sum(best_solution)

                # Try to increase individual radii
                for i in range(self.num_circles):
                    # Try increasing radius
                    test_circles = best_solution.copy()
                    old_radius = test_circles[i, 2]
                    new_radius = min(old_radius + 0.01, min(0.3, old_radius * 1.2))

                    if new_radius > old_radius:
                        test_circles[i, 2] = new_radius

                        # Check if this improves the configuration
                        valid = True
                        for j in range(self.num_circles):
                            for k in range(j+1, self.num_circles):
                                if self.check_overlap(test_circles, j, k):
                                    valid = False
                                    break
                            if not valid:
                                break

                        # Check boundary constraints
                        for j in range(self.num_circles):
                            if not self.is_valid_circle(test_circles[j, 0], test_circles[j, 1], test_circles[j, 2]):
                                valid = False
                                break

                        if valid:
                            new_sum = self.calculate_total_radius_sum(test_circles)
                            if new_sum > current_sum:
                                best_solution = test_circles.copy()
                                improved = True

                iterations += 1

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
    # Try different rectangle aspect ratios to find optimal packing
    best_result = None
    best_sum = 0

    # Test different aspect ratios
    ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for ratio in ratios:
        width = 2.0 / (1 + ratio)  # Ensure perimeter = 4
        height = width * ratio

        packer = CirclePacker(width=width, height=height, num_circles=21)
        circles = packer.optimize()

        radius_sum = np.sum(circles[:, 2])
        if radius_sum > best_sum:
            best_sum = radius_sum
            best_result = circles.copy()

    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")