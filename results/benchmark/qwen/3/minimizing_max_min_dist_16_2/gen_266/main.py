# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math
import time

# Configuration constants
MAX_ITERATIONS = 100
POPULATION_SIZE = 15
LOCAL_REFINEMENT_ITERATIONS = 200
BOUNDARY_PENALTY = 0.01
MIN_BOUNDARY_DISTANCE = 0.02

class PointOptimizer:
    """Optimizes point placement to maximize min/max distance ratio using modular approach."""

    def __init__(self):
        self.best_points = None
        self.best_ratio = 0.0
        self.eval_time = 0.0

    def compute_distance_matrix(self, points):
        """Compute pairwise distance matrix for given points."""
        return squareform(pdist(points))

    def calculate_min_max_ratio(self, distance_matrix):
        """Calculate the ratio of minimum to maximum distances."""
        # Exclude diagonal (distance to self)
        off_diagonal = distance_matrix[distance_matrix > 0]
        if len(off_diagonal) == 0:
            return 0.0
        d_min = np.min(off_diagonal)
        d_max = np.max(off_diagonal)
        return d_min / d_max if d_max > 0 else 0.0

    def initialize_hexagonal_grid(self):
        """Initialize points using enhanced hexagonal packing principles."""
        points = []
        rows = 4
        cols = 4

        # Create a more optimized hexagonal pattern with better spacing
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2
        col_spacing = 1.0

        # Create hexagonal grid with improved spacing
        for i in range(rows):
            for j in range(cols):
                # Offset every other row for proper hexagonal packing
                x = j * col_spacing + (i % 2) * col_spacing / 2
                y = i * row_spacing
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points)

        # Normalize to fit within [0,1] square with better scaling
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        # Avoid division by zero and scale properly
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Apply more sophisticated boundary-aware scaling
        scale_factor = 0.95
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        points[:, 0] = 0.025 + scale_factor * (points[:, 0] - center_x) + 0.5
        points[:, 1] = 0.025 + scale_factor * (points[:, 1] - center_y) + 0.5

        # Add sophisticated symmetry-breaking perturbations using mathematical patterns
        np.random.seed(42)
        # Apply mathematically designed perturbations to break all symmetries
        for i in range(len(points)):
            # Use a combination of trigonometric functions and position-based factors
            angle = i * 0.785398  # pi/4 increments for good symmetry breaking
            # Position-dependent amplitude that varies with the index
            amplitude = 0.008 + 0.002 * math.sin(i * 0.5) + 0.001 * math.cos(i * 0.3)

            # Add both deterministic and random components
            deterministic_x = amplitude * math.sin(angle) * 0.5
            deterministic_y = amplitude * math.cos(angle) * 0.5

            # Add random component with position-specific variance
            random_x = np.random.normal(0, amplitude * 0.3)
            random_y = np.random.normal(0, amplitude * 0.3)

            points[i] += [deterministic_x + random_x, deterministic_y + random_y]

        # Keep within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)

        return points

    def initialize_random_points(self):
        """Initialize points randomly with better distribution properties."""
        np.random.seed(42)
        return np.random.uniform(0, 1, (16, 2))

    def initialize_voronoi_based(self):
        """Initialize points using a Voronoi-inspired approach."""
        # Create a 4x4 grid with slight perturbations
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Add some randomness to avoid perfect symmetry
                x = j + np.random.normal(0, 0.02)
                y = i + np.random.normal(0, 0.02)
                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] range
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Apply boundary constraints
        points[:, 0] = np.clip(points[:, 0], MIN_BOUNDARY_DISTANCE, 1 - MIN_BOUNDARY_DISTANCE)
        points[:, 1] = np.clip(points[:, 1], MIN_BOUNDARY_DISTANCE, 1 - MIN_BOUNDARY_DISTANCE)

        return points

    def evaluate_initialization_strategy(self, initial_points):
        """Evaluate a given initialization strategy and return the resulting ratio."""
        try:
            # Apply local refinement first to get a better starting point
            refined_points = self.local_refinement(initial_points, LOCAL_REFINEMENT_ITERATIONS)

            # Calculate ratio
            dist_matrix = self.compute_distance_matrix(refined_points)
            ratio = self.calculate_min_max_ratio(dist_matrix)

            return ratio, refined_points
        except Exception:
            return 0.0, initial_points

    def local_refinement(self, initial_points, max_iter):
        """Apply local optimization using L-BFGS-B."""
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            points = np.clip(points, 0, 1)

            try:
                dist_matrix = self.compute_distance_matrix(points)
                ratio = self.calculate_min_max_ratio(dist_matrix)

                # Penalty for points too close to boundary
                penalty = 0
                if np.any(points < MIN_BOUNDARY_DISTANCE) or np.any(points > 1 - MIN_BOUNDARY_DISTANCE):
                    penalty = -BOUNDARY_PENALTY

                return -(ratio + penalty)
            except Exception:
                return 1e6  # Return large value for invalid configurations

        x0 = initial_points.flatten()
        bounds = [(0, 1) for _ in range(len(x0))]

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5},
                callback=None
            )
            return result.x.reshape(-1, 2)
        except Exception:
            return initial_points

    def optimize_with_differential_evolution(self, initial_points):
        """Optimize using differential evolution for global search."""
        def objective_function(points_flat):
            points = points_flat.reshape((16, 2))
            points = np.clip(points, 0, 1)

            try:
                distances = pdist(points)
                d_min = np.min(distances)
                d_max = np.max(distances)

                if d_max == 0:
                    return float('inf')

                # Add penalty for boundary proximity
                penalty = 0
                if np.any(points < MIN_BOUNDARY_DISTANCE) or np.any(points > 1 - MIN_BOUNDARY_DISTANCE):
                    penalty = -BOUNDARY_PENALTY

                return -(d_min / d_max + penalty)
            except Exception:
                return 1e6

        bounds = [(0, 1)] * 32
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=MAX_ITERATIONS,
            popsize=POPULATION_SIZE,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )

        final_points = result.x.reshape((16, 2))
        final_points[:, 0] = np.clip(final_points[:, 0], 0, 1)
        final_points[:, 1] = np.clip(final_points[:, 1], 0, 1)

        return final_points

    def find_optimal_configuration(self):
        """Find the optimal point configuration using multi-strategy approach."""
        # Initialize strategies
        strategies = [
            ("hexagonal", self.initialize_hexagonal_grid),
            ("random", self.initialize_random_points),
            ("voronoi", self.initialize_voronoi_based)
        ]

        best_ratio = 0.0
        best_points = None

        for strategy_name, init_func in strategies:
            try:
                # Initialize using strategy
                initial_points = init_func()

                # Evaluate initialization
                ratio, evaluated_points = self.evaluate_initialization_strategy(initial_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = evaluated_points.copy()

            except Exception:
                continue

        # If no good configuration found, fallback to differential evolution
        if best_points is None:
            initial_points = self.initialize_hexagonal_grid()
            best_points = self.optimize_with_differential_evolution(initial_points)

        # Final refinement
        try:
            final_points = self.local_refinement(best_points, LOCAL_REFINEMENT_ITERATIONS)
            dist_matrix = self.compute_distance_matrix(final_points)
            final_ratio = self.calculate_min_max_ratio(dist_matrix)

            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points
        except Exception:
            pass

        return best_points

    def run_optimization(self) -> np.ndarray:
        """Run the complete optimization process."""
        start_time = time.time()

        try:
            # Find optimal configuration
            best_points = self.find_optimal_configuration()

            # Calculate final metrics
            dist_matrix = self.compute_distance_matrix(best_points)
            final_ratio = self.calculate_min_max_ratio(dist_matrix)

            self.best_ratio = final_ratio
            self.eval_time = time.time() - start_time

            return best_points

        except Exception as e:
            # Fallback to simple random initialization
            np.random.seed(42)
            fallback_points = np.random.uniform(0, 1, (16, 2))
            self.eval_time = time.time() - start_time
            return fallback_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointOptimizer()
    return optimizer.run_optimization()

# EVOLVE-BLOCK-END