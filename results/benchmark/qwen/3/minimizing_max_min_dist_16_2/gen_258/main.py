# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math
import time

# Configuration constants
MAX_ITERATIONS = 50
POPULATION_SIZE = 20
LOCAL_REFINEMENT_ITERATIONS = 300
BOUNDARY_PENALTY = 0.02
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
        """Initialize points using enhanced hexagonal packing principles with better symmetry breaking."""
        points = []

        # Create a more sophisticated hexagonal arrangement that better approximates optimal distribution
        # Using 4 rows with 4 columns but optimized for better point separation

        sqrt3 = math.sqrt(3)
        # Use golden ratio-inspired spacing for more uniform distribution
        spacing_x = 1.0 / 1.618  # Golden ratio approximation
        spacing_y = spacing_x * sqrt3 / 2

        # Create hexagonal pattern with proper offsetting
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Proper hexagonal offsetting
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y

                points.append([x, y])
                if len(points) >= 16:
                    break

        # Convert to numpy array and ensure exactly 16 points
        points = np.array(points[:16])

        # Normalize to [0,1] with better distribution preserving properties
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        # Avoid division by zero
        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        else:
            points[:, 0] = 0.5

        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        else:
            points[:, 1] = 0.5

        # Apply more sophisticated normalization that preserves aspect ratio better
        # Scale to make full use of available space while keeping points centered
        scale_factor = 0.85
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        points[:, 0] = 0.1 + scale_factor * (points[:, 0] - center_x) + 0.5
        points[:, 1] = 0.1 + scale_factor * (points[:, 1] - center_y) + 0.5

        # Apply more effective symmetry-breaking perturbations
        np.random.seed(42)

        # Use a more mathematical approach to perturbations that breaks symmetry effectively
        # Apply position-dependent perturbations with sine/cosine patterns
        for i in range(len(points)):
            # Pattern based on position that ensures good distribution
            row = i // 4
            col = i % 4

            # More structured perturbation using trigonometric functions
            angle_factor = (row + col) * 0.314  # π/10 increment
            magnitude = 0.006 + 0.003 * math.sin(angle_factor)

            # Apply both x and y perturbations with phase difference
            perturbation_x = magnitude * math.sin(angle_factor * 2 + i)
            perturbation_y = magnitude * math.cos(angle_factor + i * 0.7)

            points[i, 0] += perturbation_x
            points[i, 1] += perturbation_y

        # Keep within bounds
        points[:, 0] = np.clip(points[:, 0], 0, 1)
        points[:, 1] = np.clip(points[:, 1], 0, 1)

        return points

    def initialize_random_points(self):
        """Initialize points randomly with better distribution properties."""
        np.random.seed(42)
        return np.random.uniform(0, 1, (16, 2))

    def initialize_voronoi_based(self):
        """Initialize points using a Voronoi-inspired approach with better distribution."""
        # Create a 4x4 grid with slight perturbations and better spacing
        points = []
        rows = 4
        cols = 4

        # Create a more structured grid with better spacing
        spacing_x = 1.0
        spacing_y = 1.0 / 3.0

        for i in range(rows):
            for j in range(cols):
                # Add some randomness to avoid perfect symmetry but maintain structure
                x = j * spacing_x + np.random.normal(0, 0.03)
                y = i * spacing_y + np.random.normal(0, 0.03)
                points.append([x, y])

        points = np.array(points)

        # Normalize to [0,1] range with better control over distribution
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        # Handle edge cases where there might be no range
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Better boundary constraints with improved normalization
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
        """Apply local optimization using L-BFGS-B with enhanced penalties."""
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

                # Additional penalty to avoid numerical issues
                if ratio < 1e-10:
                    penalty -= 1.0

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
        """Optimize using differential evolution for global search with better parameters."""
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

                # Additional penalty for extreme values
                if d_min == 0:
                    penalty -= 0.1

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

        # Final refinement with enhanced local optimization
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