# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time


class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""

    def __init__(self, n_points=16, dimension=2, seed=42):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)

    def initialize_points(self) -> np.ndarray:
        """Initialize points using enhanced hexagonal grid approach with multi-start strategy."""
        # Create multiple candidate initializations and select the best one
        best_initialization = None
        best_ratio = 0

        # Try several different initialization strategies
        for init_seed in [42, 123, 456, 789, 987]:
            np.random.seed(init_seed)

            points = []
            sqrt3 = np.sqrt(3)

            # Use a 4x4 hexagonal pattern with optimized spacing
            # Calculate spacing based on mathematical optimization for 16 points
            # Ideal spacing for equidistant arrangement in unit square
            spacing_x = 0.8  # Reduced spacing to encourage better packing
            spacing_y = spacing_x * sqrt3 / 2.0

            # Create hexagonal grid with more structured pattern
            for i in range(4):
                row_offset = i * spacing_y
                for j in range(4):
                    col_offset = j * spacing_x + (i % 2) * spacing_x / 2.0
                    points.append([col_offset, row_offset])

            # Convert to numpy array and truncate to required number of points
            points = np.array(points[:self.n_points])

            # Normalize to [0,1] with careful scaling and centering
            if len(points) > 0:
                x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
                y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

                # Normalize to a reasonable range within unit square
                if x_max > x_min and y_max > y_min:
                    # Scale and center the arrangement
                    scale_x = 0.9 / (x_max - x_min) if (x_max - x_min) > 1e-10 else 1.0
                    scale_y = 0.9 / (y_max - y_min) if (y_max - y_min) > 1e-10 else 1.0

                    points[:, 0] = (points[:, 0] - x_min) * scale_x + 0.05
                    points[:, 1] = (points[:, 1] - y_min) * scale_y + 0.05
                else:
                    # Fallback for degenerate cases
                    points[:, 0] = 0.5
                    points[:, 1] = 0.5

            # Ensure bounds
            points = np.clip(points, 0, 1)

            # Strategic perturbations for symmetry breaking
            # Use a combination of structured and random perturbations
            for i in range(self.n_points):
                # Position-based perturbation with more mathematical foundation
                row = i // 4
                col = i % 4

                # Base perturbation magnitude with more strategic variation
                base_magnitude = 0.012 + (row + col) * 0.001

                # Add structured component based on position
                structured_noise = np.array([
                    ((row % 3) - 1) * 0.002,  # Row-based component
                    ((col % 3) - 1) * 0.002   # Column-based component
                ])

                # Add random component
                random_noise = np.random.normal(0, base_magnitude, 2)

                # Combine both components
                total_noise = structured_noise + random_noise

                points[i] += total_noise

            # Clip final points to ensure they stay within bounds
            points = np.clip(points, 0, 1)

            # Evaluate this initialization
            initial_ratio = self.evaluate_ratio(points)
            if initial_ratio > best_ratio:
                best_ratio = initial_ratio
                best_initialization = points.copy()

        return best_initialization if best_initialization is not None else self._default_initialization()

    def _default_initialization(self) -> np.ndarray:
        """Fallback initialization if all others fail."""
        return np.random.random((self.n_points, self.dimension))

    def calculate_distances(self, points: np.ndarray) -> tuple:
        """Calculate minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0, 0

        # Calculate pairwise distances
        distances = pdist(points)

        # Get min and max distances
        min_distance = np.min(distances)
        max_distance = np.max(distances)

        return min_distance, max_distance

    def evaluate_ratio(self, points: np.ndarray) -> float:
        """Evaluate the min/max distance ratio."""
        min_d, max_d = self.calculate_distances(points)

        if max_d <= 0:
            return 0

        return min_d / max_d

    def perturb_point(self, points: np.ndarray, idx: int, step_size: float = 0.01) -> np.ndarray:
        """Perturb a specific point."""
        new_points = points.copy()

        # Random perturbation
        delta = np.random.uniform(-step_size, step_size, self.dimension)
        new_points[idx] = points[idx] + delta

        # Boundary check
        new_points[idx] = np.clip(new_points[idx], 0, 1)

        return new_points

    def optimize(self, max_iterations: int = 10000, initial_temp: float = 1.0,
                cooling_rate: float = 0.9995) -> np.ndarray:
        """Optimize point distribution using simulated annealing."""

        # Initialize points
        current_points = self.initialize_points()
        current_ratio = self.evaluate_ratio(current_points)

        best_points = current_points.copy()
        best_ratio = current_ratio

        temperature = initial_temp

        for iteration in range(max_iterations):
            # Choose a random point to perturb
            point_idx = np.random.randint(0, self.n_points)

            # Perturb the point
            new_points = self.perturb_point(current_points, point_idx)

            # Evaluate new solution
            new_ratio = self.evaluate_ratio(new_points)

            # Accept or reject the move
            if new_ratio > current_ratio:
                # Always accept better solutions
                current_points = new_points
                current_ratio = new_ratio

                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
            else:
                # Accept worse solutions with probability based on temperature
                if np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                    current_points = new_points
                    current_ratio = new_ratio

            # Cool down
            temperature *= cooling_rate

            # Early stopping condition
            if iteration > 1000 and abs(current_ratio - best_ratio) < 1e-8:
                break

        return best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Create optimizer instance
    optimizer = PointDispersionOptimizer(n_points=16, dimension=2, seed=42)

    # Run optimization
    start_time = time.time()
    optimized_points = optimizer.optimize(max_iterations=5000)
    end_time = time.time()

    # Debug output
    final_ratio = optimizer.evaluate_ratio(optimized_points)

    return optimized_points


# EVOLVE-BLOCK-END