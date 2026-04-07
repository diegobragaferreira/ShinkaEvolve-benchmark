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
        """Initialize points using a more sophisticated hexagonal grid approach."""
        # Create a hexagonal arrangement that better approximates optimal distribution
        # Using 4 rows with alternating columns but with optimized spacing

        points = []
        sqrt3 = np.sqrt(3)

        # Optimal spacing parameters for 16 points in a roughly hexagonal pattern
        # These values are chosen to balance minimum and maximum distances
        spacing_x = 0.8  # Reduced from 1.0 to allow better packing
        spacing_y = spacing_x * sqrt3 / 2.0

        # Create hexagonal grid with 4 rows and 4 columns
        for i in range(4):
            row_offset = i * spacing_y
            for j in range(4):
                col_offset = j * spacing_x + (i % 2) * spacing_x / 2.0
                points.append([col_offset, row_offset])

        # Convert to numpy array and truncate to required number of points
        points = np.array(points[:self.n_points])

        # Normalize to [0,1] range more carefully to avoid extreme scaling
        if len(points) > 0:
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

            # Avoid division by zero
            if x_max > x_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            else:
                points[:, 0] = 0.5

            if y_max > y_min:
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
            else:
                points[:, 1] = 0.5

        # Ensure bounds
        points = np.clip(points, 0, 1)

        # Add more strategic perturbations to break symmetry effectively
        # Use different noise levels for different points to prevent symmetric solutions
        np.random.seed(self.seed)  # Ensure reproducibility
        noise_magnitudes = np.linspace(0.01, 0.005, self.n_points)

        for i in range(self.n_points):
            noise = np.random.normal(0, noise_magnitudes[i], self.dimension)
            points[i] += noise

        # Clip final points to ensure they stay within bounds
        points = np.clip(points, 0, 1)

        return points

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