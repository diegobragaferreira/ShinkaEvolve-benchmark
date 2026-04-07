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
        """Initialize points using an enhanced hexagonal grid approach."""
        # Create a more sophisticated hexagonal grid pattern for better initial distribution
        # Using the mathematical properties of hexagonal close packing

        # For 16 points, we can arrange in a hexagonal pattern with 4 points per row
        # Create points in a proper hexagonal lattice
        points = []

        # Hexagonal lattice parameters
        sqrt3 = np.sqrt(3)
        row_spacing = sqrt3 / 2  # Vertical spacing between rows
        col_spacing = 1.0        # Horizontal spacing between columns

        # Create multiple rows with alternating column offsets
        rows = 4
        cols_per_row = 4

        for i in range(rows):
            for j in range(cols_per_row):
                # Alternate column offset for hexagonal packing
                x_offset = (i % 2) * 0.5
                x = j * col_spacing + x_offset
                y = i * row_spacing

                points.append([x, y])

        # Convert to numpy array
        points = np.array(points[:self.n_points])

        # Normalize to fit within [0,1] bounds properly
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Apply more sophisticated perturbations to break symmetry
        # Use different noise magnitudes and patterns to avoid symmetric traps
        np.random.seed(self.seed)  # Ensure reproducible results

        # Apply systematic perturbations with varying amplitudes
        for i in range(self.n_points):
            # Different noise intensity for each point to break symmetry
            noise_intensity = 0.01 + i * 0.001
            noise_x = np.random.normal(0, noise_intensity, 1)[0]
            noise_y = np.random.normal(0, noise_intensity, 1)[0]
            points[i] += [noise_x, noise_y]

        # Ensure all points are clipped to [0,1] range
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
        stagnation_counter = 0
        last_best_ratio = best_ratio

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
                    stagnation_counter = 0  # Reset stagnation counter
            else:
                # Accept worse solutions with probability based on temperature
                if np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                    current_points = new_points
                    current_ratio = new_ratio
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1

            # Adaptive cooling schedule
            if stagnation_counter > 50:
                # If we're stagnating, cool faster to escape local optima
                temperature *= 0.995
            else:
                # Normal cooling
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