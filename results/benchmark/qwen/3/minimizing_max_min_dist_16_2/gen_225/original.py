# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

class PointInitializer:
    """Handles various point initialization strategies for optimal starting configurations."""

    @staticmethod
    def hexagonal_grid(num_points=16):
        """Initialize points using a hexagonal grid pattern with mathematical precision."""
        points = np.zeros((num_points, 2))

        # Create true hexagonal arrangement
        row_spacing = np.sqrt(3) / 2
        col_spacing = 1.0

        # Generate 4x4 hexagonal grid
        idx = 0
        for row in range(4):
            for col in range(4):
                if idx >= num_points:
                    break
                x = col * col_spacing + (row % 2) * col_spacing * 0.5
                y = row * row_spacing
                points[idx, 0] = x
                points[idx, 1] = y
                idx += 1

        # Normalize to [0,1] x [0,1]
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Apply controlled symmetry breaking with deterministic pattern
        np.random.seed(42)
        for i in range(len(points)):
            # Apply position-based perturbation strengths
            if i % 4 == 0:  # Corner points
                scale = 0.02
            elif i % 3 == 0:  # Strategic points
                scale = 0.015
            else:  # Others
                scale = 0.01

            points[i, 0] += np.random.normal(0, scale * 0.7)
            points[i, 1] += np.random.normal(0, scale * 0.7)

        return np.clip(points, 0, 1)

    @staticmethod
    def triangular_lattice(num_points=16):
        """Initialize points using triangular lattice arrangement."""
        points = []

        # Create triangular lattice (approximately 16 points)
        rows = 5
        cols = 4

        for i in range(rows):
            for j in range(cols):
                if len(points) >= num_points:
                    break
                x = j + (i % 2) * 0.5
                y = i * np.sqrt(3) / 2
                points.append([x, y])

        points = np.array(points[:num_points])

        # Normalize to [0,1] x [0,1]
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Add noise to break symmetry
        points += np.random.normal(0, 0.01, points.shape)
        return np.clip(points, 0, 1)

    @staticmethod
    def random_points(num_points=16):
        """Initialize points using completely random distribution."""
        return np.random.rand(num_points, 2)

    @staticmethod
    def perturbed_hexagonal(num_points=16, noise_level=0.015):
        """Initialize with hexagonal grid perturbed with varying noise levels."""
        points = PointInitializer.hexagonal_grid(num_points)

        # Apply different noise levels to create diversity
        for i in range(len(points)):
            noise_scale = noise_level * (0.5 + 0.5 * np.random.random())
            points[i] += np.random.normal(0, noise_scale, 2)

        return np.clip(points, 0, 1)

class Optimizer:
    """Handles the simulated annealing optimization process."""

    def __init__(self):
        self.best_points = None
        self.best_ratio = 0.0

    def compute_min_max_ratio(self, points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Efficient distance computation using cdist
        distances = cdist(points, points)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances (excluding infinity values)
        d_min = np.min(distances[np.isfinite(distances)])
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max <= 0:
            return 0.0

        return d_min / d_max

    def compute_boundary_penalty(self, points, penalty_factor=1e6):
        """Compute penalty for points near boundaries."""
        penalty = 0
        boundary_threshold = 0.01

        for point in points:
            min_dist_to_boundaries = min(point[0], 1-point[0], point[1], 1-point[1])
            if min_dist_to_boundaries < boundary_threshold:
                penalty += penalty_factor * (boundary_threshold - min_dist_to_boundaries)

        return penalty

    def compute_min_max_ratio_with_penalty(self, points, penalty_factor=1e6):
        """Compute ratio with boundary penalty."""
        base_ratio = self.compute_min_max_ratio(points)
        penalty = self.compute_boundary_penalty(points, penalty_factor)
        return base_ratio - penalty

    def perturb_points(self, points, magnitude=0.01, method='individual'):
        """Apply small perturbations to points with boundary checking."""
        new_points = points.copy()

        if method == 'individual':
            # Select random subset of points to perturb
            num_perturb = max(1, len(points) // 4)
            indices = np.random.choice(len(points), size=num_perturb, replace=False)

            for idx in indices:
                delta = np.random.uniform(-magnitude, magnitude, 2)
                new_points[idx] += delta
                new_points[idx] = np.clip(new_points[idx], 0, 1)

        elif method == 'cluster':
            # Perturb small clusters together
            cluster_size = min(3, len(points) // 4)
            num_clusters = len(points) // cluster_size

            for i in range(num_clusters):
                start_idx = i * cluster_size
                end_idx = min(start_idx + cluster_size, len(points))

                # Find centroid of cluster
                cluster_center = np.mean(new_points[start_idx:end_idx], axis=0)

                # Apply same perturbation to whole cluster
                perturbation = np.random.uniform(-magnitude, magnitude, 2)
                for idx in range(start_idx, end_idx):
                    new_points[idx] += perturbation
                    new_points[idx] = np.clip(new_points[idx], 0, 1)

        return new_points

    def adaptive_cooling_schedule(self, iteration, max_iterations, base_cooling_rate=0.9995):
        """Provide adaptive cooling based on progress."""
        # Start with a high temperature and cool based on iteration
        temperature = 0.1 * (base_cooling_rate ** iteration)

        # Accelerate cooling if we've made significant progress recently
        if iteration > max_iterations * 0.7:
            temperature *= (0.999 ** (iteration - max_iterations * 0.7))

        return max(temperature, 1e-6)

    def optimize_single(self, initial_points, max_iterations=5000):
        """Run simulated annealing optimization from given initial points."""
        points = initial_points.copy()
        current_ratio = self.compute_min_max_ratio_with_penalty(points)

        best_points = points.copy()
        best_ratio = current_ratio

        # Tracking variables for adaptive cooling
        recent_improvements = []
        max_recent = 50

        # Optimization loop
        for iteration in range(max_iterations):
            # Alternate perturbation methods
            perturbation_method = 'cluster' if iteration % 3 == 0 else 'individual'

            # Perturb points with adaptive magnitude
            temperature = self.adaptive_cooling_schedule(iteration, max_iterations)
            new_points = self.perturb_points(
                points,
                magnitude=temperature * 0.5,
                method=perturbation_method
            )

            # Evaluate new configuration
            new_ratio = self.compute_min_max_ratio_with_penalty(new_points)

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                points = new_points
                current_ratio = new_ratio

                # Update best solution
                if current_ratio > best_ratio:
                    best_points = points.copy()
                    best_ratio = current_ratio
                    recent_improvements = []  # Reset improvement tracking
                else:
                    # Track recent improvements
                    if len(recent_improvements) < max_recent:
                        recent_improvements.append(current_ratio)
                    else:
                        recent_improvements.pop(0)
                        recent_improvements.append(current_ratio)

            # Early stopping if we're getting very good results
            if best_ratio > 0.3:
                break

        return best_points, best_ratio

class PointDispersionOptimizer:
    """Main optimizer class that orchestrates the optimization process."""

    def __init__(self):
        self.initializer = PointInitializer()
        self.optimizer = Optimizer()

    def optimize(self, max_iterations=5000):
        """Run multi-start optimization with diverse initialization strategies."""
        # Define multiple initialization strategies
        initializations = [
            self.initializer.hexagonal_grid,
            self.initializer.triangular_lattice,
            self.initializer.random_points,
            self.initializer.perturbed_hexagonal,
            lambda: self.initializer.hexagonal_grid() + np.random.normal(0, 0.005, (16, 2))
        ]

        best_overall_points = None
        best_overall_ratio = 0.0

        # Run optimization from each initialization
        for i, init_func in enumerate(initializations):
            # Set different seed for each initialization
            np.random.seed(i * 100 + 42)

            try:
                # Initialize points
                initial_points = init_func()

                # Optimize from this initialization
                optimized_points, optimized_ratio = self.optimizer.optimize_single(
                    initial_points, max_iterations=max_iterations
                )

                # Update global best if this run was better
                if optimized_ratio > best_overall_ratio:
                    best_overall_ratio = optimized_ratio
                    best_overall_points = optimized_points.copy()

            except Exception as e:
                continue  # Skip failed optimizations

        # Final safeguard
        if best_overall_points is None:
            best_overall_points = self.initializer.hexagonal_grid()

        return best_overall_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Create optimizer instance
    optimizer = PointDispersionOptimizer()

    # Run optimization
    result = optimizer.optimize(max_iterations=5000)

    return result

# EVOLVE-BLOCK-END