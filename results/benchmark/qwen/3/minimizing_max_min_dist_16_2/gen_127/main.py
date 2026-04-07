# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform


class PointDispersionOptimizer:
    """Optimizes point placement to maximize min/max distance ratio using simulated annealing."""

    def __init__(self, num_points=16, dimension=2):
        self.num_points = num_points
        self.dimension = dimension
        self.best_points = None
        self.best_ratio = 0.0
        self.iteration_count = 0

    def compute_min_max_ratio(self, points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max <= 0:
            return 0.0

        return d_min / d_max

    def compute_min_max_ratio_with_boundary_penalty(self, points, penalty_factor=1e6):
        """Compute ratio with penalty for points near boundaries."""
        # Check if any point is too close to boundary
        boundary_penalties = []
        for point in points:
            min_dist_to_boundaries = min(point[0], 1-point[0], point[1], 1-point[1])
            if min_dist_to_boundaries < 0.01:
                boundary_penalties.append(min_dist_to_boundaries)

        # Compute base ratio
        ratio = self.compute_min_max_ratio(points)

        # Apply penalty if needed
        if boundary_penalties:
            penalty = penalty_factor * sum(boundary_penalties)
            return ratio - penalty

        return ratio

    def initialize_hexagonal_points(self):
        """Initialize points using a more sophisticated hexagonal arrangement with mathematical precision."""
        # Create a true hexagonal lattice for 16 points
        # Using 4 rows with 4 columns, but arranged in hexagonal pattern
        points = []

        # Hexagonal packing constants
        phi = np.sqrt(3) / 2  # Vertical spacing factor
        row_spacing = 1.0
        col_spacing = 1.0

        # Generate hexagonal grid - optimized for 16 points
        # Arrange in 4 rows with alternating column offsets
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Hexagonal offset pattern
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * phi

                points.append([x, y])

        # Convert to numpy array
        points = np.array(points[:self.num_points])  # Take only required number

        # Normalize to [0,1] x [0,1] with proper scaling
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        # Avoid division by zero
        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Apply precise symmetry breaking with deterministic pattern
        # Use a more sophisticated approach to break symmetries
        np.random.seed(42)  # Fixed seed for reproducibility

        # Apply asymmetric perturbations that are mathematically designed
        # to break hexagonal symmetry while maintaining good geometric properties
        for i in range(len(points)):
            # Apply different perturbation strengths based on position
            if i % 4 == 0:  # Corner points - larger perturbations
                scale = 0.02
            elif i % 3 == 0:  # Some strategic points
                scale = 0.01
            else:  # Others - smaller perturbations
                scale = 0.005

            # Apply perturbations with controlled randomness
            points[i, 0] += np.random.normal(0, scale * 0.7)
            points[i, 1] += np.random.normal(0, scale * 0.7)

        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)

        return points

    def perturb_points(self, points, magnitude=0.01, method='individual'):
        """Apply small perturbations to points with boundary checking."""
        new_points = points.copy()

        if method == 'individual':
            # Select random subset of points to perturb
            num_perturb = max(1, len(points) // 4)
            indices = np.random.choice(len(points), size=num_perturb, replace=False)

            for idx in indices:
                # Apply perturbation
                delta = np.random.uniform(-magnitude, magnitude, 2)
                new_points[idx] += delta

                # Keep within bounds
                new_points[idx] = np.clip(new_points[idx], 0, 1)
        elif method == 'cluster':
            # Perturb small clusters together to maintain local structure
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

                    # Keep within bounds
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

    def initialize_random_points(self):
        """Initialize points using completely random distribution."""
        points = np.random.rand(self.num_points, self.dimension)
        return points

    def initialize_triangular_points(self):
        """Initialize points using a triangular lattice arrangement."""
        points = []
        # Create triangular lattice with approximately 16 points
        rows = 5
        cols = 4

        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.num_points:
                    break
                # Triangular offset pattern
                x = j + (i % 2) * 0.5
                y = i * np.sqrt(3) / 2
                points.append([x, y])

        points = np.array(points[:self.num_points])

        # Normalize to [0,1] x [0,1]
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Add small noise to break symmetry
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)

        return points

    def optimize(self, max_iterations=5000):
        """Run the simulated annealing optimization with multi-start strategy."""
        best_overall_points = None
        best_overall_ratio = 0.0

        # Run optimization from multiple starting points
        # This includes 4 different initialization strategies
        initializations = [
            self.initialize_hexagonal_points,
            self.initialize_random_points,
            self.initialize_triangular_points,
            lambda: self.initialize_hexagonal_points() + np.random.normal(0, 0.01, (self.num_points, self.dimension))
        ]

        # Run with different random seeds for reproducibility
        seeds = [42, 123, 456, 789]

        for seed_idx, (initialization_func, seed) in enumerate(zip(initializations, seeds)):
            np.random.seed(seed)

            # Initialize points
            current_points = initialization_func()
            current_ratio = self.compute_min_max_ratio_with_boundary_penalty(current_points)

            # Local best for this starting point
            local_best_points = current_points.copy()
            local_best_ratio = current_ratio

            # Tracking variables for adaptive cooling
            recent_improvements = []
            max_recent = 50

            # Optimization loop
            for iteration in range(max_iterations):
                # Alternate perturbation methods
                perturbation_method = 'cluster' if iteration % 3 == 0 else 'individual'

                # Perturb points
                new_points = self.perturb_points(
                    current_points,
                    magnitude=self.adaptive_cooling_schedule(iteration, max_iterations) * 0.5,
                    method=perturbation_method
                )

                # Evaluate new configuration
                new_ratio = self.compute_min_max_ratio_with_boundary_penalty(new_points)

                # Accept or reject based on Metropolis criterion
                temperature = self.adaptive_cooling_schedule(iteration, max_iterations)

                if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                    current_points = new_points
                    current_ratio = new_ratio

                    # Update local best solution
                    if current_ratio > local_best_ratio:
                        local_best_points = current_points.copy()
                        local_best_ratio = current_ratio
                        recent_improvements = []  # Reset improvement tracking
                    else:
                        # Track recent improvements
                        if len(recent_improvements) < max_recent:
                            recent_improvements.append(current_ratio)
                        else:
                            recent_improvements.pop(0)
                            recent_improvements.append(current_ratio)

                # Adaptive cooling based on recent improvement variance
                if len(recent_improvements) >= 10:
                    recent_std = np.std(recent_improvements[-10:])
                    if recent_std < 0.001 * local_best_ratio:  # Very little variation
                        # Slow down cooling to allow more exploration
                        pass  # Already handled by cooling schedule

                # Early stopping if we're getting very good results
                if local_best_ratio > 0.3:  # Early exit threshold
                    break

            # Update overall best if this run was better
            if local_best_ratio > best_overall_ratio:
                best_overall_ratio = local_best_ratio
                best_overall_points = local_best_points.copy()

        self.best_points = best_overall_points
        self.best_ratio = best_overall_ratio
        return self.best_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Create optimizer instance
    optimizer = PointDispersionOptimizer(num_points=16, dimension=2)

    # Run optimization
    result = optimizer.optimize(max_iterations=5000)

    return result


# EVOLVE-BLOCK-END