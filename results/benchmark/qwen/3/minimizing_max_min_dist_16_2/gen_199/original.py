# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import time


class PointDispersionOptimizer:
    """Optimizes point distribution to maximize min/max distance ratio."""

    def __init__(self, n_points=16, dimension=2, seed=42):
        self.n_points = n_points
        self.dimension = dimension
        self.seed = seed
        np.random.seed(seed)

    def _initialize_hexagonal_packing(self) -> np.ndarray:
        """Create optimized hexagonal packing pattern with systematic symmetry breaking."""
        # Mathematical constants for proper hexagonal packing
        sqrt3 = math.sqrt(3)
        row_spacing = sqrt3 / 2  # Vertical spacing between rows
        col_spacing = 1.0        # Horizontal spacing between columns

        points = []

        # Create hexagonal lattice pattern (more precise than previous version)
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Proper hexagonal offset for alternating rows
                x = j * col_spacing + (i % 2) * 0.5
                y = i * row_spacing
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points[:self.n_points])

        # Normalize to [0,1] bounds properly
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

        # Apply structured perturbations to break symmetry systematically
        for i in range(self.n_points):
            # Apply different noise patterns based on position and mathematical functions
            noise_intensity = 0.01 + 0.005 * math.sin(i * 0.785398)  # pi/4 increments
            noise_x = np.random.normal(0, noise_intensity, 1)[0]
            noise_y = np.random.normal(0, noise_intensity, 1)[0]
            points[i] += [noise_x, noise_y]

        # Clip to ensure bounds
        points = np.clip(points, 0, 1)

        return points

    def _initialize_random_points(self) -> np.ndarray:
        """Create random initial points."""
        return np.random.rand(self.n_points, self.dimension)

    def _initialize_grid_points(self) -> np.ndarray:
        """Create grid-based initial points."""
        n_per_side = int(np.ceil(np.sqrt(self.n_points)))
        x = np.linspace(0.1, 0.9, n_per_side)
        y = np.linspace(0.1, 0.9, n_per_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_points]
        return points

    def _calculate_distances(self, points: np.ndarray) -> tuple:
        """Calculate minimum and maximum distances efficiently."""
        if len(points) < 2:
            return 0, 0

        try:
            distances = pdist(points)

            if len(distances) == 0:
                return 0, 0

            min_distance = np.min(distances)
            max_distance = np.max(distances)

            return min_distance, max_distance
        except Exception:
            return 0, 0

    def _evaluate_ratio(self, points: np.ndarray) -> float:
        """Evaluate the min/max distance ratio with safety checks."""
        min_d, max_d = self._calculate_distances(points)

        if max_d <= 1e-12:
            return 0

        return min_d / max_d

    def _perturb_single(self, points: np.ndarray, idx: int, step_size: float = 0.005) -> np.ndarray:
        """Perturb a single point with boundary handling."""
        new_points = points.copy()
        delta = np.random.uniform(-step_size, step_size, self.dimension)
        new_points[idx] = points[idx] + delta
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points

    def _perturb_neighborhood(self, points: np.ndarray, indices: list, step_size: float = 0.005) -> np.ndarray:
        """Perturb a group of points together while preserving structure."""
        new_points = points.copy()
        centroid = np.mean(points[indices], axis=0)

        for idx in indices:
            # Apply coordinated perturbations
            delta = np.random.uniform(-step_size, step_size, self.dimension)
            new_points[idx] = points[idx] + delta
            new_points[idx] = np.clip(new_points[idx], 0, 1)

        return new_points

    def _adaptive_simulated_annealing(self, initial_points: np.ndarray, max_iterations: int = 5000) -> np.ndarray:
        """Enhanced adaptive simulated annealing optimization."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = self._evaluate_ratio(current_points)

        # Adaptive cooling schedule
        temperature = 1.0
        cooling_rate = 0.9995
        stagnation_counter = 0
        previous_best = best_ratio

        # Phase-based adjustments
        phase = 0
        phase_thresholds = [1000, 3000]

        for iteration in range(max_iterations):
            # Decide between single point and neighborhood perturbation
            if np.random.random() < 0.7:  # 70% chance of neighborhood move
                # Choose a random group of points to perturb together
                # Size of neighborhood: 2-4 points, adaptively chosen
                if iteration < 1000:
                    neighborhood_size = 2
                elif iteration < 3000:
                    neighborhood_size = np.random.randint(2, 4)
                else:
                    neighborhood_size = np.random.randint(2, min(5, self.n_points))

                indices = np.random.choice(self.n_points, neighborhood_size, replace=False).tolist()
                neighbor_points = self._perturb_neighborhood(current_points, indices, step_size=temperature * 0.05)
            else:
                # Single point perturbation (traditional approach)
                point_idx = np.random.randint(0, self.n_points)
                neighbor_points = self._perturb_single(current_points, point_idx, step_size=temperature * 0.05)

            # Evaluate new solution
            neighbor_ratio = self._evaluate_ratio(neighbor_points)

            # Accept or reject the move
            if neighbor_ratio > best_ratio:
                # Always accept better solutions
                current_points = neighbor_points
                best_points = neighbor_points
                best_ratio = neighbor_ratio
                stagnation_counter = 0  # Reset stagnation counter
            elif np.random.rand() < math.exp((neighbor_ratio - best_ratio) / temperature):
                current_points = neighbor_points
                stagnation_counter = 0  # Reset stagnation counter
            else:
                stagnation_counter += 1

            # Adaptive cooling: slow down when progress stalls
            if stagnation_counter > 50:
                temperature *= 0.995  # Faster cooling when stagnating
            else:
                # Phase-dependent cooling
                phase_cooling = cooling_rate * (0.95 if phase > 0 else 1.0)
                temperature *= phase_cooling

            # Phase transitions
            if iteration in phase_thresholds:
                phase += 1

            # Early stopping condition
            if iteration % 100 == 0 and iteration > 0:
                current_ratio = self._evaluate_ratio(best_points)
                if abs(previous_best - current_ratio) < 1e-8:
                    break
                previous_best = current_ratio

            # Early termination for slow progress
            if iteration > 1000 and temperature < 0.001:
                break

        return best_points

    def optimize_multiple_starts(self, max_iterations: int = 5000) -> np.ndarray:
        """Run optimization from multiple starting points and return best result."""
        initial_configs = [
            self._initialize_hexagonal_packing(),
            self._initialize_random_points(),
            self._initialize_grid_points()
        ]

        best_points = None
        best_ratio = -float('inf')

        for i, initial_config in enumerate(initial_configs):
            try:
                optimized_points = self._adaptive_simulated_annealing(initial_config, max_iterations)
                current_ratio = self._evaluate_ratio(optimized_points)

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = optimized_points.copy()

            except Exception as e:
                print(f"Warning: Optimization from start {i} failed: {e}")
                continue

        return best_points if best_points is not None else initial_configs[0]


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Create optimizer instance
    optimizer = PointDispersionOptimizer(n_points=16, dimension=2, seed=42)

    # Run optimization with multiple starts
    start_time = time.time()
    optimized_points = optimizer.optimize_multiple_starts(max_iterations=5000)
    end_time = time.time()

    # Debug output
    final_ratio = optimizer._evaluate_ratio(optimized_points)

    return optimized_points


# EVOLVE-BLOCK-END