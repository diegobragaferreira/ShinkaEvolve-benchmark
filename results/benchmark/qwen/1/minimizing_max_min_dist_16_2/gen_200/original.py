# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List, Optional
import math

class PointEvolutionOptimizer:
    """Advanced optimizer for point distribution maximizing min/max distance ratio."""

    def __init__(self, n_points: int = 16, dimensions: int = 2, max_time: float = 180.0):
        self.n_points = n_points
        self.dimensions = dimensions
        self.benchmark_ratio = 1 / np.sqrt(12.889266112)  # ~0.2786
        self.max_time = max_time
        self.start_time = time.time()

    def _calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

        try:
            distances = pdist(points)

            # Handle edge cases
            if len(distances) == 0 or np.max(distances) <= 0:
                return 0.0

            d_min = np.min(distances)
            d_max = np.max(distances)

            # Avoid division by zero
            if d_max <= 0:
                return 0.0

            return d_min / d_max
        except Exception:
            return 0.0

    def _generate_hexagonal_grid(self) -> np.ndarray:
        """Generate points in a hexagonal grid pattern."""
        # Create a grid that approximates hexagonal packing
        rows = int(np.ceil(np.sqrt(self.n_points)))
        cols = int(np.ceil(self.n_points / rows))

        points = []
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows

        for i in range(rows):
            for j in range(cols):
                if len(points) >= self.n_points:
                    break

                # Offset odd rows for hexagonal arrangement
                offset = 0.5 * (i % 2)
                x = (j + offset) * spacing_x
                y = i * spacing_y

                # Ensure we don't exceed bounds
                x = min(x, 0.99)
                y = min(y, 0.99)

                points.append([x, y])

        # Trim to exact number of points
        points = np.array(points[:self.n_points])

        # Normalize to fit properly in [0,1] box
        if len(points) > 0:
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05

        return points

    def _generate_spiral_pattern(self) -> np.ndarray:
        """Generate points in a spiral pattern."""
        points = []
        angle_step = 2 * np.pi / 10
        radius_step = 1.0 / 10

        for i in range(self.n_points):
            if i == 0:
                points.append([0.5, 0.5])  # Center point
            else:
                angle = i * angle_step
                radius = min(0.45, i * radius_step)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])

        # Fill remaining points with random if needed
        while len(points) < self.n_points:
            points.append([np.random.rand(), np.random.rand()])

        return np.array(points[:self.n_points])

    def _generate_fibonacci_sphere(self) -> np.ndarray:
        """Generate points using fibonacci sphere algorithm and project to 2D."""
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        # Project 3D sphere points to 2D using stereographic projection
        points_2d = []
        for x, y, z in points:
            # Stereographic projection from south pole
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])

        # Normalize to unit square
        points_2d = np.array(points_2d)

        # Scale and center the points
        x_min, y_min = np.min(points_2d, axis=0)
        x_max, y_max = np.max(points_2d, axis=0)

        if x_max > x_min and y_max > y_min:
            points_2d[:, 0] = (points_2d[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            points_2d[:, 1] = (points_2d[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05

        return points_2d

    def _generate_grid_with_jitter(self) -> np.ndarray:
        """Generate grid pattern with random jitter."""
        points = []
        grid_size = int(np.ceil(np.sqrt(self.n_points)))

        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= self.n_points:
                    break
                x = i / (grid_size - 1) if grid_size > 1 else 0.5
                y = j / (grid_size - 1) if grid_size > 1 else 0.5
                # Add slight randomness to avoid perfect grid
                x += (np.random.rand() - 0.5) * 0.1
                y += (np.random.rand() - 0.5) * 0.1
                points.append([x, y])

        return np.clip(np.array(points[:self.n_points]), 0, 1)

    def _generate_initial_population(self) -> List[np.ndarray]:
        """Generate diverse initial configurations."""
        population = []
        np.random.seed(42)

        # Strategy 1: Hexagonal grid
        try:
            population.append(self._generate_hexagonal_grid())
        except Exception:
            population.append(np.random.rand(self.n_points, self.dimensions))

        # Strategy 2: Spiral pattern
        try:
            population.append(self._generate_spiral_pattern())
        except Exception:
            population.append(np.random.rand(self.n_points, self.dimensions))

        # Strategy 3: Fibonacci sphere projection
        try:
            population.append(self._generate_fibonacci_sphere())
        except Exception:
            population.append(np.random.rand(self.n_points, self.dimensions))

        # Strategy 4: Grid with jitter
        try:
            population.append(self._generate_grid_with_jitter())
        except Exception:
            population.append(np.random.rand(self.n_points, self.dimensions))

        # Strategy 5: Pure random
        population.append(np.random.rand(self.n_points, self.dimensions))

        return population

    def _time_remaining(self) -> float:
        """Check if there's enough time remaining."""
        return self.max_time - (time.time() - self.start_time)

    def _evaluate_initial_configurations(self, configs: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """Evaluate initial configurations and return sorted by performance."""
        evaluations = []
        for config in configs:
            ratio = self._calculate_min_max_ratio(config)
            evaluations.append((config, ratio))

        # Sort by ratio descending
        evaluations.sort(key=lambda x: x[1], reverse=True)
        return evaluations

    def _global_search(self, initial_points: np.ndarray, max_iter: int = 200) -> Tuple[np.ndarray, float]:
        """Use global optimization to explore promising regions."""
        if self._time_remaining() < 10:
            return initial_points, self._calculate_min_max_ratio(initial_points)

        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self._calculate_min_max_ratio(points)

        bounds = [(0, 1) for _ in range(len(initial_points.flatten()))]

        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self._calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception as e:
            pass

        return initial_points, self._calculate_min_max_ratio(initial_points)

    def _local_refinement(self, points: np.ndarray, max_iter: int = 300) -> Tuple[np.ndarray, float]:
        """Apply local refinement using multiple methods."""
        if self._time_remaining() < 10:
            return points, self._calculate_min_max_ratio(points)

        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, self.dimensions)
            return -self._calculate_min_max_ratio(points_candidate)

        best_points = points.copy()
        best_ratio = self._calculate_min_max_ratio(best_points)

        # Method 1: L-BFGS-B - most reliable
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(points.flatten()))],
                options={'maxiter': max_iter},
                tol=1e-6
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self._calculate_min_max_ratio(optimized_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass

        # Method 2: Nelder-Mead for additional exploration
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter // 2, 'adaptive': True}
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self._calculate_min_max_ratio(optimized_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass

        return best_points, best_ratio

    def optimize(self) -> np.ndarray:
        """Main optimization process with hierarchical approach."""
        if self._time_remaining() < 10:
            # Fallback to simple approach
            return np.random.rand(self.n_points, self.dimensions)

        # Step 1: Generate initial population
        initial_configs = self._generate_initial_population()

        # Step 2: Evaluate initial configurations
        evaluated_configs = self._evaluate_initial_configurations(initial_configs)

        # Step 3: Select best initial configuration
        best_initial_config = evaluated_configs[0][0]
        best_ratio = evaluated_configs[0][1]

        # Step 4: Apply global search to the best initial
        if self._time_remaining() > 20:
            global_points, global_ratio = self._global_search(best_initial_config)

            if global_ratio > best_ratio:
                best_ratio = global_ratio
                best_initial_config = global_points

        # Step 5: Apply progressive local refinement
        if self._time_remaining() > 15:
            # Coarse refinement
            coarse_points, coarse_ratio = self._local_refinement(best_initial_config, max_iter=100)
            if coarse_ratio > best_ratio:
                best_ratio = coarse_ratio
                best_initial_config = coarse_points

            # Medium refinement
            medium_points, medium_ratio = self._local_refinement(best_initial_config, max_iter=150)
            if medium_ratio > best_ratio:
                best_ratio = medium_ratio
                best_initial_config = medium_points

            # Fine refinement
            fine_points, fine_ratio = self._local_refinement(best_initial_config, max_iter=200)
            if fine_ratio > best_ratio:
                best_ratio = fine_ratio
                best_initial_config = fine_points

        # Step 6: Final local refinement with adaptive parameters
        if self._time_remaining() > 10:
            final_points, final_ratio = self._local_refinement(best_initial_config, max_iter=300)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_initial_config = final_points

        return best_initial_config

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointEvolutionOptimizer(n_points=16, dimensions=2)
    return optimizer.optimize()

# EVOLVE-BLOCK-END