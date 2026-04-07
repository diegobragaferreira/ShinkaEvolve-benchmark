# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List
import copy
import math

class PointEvolutionOptimizer:
    """Advanced optimizer for point distribution maximizing min/max distance ratio."""

    def __init__(self, n_points: int = 16, dimensions: int = 2):
        self.n_points = n_points
        self.dimensions = dimensions
        self.benchmark_ratio = 1 / np.sqrt(12.889266112)  # 0.2786
        self.max_time = 180.0  # seconds

    def calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

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

    def initialize_spherical_projection(self) -> np.ndarray:
        """Initialize points using a spherical arrangement projected to 2D."""
        # Generate points on a sphere (fibonacci sphere algorithm)
        points_sphere = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points_sphere.append([x, y, z])

        # Project 3D sphere points to 2D using stereographic projection
        points_2d = []
        for x, y, z in points_sphere:
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

    def initialize_hexagonal_grid(self) -> np.ndarray:
        """Initialize points using hexagonal grid pattern with proper scaling."""
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

    def initialize_spiral_pattern(self) -> np.ndarray:
        """Initialize points using spiral pattern."""
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

    def initialize_grid_points(self) -> np.ndarray:
        """Initialize points using regular grid."""
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) >= self.n_points:
                    break
                points.append([i * 0.25 + 0.125, j * 0.25 + 0.125])
        return np.array(points[:self.n_points])

    def initialize_population(self) -> List[np.ndarray]:
        """Create diverse initial population using multiple strategies."""
        np.random.seed(42)

        # Strategy 1: Spherical projection
        init1 = self.initialize_spherical_projection()

        # Strategy 2: Hexagonal grid
        init2 = self.initialize_hexagonal_grid()

        # Strategy 3: Spiral pattern
        init3 = self.initialize_spiral_pattern()

        # Strategy 4: Grid pattern
        init4 = self.initialize_grid_points()

        # Strategy 5: Random uniform distribution
        init5 = np.random.rand(self.n_points, self.dimensions)

        # Strategy 6: Perturbed grid
        init6 = self.initialize_grid_points() + np.random.normal(0, 0.02, (self.n_points, self.dimensions))
        init6 = np.clip(init6, 0, 1)

        # Combine strategies
        init_points = [
            init1,
            init2,
            init3,
            init4,
            init5,
            init6
        ]

        return init_points

    def global_optimization(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Use global optimization to explore promising regions."""
        def objective(x_flat):
            points = x_flat.reshape(-1, self.dimensions)
            return -self.calculate_min_max_ratio(points)

        # Use differential evolution for broad exploration
        bounds = [(0, 1) for _ in range(len(initial_points.flatten()))]

        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=200,
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
                ratio = self.calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass

        return initial_points, self.calculate_min_max_ratio(initial_points)

    def local_refinement(self, points: np.ndarray, max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """Apply local refinement with multiple methods."""
        def objective(x_flat):
            points_candidate = x_flat.reshape(-1, self.dimensions)
            return -self.calculate_min_max_ratio(points_candidate)

        best_points = points.copy()
        best_ratio = self.calculate_min_max_ratio(best_points)

        # Method 1: L-BFGS-B - most reliable
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(points.flatten()))],
                options={'maxiter': max_iter // 2, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)

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
                options={'maxiter': max_iter // 4, 'adaptive': True, 'fatol': 1e-10, 'xatol': 1e-10}
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass

        # Method 3: Additional L-BFGS-B with fine tuning
        try:
            # Add small noise to escape local minima
            noisy_points = points + np.random.normal(0, 0.005, points.shape)
            noisy_points = np.clip(noisy_points, 0, 1)

            result = minimize(
                objective,
                noisy_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(noisy_points.flatten()))],
                options={'maxiter': max_iter // 4, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            if result.success:
                optimized_points = result.x.reshape(-1, self.dimensions)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = self.calculate_min_max_ratio(optimized_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass

        return best_points, best_ratio

    def progressive_refinement(self, initial_points: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Apply progressive refinement with increasing precision."""
        current_points = initial_points.copy()
        current_ratio = self.calculate_min_max_ratio(current_points)

        # Stage 1: Coarse optimization
        coarse_points, coarse_ratio = self.local_refinement(current_points, max_iter // 4)
        if coarse_ratio > current_ratio:
            current_points = coarse_points
            current_ratio = coarse_ratio

        # Stage 2: Medium optimization
        medium_points, medium_ratio = self.local_refinement(current_points, max_iter // 2)
        if medium_ratio > current_ratio:
            current_points = medium_points
            current_ratio = medium_ratio

        # Stage 3: Fine optimization
        fine_points, fine_ratio = self.local_refinement(current_points, max_iter)
        if fine_ratio > current_ratio:
            current_points = fine_points
            current_ratio = fine_ratio

        return current_points, current_ratio

    def optimize(self) -> np.ndarray:
        """Main optimization process."""
        start_time = time.time()

        # Initialize using diverse strategies
        initial_configs = self.initialize_population()

        best_ratio = -np.inf
        best_points = None

        # Try each initial configuration
        for i, initial_config in enumerate(initial_configs):
            try:
                # Global optimization to find promising regions
                global_points, global_ratio = self.global_optimization(initial_config)

                # Progressive refinement
                refined_points, refined_ratio = self.progressive_refinement(global_points, max_iter=800)

                # Additional local refinement with different methods
                final_points, final_ratio = self.local_refinement(refined_points, max_iter=400)

                # Use the best result from this initial configuration
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()

            except Exception:
                continue

        # If nothing worked properly, fallback to simple approach
        if best_points is None:
            # Simple approach: start with hexagonal grid and refine locally
            initial_points = self.initialize_hexagonal_grid()
            best_points, _ = self.local_refinement(initial_points, max_iter=1000)

        return best_points

    def evolve(self) -> np.ndarray:
        """Main evolutionary optimization loop."""
        return self.optimize()

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    optimizer = PointEvolutionOptimizer(n_points=16, dimensions=2)
    return optimizer.evolve()

# EVOLVE-BLOCK-END