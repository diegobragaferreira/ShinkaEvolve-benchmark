# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple, Optional
import warnings

class PointInitializer:
    """Handles various point initialization strategies."""

    @staticmethod
    def fibonacci_spherical(n: int) -> np.ndarray:
        """Initialize points using Fibonacci spiral on sphere."""
        points = []
        for i in range(n):
            phi = np.arccos(1 - 2 * (i / (n - 1)))
            theta = np.sqrt(n) * phi
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])

        return np.array(points)

    @classmethod
    def initialize_points(cls, n: int = 14, d: int = 3, seed: int = 42) -> np.ndarray:
        """Initialize points with enhanced distribution strategy."""
        np.random.seed(seed)

        # Start with Fibonacci spiral on sphere
        points = cls.fibonacci_spherical(n)

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.max(norms)

        # Scale and shift to [0,1]^3
        points = points * 0.5 + 0.5

        # Add small random perturbation for better mixing
        points += np.random.normal(0, 0.01, points.shape)

        return np.clip(points, 0, 1)

class DistanceCalculator:
    """Handles distance computations and metrics calculation."""

    @staticmethod
    def compute_min_max_distances(points: np.ndarray) -> Tuple[float, float]:
        """Compute minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0, 0.0

        try:
            # Compute pairwise distances
            distances = pdist(points)

            if len(distances) == 0:
                return 0.0, 0.0

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            # Avoid division by zero
            if max_dist <= 0:
                return 0.0, 0.0

            return min_dist, max_dist
        except Exception:
            return 0.0, 0.0

class OptimizationStrategy:
    """Manages different optimization approaches."""

    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.total_variables = num_points * dimension

    def _penalty_objective(self, x_flat: np.ndarray, penalty_weight: float = 1000.0) -> float:
        """Objective with penalty for constraint violations."""
        points = x_flat.reshape(-1, self.dimension)

        # Apply boundary penalties
        penalty = 0.0
        for i, coord in enumerate(points.flat):
            if coord < 0:
                penalty += penalty_weight * (0 - coord)**2
            elif coord > 1:
                penalty += penalty_weight * (coord - 1)**2

        # Compute distance ratio
        min_dist, max_dist = DistanceCalculator.compute_min_max_distances(points)

        if max_dist <= 0:
            return penalty + 1e6  # Large penalty for invalid configuration

        ratio = -min_dist / max_dist  # Negative for minimization
        return ratio + penalty

    def differential_evolution_optimize(self, initial_points: np.ndarray,
                                     max_time: float = 300.0) -> np.ndarray:
        """Optimize using differential evolution for global search."""
        x0 = initial_points.flatten()
        bounds = [(0, 1)] * self.total_variables

        # Set up DE parameters
        de_options = {
            'maxiter': 500,
            'popsize': 15,
            'tol': 1e-6,
            'mutation': (0.5, 1.0),
            'recombination': 0.7,
            'seed': 42,
            'disp': False
        }

        try:
            result = differential_evolution(
                self._penalty_objective,
                bounds,
                **de_options
            )
            return result.x.reshape(-1, self.dimension)
        except Exception:
            return initial_points.reshape(-1, self.dimension)

    def local_optimize(self, initial_points: np.ndarray,
                      max_time: float = 60.0) -> np.ndarray:
        """Optimize using L-BFGS-B for local refinement."""
        x0 = initial_points.flatten()
        bounds = [(0, 1)] * self.total_variables

        try:
            result = minimize(
                self._penalty_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            return result.x.reshape(-1, self.dimension)
        except Exception:
            return initial_points.reshape(-1, self.dimension)

class PointOptimizer:
    """Main optimization class orchestrating the complete process."""

    def __init__(self, num_points: int = 14, dimension: int = 3):
        self.num_points = num_points
        self.dimension = dimension
        self.initializer = PointInitializer()
        self.optimizer = OptimizationStrategy(num_points, dimension)
        self.evaluator = DistanceCalculator()

    def validate_and_correct(self, points: np.ndarray) -> np.ndarray:
        """Ensure points are valid and within bounds."""
        if points.shape != (self.num_points, self.dimension):
            raise ValueError("Invalid point array shape")

        # Clip to valid range
        points = np.clip(points, 0, 1)

        # Handle degenerate cases
        if np.allclose(points[0], points):
            # All points are identical, regenerate
            points = self.initializer.initialize_points(self.num_points, self.dimension)

        return points

    def evaluate_solution(self, points: np.ndarray) -> float:
        """Evaluate solution quality."""
        min_dist, max_dist = self.evaluator.compute_min_max_distances(points)

        if max_dist <= 0:
            return 0.0

        return min_dist / max_dist

    def optimize(self) -> np.ndarray:
        """Main optimization routine with multi-stage approach."""
        # Phase 1: Initialize points
        try:
            initial_points = self.initializer.initialize_points(
                self.num_points, self.dimension
            )
        except Exception:
            # Fallback to random initialization
            np.random.seed(42)
            initial_points = np.random.rand(self.num_points, self.dimension)

        # Phase 2: Global optimization with Differential Evolution
        try:
            global_optimized = self.optimizer.differential_evolution_optimize(
                initial_points
            )
        except Exception:
            global_optimized = initial_points

        # Phase 3: Local refinement with L-BFGS-B
        try:
            local_optimized = self.optimizer.local_optimize(global_optimized)
        except Exception:
            local_optimized = global_optimized

        # Phase 4: Validation and final correction
        final_points = self.validate_and_correct(local_optimized)

        # Phase 5: Quality check and potential fallback
        ratio = self.evaluate_solution(final_points)

        # If quality is poor, fallback to initialization
        if ratio < 1e-6:
            try:
                fallback_points = self.initializer.initialize_points(
                    self.num_points, self.dimension
                )
                fallback_ratio = self.evaluate_solution(fallback_points)
                if fallback_ratio > ratio:
                    final_points = fallback_points
            except Exception:
                pass

        return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    try:
        optimizer = PointOptimizer(num_points=14, dimension=3)
        return optimizer.optimize()
    except Exception:
        # Last resort: random initialization
        np.random.seed(42)
        return np.random.rand(14, 3)

# EVOLVE-BLOCK-END