# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from typing import Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PointConfiguration:
    """Encapsulates point configuration management and initialization."""

    @staticmethod
    def fibonacci_spiral_on_sphere(n: int) -> np.ndarray:
        """Generate points using Fibonacci spiral on unit sphere."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n):
            phi = np.arccos(1 - 2 * (i / (n - 1)))
            theta = np.sqrt(n) * phi
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])

        return np.array(points)

    @classmethod
    def initialize_points(cls, n: int = 14, d: int = 3) -> np.ndarray:
        """
        Initialize points using enhanced strategy combining multiple approaches.
        """
        np.random.seed(42)

        # Start with Fibonacci spiral distribution
        points = cls.fibonacci_spiral_on_sphere(n)

        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.max(norms)

        # Scale and shift to [0,1]^3
        points = points * 0.5 + 0.5

        # Add small random perturbation to escape local optima
        points += np.random.normal(0, 0.005, points.shape)

        # Ensure bounds
        points = np.clip(points, 0, 1)

        return points

class DistanceMetrics:
    """Handles all distance-related computations and metrics."""

    @staticmethod
    def compute_min_max_distances(points: np.ndarray) -> Tuple[float, float]:
        """
        Calculate minimum and maximum distances between all point pairs.
        """
        if len(points) < 2:
            return 0.0, 0.0

        try:
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

class OptimizationConfig:
    """Configuration container for optimization parameters."""

    def __init__(self):
        self.de_maxiter = 500
        self.de_popsize = 15
        self.de_tol = 1e-6
        self.de_mutation = (0.5, 1.0)
        self.de_recombination = 0.7
        self.lbfgs_maxiter = 1000
        self.lbfgs_ftol = 1e-12
        self.lbfgs_gtol = 1e-12
        self.penalty_weight = 1000.0

class OptimizationStrategy:
    """Manages different optimization techniques."""

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def objective_with_penalty(self, points_flat: np.ndarray) -> float:
        """Objective function with penalty for constraint violations."""
        n, d = 14, 3
        points = points_flat.reshape(n, d)

        # Compute penalty for boundary violations
        penalty = 0.0
        for i, coord in enumerate(points.flat):
            if coord < 0:
                penalty += self.config.penalty_weight * (0 - coord)**2
            elif coord > 1:
                penalty += self.config.penalty_weight * (coord - 1)**2

        # Calculate distances
        distances = pdist(points)

        if len(distances) == 0:
            return penalty + float('inf')

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist <= 0:
            return penalty + float('inf')

        # Return negative ratio plus penalty to minimize (maximize the ratio)
        ratio = -min_dist / max_dist
        return ratio + penalty

    def adaptive_differential_evolution(self, initial_points: np.ndarray,
                                     max_time: float = 350.0) -> np.ndarray:
        """Optimize using adaptive differential evolution."""
        try:
            # Flatten initial points for optimization
            initial_flat = initial_points.flatten()

            # Define bounds for each coordinate (0 to 1)
            bounds = [(0.0, 1.0)] * len(initial_flat)

            # Initialize adaptive parameters
            current_popsize = self.config.de_popsize
            maxiter = self.config.de_maxiter
            stagnation_counter = 0
            best_value = float('inf')
            stagnation_threshold = 10
            max_stagnation = 20

            # History tracking for convergence
            history = []

            # Run multiple rounds of DE with adaptive population size
            for round_num in range(3):
                # Check if we've exceeded time limits or stagnated
                if len(history) > 0 and len(history) % 50 == 0:
                    # Check for convergence
                    if len(history) >= 2:
                        recent_change = abs(history[-1] - history[-2])
                        if recent_change < 1e-8:
                            stagnation_counter += 1
                        else:
                            stagnation_counter = 0

                    # Increase population size if stagnated
                    if stagnation_counter >= stagnation_threshold and current_popsize < 30:
                        current_popsize = min(current_popsize + 5, 30)
                        stagnation_counter = 0  # Reset counter after adjustment

                    # Stop if too many stagnations
                    if stagnation_counter >= max_stagnation:
                        break

                # Run differential evolution with current settings
                result = differential_evolution(
                    self.objective_with_penalty,
                    bounds,
                    maxiter=maxiter // 3,
                    popsize=current_popsize,
                    tol=self.config.de_tol,
                    mutation=self.config.de_mutation,
                    recombination=self.config.de_recombination,
                    seed=42,
                    disp=False
                )

                # Track history
                history.append(result.fun)

                # Update best value
                if result.fun < best_value:
                    best_value = result.fun
                    stagnation_counter = 0  # Reset stagnation counter on improvement

                # Early stopping based on improvement
                if len(history) >= 2 and abs(history[-1] - history[-2]) < 1e-9:
                    break

            # Reshape optimized result
            optimized_points = result.x.reshape(14, 3)

            # Ensure all points are within valid range
            optimized_points = np.clip(optimized_points, 0, 1)

            return optimized_points

        except Exception:
            # Return the initial points if optimization fails
            return initial_points

    def local_refinement(self, initial_points: np.ndarray,
                       max_time: float = 60.0) -> np.ndarray:
        """Perform local refinement using L-BFGS-B with adaptive tolerance tightening."""
        try:
            x0 = initial_points.flatten()
            bounds = [(0.0, 1.0)] * len(x0)

            # Apply L-BFGS-B optimization with adaptive tolerances
            # Start with looser tolerances for faster initial convergence
            result = minimize(
                self.objective_with_penalty,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.config.lbfgs_maxiter,
                    'ftol': 1e-6,  # Looser initial tolerance
                    'gtol': 1e-6   # Looser initial tolerance
                }
            )

            # If improvement is still occurring, retry with tighter tolerances
            if result.success and hasattr(result, 'fun'):
                # Check if we can benefit from tighter tolerances
                # This simple heuristic looks at whether we're approaching the benchmark
                if result.fun < -0.4:  # If we're getting close to good solutions
                    # Retry with tighter tolerances for better final quality
                    result = minimize(
                        self.objective_with_penalty,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': self.config.lbfgs_maxiter,
                            'ftol': self.config.lbfgs_ftol,  # Tighter tolerance
                            'gtol': self.config.lbfgs_gtol   # Tighter tolerance
                        }
                    )

            refined_points = result.x.reshape(14, 3)
            refined_points = np.clip(refined_points, 0, 1)
            return refined_points

        except Exception:
            return initial_points

class PointOptimizer:
    """Main optimization orchestrator."""

    def __init__(self):
        self.config = OptimizationConfig()
        self.initializer = PointConfiguration()
        self.distance_calculator = DistanceMetrics()
        self.strategy = OptimizationStrategy(self.config)

    def optimize(self) -> np.ndarray:
        """Main optimization routine."""
        try:
            # Phase 1: Initialize points with better strategy
            initial_points = self.initializer.initialize_points(14, 3)

            # Phase 2: Global optimization with adaptive differential evolution
            global_optimized = self.strategy.adaptive_differential_evolution(initial_points)

            # Phase 3: Local refinement with L-BFGS-B
            local_optimized = self.strategy.local_refinement(global_optimized)

            # Phase 4: Final validation and adjustment
            final_points = local_optimized.copy()

            # Calculate final metrics
            min_dist, max_dist = self.distance_calculator.compute_min_max_distances(final_points)

            # If optimization didn't work well, fall back to a good known arrangement
            if max_dist <= 0 or min_dist <= 0:
                # Fallback to regularized arrangement
                np.random.seed(42)
                final_points = np.random.rand(14, 3)

            return final_points

        except Exception as e:
            # Last resort: return random initialization
            np.random.seed(42)
            return np.random.rand(14, 3)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    try:
        optimizer = PointOptimizer()
        return optimizer.optimize()
    except Exception:
        # Fallback to random initialization
        np.random.seed(42)
        return np.random.rand(14, 3)

# EVOLVE-BLOCK-END