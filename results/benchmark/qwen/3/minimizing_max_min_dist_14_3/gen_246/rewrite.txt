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
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    @staticmethod
    def spherical_voronoi(n: int, seed: int = 42) -> np.ndarray:
        """Initialize points using spherical Voronoi approach."""
        np.random.seed(seed)
        points = np.random.randn(n, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms
        return points

    @staticmethod
    def grid_with_jitter(n: int, seed: int = 42) -> np.ndarray:
        """Initialize points using grid with jitter for better distribution."""
        np.random.seed(seed)
        # Create a 3x3x3 grid and sample points
        grid_coords = np.linspace(0.05, 0.95, 3)
        grid_points = []
        for x in grid_coords:
            for y in grid_coords:
                for z in grid_coords:
                    grid_points.append([x, y, z])
        # Take first N points and add jitter
        points = np.array(grid_points[:n]) + np.random.normal(0, 0.02, (n, 3))
        # Clamp to [0,1]^3
        points = np.clip(points, 0, 1)
        return points

    @staticmethod
    def random_uniform(n: int, d: int = 3, seed: int = 42) -> np.ndarray:
        """Initialize points using uniform random distribution."""
        np.random.seed(seed)
        return np.random.rand(n, d)

    @classmethod
    def initialize_points(cls, n: int = 14, d: int = 3, seed: int = 42) -> np.ndarray:
        """Initialize points with multiple strategies and selection based on diversity."""
        np.random.seed(seed)

        # Strategy 1: Fibonacci spiral on sphere
        fib_points = cls.fibonacci_spherical(n)
        # Normalize to unit sphere
        norms = np.linalg.norm(fib_points, axis=1, keepdims=True)
        fib_points = fib_points / np.max(norms)
        # Scale and shift to [0,1]^3
        fib_points = fib_points * 0.5 + 0.5
        # Add small random perturbation for better mixing
        fib_points += np.random.normal(0, 0.01, fib_points.shape)
        fib_points = np.clip(fib_points, 0, 1)

        # Strategy 2: Spherical Voronoi
        sv_points = cls.spherical_voronoi(n, seed+1)
        # Scale and shift to [0,1]^3
        sv_points = sv_points * 0.5 + 0.5
        sv_points = np.clip(sv_points, 0, 1)

        # Strategy 3: Grid with jitter
        grid_points = cls.grid_with_jitter(n, seed+2)

        # Strategy 4: Random uniform
        rand_points = cls.random_uniform(n, d, seed+3)

        # Evaluate which initialization gives better spread
        # Using minimum distance as proxy for spread quality
        def compute_min_distance(points):
            if len(points) < 2:
                return 0.0
            try:
                distances = pdist(points)
                if len(distances) == 0:
                    return 0.0
                return np.min(distances)
            except:
                return 0.0

        fib_quality = compute_min_distance(fib_points)
        sv_quality = compute_min_distance(sv_points)
        grid_quality = compute_min_distance(grid_points)
        rand_quality = compute_min_distance(rand_points)

        # Select the best initialization based on quality
        max_quality = max(fib_quality, sv_quality, grid_quality, rand_quality)
        if max_quality == fib_quality:
            return fib_points
        elif max_quality == sv_quality:
            return sv_points
        elif max_quality == grid_quality:
            return grid_points
        else:
            return rand_points

class DistanceCalculator:
    """Handles distance computations and metrics calculation."""

    @staticmethod
    def compute_min_max_distances(points: np.ndarray) -> Tuple[float, float]:
        """Compute minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0, 0.0

        try:
            # Compute pairwise distances using vectorized operations
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

        # Apply boundary penalties efficiently
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

    def adaptive_differential_evolution_optimize(self, initial_points: np.ndarray,
                                              max_time: float = 300.0) -> np.ndarray:
        """Optimize using differential evolution with adaptive parameters and multi-start strategy."""
        x0 = initial_points.flatten()
        bounds = [(0, 1)] * self.total_variables
        
        # Multi-start optimization with different populations
        best_solution = None
        best_value = float('inf')
        start_time = time.time()

        # Create multiple initial configurations for multi-start
        configs = []
        configs.append(initial_points)
        
        # Add variations of initial points
        np.random.seed(42)
        for _ in range(2):
            # Add small noise to initial points
            noisy_points = initial_points + np.random.normal(0, 0.01, initial_points.shape)
            noisy_points = np.clip(noisy_points, 0, 1)
            configs.append(noisy_points)
        
        # Add completely random initialization
        random_points = np.random.rand(self.num_points, self.dimension)
        configs.append(random_points)

        # Run optimization with different strategies
        for i, config in enumerate(configs):
            if time.time() - start_time > max_time - 30:  # Leave some buffer
                break
                
            config_flat = config.flatten()
            
            # Adaptive parameters based on configuration
            popsize = 15 + i * 3
            maxiter = 60 + i * 20
            
            try:
                # Run DE with current configuration
                result = differential_evolution(
                    self._penalty_objective,
                    bounds,
                    seed=42 + i,
                    maxiter=maxiter,
                    popsize=popsize,
                    mutation=(0.7, 1.0),
                    recombination=0.8,
                    atol=1e-7,
                    tol=1e-7,
                    disp=False,
                    polish=True
                )
                
                if result.success and result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
                    
            except Exception as e:
                warnings.warn(f"DE optimization failed for config {i}: {e}")
                continue

        if best_solution is not None:
            return best_solution.reshape(-1, self.dimension)
        else:
            return initial_points.reshape(-1, self.dimension)

    def local_optimize(self, initial_points: np.ndarray,
                      max_time: float = 60.0) -> np.ndarray:
        """Optimize using L-BFGS-B for local refinement with adaptive tolerance tightening."""
        x0 = initial_points.flatten()
        bounds = [(0, 1)] * self.total_variables

        try:
            # Try different tolerance levels for progressive refinement
            tolerance_settings = [
                {'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 100},
                {'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 200},
                {'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 300}
            ]
            
            current_solution = x0
            
            for i, tol_params in enumerate(tolerance_settings):
                if time.time() - time.time() > max_time - 10:  # Leave buffer
                    break
                    
                try:
                    result = minimize(
                        self._penalty_objective,
                        current_solution,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={**tol_params}
                    )
                    
                    if result.success:
                        current_solution = result.x
                    else:
                        break
                        
                except Exception as e:
                    warnings.warn(f"Local optimization failed at stage {i}: {e}")
                    break

            return current_solution.reshape(-1, self.dimension)
        except Exception as e:
            warnings.warn(f"Local optimization failed completely: {e}")
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

        # Phase 2: Global optimization with Adaptive Differential Evolution
        try:
            global_optimized = self.optimizer.adaptive_differential_evolution_optimize(
                initial_points, max_time=280.0
            )
        except Exception as e:
            warnings.warn(f"Global optimization failed: {e}")
            global_optimized = initial_points

        # Phase 3: Local refinement with L-BFGS-B
        try:
            local_optimized = self.optimizer.local_optimize(global_optimized, max_time=40.0)
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
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