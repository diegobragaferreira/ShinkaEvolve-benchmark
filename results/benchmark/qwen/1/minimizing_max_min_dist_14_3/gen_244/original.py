# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Fixed seed for reproducibility
    np.random.seed(42)

    class PointOptimizer:
        def __init__(self, n_points=14):
            self.n_points = n_points
            self.best_ratio = -np.inf
            self.best_points = None

        def fibonacci_sphere(self, n):
            """Generate n points on a sphere using Fibonacci spiral method"""
            points = []
            phi = np.pi * (3 - np.sqrt(5))  # golden angle

            for i in range(n):
                y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y

                theta = phi * i  # golden angle increment

                x = np.cos(theta) * radius
                z = np.sin(theta) * radius

                points.append([x, y, z])

            return np.array(points)

        def sobol_points(self, n, seed=42):
            """Generate n points using Sobol sequence for better space-filling"""
            try:
                sampler = qmc.Sobol(d=3, seed=seed)
                points = sampler.random(n)
                return points
            except ImportError:
                # Fallback to random points if qmc not available
                return np.random.rand(n, 3)

        def adaptive_constraint_tightening(self, x, iteration_step=0, max_steps=100):
            """Apply adaptive constraint tightening during optimization"""
            # Reshape x into points
            points = x.reshape(-1, 3)

            # Calculate pairwise distances
            distances = pdist(points)

            # Get min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)

            # Apply adaptive constraint tightening
            if iteration_step < max_steps:
                # Linear decay from 2.8 to 2.0 diameter constraint
                max_diameter = 2.8 - (iteration_step / max_steps) * 0.8
                # Penalize solutions that exceed the constraint
                if d_max > max_diameter:
                    penalty = (d_max - max_diameter) * 1000
                    return -d_min / d_max + penalty

            # Return negative ratio since we want to maximize
            if d_max < 1e-10:
                return -1e10
            return -d_min / d_max

        def enhanced_initialization(self, n_points, seed=42):
            """Enhanced initialization using multiple strategies"""
            np.random.seed(seed)

            # Strategy 1: Fibonacci points on sphere
            fib_points = self.fibonacci_sphere(n_points)

            # Strategy 2: Sobol points for better space filling
            sobol_points_generated = self.sobol_points(n_points, seed=seed+1000)

            # Strategy 3: Random points
            random_points = np.random.rand(n_points, 3)

            # Use a weighted combination to get better diversity
            mixed_points = (
                0.5 * fib_points +
                0.3 * sobol_points_generated +
                0.2 * random_points
            )

            # Normalize to unit cube [0,1]^3
            # First center around origin and scale appropriately
            mixed_points = mixed_points - np.mean(mixed_points, axis=0)
            max_coord = np.max(np.abs(mixed_points))
            if max_coord > 0:
                mixed_points = mixed_points / max_coord * 0.5
            # Then shift to [0,1]^3
            mixed_points = mixed_points + 0.5

            # Add controlled perturbation to break symmetry
            perturbation = np.random.normal(0, 0.02, mixed_points.shape)
            mixed_points += perturbation
            mixed_points = np.clip(mixed_points, 0, 1)

            return mixed_points

        def evaluate_solution(self, points):
            """Evaluate a solution and return the ratio"""
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max < 1e-10:
                return 0.0
            return d_min / d_max

        def optimize_single_start(self, seed):
            """Perform optimization from a single starting point"""
            try:
                # Generate enhanced initial points
                initial_points = self.enhanced_initialization(self.n_points, seed=seed)

                # Flatten for optimization
                x0 = initial_points.flatten()

                # Set up bounds for optimization (0 to 1 for all coordinates)
                bounds = [(0.0, 1.0)] * self.n_points * 3

                # First stage: Differential Evolution for global search
                de_result = differential_evolution(
                    self.adaptive_constraint_tightening,
                    bounds,
                    seed=seed,
                    maxiter=250,
                    popsize=12,
                    tol=1e-8,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False
                )

                # Second stage: Local refinement with L-BFGS-B
                refined_result = minimize(
                    self.adaptive_constraint_tightening,
                    de_result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 200},
                    callback=None
                )

                # Evaluate final result
                final_points = refined_result.x.reshape(-1, 3)
                ratio = self.evaluate_solution(final_points)

                return ratio, final_points

            except Exception as e:
                return -np.inf, None

        def run_multi_start_optimization(self):
            """Run optimization with multiple starting points"""
            # Try multiple initializations with different seeds
            seeds = [42, 123, 456, 789, 999]

            for seed in seeds:
                ratio, points = self.optimize_single_start(seed)
                if ratio > self.best_ratio and points is not None:
                    self.best_ratio = ratio
                    self.best_points = points.copy()

            # If no good result was found, fallback to simple random initialization
            if self.best_points is None:
                np.random.seed(42)
                points = np.random.rand(self.n_points, 3) * 0.8 + 0.1
                ratio = self.evaluate_solution(points)
                if ratio > self.best_ratio:
                    self.best_ratio = ratio
                    self.best_points = points.copy()

            return self.best_points

    # Create optimizer instance and run optimization
    optimizer = PointOptimizer(n_points=14)
    result = optimizer.run_multi_start_optimization()

    return result


# EVOLVE-BLOCK-END