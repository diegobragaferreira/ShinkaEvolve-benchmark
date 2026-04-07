# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time
from typing import Tuple, Optional

class HybridPointOptimizer:
    """A hybrid optimizer combining global and local search strategies for point distribution in 3D space."""

    def __init__(self, n_points: int = 14, d: int = 3):
        self.n_points = n_points
        self.d = d
        self.best_ratio = -np.inf
        self.best_points = None

    def fibonacci_spiral_sphere(self) -> np.ndarray:
        """Generate points on a sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def sobol_initialization(self) -> np.ndarray:
        """Initialize points using Sobol sequence for better space-filling properties."""
        sampler = qmc.Sobol(d=self.d, seed=42)
        points = sampler.random(n=self.n_points)
        # Scale to unit sphere
        points = points * 2 - 1  # Map to [-1, 1]^3
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms
        return points

    def adaptive_initialization(self, method: str = 'fibonacci') -> np.ndarray:
        """Create diverse initial point configurations with improved perturbations."""
        if method == 'fibonacci':
            base_points = self.fibonacci_spiral_sphere()
            # Add Gaussian perturbation with controlled magnitude to break symmetry
            np.random.seed(42)
            perturbation = np.random.normal(0, 0.02, (self.n_points, self.d))
            perturbed_points = base_points + perturbation
        elif method == 'sobol':
            perturbed_points = self.sobol_initialization()
        else:
            # Random initialization with better spread
            np.random.seed(42)
            perturbed_points = np.random.rand(self.n_points, self.d) * 2 - 1

        # Normalize to unit sphere
        norms = np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return perturbed_points / norms

    def min_max_dist_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance with early termination and variance regularization."""
        if len(points) < 2:
            return 0.0

        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0

            # Early termination: if we have very small distances already,
            # we don't need to compute the full ratio
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            mean_dist = np.mean(distances)
            dist_variance = np.var(distances)

            # Avoid division by zero
            if max_dist <= 0:
                return 0.0

            # Add small epsilon to prevent numerical issues
            epsilon = 1e-15

            # Regularization factor to penalize high variance in distances
            # This encourages more uniform distribution
            lambda_reg = 0.15
            variance_penalty = lambda_reg * (dist_variance / (mean_dist**2 + epsilon))

            # Return ratio minus penalty
            ratio = min_dist / (max_dist + epsilon)
            return ratio - variance_penalty
        except Exception:
            return 0.0

    def setup_constraint(self) -> dict:
        """Setup constraint dictionary for optimization."""
        def constraint_sphere(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            norms = np.linalg.norm(points_reshaped, axis=1)
            return norms - 1.0

        return {'type': 'eq', 'fun': constraint_sphere}

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere ensuring numerical stability."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def optimize_with_dual_annealing(self, initial_points: np.ndarray) -> np.ndarray:
        """Optimize using dual annealing for global search."""
        # Define objective function for optimization (negative because we want to maximize)
        def objective(x):
            points_reshaped = x.reshape(self.n_points, self.d)
            return -self.min_max_dist_ratio(points_reshaped)

        # Define bounds for optimization (points in [-2, 2]^3 to allow some flexibility)
        bounds = [(-2, 2) for _ in range(self.n_points * self.d)]

        # Use dual annealing for global optimization with tuned parameters
        result = dual_annealing(objective, bounds, maxiter=300, seed=42,
                               initial_temp=1000, restart_temp_ratio=0.9)

        # Get optimized points
        optimized_points = result.x.reshape(self.n_points, self.d)
        # Normalize to unit sphere
        normalized_points = self.normalize_points(optimized_points)

        return normalized_points

    def optimize_single_start(self, initial_points: np.ndarray,
                            max_iter: int = 1000,
                            method: str = 'L-BFGS-B',
                            coarse: bool = False) -> Tuple[np.ndarray, float]:
        """Optimize a single start point configuration with adaptive constraints."""
        x0 = initial_points.flatten()

        def objective(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            normalized_points = self.normalize_points(points_reshaped)
            return -self.min_max_dist_ratio(normalized_points)

        constraints = self.setup_constraint()
        bounds = [(-2, 2) for _ in range(self.n_points * self.d)]

        # Adjust tolerances based on coarse/fine mode
        ftol = 1e-8 if coarse else 1e-12
        gtol = 1e-8 if coarse else 1e-12
        maxiter = 300 if coarse else max_iter

        # Use optimization method with appropriate tolerances
        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
            tol=ftol
        )

        # Extract and normalize optimized points
        optimized_points = result.x.reshape(self.n_points, self.d)
        final_points = self.normalize_points(optimized_points)
        final_ratio = self.min_max_dist_ratio(final_points)

        return final_points, final_ratio

    def multi_start_optimization(self) -> np.ndarray:
        """Perform multi-start optimization with hybrid approach."""
        # Try multiple initialization methods with varied seeds
        init_methods = ['fibonacci', 'sobol', 'random']
        num_starts = 15  # Increased number of starts for better exploration

        # Optimization methods in order of preference
        optimization_methods = ['L-BFGS-B', 'SLSQP', 'TNC']

        # Phase 1: Multi-start with different initialization strategies
        for method_idx, method in enumerate(init_methods):
            # For each method, try multiple random seeds
            for start_idx in range(num_starts // len(init_methods)):
                try:
                    # Different seeding for better exploration
                    seed_val = 42 + method_idx * 100 + start_idx * 10
                    np.random.seed(seed_val)

                    if method == 'fibonacci' and start_idx == 0:
                        # First Fibonacci start uses fixed seed for reproducibility
                        points = self.adaptive_initialization('fibonacci')
                    else:
                        # Other starts use different seeds
                        points = self.adaptive_initialization(method)

                    # Phase 1a: Coarse optimization to find promising regions
                    best_local_points = None
                    best_local_ratio = -np.inf

                    for opt_method in optimization_methods:
                        try:
                            optimized_points, current_ratio = self.optimize_single_start(
                                points, max_iter=300, method=opt_method, coarse=True
                            )

                            if current_ratio > best_local_ratio:
                                best_local_ratio = current_ratio
                                best_local_points = optimized_points.copy()

                        except Exception:
                            continue

                    # Phase 1b: Fine-grained optimization on best local solution
                    if best_local_points is not None:
                        fine_best_ratio = best_local_ratio
                        fine_best_points = best_local_points.copy()

                        for opt_method in optimization_methods:
                            try:
                                refined_points, refined_ratio = self.optimize_single_start(
                                    fine_best_points, max_iter=500, method=opt_method, coarse=False
                                )

                                if refined_ratio > fine_best_ratio:
                                    fine_best_ratio = refined_ratio
                                    fine_best_points = refined_points.copy()

                            except Exception:
                                continue

                        # Update global best if we found something better
                        if fine_best_ratio > self.best_ratio:
                            self.best_ratio = fine_best_ratio
                            self.best_points = fine_best_points.copy()

                except Exception as e:
                    # Skip failed optimizations but continue
                    continue

        # Phase 2: Global search with dual annealing as backup
        if self.best_points is None:
            print("Using global optimization as fallback...")
            # Try global search with dual annealing on Sobol initialization
            try:
                global_points = self.adaptive_initialization('sobol')
                optimized_global = self.optimize_with_dual_annealing(global_points)
                global_ratio = self.min_max_dist_ratio(optimized_global)

                if global_ratio > self.best_ratio:
                    self.best_ratio = global_ratio
                    self.best_points = optimized_global.copy()
            except Exception:
                pass

        # Phase 3: Final local refinement of the best solution
        if self.best_points is not None:
            # Try multiple local optimization methods on the best solution
            refinement_methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            for method in refinement_methods:
                try:
                    refined_points, refined_ratio = self.optimize_single_start(
                        self.best_points, max_iter=500, method=method, coarse=False
                    )

                    if refined_ratio > self.best_ratio:
                        self.best_ratio = refined_ratio
                        self.best_points = refined_points.copy()

                except Exception:
                    continue

        # Return the best solution found, fallback to Fibonacci if needed
        if self.best_points is not None:
            return self.best_points
        else:
            return self.adaptive_initialization('fibonacci')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = HybridPointOptimizer(n_points=14, d=3)
    return optimizer.multi_start_optimization()

# EVOLVE-BLOCK-END