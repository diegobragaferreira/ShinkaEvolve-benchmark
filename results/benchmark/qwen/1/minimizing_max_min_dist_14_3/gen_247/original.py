# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time
from typing import Tuple, Optional

class AdaptiveHybridEvolve:
    """An advanced optimizer for point distribution in 3D space with adaptive hybrid strategies."""

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

    def improved_sobol_initialization(self) -> np.ndarray:
        """Improved Sobol initialization with better spherical distribution."""
        # Generate more points than needed to allow for rejection sampling
        sampler = qmc.Sobol(d=self.d, seed=42)
        points = sampler.random(n=self.n_points * 2)

        # Apply rejection sampling to get points on the surface of the unit sphere
        points = points * 2 - 1  # Map to [-1, 1]^3
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Keep only points that are reasonably close to unit sphere
        mask = np.abs(norms - 1.0) < 0.5
        valid_points = points[mask].reshape(-1, self.d)

        # If we don't have enough valid points, use all points and normalize
        if len(valid_points) < self.n_points:
            valid_points = points[:self.n_points]

        # Ensure we have exactly n_points
        if len(valid_points) > self.n_points:
            valid_points = valid_points[:self.n_points]
        elif len(valid_points) < self.n_points:
            # Fill remaining with random points
            remaining = self.n_points - len(valid_points)
            extra_points = np.random.rand(remaining, self.d) * 2 - 1
            valid_points = np.vstack([valid_points, extra_points])

        # Normalize to unit sphere
        norms = np.linalg.norm(valid_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return valid_points / norms

    def adaptive_initialization(self, method: str = 'sobol') -> np.ndarray:
        """Create diverse initial point configurations with improved strategies."""
        if method == 'fibonacci':
            base_points = self.fibonacci_spiral_sphere()
            # Add controlled Gaussian perturbation to break symmetry
            np.random.seed(42)
            perturbation = np.random.normal(0, 0.025, (self.n_points, self.d))
            perturbed_points = base_points + perturbation
        elif method == 'sobol':
            perturbed_points = self.sobol_initialization()
        else:
            # Random initialization with better spread
            np.random.seed(42)
            perturbed_points = np.random.rand(self.n_points, self.d) * 2 - 1

        # Normalize to unit sphere with numerical stability
        norms = np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return perturbed_points / norms

    def min_max_dist_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance with early termination and numerical stability."""
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

            # Avoid division by zero or near-zero values
            if max_dist <= 1e-12:
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
        """Setup constraint dictionary for optimization with numerical stability."""
        def constraint_sphere(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            norms = np.linalg.norm(points_reshaped, axis=1)
            # Add small tolerance for numerical precision
            return norms - 1.0

        return {'type': 'eq', 'fun': constraint_sphere}

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere ensuring numerical stability."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero with epsilon
        epsilon = 1e-15
        norms = np.where(norms < epsilon, 1.0, norms)
        return points / norms

    def optimize_single_start(self, initial_points: np.ndarray,
                            max_iter: int = 1000,
                            method: str = 'L-BFGS-B',
                            coarse: bool = False,
                            phase: str = 'main') -> Tuple[np.ndarray, float]:
        """Optimize a single start point configuration with adaptive parameters."""
        x0 = initial_points.flatten()

        def objective(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            normalized_points = self.normalize_points(points_reshaped)
            return -self.min_max_dist_ratio(normalized_points)

        bounds = [(-2, 2) for _ in range(self.n_points * self.d)]

        # Adaptive tolerances based on phase and optimization needs
        if coarse:
            ftol = 1e-6
            gtol = 1e-6
            maxiter = 200 if phase == 'coarse' else 400
        else:
            ftol = 1e-12
            gtol = 1e-12
            maxiter = max_iter

        # Adjust for specific methods
        if method == 'TNC':
            options = {'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'disp': 0}
        else:
            options = {'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}

        constraints = self.setup_constraint()

        # Use optimization method with appropriate tolerances
        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options,
                tol=ftol
            )
        except Exception:
            # Fallback to a simpler optimization if complex one fails
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
                    tol=ftol
                )
            except Exception:
                # Last resort: return initial points
                return initial_points, self.min_max_dist_ratio(initial_points)

        # Extract and normalize optimized points
        optimized_points = result.x.reshape(self.n_points, self.d)
        final_points = self.normalize_points(optimized_points)
        final_ratio = self.min_max_dist_ratio(final_points)

        return final_points, final_ratio

    def multi_stage_optimization(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Perform multi-stage optimization with progressive refinement."""
        best_points = initial_points.copy()
        best_ratio = self.min_max_dist_ratio(best_points)

        # Stage 1: Coarse global search with multiple methods
        optimization_methods = ['L-BFGS-B', 'SLSQP', 'TNC']
        stage1_results = []

        for method in optimization_methods:
            try:
                # Use coarse parameters for faster exploration
                coarse_points, coarse_ratio = self.optimize_single_start(
                    best_points, max_iter=200, method=method, coarse=True, phase='coarse'
                )
                stage1_results.append((coarse_points, coarse_ratio))
                if coarse_ratio > best_ratio:
                    best_ratio = coarse_ratio
                    best_points = coarse_points.copy()
            except Exception:
                continue

        # Stage 2: Refinement with tighter tolerances
        if len(stage1_results) > 0:
            # Pick the best from stage 1
            best_from_stage1 = max(stage1_results, key=lambda x: x[1])
            refined_points, refined_ratio = self.optimize_single_start(
                best_from_stage1[0], max_iter=500, method='L-BFGS-B', coarse=False, phase='refine'
            )
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()

        # Stage 3: Final polishing with multiple methods
        polish_methods = ['SLSQP', 'L-BFGS-B']
        for method in polish_methods:
            try:
                polished_points, polished_ratio = self.optimize_single_start(
                    best_points, max_iter=300, method=method, coarse=False, phase='polish'
                )
                if polished_ratio > best_ratio:
                    best_ratio = polished_ratio
                    best_points = polished_points.copy()
            except Exception:
                continue

        return best_points, best_ratio

    def multi_start_optimization(self) -> np.ndarray:
        """Perform enhanced multi-start optimization with hybrid approach."""
        # Try multiple initialization methods with varied seeds and strategies
        init_methods = ['sobol', 'fibonacci', 'random']
        # Give more weight to Sobol which generally performs better
        num_starts_dict = {'sobol': 20, 'fibonacci': 10, 'random': 10}
        total_num_starts = sum(num_starts_dict.values())

        # Optimization methods for diversity
        optimization_methods = ['L-BFGS-B', 'SLSQP', 'TNC']

        for method_idx, method in enumerate(init_methods):
            # For each method, try multiple random seeds
            for start_idx in range(num_starts_dict[method]):
                try:
                    # Different seeding for better exploration
                    seed_val = 42 + method_idx * 100 + start_idx * 10
                    np.random.seed(seed_val)

                    if method == 'sobol' and start_idx == 0:
                        # First Sobol start uses fixed seed for reproducibility
                        points = self.improved_sobol_initialization()
                    elif method == 'fibonacci' and start_idx == 0:
                        # First Fibonacci start uses fixed seed for reproducibility
                        points = self.adaptive_initialization('fibonacci')
                    else:
                        # Other starts use different seeds
                        points = self.adaptive_initialization(method)

                    # Multi-stage optimization for better convergence
                    optimized_points, current_ratio = self.multi_stage_optimization(points)

                    # Additional refinement step - try one more optimization with different method
                    if current_ratio > self.best_ratio and current_ratio > 0.1:
                        # Try optimizing again with different method for possible improvement
                        try:
                            alternate_points, alternate_ratio = self.optimize_single_start(
                                optimized_points, max_iter=500, method='SLSQP', coarse=False, phase='secondary'
                            )
                            if alternate_ratio > current_ratio:
                                optimized_points = alternate_points
                                current_ratio = alternate_ratio
                        except Exception:
                            pass

                    if current_ratio > self.best_ratio:
                        self.best_ratio = current_ratio
                        self.best_points = optimized_points.copy()

                except Exception as e:
                    # Skip failed optimizations but continue
                    continue

        # Return the best solution found, fallback to improved Sobol if needed
        if self.best_points is not None:
            return self.best_points
        else:
            return self.improved_sobol_initialization()

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = AdaptiveHybridEvolve(n_points=14, d=3)
    return optimizer.multi_start_optimization()

# EVOLVE-BLOCK-END