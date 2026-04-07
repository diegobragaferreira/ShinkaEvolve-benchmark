# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time
from typing import Tuple, Optional

class AdaptiveHybridOptimizer:
    """An improved optimizer for point distribution in 3D space with adaptive hybrid strategies."""

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
        """Calculate the ratio of minimum to maximum distance with early termination."""
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

            # Avoid division by zero
            if max_dist <= 0:
                return 0.0

            return min_dist / max_dist
        except Exception:
            return 0.0

    def setup_adaptive_constraint(self, iteration: int = 0, max_iterations: int = 1000) -> dict:
        """Setup adaptive constraint with dynamic radius based on iteration."""
        # Start with a larger radius to allow more exploration
        initial_radius = 1.2
        final_radius = 1.0

        # Interpolate radius based on iteration progress
        progress = min(iteration / max_iterations, 1.0)
        current_radius = initial_radius + (final_radius - initial_radius) * progress

        def constraint_adaptive_sphere(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            norms = np.linalg.norm(points_reshaped, axis=1)
            return norms - current_radius

        return {'type': 'eq', 'fun': constraint_adaptive_sphere}

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere ensuring numerical stability."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

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
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
            tol=ftol
        )

        # Extract and normalize optimized points
        optimized_points = result.x.reshape(self.n_points, self.d)
        final_points = self.normalize_points(optimized_points)
        final_ratio = self.min_max_dist_ratio(final_points)

        return final_points, final_ratio

    def multi_start_optimization(self) -> np.ndarray:
        """Perform enhanced multi-start optimization with hybrid approach."""
        # Try multiple initialization methods with varied seeds
        init_methods = ['fibonacci', 'sobol', 'random']  
        num_starts = 12  # Increased number of starts for better exploration

        # Optimization methods in order of preference
        optimization_methods = ['L-BFGS-B', 'SLSQP', 'TNC']

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

                    # Phase 1: Coarse optimization to find promising regions 
                    best_local_points = None
                    best_local_ratio = -np.inf
                    
                    for opt_method in optimization_methods:
                        try:
                            if start_idx == 0 and method_idx == 0:
                                # First optimization: coarse with relaxed tolerances
                                optimized_points, current_ratio = self.optimize_single_start(
                                    points, max_iter=300, method=opt_method, coarse=True
                                )
                            else:
                                # Other optimizations: standard
                                optimized_points, current_ratio = self.optimize_single_start(
                                    points, max_iter=300, method=opt_method, coarse=True
                                )

                            if current_ratio > best_local_ratio:
                                best_local_ratio = current_ratio
                                best_local_points = optimized_points.copy()

                        except Exception:
                            continue

                    # Phase 2: Fine-grained optimization on best local solution
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
    optimizer = AdaptiveHybridOptimizer(n_points=14, d=3)
    return optimizer.multi_start_optimization()

# EVOLVE-BLOCK-END