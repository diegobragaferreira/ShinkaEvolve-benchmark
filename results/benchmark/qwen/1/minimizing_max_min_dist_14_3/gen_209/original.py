# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import warnings
from typing import Tuple, Optional
import time

class SphericalOptimizer:
    """A specialized optimizer for maximizing min/max distance ratio on a sphere."""

    def __init__(self, num_points: int = 14):
        self.n = num_points
        self.golden_ratio = (1 + np.sqrt(5)) / 2
        self.best_solution = None
        self.best_ratio = -np.inf

    def _calculate_ratio(self, points: np.ndarray) -> Tuple[float, float]:
        """Calculate min/max distance ratio for given points."""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist, max_dist

    def _objective(self, x: np.ndarray) -> float:
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, 3)
        min_dist, max_dist = self._calculate_ratio(points)

        if max_dist == 0:
            return 0
        return -min_dist / max_dist

    def _constraint_func(self, x: np.ndarray) -> np.ndarray:
        """Constraint function ensuring points lie on unit sphere."""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    def _generate_enhanced_fibonacci_points(self) -> np.ndarray:
        """Generate enhanced points using modified Fibonacci spiral with better distribution properties."""
        points = []
        # Use a more sophisticated approach for 14 points
        # Start with Fibonacci spiral but adjust for better spacing
        for i in range(self.n):
            # Modified Fibonacci distribution with better spread
            if i == 0:
                # Place first point at north pole
                phi = 0
                theta = 0
            elif i == self.n - 1:
                # Place last point at south pole
                phi = np.pi
                theta = 0
            else:
                # Distribute remaining points using Fibonacci-like spacing but with adjustments
                phi = np.arccos(1 - 2 * i / (self.n - 1))
                # Add small perturbation to break perfect symmetry and improve optimization
                theta = 2 * np.pi * i / self.golden_ratio + 0.1 * np.sin(i * 0.5)

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])

        points = np.array(points)

        # Add strategic perturbations to improve initial distribution
        # Use a small amount of noise to avoid symmetric local minima
        np.random.seed(42)
        noise_magnitude = 0.02
        points += np.random.normal(0, noise_magnitude, points.shape)

        # Ensure points remain on unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def _perturb_points(self, points: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add controlled noise to points and normalize them."""
        noisy_points = points + np.random.normal(0, noise_level, points.shape)
        return noisy_points / np.linalg.norm(noisy_points, axis=1, keepdims=True)

    def _optimize_with_method(self, x0: np.ndarray, method: str,
                            options: dict) -> Optional[np.ndarray]:
        """Optimize using specified method with error handling."""
        try:
            cons = {'type': 'eq', 'fun': self._constraint_func}
            result = minimize(self._objective, x0, method=method, constraints=cons,
                            options=options)

            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                # Ensure normalization
                optimized_points = optimized_points / np.linalg.norm(optimized_points, axis=1, keepdims=True)
                return optimized_points
        except Exception as e:
            warnings.warn(f"Optimization with {method} failed: {e}")
        return None

    def _evaluate_and_update_best(self, points: np.ndarray) -> bool:
        """Evaluate solution and update best if better."""
        min_dist, max_dist = self._calculate_ratio(points)
        if max_dist > 0:
            ratio = min_dist / max_dist
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_solution = points.copy()
                return True
        return False

    def _generate_spherical_code_points(self) -> np.ndarray:
        """Generate points based on known spherical code configurations for 14 points."""
        # Use an icosahedral-based configuration with modifications for 14 points
        # Vertices of regular icosahedron (12 vertices)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])

        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Add 2 additional points to make 14 total
        # Place them at poles for better symmetry
        additional_points = np.array([[0, 0, 1], [0, 0, -1]])

        # Combine and adjust for better distribution
        points = np.vstack([vertices, additional_points])

        # Apply small random perturbations to break perfect symmetry
        np.random.seed(42)
        noise_magnitude = 0.015
        points += np.random.normal(0, noise_magnitude, points.shape)

        # Ensure normalization
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def _adaptive_restart_strategy(self) -> np.ndarray:
        """Execute multi-phase optimization with adaptive strategies."""

        # Phase 1: Initial population with enhanced Fibonacci + spherical code + perturbations
        initial_points_fib = self._generate_enhanced_fibonacci_points()
        initial_points_code = self._generate_spherical_code_points()

        # Evaluate both initializations
        self._evaluate_and_update_best(initial_points_fib)
        self._evaluate_and_update_best(initial_points_code)

        # Create diverse initial population
        population = [initial_points_fib, initial_points_code]
        for i in range(3):  # Create 3 perturbed versions of each
            np.random.seed(i)
            perturbed_fib = self._perturb_points(initial_points_fib, 0.01)
            perturbed_code = self._perturb_points(initial_points_code, 0.01)
            population.append(perturbed_fib)
            population.append(perturbed_code)

        # Phase 2: Multi-stage optimization with varying strategies
        optimization_phases = [
            # Phase 1: Coarse optimization
            {
                'methods': ['L-BFGS-B', 'SLSQP'],
                'options': {'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 200}
            },
            # Phase 2: Refinement with stricter tolerances
            {
                'methods': ['L-BFGS-B', 'SLSQP', 'TNC'],
                'options': {'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 400}
            },
            # Phase 3: Aggressive refinement
            {
                'methods': ['L-BFGS-B'],
                'options': {'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 600}
            }
        ]

        # Execute optimization phases
        for phase_idx, phase in enumerate(optimization_phases):
            for method in phase['methods']:
                for pop_idx, individual in enumerate(population):
                    np.random.seed(phase_idx * 100 + pop_idx * 10)
                    x0 = individual.flatten()

                    optimized = self._optimize_with_method(x0, method, phase['options'])
                    if optimized is not None:
                        self._evaluate_and_update_best(optimized)

        # Phase 3: Final aggressive refinement
        if self.best_solution is not None:
            final_refinement_options = {
                'ftol': 1e-14,
                'gtol': 1e-14,
                'maxiter': 1000
            }

            # Try multiple refinement attempts
            for attempt in range(3):
                np.random.seed(100 + attempt)
                refined_x0 = self.best_solution + np.random.normal(0, 0.001, self.best_solution.shape)
                refined_x0 = refined_x0 / np.linalg.norm(refined_x0, axis=1, keepdims=True)

                refined = self._optimize_with_method(
                    refined_x0.flatten(), 'L-BFGS-B', final_refinement_options
                )
                if refined is not None:
                    self._evaluate_and_update_best(refined)

        return self.best_solution if self.best_solution is not None else initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Set fixed seed for reproducibility
    np.random.seed(42)

    # Create optimizer instance
    optimizer = SphericalOptimizer(14)

    # Execute optimization
    result = optimizer._adaptive_restart_strategy()

    return result

# EVOLVE-BLOCK-END