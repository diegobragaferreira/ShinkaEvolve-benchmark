# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import qmc
import time
from typing import Tuple, Optional

class SobolEvolutionOptimizer:
    """An enhanced optimizer using Sobol sequence initialization with adaptive constraints and hybrid refinement."""

    def __init__(self, n_points: int = 14, d: int = 3, seed: int = 42):
        self.n_points = n_points
        self.d = d
        self.seed = seed
        np.random.seed(seed)
        self.best_ratio = -np.inf
        self.best_points = None

    def sobol_initialization(self) -> np.ndarray:
        """Initialize points using 3D Sobol sequence for better space-filling properties."""
        sampler = qmc.Sobol(d=self.d, seed=self.seed)
        points = sampler.random(n=self.n_points)
        # Scale to unit sphere
        points = points * 2 - 1  # Map to [-1, 1]^3
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms
        return points

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

    def adaptive_initialization(self) -> list:
        """Create diverse initial point configurations with improved perturbations."""
        initial_configs = []
        
        # 1. Sobol sequence initialization
        sobol_points = self.sobol_initialization()
        initial_configs.append(sobol_points.copy())
        
        # 2. Fibonacci spiral with noise
        fib_points = self.fibonacci_spiral_sphere()
        np.random.seed(self.seed)
        perturbation = np.random.normal(0, 0.015, (self.n_points, self.d))
        fib_perturbed = fib_points + perturbation
        norms = np.linalg.norm(fib_perturbed, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        fib_perturbed = fib_perturbed / norms
        initial_configs.append(fib_perturbed.copy())
        
        # 3. Additional Fibonacci variants with different noise scales
        for i in range(3):  # 3 more variants
            np.random.seed(self.seed + i + 100)
            perturbation = np.random.normal(0, 0.01, (self.n_points, self.d))
            fib_variant = fib_points + perturbation
            norms = np.linalg.norm(fib_variant, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            fib_variant = fib_variant / norms
            initial_configs.append(fib_variant.copy())
        
        # 4. Random initialization with better spread
        np.random.seed(self.seed + 200)
        random_points = np.random.rand(self.n_points, self.d) * 2 - 1
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        random_points = random_points / norms
        initial_configs.append(random_points.copy())
        
        return initial_configs

    def min_max_dist_ratio(self, points: np.ndarray) -> float:
        """Calculate the ratio of minimum to maximum distance."""
        if len(points) < 2:
            return 0.0

        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0

            # Remove near-zero distances that might occur due to numerical errors
            distances = distances[distances > 1e-12]
            
            if len(distances) == 0:
                return 0.0

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            # Avoid division by zero
            if max_dist <= 0:
                return 0.0

            return min_dist / max_dist
        except Exception:
            return 0.0

    def setup_constraint(self, tightness_factor: float = 1.0) -> dict:
        """Setup constraint dictionary with adaptive tightness."""
        def constraint_sphere(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            norms = np.linalg.norm(points_reshaped, axis=1)
            # Apply adaptive constraint tightening
            return norms - 1.0 * tightness_factor
        
        return {'type': 'eq', 'fun': constraint_sphere}

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere ensuring numerical stability."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def objective_with_regularization(self, points: np.ndarray, lambda_reg: float = 0.15) -> float:
        """Objective function with distance variance regularization."""
        distances = pdist(points)
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -np.inf
            
        # Base ratio
        ratio = d_min / d_max
        
        # Regularization to penalize distance variance
        distance_variance = np.var(distances)
        regularization_term = lambda_reg * distance_variance
        
        # Return negative since we're minimizing
        return -(ratio - regularization_term)

    def adaptive_constraint_tightening_optimize(self, initial_points: np.ndarray,
                                              max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Optimize with adaptive constraint tightening."""
        x0 = initial_points.flatten()
        
        # Define constraint with progressive tightening
        def adaptive_constraint_factory(factor):
            def constraint_func(x_flat):
                points_reshaped = x_flat.reshape(self.n_points, self.d)
                norms = np.linalg.norm(points_reshaped, axis=1)
                return norms - 1.0 * factor
            return constraint_func

        # Initial optimization with relaxed constraints
        result = minimize(
            lambda x: -self.min_max_dist_ratio(x.reshape(self.n_points, self.d)),
            x0,
            method='L-BFGS-B',
            bounds=[(-2, 2) for _ in range(self.n_points * self.d)],
            constraints={'type': 'eq', 'fun': adaptive_constraint_factory(0.9)},
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if not result.success:
            # Try with tighter constraints
            result = minimize(
                lambda x: -self.min_max_dist_ratio(x.reshape(self.n_points, self.d)),
                x0,
                method='L-BFGS-B',
                bounds=[(-2, 2) for _ in range(self.n_points * self.d)],
                constraints={'type': 'eq', 'fun': adaptive_constraint_factory(0.95)},
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
        
        optimized_points = result.x.reshape(self.n_points, self.d)
        final_points = self.normalize_points(optimized_points)
        final_ratio = self.min_max_dist_ratio(final_points)
        
        return final_points, final_ratio

    def hybrid_refinement(self, initial_points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply hybrid refinement with secondary optimization."""
        # First phase: optimize with SLSQP 
        x0 = initial_points.flatten()
        
        result = minimize(
            lambda x: -self.min_max_dist_ratio(x.reshape(self.n_points, self.d)),
            x0,
            method='SLSQP',
            bounds=[(-2, 2) for _ in range(self.n_points * self.d)],
            constraints=self.setup_constraint(),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            refined_points = result.x.reshape(self.n_points, self.d)
            refined_points = self.normalize_points(refined_points)
            ratio_slsqp = self.min_max_dist_ratio(refined_points)
        else:
            refined_points = initial_points.copy()
            ratio_slsqp = self.min_max_dist_ratio(initial_points)
        
        # Second phase: refine with L-BFGS-B for even tighter local search
        x0_lbfgs = refined_points.flatten()
        
        result_lbfgs = minimize(
            lambda x: -self.min_max_dist_ratio(x.reshape(self.n_points, self.d)),
            x0_lbfgs,
            method='L-BFGS-B',
            bounds=[(-2, 2) for _ in range(self.n_points * self.d)],
            constraints=self.setup_constraint(tightness_factor=1.0),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result_lbfgs.success:
            final_points = result_lbfgs.x.reshape(self.n_points, self.d)
            final_points = self.normalize_points(final_points)
            ratio_final = self.min_max_dist_ratio(final_points)
        else:
            final_points = refined_points.copy()
            ratio_final = ratio_slsqp
            
        return final_points, ratio_final

    def multi_start_optimization(self) -> np.ndarray:
        """Perform enhanced multi-start optimization with improved strategies."""
        # Get diverse initial configurations
        initial_configs = self.adaptive_initialization()
        
        # Optimization parameters
        optimization_methods = ['L-BFGS-B', 'SLSQP', 'TNC']
        num_starts = len(initial_configs) * 3  # Increased diversity
        
        for config_idx, initial_points in enumerate(initial_configs):
            # Different seeding for better exploration
            seed_val = self.seed + config_idx * 10
            np.random.seed(seed_val)
            
            # Apply hybrid refinement approach
            refined_points, ratio = self.hybrid_refinement(initial_points)
            
            # Update global best
            if ratio > self.best_ratio:
                self.best_ratio = ratio
                self.best_points = refined_points.copy()
                
            # Also try with adaptive constraint tightening for this configuration
            try:
                tight_refined_points, tight_ratio = self.adaptive_constraint_tightening_optimize(initial_points)
                if tight_ratio > self.best_ratio:
                    self.best_ratio = tight_ratio
                    self.best_points = tight_refined_points.copy()
            except Exception:
                continue
        
        # Return the best solution found
        if self.best_points is not None:
            return self.best_points
        else:
            # Fallback to Sobol initialization
            return self.sobol_initialization()

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = SobolEvolutionOptimizer(n_points=14, d=3, seed=42)
    return optimizer.multi_start_optimization()

# EVOLVE-BLOCK-END