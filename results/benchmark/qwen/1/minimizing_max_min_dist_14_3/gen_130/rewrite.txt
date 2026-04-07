# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time
from typing import Tuple, Optional, List, Callable
import warnings

class PointInitializationStrategy:
    """Handles various point initialization strategies for 3D distributions."""
    
    @staticmethod
    def fibonacci_spiral_sphere(n_points: int) -> np.ndarray:
        """Generate points on a sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    @staticmethod
    def sobol_initialization(n_points: int, seed: int = 42) -> np.ndarray:
        """Initialize points using Sobol sequence for better space-filling properties."""
        sampler = qmc.Sobol(d=3, seed=seed)
        points = sampler.random(n=n_points)
        # Scale to unit sphere
        points = points * 2 - 1  # Map to [-1, 1]^3
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms
        return points

    @staticmethod
    def random_initialization(n_points: int, seed: int = 42) -> np.ndarray:
        """Generate random points on unit sphere."""
        np.random.seed(seed)
        points = np.random.randn(n_points, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

class OptimizationPipeline:
    """Manages the complete optimization workflow with staged approaches."""
    
    def __init__(self, n_points: int = 14, d: int = 3):
        self.n_points = n_points
        self.d = d
        self.best_ratio = -np.inf
        self.best_points = None

    def _setup_constraint(self) -> dict:
        """Setup constraint dictionary for optimization."""
        def constraint_sphere(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            norms = np.linalg.norm(points_reshaped, axis=1)
            return norms - 1.0
        
        return {'type': 'eq', 'fun': constraint_sphere}

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit sphere ensuring numerical stability."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms

    def _calculate_ratio(self, points: np.ndarray) -> float:
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

    def _single_optimization_step(self, 
                                initial_points: np.ndarray,
                                method: str = 'L-BFGS-B',
                                max_iter: int = 1000,
                                coarse: bool = False) -> Tuple[np.ndarray, float]:
        """Single optimization step with specified method and tolerances."""
        x0 = initial_points.flatten()

        def objective(x_flat):
            points_reshaped = x_flat.reshape(self.n_points, self.d)
            normalized_points = self._normalize_points(points_reshaped)
            return -self._calculate_ratio(normalized_points)

        constraints = self._setup_constraint()
        bounds = [(-2, 2) for _ in range(self.n_points * self.d)]

        # Adjust tolerances based on coarse/fine mode
        ftol = 1e-8 if coarse else 1e-12
        gtol = 1e-8 if coarse else 1e-12
        maxiter = 300 if coarse else max_iter

        # Use optimization method with appropriate tolerances
        try:
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
            final_points = self._normalize_points(optimized_points)
            final_ratio = self._calculate_ratio(final_points)
            
            return final_points, final_ratio
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            return initial_points, 0.0

    def _refinement_stage(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply multiple refinement techniques to improve solution."""
        refinement_methods = ['L-BFGS-B', 'SLSQP', 'TNC']
        best_points = points.copy()
        best_ratio = self._calculate_ratio(best_points)
        
        for method in refinement_methods:
            try:
                refined_points, refined_ratio = self._single_optimization_step(
                    best_points, method=method, max_iter=500, coarse=False
                )
                
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
                    
            except Exception:
                continue
                
        return best_points, best_ratio

    def run_multi_start_optimization(self) -> np.ndarray:
        """Main optimization loop with multiple start strategies."""
        # Define initial strategies and their parameters
        strategies = [
            ('fibonacci', lambda: PointInitializationStrategy.fibonacci_spiral_sphere(self.n_points)),
            ('sobol', lambda: PointInitializationStrategy.sobol_initialization(self.n_points, 42)),
            ('random', lambda: PointInitializationStrategy.random_initialization(self.n_points, 42))
        ]
        
        # Number of attempts per strategy
        attempts_per_strategy = 5
        
        for strategy_name, strategy_func in strategies:
            for attempt in range(attempts_per_strategy):
                try:
                    # Generate initial points with unique seed
                    seed = hash(f"{strategy_name}_{attempt}") % (2**32)
                    np.random.seed(seed)
                    
                    # Get initial points
                    initial_points = strategy_func()
                    
                    # Add slight perturbation to break symmetry
                    if strategy_name != 'fibonacci' or attempt > 0:
                        perturbation = np.random.normal(0, 0.02, (self.n_points, self.d))
                        initial_points += perturbation
                    
                    # Normalize to unit sphere
                    initial_points = self._normalize_points(initial_points)
                    
                    # Phase 1: Coarse optimization to explore region
                    coarse_points, coarse_ratio = self._single_optimization_step(
                        initial_points, method='L-BFGS-B', max_iter=300, coarse=True
                    )
                    
                    # Phase 2: Fine optimization on promising solution
                    if coarse_ratio > self.best_ratio:
                        fine_points, fine_ratio = self._single_optimization_step(
                            coarse_points, method='L-BFGS-B', max_iter=500, coarse=False
                        )
                        
                        # Final refinement
                        if fine_ratio > self.best_ratio:
                            refined_points, refined_ratio = self._refinement_stage(fine_points)
                            
                            if refined_ratio > self.best_ratio:
                                self.best_ratio = refined_ratio
                                self.best_points = refined_points.copy()
                                
                except Exception as e:
                    warnings.warn(f"Strategy {strategy_name} attempt {attempt} failed: {str(e)}")
                    continue
        
        # Return best solution found or fallback to fibonacci
        if self.best_points is not None:
            return self.best_points
        else:
            return PointInitializationStrategy.fibonacci_spiral_sphere(self.n_points)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = OptimizationPipeline(n_points=14, d=3)
    return optimizer.run_multi_start_optimization()

# EVOLVE-BLOCK-END