# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import time
from typing import Tuple, List
import warnings

class GeometricOptimizer:
    """Core geometric optimization engine for point distribution."""
    
    def __init__(self, n_points: int = 14, d: int = 3):
        self.n_points = n_points
        self.d = d
        self.best_ratio = -np.inf
        self.best_points = None
        self.eval_time = 0.0
    
    def _normalize_to_unit_sphere(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to lie on unit sphere with numerical stability."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Prevent division by zero
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms
    
    def _calculate_min_max_ratio(self, points: np.ndarray) -> float:
        """Calculate minimum to maximum distance ratio efficiently."""
        if len(points) < 2:
            return 0.0
            
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist <= 0:
                return 0.0
                
            return min_dist / max_dist
        except Exception:
            return 0.0
    
    def _constraint_sphere(self, points_flat: np.ndarray) -> np.ndarray:
        """Constraint function for unit sphere constraint."""
        points = points_flat.reshape(-1, self.d)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    def _objective_function(self, points_flat: np.ndarray) -> float:
        """Objective function that minimizes negative ratio (maximizes ratio)."""
        points = points_flat.reshape(-1, self.d)
        normalized_points = self._normalize_to_unit_sphere(points)
        ratio = self._calculate_min_max_ratio(normalized_points)
        return -ratio

class PointDistributionInitializer:
    """Handles various point initialization strategies."""
    
    @staticmethod
    def fibonacci_spiral(n_points: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
            
        return np.array(points)
    
    @staticmethod
    def sobol_sequence_points(n_points: int, seed: int = 42) -> np.ndarray:
        """Generate points using Sobol sequence with proper sphere mapping."""
        try:
            sampler = qmc.Sobol(d=3, seed=seed)
            points = sampler.random(n=n_points)
            # Map to [-1, 1]^3 first
            points = points * 2 - 1
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1.0, norms)
            return points / safe_norms
        except ImportError:
            # Fallback to random initialization if qmc not available
            return PointDistributionInitializer.random_points(n_points, seed)
    
    @staticmethod
    def random_points(n_points: int, seed: int = 42) -> np.ndarray:
        """Generate random points uniformly distributed on unit sphere."""
        np.random.seed(seed)
        points = np.random.randn(n_points, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return points / safe_norms
    
    @staticmethod
    def perturbed_points(initial_points: np.ndarray, sigma: float = 0.05) -> np.ndarray:
        """Add controlled perturbation to existing point set."""
        noise = np.random.normal(0, sigma, initial_points.shape)
        perturbed = initial_points + noise
        # Re-normalize to unit sphere
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return perturbed / safe_norms

class OptimizationStages:
    """Encapsulates different optimization phases with appropriate parameters."""
    
    @staticmethod
    def coarse_optimization(points: np.ndarray, max_iter: int = 300) -> Tuple[np.ndarray, float]:
        """Coarse global search with relaxed tolerances."""
        bounds = [(-2, 2) for _ in range(len(points) * 3)]
        
        result = minimize(
            lambda x: -GeometricOptimizer()._calculate_min_max_ratio(
                GeometricOptimizer()._normalize_to_unit_sphere(x.reshape(-1, 3))
            ),
            points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        optimized_points = result.x.reshape(-1, 3)
        normalized_points = GeometricOptimizer()._normalize_to_unit_sphere(optimized_points)
        ratio = GeometricOptimizer()._calculate_min_max_ratio(normalized_points)
        
        return normalized_points, ratio
    
    @staticmethod
    def fine_optimization(points: np.ndarray, max_iter: int = 500) -> Tuple[np.ndarray, float]:
        """Fine-grained local optimization with high precision."""
        bounds = [(-2, 2) for _ in range(len(points) * 3)]
        
        result = minimize(
            lambda x: -GeometricOptimizer()._calculate_min_max_ratio(
                GeometricOptimizer()._normalize_to_unit_sphere(x.reshape(-1, 3))
            ),
            points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        optimized_points = result.x.reshape(-1, 3)
        normalized_points = GeometricOptimizer()._normalize_to_unit_sphere(optimized_points)
        ratio = GeometricOptimizer()._calculate_min_max_ratio(normalized_points)
        
        return normalized_points, ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    optimizer = GeometricOptimizer(n_points=14, d=3)
    
    # Initialize strategies with proper parameters
    strategies = [
        ("fibonacci", PointDistributionInitializer.fibonacci_spiral),
        ("sobol", lambda n: PointDistributionInitializer.sobol_sequence_points(n, 42)),
        ("random", lambda n: PointDistributionInitializer.random_points(n, 42))
    ]
    
    # Multi-start approach with diverse initializations
    for strategy_name, strategy_func in strategies:
        if time.time() - start_time > 340:  # Leave buffer time
            break
            
        for attempt in range(8):  # 8 attempts per strategy
            if time.time() - start_time > 340:
                break
                
            try:
                # Generate initial points
                initial_points = strategy_func(14)
                
                # Add slight perturbation to break symmetry
                if strategy_name != "fibonacci" or attempt > 0:
                    initial_points = PointDistributionInitializer.perturbed_points(initial_points, 0.02)
                
                # Normalize to unit sphere
                initial_points = optimizer._normalize_to_unit_sphere(initial_points)
                
                # Phase 1: Coarse optimization
                coarse_points, coarse_ratio = OptimizationStages.coarse_optimization(initial_points)
                
                # Phase 2: Fine optimization
                fine_points, fine_ratio = OptimizationStages.fine_optimization(coarse_points)
                
                # Track best solution
                if fine_ratio > optimizer.best_ratio:
                    optimizer.best_ratio = fine_ratio
                    optimizer.best_points = fine_points.copy()
                    
            except Exception as e:
                warnings.warn(f"Strategy {strategy_name} attempt {attempt} failed: {str(e)}")
                continue
    
    # If no solution found, fallback to Fibonacci spiral
    if optimizer.best_points is None:
        fallback_points = PointDistributionInitializer.fibonacci_spiral(14)
        optimizer.best_points = optimizer._normalize_to_unit_sphere(fallback_points)
    
    # Final refinement with all methods
    refinement_methods = ["L-BFGS-B", "SLSQP", "TNC"]
    for method in refinement_methods:
        if time.time() - start_time > 350:
            break
            
        try:
            bounds = [(-2, 2) for _ in range(14 * 3)]
            
            result = minimize(
                optimizer._objective_function,
                optimizer.best_points.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            refined_points = result.x.reshape(-1, 3)
            refined_points = optimizer._normalize_to_unit_sphere(refined_points)
            refined_ratio = optimizer._calculate_min_max_ratio(refined_points)
            
            if refined_ratio > optimizer.best_ratio:
                optimizer.best_ratio = refined_ratio
                optimizer.best_points = refined_points.copy()
                
        except Exception:
            continue
    
    optimizer.eval_time = time.time() - start_time
    return optimizer.best_points

# EVOLVE-BLOCK-END